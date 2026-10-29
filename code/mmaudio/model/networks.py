from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaudio.ext.rotary_embeddings import compute_rope_rotations
from mmaudio.model.embeddings import TimestepEmbedder
from mmaudio.model.low_level import MLP, ChannelLastConv1d, ConvMLP
from mmaudio.model.transformer_layers import (FinalBlock, JointBlock, JointBlockD,
                                               MMDitSingleBlock)

log = logging.getLogger()


@dataclass
class PreprocessedConditions:
    clip_f: torch.Tensor
    sync_f: torch.Tensor
    text_f: torch.Tensor
    clip_f_c: torch.Tensor
    text_f_c: torch.Tensor
    # Student conditioning (CMOD onset). None when use_student=False.
    student_f: Optional[torch.Tensor] = None
    # Scalogram (wavelet) conditioning. None when use_scalo=False.
    scalo_f: Optional[torch.Tensor] = None


# Partially from https://github.com/facebookresearch/DiT
class MMAudio(nn.Module):

    def __init__(self,
                 *,
                 latent_dim: int,
                 clip_dim: int,
                 sync_dim: int,
                 text_dim: int,
                 hidden_dim: int,
                 depth: int,
                 fused_depth: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 latent_seq_len: int,
                 clip_seq_len: int,
                 sync_seq_len: int,
                 text_seq_len: int = 77,
                 latent_mean: Optional[torch.Tensor] = None,
                 latent_std: Optional[torch.Tensor] = None,
                 empty_string_feat: Optional[torch.Tensor] = None,
                 v2: bool = False,
                 # CMOD Student conditioning (Position B: residual into c_f)
                 use_student: bool = False,
                 student_dim: int = 1536,
                 student_seq_len: int = 392,
                 # Initial value for the student_alpha residual gate. Default 0.0
                 # follows DiT zero-init convention, but in practice the gate
                 # often stays near zero and the student path is ignored.
                 # Set to 1.0 to bypass zero-init and force the student to
                 # contribute from step 0 (lets gradients shape both paths jointly).
                 student_alpha_init: float = 0.0,
                 # Tier-2 auxiliary onset-distillation head (only sensible when
                 # use_student=True AND student variant produces 1-D onset logits)
                 use_aux_onset_loss: bool = False,
                 # Position D: add Student as a 4th modality stream in joint
                 # attention. Values: 'B' (default, residual into c_f), 'D'
                 # (4th stream, no residual), 'BD' (both).
                 student_position: str = 'B',
                 # Scalogram (wavelet) conditioning — independent of student.
                 # Same Position-B injection pattern: project to hidden_dim,
                 # upsample to latent_seq_len, residual-add with scalo_alpha gate.
                 use_scalo: bool = False,
                 scalo_dim: int = 64,
                 scalo_seq_len: int = 392,
                 scalo_alpha_init: float = 0.0,
                 # Tier-2 style scalogram distillation. Predict scalogram from
                 # the model's final latent and MSE against GT. Lets training
                 # benefit from GT-audio scalogram WITHOUT requiring it at
                 # inference (silent-video deployment). Independent of use_scalo.
                 use_scalo_aux_loss: bool = False) -> None:
        super().__init__()

        self.v2 = v2
        self.latent_dim = latent_dim
        self._latent_seq_len = latent_seq_len
        self._clip_seq_len = clip_seq_len
        self._sync_seq_len = sync_seq_len
        self._text_seq_len = text_seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_student = bool(use_student)
        self._student_dim = int(student_dim)
        self._student_seq_len = int(student_seq_len)
        self._student_alpha_init = float(student_alpha_init)
        self.use_scalo = bool(use_scalo)
        self._scalo_dim = int(scalo_dim)
        self._scalo_seq_len = int(scalo_seq_len)
        self._scalo_alpha_init = float(scalo_alpha_init)
        self.use_scalo_aux_loss = bool(use_scalo_aux_loss)
        # student_position: 'B' (residual into c_f, original/default),
        # 'D' (4th joint-attention stream, no residual),
        # 'BD' (both — residual + joint-attention stream).
        sp = str(student_position).upper()
        assert sp in ('B', 'D', 'BD'), f'student_position must be B|D|BD, got {sp}'
        self.student_position = sp if self.use_student else 'B'
        self._student_in_joint = self.use_student and ('D' in self.student_position)
        self._student_in_residual = self.use_student and ('B' in self.student_position)

        if v2:
            self.audio_input_proj = nn.Sequential(
                ChannelLastConv1d(latent_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SiLU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
            )

            self.clip_input_proj = nn.Sequential(
                nn.Linear(clip_dim, hidden_dim),
                nn.SiLU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.sync_input_proj = nn.Sequential(
                ChannelLastConv1d(sync_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SiLU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.text_input_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.SiLU(),
                MLP(hidden_dim, hidden_dim * 4),
            )
        else:
            self.audio_input_proj = nn.Sequential(
                ChannelLastConv1d(latent_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SELU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
            )

            self.clip_input_proj = nn.Sequential(
                nn.Linear(clip_dim, hidden_dim),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.sync_input_proj = nn.Sequential(
                ChannelLastConv1d(sync_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SELU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.text_input_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                MLP(hidden_dim, hidden_dim * 4),
            )

        self.clip_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.global_cond_mlp = MLP(hidden_dim, hidden_dim * 4)
        # each synchformer output segment has 8 feature frames
        self.sync_pos_emb = nn.Parameter(torch.zeros((1, 1, 8, sync_dim)))

        self.final_layer = FinalBlock(hidden_dim, latent_dim)

        if v2:
            self.t_embed = TimestepEmbedder(hidden_dim,
                                            frequency_embedding_size=hidden_dim,
                                            max_period=1)
        else:
            self.t_embed = TimestepEmbedder(hidden_dim,
                                            frequency_embedding_size=256,
                                            max_period=10000)
        if self._student_in_joint:
            # Position D: use 4-stream joint blocks
            self.joint_blocks = nn.ModuleList([
                JointBlockD(hidden_dim,
                            num_heads,
                            mlp_ratio=mlp_ratio,
                            pre_only=(i == depth - fused_depth - 1))
                for i in range(depth - fused_depth)
            ])
        else:
            self.joint_blocks = nn.ModuleList([
                JointBlock(hidden_dim,
                           num_heads,
                           mlp_ratio=mlp_ratio,
                           pre_only=(i == depth - fused_depth - 1))
                for i in range(depth - fused_depth)
            ])

        self.fused_blocks = nn.ModuleList([
            MMDitSingleBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, kernel_size=3, padding=1)
            for i in range(fused_depth)
        ])

        if latent_mean is None:
            # these values are not meant to be used
            # if you don't provide mean/std here, we should load them later from a checkpoint
            assert latent_std is None
            latent_mean = torch.ones(latent_dim).view(1, 1, -1).fill_(float('nan'))
            latent_std = torch.ones(latent_dim).view(1, 1, -1).fill_(float('nan'))
        else:
            assert latent_std is not None
            assert latent_mean.numel() == latent_dim, f'{latent_mean.numel()=} != {latent_dim=}'
        if empty_string_feat is None:
            empty_string_feat = torch.zeros((text_seq_len, text_dim))
        self.latent_mean = nn.Parameter(latent_mean.view(1, 1, -1), requires_grad=False)
        self.latent_std = nn.Parameter(latent_std.view(1, 1, -1), requires_grad=False)

        self.empty_string_feat = nn.Parameter(empty_string_feat, requires_grad=False)
        self.empty_clip_feat = nn.Parameter(torch.zeros(1, clip_dim), requires_grad=True)
        self.empty_sync_feat = nn.Parameter(torch.zeros(1, sync_dim), requires_grad=True)

        # ---- CMOD Student conditioning (Position B: residual into c_f) ----
        # Project D_s -> hidden_dim with the same conv-MLP recipe used for sync.
        # alpha is a learnable scalar gate, init=0 so the model starts identical
        # to the no-Student baseline; alpha grows during training if Student helps.
        if self.use_student:
            self.student_input_proj = nn.Sequential(
                ChannelLastConv1d(self._student_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SELU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )
            self.student_alpha = nn.Parameter(torch.zeros(1))
            self.empty_student_feat = nn.Parameter(
                torch.zeros(1, self._student_dim), requires_grad=True
            )

        # ---- Scalogram (wavelet) conditioning (Position B: residual into c_f) ----
        # Project scalo_dim -> hidden_dim, residual-add with learnable alpha gate.
        # Independent of student — both can be active simultaneously.
        if self.use_scalo:
            self.scalo_input_proj = nn.Sequential(
                ChannelLastConv1d(self._scalo_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SELU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )
            self.scalo_alpha = nn.Parameter(torch.zeros(1))
            self.empty_scalo_feat = nn.Parameter(
                torch.zeros(1, self._scalo_dim), requires_grad=True
            )

        # ---- Tier-2 style scalogram distillation head ----
        # Predicts scalogram from the model's final latent. Supervised against
        # the GT-audio scalogram. Lets the model leverage scalogram info during
        # training without needing scalogram at inference (silent-video OK).
        if self.use_scalo_aux_loss:
            self.scalo_aux_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, self._scalo_dim),
            )

        # ---- Tier-2 auxiliary onset head ----
        # Predicts onset logits from the model's final latent hidden state.
        # Supervised against the Student's onset logits (downsampled to latent_seq_len).
        self.use_aux_onset_loss = bool(use_aux_onset_loss) and self.use_student
        if self.use_aux_onset_loss:
            self.onset_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1),
            )

        self.initialize_weights()
        self.initialize_rotations()

    def initialize_rotations(self):
        base_freq = 1.0
        latent_rot = compute_rope_rotations(self._latent_seq_len,
                                            self.hidden_dim // self.num_heads,
                                            10000,
                                            freq_scaling=base_freq,
                                            device=self.device)
        clip_rot = compute_rope_rotations(self._clip_seq_len,
                                          self.hidden_dim // self.num_heads,
                                          10000,
                                          freq_scaling=base_freq * self._latent_seq_len /
                                          self._clip_seq_len,
                                          device=self.device)

        self.register_buffer('latent_rot', latent_rot, persistent=False)
        self.register_buffer('clip_rot', clip_rot, persistent=False)

    def update_seq_lengths(self, latent_seq_len: int, clip_seq_len: int, sync_seq_len: int) -> None:
        self._latent_seq_len = latent_seq_len
        self._clip_seq_len = clip_seq_len
        self._sync_seq_len = sync_seq_len
        self.initialize_rotations()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.joint_blocks:
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.clip_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.clip_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].bias, 0)
            # Position D: the extra student_block exists only on JointBlockD.
            if hasattr(block, 'student_block'):
                nn.init.constant_(block.student_block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.student_block.adaLN_modulation[-1].bias, 0)
        for block in self.fused_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

        # empty string feat shall be initialized by a CLIP encoder
        nn.init.constant_(self.sync_pos_emb, 0)
        nn.init.constant_(self.empty_clip_feat, 0)
        nn.init.constant_(self.empty_sync_feat, 0)

        # Student residual gate: alpha init from config (default 0.0 = DiT zero-init).
        # 1.0 forces student to contribute from step 0 — useful when the gate
        # would otherwise stay near zero (observed empirically with feat_tcn variant).
        if getattr(self, 'use_student', False):
            nn.init.constant_(self.student_alpha, self._student_alpha_init)
            nn.init.constant_(self.empty_student_feat, 0.0)
        if getattr(self, 'use_scalo', False):
            nn.init.constant_(self.scalo_alpha, self._scalo_alpha_init)
            nn.init.constant_(self.empty_scalo_feat, 0.0)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # return (x - self.latent_mean) / self.latent_std
        return x.sub_(self.latent_mean).div_(self.latent_std)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        # return x * self.latent_std + self.latent_mean
        return x.mul_(self.latent_std).add_(self.latent_mean)

    def preprocess_conditions(
        self,
        clip_f: torch.Tensor,
        sync_f: torch.Tensor,
        text_f: torch.Tensor,
        student_f: Optional[torch.Tensor] = None,
        scalo_f: Optional[torch.Tensor] = None,
    ) -> PreprocessedConditions:
        """
        cache computations that do not depend on the latent/time step
        i.e., the features are reused over steps during inference
        """
        assert clip_f.shape[1] == self._clip_seq_len, f'{clip_f.shape=} {self._clip_seq_len=}'
        assert sync_f.shape[1] == self._sync_seq_len, f'{sync_f.shape=} {self._sync_seq_len=}'
        assert text_f.shape[1] == self._text_seq_len, f'{text_f.shape=} {self._text_seq_len=}'

        bs = clip_f.shape[0]

        # B * num_segments (24) * 8 * 768
        num_sync_segments = self._sync_seq_len // 8
        sync_f = sync_f.view(bs, num_sync_segments, 8, -1) + self.sync_pos_emb
        sync_f = sync_f.flatten(1, 2)  # (B, VN, D)

        # extend vf to match x
        clip_f = self.clip_input_proj(clip_f)  # (B, VN, D)
        sync_f = self.sync_input_proj(sync_f)  # (B, VN, D)
        text_f = self.text_input_proj(text_f)  # (B, VN, D)

        # upsample the sync features to match the audio
        sync_f = sync_f.transpose(1, 2)  # (B, D, VN)
        sync_f = F.interpolate(sync_f, size=self._latent_seq_len, mode='nearest-exact')
        sync_f = sync_f.transpose(1, 2)  # (B, N, D)

        # get conditional features from the clip side
        clip_f_c = self.clip_cond_proj(clip_f.mean(dim=1))  # (B, D)
        text_f_c = self.text_cond_proj(text_f.mean(dim=1))  # (B, D)

        # Student conditioning (Position B): project + upsample to latent rate.
        student_f_out: Optional[torch.Tensor] = None
        if self.use_student:
            if student_f is None:
                # missing-modality (audio-text data) → use learnable empty token
                student_f = self.empty_student_feat.unsqueeze(0).expand(
                    bs, self._student_seq_len, -1
                )
            assert student_f.shape[1] == self._student_seq_len, (
                f'{student_f.shape=} {self._student_seq_len=}'
            )
            student_f = self.student_input_proj(student_f)  # (B, T_s, D)
            student_f = student_f.transpose(1, 2)
            student_f = F.interpolate(
                student_f, size=self._latent_seq_len, mode='nearest-exact'
            )
            student_f_out = student_f.transpose(1, 2)  # (B, N, D)

        # Scalogram conditioning (Position B): project + upsample to latent rate.
        scalo_f_out: Optional[torch.Tensor] = None
        if self.use_scalo:
            if scalo_f is None:
                scalo_f = self.empty_scalo_feat.unsqueeze(0).expand(
                    bs, self._scalo_seq_len, -1
                )
            assert scalo_f.shape[1] == self._scalo_seq_len, (
                f'{scalo_f.shape=} {self._scalo_seq_len=}'
            )
            scalo_f = self.scalo_input_proj(scalo_f)            # (B, T_s, D)
            scalo_f = scalo_f.transpose(1, 2)
            scalo_f = F.interpolate(
                scalo_f, size=self._latent_seq_len, mode='nearest-exact'
            )
            scalo_f_out = scalo_f.transpose(1, 2)                # (B, N, D)

        return PreprocessedConditions(clip_f=clip_f,
                                      sync_f=sync_f,
                                      text_f=text_f,
                                      clip_f_c=clip_f_c,
                                      text_f_c=text_f_c,
                                      student_f=student_f_out,
                                      scalo_f=scalo_f_out)

    def predict_flow(self, latent: torch.Tensor, t: torch.Tensor,
                     conditions: PreprocessedConditions) -> torch.Tensor:
        """
        for non-cacheable computations
        """
        assert latent.shape[1] == self._latent_seq_len, f'{latent.shape=} {self._latent_seq_len=}'

        clip_f = conditions.clip_f
        sync_f = conditions.sync_f
        text_f = conditions.text_f
        clip_f_c = conditions.clip_f_c
        text_f_c = conditions.text_f_c

        latent = self.audio_input_proj(latent)  # (B, N, D)
        global_c = self.global_cond_mlp(clip_f_c + text_f_c)  # (B, D)

        global_c = self.t_embed(t).unsqueeze(1) + global_c.unsqueeze(1)  # (B, D)
        extended_c = global_c + sync_f
        # Position B: residual from CMOD Student feature, gated by alpha (init 0).
        if self._student_in_residual and conditions.student_f is not None:
            extended_c = extended_c + self.student_alpha * conditions.student_f

        # Scalogram conditioning (Position B-equivalent): residual into c_f.
        if self.use_scalo and conditions.scalo_f is not None:
            extended_c = extended_c + self.scalo_alpha * conditions.scalo_f

        if self._student_in_joint and conditions.student_f is not None:
            # Position D: pass student as 4th stream through joint attention.
            student_stream = conditions.student_f
            for block in self.joint_blocks:
                latent, clip_f, text_f, student_stream = block(
                    latent, clip_f, text_f, student_stream,
                    global_c, extended_c, self.latent_rot, self.clip_rot)
        else:
            for block in self.joint_blocks:
                latent, clip_f, text_f = block(latent, clip_f, text_f, global_c, extended_c,
                                               self.latent_rot, self.clip_rot)

        for block in self.fused_blocks:
            latent = block(latent, extended_c, self.latent_rot)

        # Tier-2: predict onset logits from the final hidden BEFORE the flow head.
        aux_onset_logits: Optional[torch.Tensor] = None
        if self.use_aux_onset_loss:
            aux_onset_logits = self.onset_head(latent)  # (B, N=latent_seq_len, 1)

        # Tier-2 style scalogram distillation prediction (also from final hidden).
        aux_scalo_pred: Optional[torch.Tensor] = None
        if self.use_scalo_aux_loss:
            x = latent.transpose(1, 2)                                       # (B, H, N)
            x = F.interpolate(x, size=self._scalo_seq_len, mode='linear',
                              align_corners=False)                             # (B, H, T_w)
            x = x.transpose(1, 2)                                            # (B, T_w, H)
            aux_scalo_pred = self.scalo_aux_head(x)                          # (B, T_w, D_w)

        # should be extended_c; this is a minor implementation error #55
        flow = self.final_layer(latent, global_c)  # (B, N, out_dim), remove t
        # Build aux dict and return tuple when any aux is active. Backward
        # compatibility: if only onset is active, return (flow, onset) as before.
        if self.use_aux_onset_loss and self.use_scalo_aux_loss:
            return flow, {'onset_logits': aux_onset_logits, 'pred_scalo': aux_scalo_pred}
        if self.use_aux_onset_loss:
            return flow, aux_onset_logits
        if self.use_scalo_aux_loss:
            return flow, {'pred_scalo': aux_scalo_pred}
        return flow

    def forward(self, latent: torch.Tensor, clip_f: torch.Tensor, sync_f: torch.Tensor,
                text_f: torch.Tensor, t: torch.Tensor,
                student_f: Optional[torch.Tensor] = None,
                scalo_f: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        latent: (B, N, C)
        vf: (B, T, C_V)
        t: (B,)
        student_f: (B, T_s, D_s) optional — CMOD onset feature for Position B
                   conditioning. Pass None to disable (use empty token).
        scalo_f:   (B, T_w, D_w) optional — wavelet scalogram feature for
                   Position B conditioning. Pass None when use_scalo=False.
        """
        conditions = self.preprocess_conditions(clip_f, sync_f, text_f, student_f, scalo_f)
        out = self.predict_flow(latent, t, conditions)
        # predict_flow returns either flow OR (flow, aux_onset_logits) — preserve
        return out

    def get_empty_string_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_string_feat.unsqueeze(0).expand(bs, -1, -1)

    def get_empty_clip_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_clip_feat.unsqueeze(0).expand(bs, self._clip_seq_len, -1)

    def get_empty_sync_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_sync_feat.unsqueeze(0).expand(bs, self._sync_seq_len, -1)

    def get_empty_student_sequence(self, bs: int) -> torch.Tensor:
        assert self.use_student, 'Student conditioning is disabled'
        return self.empty_student_feat.unsqueeze(0).expand(bs, self._student_seq_len, -1)

    def get_empty_conditions(
            self,
            bs: int,
            *,
            negative_text_features: Optional[torch.Tensor] = None) -> PreprocessedConditions:
        if negative_text_features is not None:
            empty_text = negative_text_features
        else:
            empty_text = self.get_empty_string_sequence(1)

        empty_clip = self.get_empty_clip_sequence(1)
        empty_sync = self.get_empty_sync_sequence(1)
        empty_student = self.get_empty_student_sequence(1) if self.use_student else None
        empty_scalo = (self.empty_scalo_feat.unsqueeze(0).expand(1, self._scalo_seq_len, -1)
                       if self.use_scalo else None)
        conditions = self.preprocess_conditions(empty_clip, empty_sync, empty_text,
                                                empty_student, empty_scalo)
        conditions.clip_f = conditions.clip_f.expand(bs, -1, -1)
        conditions.sync_f = conditions.sync_f.expand(bs, -1, -1)
        conditions.clip_f_c = conditions.clip_f_c.expand(bs, -1)
        if negative_text_features is None:
            conditions.text_f = conditions.text_f.expand(bs, -1, -1)
            conditions.text_f_c = conditions.text_f_c.expand(bs, -1)
        if conditions.student_f is not None:
            conditions.student_f = conditions.student_f.expand(bs, -1, -1)
        if conditions.scalo_f is not None:
            conditions.scalo_f = conditions.scalo_f.expand(bs, -1, -1)

        return conditions

    def ode_wrapper(self, t: torch.Tensor, latent: torch.Tensor, conditions: PreprocessedConditions,
                    empty_conditions: PreprocessedConditions, cfg_strength: float) -> torch.Tensor:
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)

        def _flow(c):
            out = self.predict_flow(latent, t, c)
            return out[0] if isinstance(out, tuple) else out

        if cfg_strength < 1.0:
            return _flow(conditions)
        return cfg_strength * _flow(conditions) + (1 - cfg_strength) * _flow(empty_conditions)

    def load_weights(self, src_dict) -> None:
        if 't_embed.freqs' in src_dict:
            del src_dict['t_embed.freqs']
        if 'latent_rot' in src_dict:
            del src_dict['latent_rot']
        if 'clip_rot' in src_dict:
            del src_dict['clip_rot']

        # Allowed-to-be-missing prefixes when warm-starting a NEW feature from a
        # ckpt that didn't have it. Strict=False with a whitelist (so any other
        # missing key still raises). Also strip incoming keys we no longer have.
        allow_missing_prefixes = []
        if self.use_aux_onset_loss:
            allow_missing_prefixes.append('onset_head.')
        if self._student_in_joint:
            # Position D adds a per-block student_block sub-module.
            allow_missing_prefixes.append('joint_blocks.')  # filtered below

        src_has_aux = any(k.startswith('onset_head.') for k in src_dict)
        src_has_student = any(k.startswith('student_') or k == 'empty_student_feat' for k in src_dict)
        src_has_student_block = any('.student_block.' in k for k in src_dict)
        need_relaxed_load = (
            (self.use_aux_onset_loss and not src_has_aux)
            or (self._student_in_joint and not src_has_student_block)
            or (self.use_student and not src_has_student)
        )

        if not self.use_aux_onset_loss:
            src_dict = {k: v for k, v in src_dict.items()
                        if not k.startswith('onset_head.')}
        if not self.use_scalo_aux_loss:
            src_dict = {k: v for k, v in src_dict.items()
                        if not k.startswith('scalo_aux_head.')}
        if not self._student_in_joint:
            src_dict = {k: v for k, v in src_dict.items()
                        if not ('.student_block.' in k and k.startswith('joint_blocks.'))}

        if need_relaxed_load:
            incompatible = self.load_state_dict(src_dict, strict=False)
            def _is_ok(k):
                if k.startswith('onset_head.') and self.use_aux_onset_loss:
                    return True
                if (k.startswith('joint_blocks.') and '.student_block.' in k
                        and self._student_in_joint):
                    return True
                # CMOD Position B params (zero-init) — expected missing from vanilla MMAudio ckpt
                if self.use_student and (k.startswith('student_') or k == 'empty_student_feat'):
                    return True
                return False
            bad_missing = [k for k in incompatible.missing_keys if not _is_ok(k)]
            assert not bad_missing, f'Unexpected missing keys: {bad_missing}'
            assert not incompatible.unexpected_keys, \
                f'Unexpected extra keys: {incompatible.unexpected_keys}'
            return

        self.load_state_dict(src_dict, strict=True)

    @property
    def device(self) -> torch.device:
        return self.latent_mean.device

    @property
    def latent_seq_len(self) -> int:
        return self._latent_seq_len

    @property
    def clip_seq_len(self) -> int:
        return self._clip_seq_len

    @property
    def sync_seq_len(self) -> int:
        return self._sync_seq_len


def small_16k(**kwargs) -> MMAudio:
    num_heads = 7
    return MMAudio(latent_dim=20,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=12,
                   fused_depth=8,
                   num_heads=num_heads,
                   latent_seq_len=250,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   **kwargs)


def small_44k(**kwargs) -> MMAudio:
    num_heads = 7
    return MMAudio(latent_dim=40,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=12,
                   fused_depth=8,
                   num_heads=num_heads,
                   latent_seq_len=345,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   **kwargs)


def medium_44k(**kwargs) -> MMAudio:
    num_heads = 14
    return MMAudio(latent_dim=40,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=12,
                   fused_depth=8,
                   num_heads=num_heads,
                   latent_seq_len=345,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   **kwargs)


def large_44k(**kwargs) -> MMAudio:
    num_heads = 14
    return MMAudio(latent_dim=40,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=21,
                   fused_depth=14,
                   num_heads=num_heads,
                   latent_seq_len=345,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   **kwargs)


def large_44k_v2(**kwargs) -> MMAudio:
    num_heads = 14
    return MMAudio(latent_dim=40,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=21,
                   fused_depth=14,
                   num_heads=num_heads,
                   latent_seq_len=345,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   v2=True,
                   **kwargs)


def get_my_mmaudio(name: str, **kwargs) -> MMAudio:
    if name == 'small_16k':
        return small_16k(**kwargs)
    if name == 'small_44k':
        return small_44k(**kwargs)
    if name == 'medium_44k':
        return medium_44k(**kwargs)
    if name == 'large_44k':
        return large_44k(**kwargs)
    if name == 'large_44k_v2':
        return large_44k_v2(**kwargs)

    raise ValueError(f'Unknown model name: {name}')


if __name__ == '__main__':
    network = get_my_mmaudio('small_16k')

    # print the number of parameters in terms of millions
    num_params = sum(p.numel() for p in network.parameters()) / 1e6
    print(f'Number of parameters: {num_params:.2f}M')
