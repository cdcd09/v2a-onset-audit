"""Feature-conditioned latent sampling using the released architecture.

The caller supplies CLIP, Synchformer and text features, plus onset-student
logits only for conditions that use them. Encoding raw video and decoding audio
require separately licensed upstream feature extractors and codec weights.
"""
from functools import partial
import torch
from mmaudio.model.flow_matching import FlowMatching


@torch.inference_mode()
def sample_latents(model, clip_features, sync_features, text_features, *,
                   student_logits=None, seed=0, num_steps=25, cfg_strength=4.5):
    """Return unnormalized audio latents; this is not a waveform API."""
    batch = clip_features.shape[0]
    conditions = model.preprocess_conditions(clip_features, sync_features,
                                             text_features, student_logits)
    empty = model.get_empty_conditions(batch)
    generator = torch.Generator(device=clip_features.device).manual_seed(seed)
    x0 = torch.randn((batch, model.latent_seq_len, model.latent_dim),
                     device=clip_features.device, dtype=clip_features.dtype,
                     generator=generator)
    flow = partial(model.ode_wrapper, conditions=conditions,
                   empty_conditions=empty, cfg_strength=cfg_strength)
    return model.unnormalize(FlowMatching(num_steps=num_steps).to_data(flow, x0))
