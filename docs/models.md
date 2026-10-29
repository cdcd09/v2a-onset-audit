# Model architecture and weight utilities

The [MMAudio-derived architecture](../code/mmaudio/) includes the scalar student-conditioning
and auxiliary-onset modules used in this study. Constructor settings are in
[`configs/model_configs.json`](../configs/model_configs.json).

Model weights are part of the planned companion data release, not this Git repository.
The following tools require those weights or your own compatible local checkpoint.

## Install optional dependencies

```bash
python -m pip install -e '.[models]'
```

Use a PyTorch 2.6.0 build suitable for your machine. Schema checks run on CPU and do not
require a GPU. The preparation host used PyTorch 2.6.0+cu124 for CPU-only checks.

## Validate a checkpoint

```bash
python -m mmaudio.utils.release_weights \
  --weights /path/to/curated_150000_ema_sigma005.safetensors --run curated \
  --out outputs/curated_weight_schema.json
```

Use the exact filename in the data package's weight manifest. The utility checks model
state keys and shapes on the PyTorch meta device and hashes the file. This is a schema check,
not an inference-quality test. Programmatic loading is available through
`load_model(path, model_config("configs/model_configs.json", run))` in
`mmaudio.utils.release_weights`.

The student and fullstack configurations use `student_alpha_init=1.0`,
`freeze_student_with_sync=true`, and a freeze trigger at 20K. Loading weights restores the
trained gate value. A gradient-zeroing hook under AdamW does not imply immediate numerical
freezing; the paper distinguishes raw parameter checks from EMA behavior.

## Inference conditions

Run IDs are `vanilla`, `curated`, `syncfreeze`, `stage1freeze`, and `fullstack`.
`stage1freeze` uses a student-capable architecture, but its historical GH/V266 generated-audio
grid did not inject onset logits. `stage1freeze_logit` is a separate same-weight inference
control. `fullstack` uses logits. Preserve these distinctions when comparing conditions.

`mmaudio.utils.latent_inference.sample_latents` provides a 25-step Euler sampler for
already-computed CLIP, Synchformer, text, and optional student features. It returns audio
latents. Feature extractors, the audio codec/vocoder, and the onset-student predictor are
separate upstream assets. Complete raw-video-to-waveform inference has not been verified
with this repository; the generated-audio package supports evaluation independently.

## EMA export

```bash
python -m mmaudio.utils.ema_export --help
```

The utility supports direct tensor-state conversion and optional post-hoc EMA synthesis.
For synthesis, install `python -m pip install -e '.[ema]'` and provide the original EMA basis
checkpoints plus an explicit schedule (`sigma_rels`, `update_every`, `checkpoint_every`,
`checkpoint_folder`). A final inference checkpoint alone is insufficient. Use trusted local
checkpoints, and retain the extraction method recorded in the asset manifest.

The study has one run per configuration. Intervention runs share a training seed; the vanilla
reference differs in seed and batch size. The included utilities do not constitute a complete
training pipeline. Original project attributions are in [licenses/README.md](licenses/README.md).
