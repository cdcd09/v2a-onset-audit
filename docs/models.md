# Model architecture and weight utilities

The [MMAudio-derived architecture](../code/mmaudio/) includes the scalar student-conditioning
and auxiliary-onset modules used in this study. Constructor settings are in
[`configs/model_configs.json`](../configs/model_configs.json).

Model weights are not included in this compact release. The following utilities are
provided for users with their own compatible local checkpoints; the released predictions,
features, and two generated-audio endpoints can be analyzed without weights.

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

Replace the example path with your compatible checkpoint. The utility checks model
state keys and shapes on the PyTorch meta device and hashes the file. This is a schema check,
not an inference-quality test. Programmatic loading is available through
`load_model(path, model_config("configs/model_configs.json", run))` in
`mmaudio.utils.release_weights`.

The student and fullstack configurations use `student_alpha_init=1.0`,
`freeze_student_with_sync=true`, and a freeze trigger at 20K. Loading weights restores the
trained gate value. For the synchronization-only freeze, raw-weight audits establish
effective freezing under AdamW: after the 20K–40K transient, all six target tensors are
identical at 40K, 70K, 100K, and 150K. The cached evaluation EMA (`sigma=0.05`) changes
by at most `2.35e-8` across the retained 40K/150K pair **before BF16 conversion**;
this comparison concerns the six freeze-target tensors, not the entire model.

## Reported training settings

All five Table 4 configurations use MMAudio-S-16kHz trained from scratch and are compared
every 10K steps over 10K–150K with evaluation EMA `sigma=0.05` and CFG 4.5.

| Paper configuration | Run ID | Seed | Effective batch | Training intervention |
| --- | --- | --- | --- | --- |
| Unmodified | `vanilla` | Different from the intervention runs | 504 | Unmodified reference |
| + curation | `curated` | 42 | 512 | Curated training corpus |
| + curation + injection | `stage1freeze` | 42 | 512 | Position-B student conditioning; synchronization/student gradient suppression from 20K |
| + curation + injection + auxiliary loss | `fullstack` | 42 | 512 | Same conditioning; auxiliary onset-loss weight 0.1 |
| + curation + freeze | `syncfreeze` | 42 | 512 | Synchronization-only gradient suppression from 20K |

These settings describe the paper's runs. The compact repository provides model and
evaluation utilities, not a training entry point. The fully matched freeze comparison
is `syncfreeze` versus `curated`; use the inference conditions below for the injection rows.

## Inference conditions

Run IDs are `vanilla`, `curated`, `syncfreeze`, `stage1freeze`, and `fullstack`.
`stage1freeze` uses a student-capable architecture, but its historical GH/V266 generated-audio
grid did not inject onset logits. Its VGGSound distributional-quality cache did use logits:
these two metric axes share weights but differ in inference conditioning.
`inference_student_logits` in the configuration file describes the GH/V266 condition.
`stage1freeze_logit` is a separate same-weight inference control. `fullstack` uses logits
throughout. Preserve these distinctions when comparing conditions.

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
checkpoints, and record the extraction method alongside your exported file.

For a compatible local EMA basis and its recorded schedule:

```bash
python -m pip install -e '.[models,ema]'
python -m mmaudio.utils.ema_export \
  --input ema_schedule.json --synthesize --run syncfreeze --sigma 0.05 --step 150000 \
  --out outputs/syncfreeze_150000_ema_sigma005.safetensors
```

`ema_schedule.json` must contain the original schedule values and the basis directory;
it is not supplied as a generic training recipe. Here `--step` is the EMA basis step,
which must correspond to the requested paper checkpoint. The script validates exported
keys and shapes; synthesis additionally depends on the caller's complete EMA basis.

Original project attributions are in [licenses/README.md](licenses/README.md).
