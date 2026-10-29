# Video-to-audio onset trajectory audit

Reproducibility code for **Training Refines Onset Placement but Erodes Event Correspondence: A Full-Trajectory Audit of Video-to-Audio Onset Alignment** (WACV paper 2143).

This repository contains the reference onset scorer, detector-free envelope measurements, annotation audits, figure scripts, and the model architecture used by the prepared inference weights.

Code: [cdcd09/v2a-onset-audit](https://github.com/cdcd09/v2a-onset-audit).

**Data release status (2026-10-29):** The companion dataset is prepared and its file hashes have been verified locally, but it is not yet publicly hosted. Its Hugging Face link will be added here after upload. Commands requiring generated audio or model weights need that separate data package. No paper DOI has been assigned here.

## Install

The evaluation was checked on Linux x86_64 with CPython **3.11.11**. Run commands from this repository root. An editable installation makes the module commands work without changing into `code/`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
```

`requirements.txt` pins the numerical packages used for verification. Other operating systems and dependency versions have not been checked. The optional model tools require `python -m pip install -e '.[models]'`; choose the PyTorch 2.6.0 build appropriate for your machine. CPU schema checks do not require a GPU. The verification host had PyTorch 2.6.0+cu124, but these checks allocated no CUDA tensors. Post-hoc EMA synthesis additionally requires `python -m pip install -e '.[ema]'` and the original EMA basis checkpoints.

## Assets and paths

After the companion dataset is published, download it using the link in the release status above. The following commands assume that package is available locally. The setup helper uses only the Python standard library. By default it installs benchmark metadata, GH test/validation IDs, and GH onset annotations; it does **not** extract generated audio. Choose audio shards explicitly, using their basenames in `assets/audio_manifest.json`:

```bash
python scripts/setup_data.py --release-root /path/to/downloaded-release --data-root data
python scripts/setup_data.py --release-root /path/to/downloaded-release --data-root data \
  --list-audio-shards

# These two shards contain the vanilla 10K and 150K endpoints used below.
python scripts/setup_data.py --release-root /path/to/downloaded-release --data-root data \
  --audio-shard audio_vanilla_00.tar --audio-shard audio_vanilla_02.tar
export V2A_DATA_ROOT="$PWD/data"
```

The helper verifies metadata/index hashes, archive hashes, exact member lists, and every extracted file's SHA-256. It rejects path traversal, archive links/special files, and existing destination symlinks. Existing identical files are verified and reused; conflicting files fail without being overwritten. `data/setup_report.json` records the selected shards and installed counts. The bundled `configs/setup_manifest.json` pins the matching data-release revision. Reference audio and source videos must be obtained under their original dataset terms; they are not required for onset scoring on generated audio.

```text
data/
  greatest_hits/
    test.txt                         # 244 clip IDs
    onsets/<id>_times.txt             # all 244 annotations
    audio/<id>.wav                   # optional for ETA reference/xcorr/control
  vggsound/
    benchmark_266.csv                # video_id, query_label, onset_times, n_onsets
    audio/<id>.wav                   # optional reference audio
  generated/
    vanilla_10000/gh/<id>.flac
    vanilla_150000/gh/<id>.flac
    curated_150000/gh/<id>.flac
    syncfreeze_150000/gh/<id>.flac
    stage1freeze_150000/gh/<id>.flac
    fullstack_150000/gh/<id>.flac
    <source_run>_<step>/v266/<id>.flac
```

```bash
export V2A_DATA_ROOT="$PWD/data"
```

`V2A_GH_ROOT`, `V2A_VGG_ROOT`, and `V2A_GEN_ROOT` can override individual directories. Source run IDs are `vanilla`, `curated`, `syncfreeze`, `stage1freeze`, and `fullstack`; use the asset registry for their manuscript display labels and exact conditions. `stage1freeze` uses a student-capable architecture, but its historical main generated-audio grid did **not** inject onset logits at inference. `stage1freeze_logit` is a separate inference control using the same architecture and weights. A directory name alone does not establish the inference condition.

The CLI validates the entire manifest before scoring. Missing annotations/audio, duplicate IDs, malformed times, and unexpected audio IDs fail explicitly. An empty scoring window is reported separately and is never inferred from a missing file. Corrupt audio raises an error during decoding. Original annotation order and repeated timestamps are preserved because changing either can change greedy matching and counts.

| Input | Full manifest | Eligible first 8 s windows | Reference events in those windows |
|---|---:|---:|---:|
| Greatest Hits test | 244 | 243 | 2,612 |
| In-the-wild benchmark | 266 | 263 | 1,826 |

Greatest Hits clip `2015-03-12-18-54-21` has its first reference onset at 9.668375 s, outside the scoring window. The three excluded in-the-wild clips are listed in `results/benchmark_stats.json`; their full clips contain onsets. All 266 full in-the-wild clips contain annotated onsets (2,225 entries, 100 classes). This benchmark does not measure false positives on globally onset-free clips. Human agreement uses 265 complete annotation pairs, of which 217 are marked usable by both lanes; the one incomplete pair remains explicit in the output.

## Reproduce event scores and threshold sensitivity

```bash
python -m eval.onset_metrics \
  --gen_dir "$V2A_DATA_ROOT/generated/vanilla_10000/gh" \
  --split gh --out outputs/vanilla_10000_onset.json

python -m eval.delta_sweep --workers 2 --check-reference \
  --out outputs/delta_sweep.json

python -m eval.onset_metrics \
  --gen_dir "$V2A_DATA_ROOT/generated/syncfreeze_150000/v266" \
  --split wild --out outputs/syncfreeze_150000_wild.json
```

The detector loads mono audio at **22,050 Hz** using the pinned librosa environment. Its historical non-exhaustive NMS implementation is deliberately retained for score compatibility. Event matching uses the original greedy one-to-one procedure and strict `abs(prediction - reference) < tolerance`; the tolerances are 50, 100, and 300 ms. Default peak-picking delta is 0.3 for Greatest Hits and 0.07 for the in-the-wild benchmark. The 16 kHz constants used by the envelope instruments do not change this detector rate.

At delta 0.3, `--check-reference` requires these six rounded values:

| Source run | F1@50 ms | F1@100 ms | F1@300 ms |
|---|---:|---:|---:|
| vanilla 10K | 0.1808 | 0.4579 | 0.7946 |
| vanilla 150K | 0.2079 | 0.4987 | 0.7640 |

The sweep evaluates delta 0.15, 0.20, 0.30, 0.40, and 0.50 at these two endpoints. It writes JSON and CSV, input/exclusion counts, code hashes, and package versions. This checks sensitivity for two endpoints; it does not validate every trajectory point or establish perceptual event presence.

## Reproduce metadata and human agreement

```bash
python -m analysis.benchmark_stats --out outputs/benchmark_stats.json
python -m analysis.annotation_agreement \
  --csv results/annotation_lanes.csv --out outputs/annotation_agreement.json
```

The reference JSON files are in `results/`. Human agreement is distinct from detector-quality Table 2: these use different populations and matching protocols. This repository does not reconstruct the historical zero-shot baseline or claim a fully controlled detector-component ablation.

## Detector-free measurements and figures

These instruments use 16 kHz audio and 10 ms envelope frames. ETA measures responses at reference onset times and control anchors at least 500 ms away; `--keep_curves` saves the averaged waveforms. ETA additionally requires a complete ±300 ms window, so the checked 10K run averages 2,503 eligible anchors across 243 clips; its anchor count differs from the 2,612 references used by event scoring. Cross-correlation additionally needs all reference WAV files. The background control needs Greatest Hits reference WAVs and a generated-audio directory.

```bash
python -m instrument.eta \
  --gen_dir "$V2A_DATA_ROOT/generated/vanilla_10000/gh" --split gh \
  --workers 2 --keep_curves --out outputs/eta_10000.json
python -m analysis.collect_eta_curves \
  --input 'vanilla 10K=outputs/eta_10000.json' --out outputs/eta_curves.json
python -m figures.plot_eta_curves --curves outputs/eta_curves.json --out outputs/eta.pdf

python -m eval.envelope_xcorr \
  --gen_dir "$V2A_DATA_ROOT/generated/vanilla_10000/gh" --split gh \
  --out outputs/xcorr_10000.json

PYTHONHASHSEED=0 python -m instrument.background_control \
  --noise_dir "$V2A_DATA_ROOT/generated/vanilla_150000/gh" \
  --targets 0.089 0.143 0.191 --workers 2 --out outputs/background.json
```

The white-noise control retains the original per-ID Python-hash seed, so set `PYTHONHASHSEED=0` when using `--noise white`. Stored reference curve/control results are included; the current verification report distinguishes recomputed results from retained historical outputs.

## Released weights and architecture

`code/mmaudio/` contains the actual experiment architecture derived from MMAudio, including the scalar student-conditioning and auxiliary-onset modules needed by `stage1freeze` and `fullstack`. `configs/model_configs.json` defines the exact constructor settings. The primary student and fullstack configurations use `student_alpha_init=1.0`, with `freeze_student_with_sync=true` and the freeze trigger at 20K; loading the checkpoint restores its trained gate value. The optional tools can validate and strictly load the released safetensors without silently initializing missing learned weights:

```bash
python -m mmaudio.utils.release_weights \
  --weights /path/to/curated_150000_ema_sigma005.safetensors --run curated \
  --out outputs/curated_weight_schema.json
```

Use the actual filename from the asset weight manifest. Validation compares every model state key and shape on the PyTorch meta device, then hashes the file. A successful schema check is not a numerical inference-quality test. Programmatic loading is available as `load_model(path, model_config("configs/model_configs.json", run))` in `mmaudio.utils.release_weights`.

`mmaudio.utils.latent_inference.sample_latents` provides the 25-step Euler latent sampler for already-computed CLIP, Synchformer, text, and optional student features. It returns audio latents, not a waveform. The feature extractors, audio codec/vocoder, and onset-student predictor are separate upstream assets; they are not bundled here. Raw-video-to-waveform inference and a complete training rerun have **not** been verified with this trimmed repository. The released generated audio supports the tested evaluation commands independently of those assets.

`python -m mmaudio.utils.ema_export --help` documents direct tensor-state conversion and optional post-hoc synthesis. Synthesis needs the original EMA basis checkpoints and an explicit JSON schedule with `sigma_rels`, `update_every`, `checkpoint_every`, and `checkpoint_folder`; a final inference checkpoint is insufficient. Use only trusted local checkpoint files. Representative inference weights preserve their exact extraction method in the asset manifest; do not treat direct-basis extraction and post-hoc synthesis as interchangeable.

The study used one run per configuration. Intervention runs share their training seed; the vanilla reference differs in seed and batch size. Gradient-zeroing hooks beginning at 20K are not equivalent to immediate numerical freezing under AdamW. Detailed training recipes and parameter-audit evidence accompany the paper and asset manifests; this code snapshot does not claim end-to-end training reproducibility.

## Attribution and scope

See `CITATION.cff`, `LICENSE`, and `THIRD_PARTY_NOTICES.md`. The onset detector/NMS adaptation retains attribution to [CondFoleyGen](https://github.com/XYPB/CondFoleyGen); the model snapshot is derived from [MMAudio](https://github.com/hkchengrex/MMAudio). Their upstream MIT notices are included in `third_party/`. Greatest Hits annotations in the companion asset package originate from the [Greatest Hits project](https://andrewowens.com/vis/), under its CC BY 4.0 dataset terms. Source media and external model assets retain their own terms.

The repository reproduces the event scorer and selected audits. It does not implement FD feature extraction, the prior work's frame-onset metric, every training intervention, or the complete pseudo-label generation ensemble. `verification/` records the checks run on the H200 preparation host: seven regression tests, all ten threshold conditions, exact metadata/agreement and 10K ETA-curve reproduction, one cross-correlation run, schema checks for all 12 prepared public weight files, and a strict CPU load of the fullstack 150K model. The baseline measurement dependencies and editable package were also installed in a fresh isolated environment, where all seven tests passed. Optional model dependencies were checked on that host; complete raw-video inference was not tested. Data-hosting status is reported at the top of this README.
