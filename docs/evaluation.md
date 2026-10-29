# Data preparation and evaluation

Run all commands from the repository root after the installation in the [README](../README.md).
Download the [companion dataset](https://huggingface.co/datasets/junhob/v2a-onset-audit).
Commands below use `RELEASE_ROOT` for that download and `V2A_DATA_ROOT` for extracted inputs.

## Prepare data

```bash
export RELEASE_ROOT=/path/to/downloaded-release
export V2A_DATA_ROOT="$PWD/data"

python scripts/setup_data.py --release-root "$RELEASE_ROOT" --data-root "$V2A_DATA_ROOT"
python scripts/setup_data.py --release-root "$RELEASE_ROOT" --data-root "$V2A_DATA_ROOT" \
  --list-audio-shards

# Vanilla 10K and 150K endpoint audio used in the examples below.
python scripts/setup_data.py --release-root "$RELEASE_ROOT" --data-root "$V2A_DATA_ROOT" \
  --audio-shard audio_vanilla_10000.tar --audio-shard audio_vanilla_150000.tar
```

The default setup installs benchmark annotations and GH validation/test IDs, without generated
audio. Select audio archives by basename from the data package's `assets/audio_manifest.json`.
The helper checks hashes and archive contents before extraction. It reuses identical files and
rejects conflicting files. `configs/setup_manifest.json` identifies the compatible data revision.

```text
data/
├── greatest_hits/
│   ├── test.txt
│   ├── onsets/<id>_times.txt
│   └── audio/<id>.wav            # optional reference audio
├── vggsound/
│   ├── benchmark_266.csv
│   └── audio/<id>.wav            # optional reference audio
└── generated/
    ├── vanilla_10000/gh/<id>.flac
    ├── vanilla_150000/gh/<id>.flac
    └── <run>_<step>/v266/<id>.flac
```

`V2A_GH_ROOT`, `V2A_VGG_ROOT`, and `V2A_GEN_ROOT` can override individual directories.
Original videos and recorded audio are not redistributed. Obtain GH media from the
[original project](https://andrewowens.com/vis/). VGGSound clips are identified by
`clip_id` (file stem), `youtube_id`, and `segment_start_s` in the companion dataset
(`metadata/v266_clip_metadata.csv`); see the
[VGGSound repository](https://github.com/hche11/VGGSound). Follow the original media terms.
Source recordings are not needed to score onsets in the generated audio.

## Event-onset scores

```bash
python -m eval.onset_metrics \
  --gen_dir "$V2A_DATA_ROOT/generated/vanilla_10000/gh" \
  --split gh --out outputs/vanilla_10000_onset.json

python -m eval.delta_sweep --workers 2 --check-reference --reference-atol 0.001 \
  --out outputs/delta_sweep.json
```

Both endpoint archives contain GH and V266. For V266, use `--split wild` with
`generated/vanilla_10000/v266` or `generated/vanilla_150000/v266`. Only these two
conditions include waveforms; all 99 conditions include detected onsets and envelope features.

The event detector uses mono **22,050 Hz** audio, the original non-exhaustive NMS, and
greedy one-to-one matching with strict `abs(prediction - reference) < tolerance`.
Tolerances are 50, 100, and 300 ms. Default peak-picking delta is 0.3 for GH and 0.07 for V266.
Missing annotations, missing audio, duplicate IDs, and malformed times cause an error.
Empty scoring windows are recorded separately; they are not inferred from missing files.

| Test set | Clips | Nonempty first-8-second windows | Onsets in those windows |
|---|---:|---:|---:|
| Greatest Hits | 244 | 243 | 2,612 |
| V266 | 266 | 263 | 1,826 |

All 266 full V266 clips contain annotated onsets (2,225 in total, across 100 classes).
These event scores exclude empty-reference windows and do not measure false positives on
globally onset-free clips.

At delta 0.3, the endpoint check expects:

| Run | F1@50 ms | F1@100 ms | F1@300 ms |
|---|---:|---:|---:|
| vanilla 10K | 0.1808 | 0.4579 | 0.7946 |
| vanilla 150K | 0.2079 | 0.4987 | 0.7640 |

The sweep checks delta 0.15, 0.20, 0.30, 0.40, and 0.50 at these two endpoints.
`--reference-atol 0.001` permits small numerical differences between platforms;
the report retains both measured values and exact-match status. Omit it for exact
agreement with the rounded reference values. On Windows with the pinned packages,
the 150K F1@50 ms was 0.2071 (difference 0.0008); the other five values matched.
The six V266 endpoint F1 values differed from the archived grid by at most 0.0009.
It does not validate every trajectory point or measure perceptual event presence.

## Benchmark statistics and annotation agreement

```bash
python -m analysis.benchmark_stats --out outputs/benchmark_stats.json
python -m analysis.annotation_agreement \
  --csv "$RELEASE_ROOT/metadata/annotation_lanes.csv" \
  --out outputs/annotation_agreement.json
python -m figures.plot_class_distribution \
  --stats outputs/benchmark_stats.json --out outputs/class_distribution.pdf
python -m figures.plot_annotation_agreement \
  --csv "$RELEASE_ROOT/metadata/annotation_lanes.csv" \
  --out outputs/annotation_agreement.pdf
```

Agreement uses 265 complete annotation pairs, with 217 clips accepted by both lanes;
one incomplete pair is reported separately. It measures full-clip annotation agreement,
with a different population and matching protocol from detector-quality Table 2.

## Detector-free measurements

These tools use 16 kHz audio and 10 ms envelope frames. Event-triggered averaging (ETA)
compares responses at reference onsets with control anchors at least 500 ms away.
Its complete ±300 ms window requirement gives 2,503 eligible anchors across 243 GH clips
for the checked 10K run, distinct from the event scorer's 2,612 reference onsets.

```bash
python -m instrument.eta \
  --gen_dir "$V2A_DATA_ROOT/generated/vanilla_10000/gh" --split gh \
  --workers 2 --keep_curves --out outputs/eta_10000.json
python -m analysis.collect_eta_curves \
  --input 'vanilla 10K=outputs/eta_10000.json' --out outputs/eta_curves.json
python -m figures.plot_eta_curves --curves outputs/eta_curves.json --out outputs/eta.pdf

# The following two commands additionally require GH reference WAV files.
python -m eval.envelope_xcorr \
  --gen_dir "$V2A_DATA_ROOT/generated/vanilla_10000/gh" --split gh \
  --out outputs/xcorr_10000.json
PYTHONHASHSEED=0 python -m instrument.background_control \
  --noise_dir "$V2A_DATA_ROOT/generated/vanilla_150000/gh" \
  --targets 0.089 0.143 0.191 --workers 2 --out outputs/background.json
```

The white-noise control retains its original per-ID Python-hash seed; set `PYTHONHASHSEED=0`
when using `--noise white`. ETA and cross-correlation characterize measured envelopes,
not perceptual sound quality or causal mechanisms.

## Paper-table index

Paths below are relative to the downloaded companion dataset. This index separates
released numerical records from commands that recompute measurements from audio.

| Main-paper table | Released records | Evaluation entry point / scope |
| --- | --- | --- |
| 1 — Full-budget event-F1 erosion | `metrics/legacy_full_budget_metrics.csv`, `metrics/legacy_run_registry.json` | Archived full-budget trajectories; `eval.onset_metrics` evaluates the released 10K/150K audio with the documented event protocol, rather than rebuilding the historical table. |
| 2 — Pseudo-label quality | `metrics/paper_numeric_audit.csv` (audited zero-shot and two-detector rows), `metadata/v266_detector_rule_split.csv` | The full detector ensemble is outside this code package; `analysis.annotation_agreement` measures human agreement and does not reproduce Table 2. |
| 3 — Validation checkpoint selection | `metrics/validation_scores.csv`, `metrics/checkpoint_metrics.csv` | Select maximum validation frame F1 within each run, then read test frame F1 and FD-PaSST from that same checkpoint; the FD penalty is relative to that run at 150K. |
| 4 — Matched-budget comparison | `metrics/checkpoint_metrics.csv`, `metrics/run_registry.json` | Average stored GH frame F1 over the shared 15-checkpoint grid, then compute paired differences and positive-sign counts. The event scorer is a separate metric. |
| 5 — Released-checkpoint contrast | `metrics/instrument_metrics.csv`, `metrics/checkpoint_metrics.csv`, `metrics/paper_numeric_audit.csv` | `instrument.eta` and `eval.envelope_xcorr` recompute envelope measurements for available audio; cross-correlation also requires reference WAVs. External-model waveforms and the original frame-F1 scorer are not bundled. |

The commands in [onset scoring](#event-onset-scores) and [detector-free measurements](#detector-free-measurements)
cover the two released audio endpoints. Read `metrics/schema.json` before aggregating:
legacy/current protocols, frame/event F1, and different envelope-instrument versions are
distinct. Missing entries remain missing; use the audited Table 5 values from
`paper_numeric_audit.csv` for reference frame F1 and released-MMAudio cross-correlation.

## Supported scope

The preparation host used Linux x86_64 with CPython 3.11.11 and the pinned dependencies
for regression tests, endpoint threshold checks, metadata/agreement calculations, ETA,
and cross-correlation. A fresh Windows CPython 3.11.11 environment additionally passed
eight regression tests, metadata/audio setup, benchmark/agreement calculations, and
the endpoint sweep with the tolerance stated above. GH 10K ETA reproduced contrast
11.913 and peak lag +70 ms. The boundary-safe NMS returned identical detections to
the previous implementation on every nonempty endpoint/threshold combination (2,427).

This repository does not provide FD feature extraction, the prior work's frame-onset metric,
the complete pseudo-labeling ensemble, or a verified end-to-end training rerun.
For architecture and weight utilities, see [models.md](models.md).
