"""Path configuration.

Nothing in this repository hard-codes a dataset location. Set the three roots
below, either by editing this file or through environment variables:

    export V2A_GH_ROOT=/path/to/greatest_hits
    export V2A_VGG_ROOT=/path/to/vggsound
    export V2A_GEN_ROOT=/path/to/generated_audio

Expected layout
---------------
$V2A_GH_ROOT/
    test.txt                       one clip id per line (244 ids)
    audio/<id>.wav                 reference audio, 16 kHz mono
    onsets/<id>_times.txt          reference onset times, one float (s) per line
    video/<id>.mp4                 (optional) only needed to regenerate audio

$V2A_VGG_ROOT/
    benchmark_266.csv              released benchmark: video_id, query_label, onset_times
    audio/<id>.wav                 reference audio, 16 kHz mono

$V2A_GEN_ROOT/
    <run>_<step>/gh/<id>.flac      generated audio, Greatest Hits split
    <run>_<step>/v266/<id>.flac    generated audio, in-the-wild benchmark

`<run>` is a configuration name (e.g. `vanilla`, `curated`, `syncfreeze`)
and `<step>` the training step, e.g. `vanilla_150000`.
"""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("V2A_DATA_ROOT", "./data"))
GH_ROOT = Path(os.environ.get("V2A_GH_ROOT", str(DATA_ROOT / "greatest_hits")))
VGG_ROOT = Path(os.environ.get("V2A_VGG_ROOT", str(DATA_ROOT / "vggsound")))
GEN_ROOT = Path(os.environ.get("V2A_GEN_ROOT", str(DATA_ROOT / "generated")))

# Envelope measurements use 16 kHz audio and 10 ms frames. The event detector
# separately preserves its original 22,050 Hz sampling rate.
SR = 16_000          # sample rate (Hz)
HOP = 160            # envelope hop (samples) -> 10 ms frames
DURATION = 8.0       # clip duration (s); reference onsets beyond this are dropped

# Onset scoring
PEAK_DELTA_GH = 0.3    # librosa peak-picking delta, Greatest Hits
PEAK_DELTA_WILD = 0.07 # librosa peak-picking delta, in-the-wild benchmark
TOLERANCES = (0.05, 0.10, 0.30)   # event-matching tolerances (s)

# Event-triggered average
ETA_HALF = 30          # +-300 ms window at 10 ms frames
ETA_EDGE = 8           # frames at each window edge defining the baseline
CONTROL_MIN_MS = 500   # control anchors sit at least this far from any onset


def split_inputs(split):
    """Return all manifest IDs, nonempty scoring windows, and explicit exclusions."""
    from validation import load_gh, load_benchmark
    if split == "gh":
        return load_gh(GH_ROOT, DURATION)
    if split == "wild":
        return load_benchmark(VGG_ROOT / "benchmark_266.csv", DURATION)
    raise ValueError(f"Unknown split: {split}")


def gh_clip_ids():
    from validation import load_ids
    return load_ids(GH_ROOT / "test.txt", expected_count=244)


def gh_onsets():
    """Validate every GH annotation; return only nonempty first-8-second windows."""
    return split_inputs("gh")[1]


def wild_onsets():
    """Validate every benchmark record; return only nonempty first-8-second windows."""
    return split_inputs("wild")[1]


def preflight(split, audio_dir, suffix=".flac"):
    from validation import validate_audio
    ids, references, excluded = split_inputs(split)
    inventory = validate_audio(audio_dir, ids, suffix)
    inventory.update(manifest_clips=len(ids), eligible_windows=len(references),
                     reference_events=sum(map(len, references.values())),
                     excluded_windows=excluded, duration_s=DURATION)
    return references, inventory
