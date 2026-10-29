"""Strict release input checks; no detection or matching math lives here."""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path


def clip_id(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value or ""):
        raise ValueError(f"Invalid clip ID: {value!r}")
    return value


def load_ids(path, expected_count=None):
    path = Path(path)
    ids = [clip_id(line.split()[0]) for line in path.read_text().splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Empty or duplicate IDs in {path.name}")
    if expected_count is not None and len(ids) != expected_count:
        raise ValueError(f"Expected {expected_count} IDs in {path.name}; found {len(ids)}")
    return ids


def onset_times(values, context):
    times = [float(value) for value in values]
    if any(not math.isfinite(t) or t < 0 for t in times):
        raise ValueError(f"Non-finite or negative onset time: {context}")
    # Preserve input order and repeated timestamps: changing them can alter
    # the submitted greedy matching and reference counts.
    return times


def load_gh(root, duration=8.0, expected_count=244):
    root = Path(root)
    ids = load_ids(root / "test.txt", expected_count)
    references, excluded = {}, []
    for vid in ids:
        path = root / "onsets" / f"{vid}_times.txt"
        # Missing annotations are errors even for an otherwise excluded window.
        raw = onset_times([line.split()[0] for line in path.read_text().splitlines()
                           if line.strip()], path.name)
        kept = [t for t in raw if t < duration]
        if kept:
            references[vid] = kept
        else:
            excluded.append({"clip_id": vid, "reason": "no_reference_onset_before_window_end",
                             "first_full_clip_onset_s": min(raw) if raw else None})
    return ids, references, excluded


def load_benchmark(path, duration=8.0, expected_count=266):
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        needed = {"video_id", "query_label", "onset_times"}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing benchmark columns: {sorted(needed - set(reader.fieldnames or []))}")
        rows = list(reader)
    ids = [clip_id(row["video_id"]) for row in rows]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("Empty or duplicate benchmark IDs")
    if expected_count is not None and len(ids) != expected_count:
        raise ValueError(f"Expected {expected_count} benchmark clips; found {len(ids)}")
    references, excluded = {}, []
    for row in rows:
        raw = onset_times([v for v in row["onset_times"].split(",") if v.strip()], row["video_id"])
        if "n_onsets" in row and int(row["n_onsets"]) != len(raw):
            raise ValueError(f"n_onsets disagrees with timestamps: {row['video_id']}")
        kept = [t for t in raw if t < duration]
        if kept:
            references[row["video_id"]] = kept
        else:
            excluded.append({"clip_id": row["video_id"],
                             "reason": "no_reference_onset_before_window_end",
                             "first_full_clip_onset_s": min(raw) if raw else None})
    return ids, references, excluded


def validate_audio(directory, ids, suffix=".flac", reject_extra=True):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Audio directory does not exist: {directory}")
    expected = set(ids)
    present = {p.name[:-len(suffix)] for p in directory.glob(f"*{suffix}") if p.is_file()}
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing or (reject_extra and extra):
        raise ValueError(f"Audio inventory mismatch: missing={missing}; extra={extra if reject_extra else []}")
    return {"expected_files": len(expected), "present_files": len(expected),
            "missing_files": [], "unexpected_files": extra}


def save_json(path, value):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
