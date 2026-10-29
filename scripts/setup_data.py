#!/usr/bin/env python3
"""Install verified metadata/annotations and explicitly selected audio shards.

Uses only the Python standard library. Audio is opt-in. Archive names, member
sets, sizes and SHA-256 values must match the accompanying release manifests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


def relative_parts(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe relative path: {name!r}")
    parts = name.split("/")
    if name.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"Unsafe relative path: {name!r}")
    return parts


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check_file(path, record):
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
        raise ValueError(f"Size or SHA-256 mismatch: {path.name}")


def safe_target(root, name):
    target = root.joinpath(*relative_parts(name))
    # Include ancestors of the data root: never follow existing output symlinks.
    for path in [target, *target.parents]:
        if path.is_symlink():
            raise ValueError(f"Refusing output symlink: {path}")
    return target


def install_stream(stream, target, record):
    if target.exists():
        check_file(target, record)
        return "already_verified"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Write a sibling temporary file; expose it only after the member hash passes.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".setup-", delete=False) as out:
            temporary = Path(out.name)
            shutil.copyfileobj(stream, out, length=1024 * 1024)
        check_file(temporary, record)
        os.chmod(temporary, 0o644)
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return "installed"


def records_by_path(records):
    result = {}
    for record in records:
        name = record["path"]
        relative_parts(name)
        if name in result:
            raise ValueError(f"Duplicate manifest member: {name}")
        if not isinstance(record.get("bytes"), int) or record["bytes"] < 0:
            raise ValueError(f"Invalid manifest size: {name}")
        if len(record.get("sha256", "")) != 64:
            raise ValueError(f"Invalid SHA-256 field: {name}")
        result[name] = record
    return result


def extract_verified(archive, records, data_root, required_prefix):
    expected = records_by_path(records)
    counts = {"installed": 0, "already_verified": 0}
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        actual = {}
        for member in members:
            relative_parts(member.name)
            if not member.name.startswith(required_prefix + "/"):
                raise ValueError(f"Unexpected archive root: {member.name}")
            # No directories, symlinks, hardlinks, devices, FIFOs or sparse files.
            if not member.isfile() or member.issparse():
                raise ValueError(f"Only regular archive members are permitted: {member.name}")
            if member.name in actual:
                raise ValueError(f"Duplicate archive member: {member.name}")
            actual[member.name] = member
        if set(actual) != set(expected):
            raise ValueError("Archive member set differs from its hash manifest")
        # Validate all names and destination conflicts before creating any files.
        for name, member in actual.items():
            if member.size != expected[name]["bytes"]:
                raise ValueError(f"Archive member size mismatch: {name}")
            target = safe_target(data_root, name)
            if target.exists():
                check_file(target, expected[name])
        for name, member in actual.items():
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read archive member: {name}")
            with source:
                status = install_stream(source, safe_target(data_root, name), expected[name])
                counts[status] += 1
    return {"archive": archive.name, "members": len(expected), **counts,
            "member_sha256_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True,
                        help="Downloaded Hugging Face release directory")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Destination used by V2A_DATA_ROOT")
    parser.add_argument("--audio-shard", action="append", default=[], metavar="NAME.tar",
                        help="Opt-in shard basename from assets/audio_manifest.json; repeatable")
    parser.add_argument("--list-audio-shards", action="store_true",
                        help="Print shard coverage and exit without installing anything")
    args = parser.parse_args()
    release = args.release_root.absolute()
    data_root = args.data_root.absolute()
    known = json.loads((Path(__file__).resolve().parents[1] /
                        "configs/setup_manifest.json").read_text())["files"]

    def verified_release_file(name):
        path = release.joinpath(*relative_parts(name))
        check_file(path, known[name])
        return path

    selections = []
    if args.audio_shard or args.list_audio_shards:
        index = json.loads(verified_release_file("assets/audio_manifest.json").read_text())
        artifacts = {PurePosixPath(row["path"]).name: row for row in index["artifacts"]}
        if len(artifacts) != len(index["artifacts"]):
            raise ValueError("Ambiguous shard basenames in audio manifest")
        if args.list_audio_shards:
            print(json.dumps([{ "shard": key, "bytes": row["bytes"],
                                "coverage": row["coverage"]}
                              for key, row in artifacts.items()], indent=2))
            return
        if len(set(args.audio_shard)) != len(args.audio_shard):
            parser.error("--audio-shard values must be unique")
        for name in args.audio_shard:
            if name not in artifacts:
                parser.error(f"Unknown audio shard: {name}; use --list-audio-shards")
            row = artifacts[name]
            archive = release / "assets" / Path(*relative_parts(row["path"]))
            manifest = release / "assets" / Path(*relative_parts(row["manifest"]))
            check_file(archive, row)
            if digest(manifest) != row["manifest_sha256"]:
                raise ValueError(f"Member-manifest SHA-256 mismatch: {manifest.name}")
            records = [json.loads(line) for line in manifest.read_text().splitlines() if line]
            if len(records) != row["members"]:
                raise ValueError(f"Member count mismatch: {name}")
            selections.append((archive, records))

    # Default installation is small: benchmark, split IDs, and GH onset annotations.
    copies = {"metadata/benchmark_266.csv": "vggsound/benchmark_266.csv",
              "metadata/gh_test_ids.txt": "greatest_hits/test.txt",
              "metadata/gh_validation_ids.txt": "greatest_hits/validation.txt"}
    inputs = {name: verified_release_file(name) for name in copies}
    annotations = verified_release_file("assets/greatest_hits_annotations.tar")
    annotation_manifest = verified_release_file("assets/greatest_hits_annotations.manifest.json")
    report = {"metadata": [], "annotations": None, "audio_shards": [],
              "audio_opt_in": bool(selections)}
    for name, destination in copies.items():
        with inputs[name].open("rb") as handle:
            status = install_stream(handle, safe_target(data_root, destination), known[name])
        report["metadata"].append({"path": destination, "status": status,
                                   "sha256": known[name]["sha256"]})
    report["annotations"] = extract_verified(annotations, json.loads(annotation_manifest.read_text()),
                                              data_root, "greatest_hits")
    for archive, records in selections:
        report["audio_shards"].append(extract_verified(archive, records, data_root, "generated"))
    destination = safe_target(data_root, "setup_report.json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
