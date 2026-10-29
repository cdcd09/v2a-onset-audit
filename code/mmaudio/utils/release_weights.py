"""Strict released-weight schema checks and loading, without GPU allocation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from mmaudio.model.networks import get_my_mmaudio


def model_config(path, run):
    configs = json.loads(Path(path).read_text())
    if run not in configs:
        raise ValueError(f"Unknown source run: {run}")
    return configs[run]


def validate_weights(path, config):
    """Check all parameter/buffer names and shapes without materializing weights."""
    with torch.device("meta"):
        model = get_my_mmaudio(config["model"], **config["kwargs"])
    expected = {name: list(value.shape) for name, value in model.state_dict().items()}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        actual = {name: handle.get_slice(name).get_shape() for name in handle.keys()}
    missing, extra = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
    wrong = {name: {"expected": expected[name], "actual": actual[name]}
             for name in expected.keys() & actual.keys() if expected[name] != actual[name]}
    if missing or extra or wrong:
        raise ValueError(f"Weight schema mismatch: missing={missing}, extra={extra}, shapes={wrong}")
    return {"tensor_count": len(actual), "parameter_elements": sum(p.numel() for p in model.parameters()),
            "strict_keys_and_shapes": True, "device": "meta (schema only)",
            "model": config["model"], "kwargs": config["kwargs"]}


def load_model(path, config, device="cpu", dtype=torch.float32):
    """Strictly load released safetensors; never silently warm-start missing modules."""
    validate_weights(path, config)
    # Construct normally so nonpersistent rotary/frequency buffers are populated;
    # assigning a state dict into a meta model would leave those buffers on meta.
    model = get_my_mmaudio(config["model"], **config["kwargs"])
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True, assign=True)
    return model.to(device=device, dtype=dtype).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--configs", type=Path, default=Path("configs/model_configs.json"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = {"run": args.run, "filename": args.weights.name,
              **validate_weights(args.weights, model_config(args.configs, args.run))}
    digest = hashlib.sha256()
    with args.weights.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    report["sha256"] = digest.hexdigest()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
