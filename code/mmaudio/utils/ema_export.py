"""Convert trusted local EMA states to strict inference safetensors.

Direct export strips only the explicit ema_model. namespace. Post-hoc synthesis
requires the original EMA basis, update schedule and checkpoint folder; a single
final inference state cannot reconstruct that history.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import save_file
from mmaudio.model.networks import get_my_mmaudio
from mmaudio.utils.release_weights import model_config, validate_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--configs", default="configs/model_configs.json")
    parser.add_argument("--synthesize", action="store_true",
                        help="Input is JSON EMA schedule; synthesize from its basis directory")
    parser.add_argument("--sigma", type=float, default=.05)
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    config = model_config(args.configs, args.run)
    if args.synthesize:
        from nitrous_ema import PostHocEMA
        schedule = json.loads(args.input.read_text())
        folder = Path(schedule["checkpoint_folder"])
        if not folder.is_absolute():
            folder = args.input.parent / folder
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        model = get_my_mmaudio(config["model"], **config["kwargs"])
        ema = PostHocEMA(model, sigma_rels=schedule["sigma_rels"],
                         update_every=schedule["update_every"],
                         checkpoint_every_num_steps=schedule["checkpoint_every"],
                         checkpoint_folder=folder)
        state = ema.synthesize_ema_model(sigma_rel=args.sigma, step=args.step,
                                         device="cpu").ema_model.state_dict()
    else:
        state = torch.load(args.input, map_location="cpu", weights_only=True)
        if any(k.startswith("ema_model.") for k in state):
            state = {k.removeprefix("ema_model."): v for k, v in state.items()
                     if k.startswith("ema_model.")}
    if not state or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a tensor-only model state or explicit ema_model.* EMA state")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(args.out))
    validate_weights(args.out, config)
    print(f"Strictly validated {args.out.name}")


if __name__ == "__main__":
    main()
