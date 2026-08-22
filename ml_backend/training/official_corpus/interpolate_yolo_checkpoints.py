from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Linearly interpolate compatible YOLO checkpoints.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True, help="Candidate weight in [0, 1].")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    baseline = torch.load(args.baseline.resolve(), map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate.resolve(), map_location="cpu", weights_only=False)
    baseline_model = copy.deepcopy(baseline["model"]).float()
    candidate_state = candidate["model"].float().state_dict()
    baseline_state = baseline_model.state_dict()
    if baseline_state.keys() != candidate_state.keys():
        raise ValueError("checkpoint architectures/state dictionaries differ")

    merged = {}
    for key, base_value in baseline_state.items():
        candidate_value = candidate_state[key]
        if base_value.shape != candidate_value.shape:
            raise ValueError(f"shape mismatch for {key}: {base_value.shape} != {candidate_value.shape}")
        if torch.is_floating_point(base_value):
            merged[key] = torch.lerp(base_value, candidate_value.to(base_value.dtype), args.alpha)
        else:
            merged[key] = base_value
    baseline_model.load_state_dict(merged, strict=True)
    output_checkpoint = copy.deepcopy(baseline)
    output_checkpoint["model"] = baseline_model.half()
    output_checkpoint["ema"] = None
    output_checkpoint["optimizer"] = None
    output_checkpoint["updates"] = None
    output_checkpoint["epoch"] = -1
    output_checkpoint["best_fitness"] = None
    output_checkpoint["interpolation"] = {
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "candidate_alpha": args.alpha,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
