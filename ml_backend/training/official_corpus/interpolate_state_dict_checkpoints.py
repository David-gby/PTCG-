from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Linearly interpolate compatible state-dict checkpoints."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    baseline = torch.load(args.baseline.resolve(), map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate.resolve(), map_location="cpu", weights_only=False)
    baseline_state = baseline["model"]
    candidate_state = candidate["model"]
    if baseline_state.keys() != candidate_state.keys():
        raise ValueError("checkpoint state dictionaries differ")

    merged = copy.deepcopy(baseline_state)
    for key, base_value in baseline_state.items():
        candidate_value = candidate_state[key]
        if base_value.shape != candidate_value.shape:
            raise ValueError(f"shape mismatch for {key}: {base_value.shape} != {candidate_value.shape}")
        if torch.is_floating_point(base_value):
            merged[key] = torch.lerp(
                base_value.float(), candidate_value.float(), args.alpha
            ).to(base_value.dtype)
        else:
            merged[key] = copy.deepcopy(base_value)

    output = copy.deepcopy(baseline)
    output["model"] = merged
    output["optimizer"] = None
    output["epoch"] = -1
    output["interpolation"] = {
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "candidate_alpha": args.alpha,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
