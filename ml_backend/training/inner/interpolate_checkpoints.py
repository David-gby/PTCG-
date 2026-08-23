from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Linearly interpolate compatible model checkpoints")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    base = torch.load(str(args.base.resolve()), map_location="cpu", weights_only=False)
    candidate = torch.load(str(args.candidate.resolve()), map_location="cpu", weights_only=False)
    base_state = base["model"]
    candidate_state = candidate["model"]
    if base_state.keys() != candidate_state.keys():
        raise ValueError("Checkpoints do not have matching model keys")
    blended: dict[str, torch.Tensor] = {}
    for key, base_value in base_state.items():
        candidate_value = candidate_state[key]
        if base_value.shape != candidate_value.shape:
            raise ValueError(f"Shape mismatch for {key}")
        if base_value.is_floating_point():
            blended[key] = base_value.lerp(candidate_value, args.alpha)
        else:
            blended[key] = candidate_value.clone() if args.alpha >= 0.5 else base_value.clone()
    output = dict(candidate)
    output["model"] = blended
    output["optimizer"] = None
    output["interpolation"] = {
        "base": str(args.base.resolve()),
        "candidate": str(args.candidate.resolve()),
        "alpha": args.alpha,
    }
    output.setdefault("config", {})["checkpoint_interpolation"] = output["interpolation"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
