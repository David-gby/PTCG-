from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np


EDGES = ("left", "right", "top", "bottom")


def read(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle) if row.get("success") == "True"}


def bootstrap(delta: np.ndarray, repeats: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(delta, size=(repeats, len(delta)), replace=True).mean(axis=1)
    return {
        "mean_delta": float(delta.mean()),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
        "probability_candidate_better": float(np.mean(draws < 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired comparison of two pipeline per-image CSV files")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    baseline = read(args.baseline.resolve())
    candidate = read(args.candidate.resolve())
    ids = sorted(baseline.keys() & candidate.keys())
    if not ids:
        raise RuntimeError("No paired successful rows")

    metrics: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "image_edge_mae_px": (
            np.asarray([float(baseline[key]["edge_mae_px"]) for key in ids]),
            np.asarray([float(candidate[key]["edge_mae_px"]) for key in ids]),
        ),
        "image_worst_edge_px": (
            np.asarray([float(baseline[key]["edge_max_px"]) for key in ids]),
            np.asarray([float(candidate[key]["edge_max_px"]) for key in ids]),
        ),
    }
    for edge in EDGES:
        metrics[f"edge_{edge}_absolute_px"] = (
            np.asarray([float(baseline[key][f"error_{edge}_px"]) for key in ids]),
            np.asarray([float(candidate[key][f"error_{edge}_px"]) for key in ids]),
        )

    report: dict[str, Any] = {
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "paired_images": len(ids),
        "metrics": {},
    }
    for index, (name, (before, after)) in enumerate(metrics.items()):
        delta = after - before
        report["metrics"][name] = {
            "baseline_mean": statistics.fmean(before),
            "candidate_mean": statistics.fmean(after),
            "wins": int(np.sum(delta < -1e-9)),
            "ties": int(np.sum(np.abs(delta) <= 1e-9)),
            "losses": int(np.sum(delta > 1e-9)),
            **bootstrap(delta, args.bootstrap, args.seed + index),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
