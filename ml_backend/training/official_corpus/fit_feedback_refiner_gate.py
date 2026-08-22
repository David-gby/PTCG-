from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np


ML_ROOT = Path(__file__).resolve().parents[2]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from card_quality_processor.outer_pose_detection import calculate_outer_pose_edge_support  # noqa: E402


def _read(path: str) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(path)
    return image


def _summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
        "max": max(values) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and blind-test a post-refiner evidence gate.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = {row["sample_id"]: row for row in csv.DictReader(handle)}
    with args.predictions.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = [row for row in csv.DictReader(handle) if row.get("success") == "True"]

    rows: list[dict] = []
    for row in source_rows:
        if not row.get("detector_points") or not row.get("prediction_points"):
            continue
        image = _read(manifest[row["sample_id"]]["original_image"])
        detector_points = json.loads(row["detector_points"])
        final_points = json.loads(row["prediction_points"])
        detector = calculate_outer_pose_edge_support(image, detector_points)
        final = calculate_outer_pose_edge_support(image, final_points)
        rows.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "learned": row.get("line_selected_source", "").startswith("learned_four_side_refiner"),
                "detector_error": float(row["detector_corner_mae_px"]),
                "final_error": float(row["corner_mae_px"]),
                "detector_support": float(detector.get("edge_support_score", 0.0)),
                "final_support": float(final.get("edge_support_score", 0.0)),
                "detector_min_support": float(detector.get("min_side_edge_support", 0.0)),
                "final_min_support": float(final.get("min_side_edge_support", 0.0)),
            }
        )

    fit_rows = [row for row in rows if row["split"] != "blind"]
    blind_rows = [row for row in rows if row["split"] == "blind"]
    trials: list[dict] = []
    for support_gain in (-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.04):
        for min_support_gain in (-0.02, -0.01, 0.0, 0.01, 0.02):
            errors: list[float] = []
            fallback_count = 0
            for row in fit_rows:
                accept_learned = bool(
                    row["learned"]
                    and row["final_support"] >= row["detector_support"] + support_gain
                    and row["final_min_support"] >= row["detector_min_support"] + min_support_gain
                )
                use_detector = row["learned"] and not accept_learned
                fallback_count += use_detector
                errors.append(row["detector_error"] if use_detector else row["final_error"])
            trials.append(
                {
                    "support_gain": support_gain,
                    "min_support_gain": min_support_gain,
                    "detector_fallback_count": fallback_count,
                    **_summary(errors),
                }
            )
    trials.sort(key=lambda row: (row["p95"], row["mean"], row["detector_fallback_count"]))
    selected = trials[0]

    def evaluate(group: list[dict]) -> dict:
        baseline_errors = [row["final_error"] for row in group]
        selected_errors: list[float] = []
        fallback_count = 0
        for row in group:
            accept_learned = bool(
                row["learned"]
                and row["final_support"] >= row["detector_support"] + selected["support_gain"]
                and row["final_min_support"] >= row["detector_min_support"] + selected["min_support_gain"]
            )
            use_detector = row["learned"] and not accept_learned
            fallback_count += use_detector
            selected_errors.append(row["detector_error"] if use_detector else row["final_error"])
        return {
            "baseline": _summary(baseline_errors),
            "guarded": _summary(selected_errors),
            "detector_fallback_count": fallback_count,
        }

    report = {
        "selected_gate": {
            "support_gain": selected["support_gain"],
            "min_support_gain": selected["min_support_gain"],
        },
        "fit": evaluate(fit_rows),
        "blind": evaluate(blind_rows),
        "top_trials": trials[:15],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
