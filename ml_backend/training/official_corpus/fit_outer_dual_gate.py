from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np


ML_ROOT = Path(__file__).resolve().parents[2]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from card_quality_processor.outer_pose_detection import calculate_outer_pose_edge_support  # noqa: E402


def _rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def _area(points: np.ndarray) -> float:
    return abs(float(cv2.contourArea(points.astype(np.float32).reshape(-1, 1, 2))))


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a conservative baseline/robust silhouette gate.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = _rows(args.baseline)
    candidate = _rows(args.candidate)
    image_paths = {path.stem: path for path in args.images.iterdir() if path.is_file()}
    features: list[dict] = []
    for sample_id, base in baseline.items():
        cand = candidate.get(sample_id)
        image_path = image_paths.get(sample_id)
        if cand is None or image_path is None or base.get("success") != "True" or cand.get("success") != "True":
            continue
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        base_points = np.asarray(json.loads(base["prediction_points"]), dtype=np.float32)
        cand_points = np.asarray(json.loads(cand["prediction_points"]), dtype=np.float32)
        base_support = calculate_outer_pose_edge_support(image, base_points)
        cand_support = calculate_outer_pose_edge_support(image, cand_points)
        base_area = _area(base_points)
        cand_area = _area(cand_points)
        diagonal = max(float(math.hypot(*image.shape[:2])), 1.0)
        features.append(
            {
                "sample_id": sample_id,
                "source": base["source"],
                "shadow_type": base.get("shadow_type", ""),
                "baseline_error": float(base["image_corner_mae_px"]),
                "candidate_error": float(cand["image_corner_mae_px"]),
                "area_ratio_baseline_to_candidate": base_area / max(cand_area, 1.0),
                "disagreement_ratio": float(np.linalg.norm(base_points - cand_points, axis=1).mean() / diagonal),
                "baseline_support": float(base_support.get("edge_support_score", 0.0)),
                "candidate_support": float(cand_support.get("edge_support_score", 0.0)),
                "baseline_min_support": float(base_support.get("min_side_edge_support", 0.0)),
                "candidate_min_support": float(cand_support.get("min_side_edge_support", 0.0)),
            }
        )

    trials: list[dict] = []
    for area_threshold in (1.002, 1.005, 1.008, 1.010, 1.015, 1.020, 1.030, 1.040):
        for support_margin in (0.0, 0.005, 0.010, 0.020, 0.040):
            for min_disagreement in (0.0, 0.002, 0.004, 0.006):
                selected_errors: list[float] = []
                history_errors: list[float] = []
                official_errors: list[float] = []
                selected_count = 0
                for row in features:
                    choose_candidate = bool(
                        row["area_ratio_baseline_to_candidate"] >= area_threshold
                        and row["disagreement_ratio"] >= min_disagreement
                        and row["candidate_support"] + support_margin >= row["baseline_support"]
                        and row["candidate_min_support"] + support_margin >= row["baseline_min_support"]
                    )
                    error = row["candidate_error"] if choose_candidate else row["baseline_error"]
                    selected_count += choose_candidate
                    selected_errors.append(error)
                    (history_errors if row["source"] == "historical_real" else official_errors).append(error)
                trials.append(
                    {
                        "area_threshold": area_threshold,
                        "support_margin": support_margin,
                        "min_disagreement": min_disagreement,
                        "selected_count": selected_count,
                        "overall_mean": statistics.fmean(selected_errors),
                        "overall_p95": _percentile(selected_errors, 95),
                        "history_mean": statistics.fmean(history_errors),
                        "history_p95": _percentile(history_errors, 95),
                        "official_mean": statistics.fmean(official_errors),
                        "official_p95": _percentile(official_errors, 95),
                    }
                )

    baseline_history = [row["baseline_error"] for row in features if row["source"] == "historical_real"]
    eligible = [
        row
        for row in trials
        if row["history_mean"] <= statistics.fmean(baseline_history) + 0.05
        and row["history_p95"] <= _percentile(baseline_history, 95) + 0.20
    ]
    eligible.sort(key=lambda row: (row["official_p95"], row["official_mean"], row["overall_mean"]))
    report = {
        "samples": len(features),
        "baseline": {
            "overall_mean": statistics.fmean(row["baseline_error"] for row in features),
            "history_mean": statistics.fmean(baseline_history),
            "history_p95": _percentile(baseline_history, 95),
            "official_mean": statistics.fmean(
                row["baseline_error"] for row in features if row["source"] != "historical_real"
            ),
        },
        "candidate": {
            "overall_mean": statistics.fmean(row["candidate_error"] for row in features),
            "history_mean": statistics.fmean(
                row["candidate_error"] for row in features if row["source"] == "historical_real"
            ),
            "official_mean": statistics.fmean(
                row["candidate_error"] for row in features if row["source"] != "historical_real"
            ),
        },
        "best_gate": eligible[0] if eligible else None,
        "top_gates": eligible[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    feature_path = args.output.with_suffix(".features.csv")
    with feature_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in features for key in row}))
        writer.writeheader()
        writer.writerows(features)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
