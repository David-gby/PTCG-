from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ML_ROOT = Path(__file__).resolve().parents[2]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from card_quality_processor.outer_detection import order_points  # noqa: E402
from card_quality_processor.outer_pose_detection import (  # noqa: E402
    apply_outer_silhouette_calibration,
    calculate_outer_pose_edge_support,
)


def _rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
    }


def _area(points: np.ndarray) -> float:
    return abs(float(cv2.contourArea(points.astype(np.float32).reshape(-1, 1, 2))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen dual-model gate.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--area-threshold", type=float, default=1.002)
    parser.add_argument("--max-area-ratio", type=float, default=1.055)
    parser.add_argument("--support-margin", type=float, default=0.04)
    parser.add_argument("--min-disagreement", type=float, default=0.004)
    parser.add_argument("--outward-canonical-px", type=float, default=1.75)
    args = parser.parse_args()

    baseline = _rows(args.baseline)
    candidate = _rows(args.candidate)
    image_paths = {path.stem: path for path in args.images.iterdir() if path.is_file()}
    calibration_config = {
        "silhouette_refinement": {
            "corner_calibration": {
                "enabled": args.outward_canonical_px > 0.0,
                "outward_canonical_px": args.outward_canonical_px,
                "canonical_width": 630.0,
                "canonical_height": 880.0,
            }
        }
    }
    rows: list[dict] = []
    for sample_id, base in baseline.items():
        cand = candidate.get(sample_id)
        image_path = image_paths.get(sample_id)
        if cand is None or image_path is None:
            continue
        base_success = base.get("success") == "True"
        candidate_success = cand.get("success") == "True"
        selected = "baseline"
        gate_reason = "baseline_default"
        if not base_success and candidate_success:
            selected = "candidate"
            gate_reason = "baseline_failed"
        elif base_success and candidate_success:
            image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            base_points = order_points(np.asarray(json.loads(base["prediction_points"]), dtype=np.float32))
            cand_points = order_points(np.asarray(json.loads(cand["prediction_points"]), dtype=np.float32))
            base_support = calculate_outer_pose_edge_support(image, base_points)
            cand_support = calculate_outer_pose_edge_support(image, cand_points)
            area_ratio = _area(base_points) / max(_area(cand_points), 1.0)
            diagonal = max(float(math.hypot(*image.shape[:2])), 1.0)
            disagreement = float(np.linalg.norm(base_points - cand_points, axis=1).mean() / diagonal)
            if (
                area_ratio >= args.area_threshold
                and area_ratio <= args.max_area_ratio
                and disagreement >= args.min_disagreement
                and float(cand_support.get("edge_support_score", 0.0)) + args.support_margin
                >= float(base_support.get("edge_support_score", 0.0))
                and float(cand_support.get("min_side_edge_support", 0.0)) + args.support_margin
                >= float(base_support.get("min_side_edge_support", 0.0))
            ):
                selected = "candidate"
                gate_reason = "larger_baseline_with_supported_candidate"

        selected_row = cand if selected == "candidate" else base
        prediction = order_points(
            np.asarray(json.loads(selected_row["prediction_points"]), dtype=np.float32)
        )
        if selected == "candidate":
            prediction = apply_outer_silhouette_calibration(prediction, calibration_config)
        target = order_points(np.asarray(json.loads(selected_row["target_points"]), dtype=np.float32))
        error = np.linalg.norm(prediction - target, axis=1)
        rows.append(
            {
                "sample_id": sample_id,
                "source": base.get("source", "unknown"),
                "shadow_type": base.get("shadow_type", ""),
                "selected": selected,
                "gate_reason": gate_reason,
                "image_corner_mae_px": float(error.mean()),
                "image_corner_max_px": float(error.max()),
            }
        )

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["source"]].append(row)
    report = {
        "gate": {
            "area_threshold": args.area_threshold,
            "max_area_ratio": args.max_area_ratio,
            "support_margin": args.support_margin,
            "min_disagreement": args.min_disagreement,
            "outward_canonical_px": args.outward_canonical_px,
        },
        "overall": {
            "samples": len(rows),
            "candidate_selected": sum(row["selected"] == "candidate" for row in rows),
            "image_corner_mae_px": _summary([row["image_corner_mae_px"] for row in rows]),
            "image_corner_max_px": _summary([row["image_corner_max_px"] for row in rows]),
        },
        "by_source": {
            source: {
                "samples": len(group),
                "candidate_selected": sum(row["selected"] == "candidate" for row in group),
                "image_corner_mae_px": _summary([row["image_corner_mae_px"] for row in group]),
                "image_corner_max_px": _summary([row["image_corner_max_px"] for row in group]),
            }
            for source, group in sorted(groups.items())
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
