from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ML_ROOT = Path(__file__).resolve().parents[2]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from card_quality_processor.outer_shadow_risk import assess_outer_shadow_risk  # noqa: E402
from card_quality_processor.outer_pose_detection import calculate_outer_pose_edge_support  # noqa: E402


def _rows(path: Path, key: str = "sample_id") -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def _read(path: str) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(path)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = _rows(args.manifest)
    baseline = _rows(args.baseline)
    candidate = _rows(args.candidate)
    rows: list[dict] = []
    for sample_id, base in baseline.items():
        cand = candidate[sample_id]
        image = _read(manifest[sample_id]["original_image"])
        points = json.loads(base["prediction_points"]) if base.get("prediction_points") else None
        candidate_points = (
            json.loads(cand["prediction_points"]) if cand.get("prediction_points") else None
        )
        risk = assess_outer_shadow_risk(image, points)
        base_support = calculate_outer_pose_edge_support(image, points) if points else {}
        candidate_support = (
            calculate_outer_pose_edge_support(image, candidate_points)
            if candidate_points
            else {}
        )
        base_error = float(base["corner_mae_px"]) if base.get("corner_mae_px") else None
        candidate_error = float(cand["corner_mae_px"]) if cand.get("corner_mae_px") else None
        rows.append(
            {
                "sample_id": sample_id,
                "split": base["split"],
                "baseline_success": base["success"] == "True",
                "candidate_success": cand["success"] == "True",
                "baseline_mae_px": base_error,
                "candidate_mae_px": candidate_error,
                "delta_candidate_minus_baseline_px": (
                    candidate_error - base_error
                    if candidate_error is not None and base_error is not None
                    else None
                ),
                "shadow_high_risk": bool(risk.get("high_risk")),
                "shadow_risk_score": risk.get("risk_score"),
                "shadow_reason": risk.get("reason"),
                "neutral_side_count": risk.get("neutral_side_count"),
                "ambiguous_neutral_side_count": risk.get("ambiguous_neutral_side_count"),
                "baseline_edge_support": base_support.get("edge_support_score"),
                "candidate_edge_support": candidate_support.get("edge_support_score"),
                "baseline_min_side_support": base_support.get("min_side_edge_support"),
                "candidate_min_side_support": candidate_support.get("min_side_edge_support"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
