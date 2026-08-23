from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np


EDGES = ("left", "right", "top", "bottom")


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate trusted-crop predictions against the confirmed 58x83 mm inner-edge spec."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-prefix", default="official_")
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        manifest = {row["id"]: row for row in csv.DictReader(stream)}
    reports: dict[str, Any] = {}
    for prediction_path in args.predictions:
        with prediction_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = [
                row
                for row in csv.DictReader(stream)
                if row.get("success") == "True"
                and (
                    args.source_prefix == "*"
                    or row.get("source", "").startswith(args.source_prefix)
                )
            ]
        per_edge = {edge: [] for edge in EDGES}
        per_image: list[float] = []
        per_image_max: list[float] = []
        signed_center_x: list[float] = []
        signed_center_y: list[float] = []
        for row in rows:
            source = manifest[row["id"]]
            width = float(source["width"])
            height = float(source["height"])
            expected_width = width * 58.0 / 63.0
            expected_height = height * 83.0 / 88.0
            target_center_x = 0.5 * (
                float(row["target_left"]) + float(row["target_right"])
            )
            target_center_y = 0.5 * (
                float(row["target_top"]) + float(row["target_bottom"])
            )
            target = {
                "left": target_center_x - 0.5 * expected_width,
                "right": target_center_x + 0.5 * expected_width,
                "top": target_center_y - 0.5 * expected_height,
                "bottom": target_center_y + 0.5 * expected_height,
            }
            prediction = {
                edge: float(row[f"prediction_{edge}"]) for edge in EDGES
            }
            errors = [abs(prediction[edge] - target[edge]) for edge in EDGES]
            per_image.append(statistics.fmean(errors))
            per_image_max.append(max(errors))
            for edge, error in zip(EDGES, errors):
                per_edge[edge].append(error)
            signed_center_x.append(
                0.5 * (prediction["left"] + prediction["right"]) - target_center_x
            )
            signed_center_y.append(
                0.5 * (prediction["top"] + prediction["bottom"]) - target_center_y
            )
        reports[prediction_path.parent.name] = {
            "prediction_file": str(prediction_path.resolve()),
            "measurement_semantics": "printed_inner_line_inner_edge",
            "physical_spec_mm": {"outer": [63.0, 88.0], "inner": [58.0, 83.0]},
            "edge_mae_px": _summary(per_image),
            "edge_max_px": _summary(per_image_max),
            "by_edge": {edge: _summary(values) for edge, values in per_edge.items()},
            "center_signed_mean_px": {
                "x": statistics.fmean(signed_center_x) if signed_center_x else None,
                "y": statistics.fmean(signed_center_y) if signed_center_y else None,
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
