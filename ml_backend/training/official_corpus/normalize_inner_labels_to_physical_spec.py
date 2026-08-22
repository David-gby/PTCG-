from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


INNER_WIDTH_FRACTION = 58.0 / 63.0
INNER_HEIGHT_FRACTION = 83.0 / 88.0


def summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize train-only inner labels to the confirmed 58x83 mm "
            "printed-line inner-edge specification while preserving center."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--splits", default="train")
    parser.add_argument(
        "--max-center-offset-fraction",
        type=float,
        default=0.035,
        help="Rows farther from the card center are kept unchanged and audited.",
    )
    args = parser.parse_args()
    selected_splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    with args.input.resolve().open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("Input manifest is empty")

    output: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    left_moves: list[float] = []
    right_moves: list[float] = []
    top_moves: list[float] = []
    bottom_moves: list[float] = []
    for row in rows:
        current = dict(row)
        status = "split_frozen"
        if row.get("split") in selected_splits:
            left = float(row["left"])
            right = float(row["right"])
            top = float(row["top"])
            bottom = float(row["bottom"])
            center_x = 0.5 * (left + right)
            center_y = 0.5 * (top + bottom)
            candidate = {
                "left": center_x - 0.5 * INNER_WIDTH_FRACTION,
                "right": center_x + 0.5 * INNER_WIDTH_FRACTION,
                "top": center_y - 0.5 * INNER_HEIGHT_FRACTION,
                "bottom": center_y + 0.5 * INNER_HEIGHT_FRACTION,
            }
            center_safe = bool(
                abs(center_x - 0.5) <= args.max_center_offset_fraction
                and abs(center_y - 0.5) <= args.max_center_offset_fraction
            )
            bounds_safe = bool(
                0.0 <= candidate["left"] < candidate["right"] <= 1.0
                and 0.0 <= candidate["top"] < candidate["bottom"] <= 1.0
            )
            if center_safe and bounds_safe:
                current.update({key: f"{value:.10f}" for key, value in candidate.items()})
                current.update(
                    {
                        "physical_prior_applied": "true",
                        "physical_measurement_semantics": "printed_inner_line_inner_edge",
                        "physical_inner_width_mm": "58.0",
                        "physical_inner_height_mm": "83.0",
                        "physical_original_left": row["left"],
                        "physical_original_right": row["right"],
                        "physical_original_top": row["top"],
                        "physical_original_bottom": row["bottom"],
                    }
                )
                width = float(row["width"])
                height = float(row["height"])
                left_moves.append(abs(candidate["left"] - left) * width)
                right_moves.append(abs(candidate["right"] - right) * width)
                top_moves.append(abs(candidate["top"] - top) * height)
                bottom_moves.append(abs(candidate["bottom"] - bottom) * height)
                status = "normalized"
            elif not center_safe:
                status = "center_outlier_kept"
            else:
                status = "bounds_outlier_kept"
        current.setdefault("physical_prior_applied", "false")
        current.setdefault("physical_measurement_semantics", "printed_inner_line_inner_edge")
        current["physical_prior_status"] = status
        status_counts[status] += 1
        by_source[str(row.get("source", "unknown"))][status] += 1
        output.append(current)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in output for key in row))
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)
    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "measurement_semantics": "printed_inner_line_inner_edge",
        "applies_to_all_layouts": True,
        "normalized_splits": sorted(selected_splits),
        "validation_and_test_labels_modified": False,
        "expected_fraction": {
            "width": INNER_WIDTH_FRACTION,
            "height": INNER_HEIGHT_FRACTION,
        },
        "rows": len(rows),
        "status": dict(status_counts),
        "by_source": {source: dict(counts) for source, counts in sorted(by_source.items())},
        "absolute_label_move_px": {
            "left": summary(left_moves),
            "right": summary(right_moves),
            "top": summary(top_moves),
            "bottom": summary(bottom_moves),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
