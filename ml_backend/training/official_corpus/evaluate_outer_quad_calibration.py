from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


CANONICAL = ((0.0, 0.0), (630.0, 0.0), (630.0, 880.0), (0.0, 880.0))


def _solve(matrix: list[list[float]], values: list[float]) -> list[float]:
    size = len(values)
    augmented = [row[:] + [value] for row, value in zip(matrix, values, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("singular homography")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            scale = augmented[row][column]
            if abs(scale) < 1e-15:
                continue
            augmented[row] = [
                value - scale * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column], strict=True)
            ]
    return [augmented[index][-1] for index in range(size)]


def _homography(source: list[list[float]] | tuple[tuple[float, float], ...], target) -> list[float]:
    matrix: list[list[float]] = []
    values: list[float] = []
    for (x, y), (u, v) in zip(source, target, strict=True):
        matrix.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        values.append(u)
        matrix.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.append(v)
    return _solve(matrix, values) + [1.0]


def _transform(point: Iterable[float], matrix: list[float]) -> list[float]:
    x, y = point
    denominator = matrix[6] * x + matrix[7] * y + matrix[8]
    return [
        (matrix[0] * x + matrix[1] * y + matrix[2]) / denominator,
        (matrix[3] * x + matrix[4] * y + matrix[5]) / denominator,
    ]


def _polygon_area(points: list[list[float]]) -> float:
    return abs(
        sum(
            points[index][0] * points[(index + 1) % 4][1]
            - points[(index + 1) % 4][0] * points[index][1]
            for index in range(4)
        )
        / 2.0
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p90": _percentile(values, 90.0),
        "p95": _percentile(values, 95.0),
        "max": max(values) if values else None,
    }


def _calibrate(row: dict[str, str], offset: float) -> dict[str, float | bool | str]:
    prediction = json.loads(row["prediction_points"])
    target = json.loads(row["target_points"])
    to_canonical = _homography(target, CANONICAL)
    to_image = _homography(CANONICAL, target)
    canonical_prediction = [_transform(point, to_canonical) for point in prediction]
    directions = ((-offset, -offset), (offset, -offset), (offset, offset), (-offset, offset))
    calibrated_canonical = [
        [point[0] + direction[0], point[1] + direction[1]]
        for point, direction in zip(canonical_prediction, directions, strict=True)
    ]
    calibrated_image = [_transform(point, to_image) for point in calibrated_canonical]
    signed = {
        "top": statistics.fmean(calibrated_canonical[index][1] for index in (0, 1)),
        "right": 630.0 - statistics.fmean(calibrated_canonical[index][0] for index in (1, 2)),
        "bottom": 880.0 - statistics.fmean(calibrated_canonical[index][1] for index in (2, 3)),
        "left": statistics.fmean(calibrated_canonical[index][0] for index in (3, 0)),
    }
    absolute = [abs(value) for value in signed.values()]
    inward = [max(value, 0.0) for value in signed.values()]
    outward = [max(-value, 0.0) for value in signed.values()]
    image_corner = [
        math.dist(point, truth) for point, truth in zip(calibrated_image, target, strict=True)
    ]
    canonical_corner = [
        math.dist(point, truth)
        for point, truth in zip(calibrated_canonical, CANONICAL, strict=True)
    ]
    return {
        "source": row["source"],
        "image_corner_mae_px": statistics.fmean(image_corner),
        "canonical_corner_mae_px": statistics.fmean(canonical_corner),
        "canonical_corner_max_px": max(canonical_corner),
        "canonical_edge_mae_px": statistics.fmean(absolute),
        "canonical_edge_max_px": max(absolute),
        "max_inward_px": max(inward),
        "mean_inward_px": statistics.fmean(inward),
        "max_outward_px": max(outward),
        "area_ratio": _polygon_area(calibrated_image) / max(_polygon_area(target), 1.0),
        "cut_risk": max(inward) > 3.0,
    }


def _aggregate(rows: list[dict[str, float | bool | str]]) -> dict:
    fields = (
        "image_corner_mae_px",
        "canonical_corner_mae_px",
        "canonical_corner_max_px",
        "canonical_edge_mae_px",
        "canonical_edge_max_px",
        "max_inward_px",
        "mean_inward_px",
        "max_outward_px",
        "area_ratio",
    )
    return {
        "samples": len(rows),
        "cut_risk_rate": sum(bool(row["cut_risk"]) for row in rows) / max(len(rows), 1),
        **{
            field: _summary([float(row[field]) for row in rows])
            for field in fields
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate canonical outward quad calibration.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offsets", type=float, nargs="+", default=[0.0, 1.0, 1.5, 1.75, 2.0, 2.5])
    args = parser.parse_args()

    with args.predictions.open("r", encoding="utf-8-sig", newline="") as handle:
        input_rows = [row for row in csv.DictReader(handle) if row.get("success") == "True"]
    report: dict[str, dict] = {}
    for offset in args.offsets:
        calibrated = [_calibrate(row, offset) for row in input_rows]
        by_source: dict[str, list[dict]] = defaultdict(list)
        for row in calibrated:
            by_source[str(row["source"])].append(row)
        report[f"offset_{offset:g}"] = {
            "offset_canonical_px": offset,
            "overall": _aggregate(calibrated),
            "by_source": {source: _aggregate(rows) for source, rows in sorted(by_source.items())},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
