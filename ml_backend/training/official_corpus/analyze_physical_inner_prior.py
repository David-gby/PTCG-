from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CANONICAL_WIDTH = 630.0
CANONICAL_HEIGHT = 880.0
OUTER_WIDTH_MM = 63.0
OUTER_HEIGHT_MM = 88.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _profile_labels(
    rows: list[dict[str, str]],
    inner_width_mm: float,
    inner_height_mm: float,
) -> dict[str, Any]:
    expected_width = CANONICAL_WIDTH * inner_width_mm / OUTER_WIDTH_MM
    expected_height = CANONICAL_HEIGHT * inner_height_mm / OUTER_HEIGHT_MM
    widths = np.asarray(
        [(float(row["right"]) - float(row["left"])) * CANONICAL_WIDTH for row in rows]
    )
    heights = np.asarray(
        [(float(row["bottom"]) - float(row["top"])) * CANONICAL_HEIGHT for row in rows]
    )
    center_x = np.asarray(
        [
            ((float(row["left"]) + float(row["right"])) / 2.0 - 0.5)
            * CANONICAL_WIDTH
            for row in rows
        ]
    )
    center_y = np.asarray(
        [
            ((float(row["top"]) + float(row["bottom"])) / 2.0 - 0.5)
            * CANONICAL_HEIGHT
            for row in rows
        ]
    )
    return {
        "rows": len(rows),
        "expected_canonical_width_px": expected_width,
        "expected_canonical_height_px": expected_height,
        "width_px": _summary(widths),
        "height_px": _summary(heights),
        "width_abs_deviation_px": _summary(np.abs(widths - expected_width)),
        "height_abs_deviation_px": _summary(np.abs(heights - expected_height)),
        "center_x_offset_px": _summary(center_x),
        "center_y_offset_px": _summary(center_y),
        "aspect_height_over_width": _summary(heights / np.maximum(widths, 1e-6)),
    }


def _evaluate_prior(
    evaluation_rows: list[dict[str, str]],
    labels_by_id: dict[str, dict[str, str]],
    split: str,
    *,
    alpha_width: float,
    alpha_height: float,
    threshold_width_px: float,
    threshold_height_px: float,
    inner_width_mm: float,
    inner_height_mm: float,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    expected_width = CANONICAL_WIDTH * inner_width_mm / OUTER_WIDTH_MM
    expected_height = CANONICAL_HEIGHT * inner_height_mm / OUTER_HEIGHT_MM
    per_image: list[dict[str, float]] = []
    applied = 0
    for row in evaluation_rows:
        if row["split"] != split or row.get("success", "").lower() != "true":
            continue
        label = labels_by_id[row["id"]]
        image_width = float(label["width"])
        image_height = float(label["height"])
        prediction = np.asarray(
            [
                float(row["prediction_left"]),
                float(row["prediction_right"]),
                float(row["prediction_top"]),
                float(row["prediction_bottom"]),
            ],
            dtype=np.float64,
        )
        target = np.asarray(
            [
                float(row["target_left"]),
                float(row["target_right"]),
                float(row["target_top"]),
                float(row["target_bottom"]),
            ],
            dtype=np.float64,
        )
        center_x = (prediction[0] + prediction[1]) / 2.0
        center_y = (prediction[2] + prediction[3]) / 2.0
        predicted_width = (prediction[1] - prediction[0]) * CANONICAL_WIDTH / image_width
        predicted_height = (prediction[3] - prediction[2]) * CANONICAL_HEIGHT / image_height
        use_width = abs(predicted_width - expected_width) > threshold_width_px
        use_height = abs(predicted_height - expected_height) > threshold_height_px
        width_alpha = alpha_width if use_width else 0.0
        height_alpha = alpha_height if use_height else 0.0
        applied += int(use_width or use_height)
        refined_width = (
            (1.0 - width_alpha) * predicted_width + width_alpha * expected_width
        ) * image_width / CANONICAL_WIDTH
        refined_height = (
            (1.0 - height_alpha) * predicted_height + height_alpha * expected_height
        ) * image_height / CANONICAL_HEIGHT
        refined = np.asarray(
            [
                center_x - refined_width / 2.0,
                center_x + refined_width / 2.0,
                center_y - refined_height / 2.0,
                center_y + refined_height / 2.0,
            ]
        )
        scale = np.asarray(
            [
                CANONICAL_WIDTH / image_width,
                CANONICAL_WIDTH / image_width,
                CANONICAL_HEIGHT / image_height,
                CANONICAL_HEIGHT / image_height,
            ]
        )
        errors = np.abs(refined - target) * scale
        per_image.append(
            {
                "edge_mean_px": float(errors.mean()),
                "edge_max_px": float(errors.max()),
            }
        )
    edge_mean = np.asarray([row["edge_mean_px"] for row in per_image])
    edge_max = np.asarray([row["edge_max_px"] for row in per_image])
    return (
        {
            "images": len(per_image),
            "applied_images": applied,
            "applied_rate": applied / max(len(per_image), 1),
            "image_edge_mean_px": _summary(edge_mean),
            "image_edge_max_px": _summary(edge_max),
        },
        per_image,
    )


def _paired_bootstrap(
    baseline: list[dict[str, float]],
    candidate: list[dict[str, float]],
    *,
    repeats: int = 2000,
    seed: int = 20260821,
) -> dict[str, Any]:
    if len(baseline) != len(candidate) or not baseline:
        return {"repeats": 0}
    rng = np.random.default_rng(seed)
    base_mean = np.asarray([row["edge_mean_px"] for row in baseline])
    cand_mean = np.asarray([row["edge_mean_px"] for row in candidate])
    base_max = np.asarray([row["edge_max_px"] for row in baseline])
    cand_max = np.asarray([row["edge_max_px"] for row in candidate])
    n = len(baseline)
    mean_delta: list[float] = []
    max_p95_delta: list[float] = []
    for _ in range(repeats):
        index = rng.integers(0, n, size=n)
        mean_delta.append(float(cand_mean[index].mean() - base_mean[index].mean()))
        max_p95_delta.append(
            float(np.percentile(cand_max[index], 95) - np.percentile(base_max[index], 95))
        )
    return {
        "repeats": repeats,
        "candidate_minus_baseline_mean_edge_px": {
            "point": float(cand_mean.mean() - base_mean.mean()),
            "ci95": [float(value) for value in np.percentile(mean_delta, [2.5, 97.5])],
        },
        "candidate_minus_baseline_image_max_p95_px": {
            "point": float(np.percentile(cand_max, 95) - np.percentile(base_max, 95)),
            "ci95": [float(value) for value in np.percentile(max_p95_delta, [2.5, 97.5])],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a physical inner-frame size prior against manual labels and replay."
    )
    parser.add_argument("--manual-manifest", type=Path, required=True)
    parser.add_argument("--official-manifest", type=Path, default=None)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inner-width-mm", type=float, default=58.0)
    parser.add_argument("--inner-height-mm", type=float, default=83.0)
    args = parser.parse_args()

    manual_rows = _read_csv(args.manual_manifest.resolve())
    official_rows = (
        _read_csv(args.official_manifest.resolve()) if args.official_manifest else []
    )
    evaluation_rows = _read_csv(args.evaluation.resolve())
    labels_by_id = {row["id"]: row for row in manual_rows}

    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manual_rows:
        by_source[row["source"]].append(row)

    baseline_val, _ = _evaluate_prior(
        evaluation_rows,
        labels_by_id,
        "val",
        alpha_width=0.0,
        alpha_height=0.0,
        threshold_width_px=1e9,
        threshold_height_px=1e9,
        inner_width_mm=args.inner_width_mm,
        inner_height_mm=args.inner_height_mm,
    )

    # Select only on validation.  The constraint deliberately allows at most a
    # 1% mean-MAE increase and optimizes the per-image worst-edge P95 tail.
    candidates: list[tuple[float, dict[str, float], dict[str, Any]]] = []
    for alpha_width in (0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0):
        for alpha_height in (0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0):
            for threshold_width in (4.0, 6.0, 8.0, 10.0, 12.0, 15.0):
                for threshold_height in (5.0, 8.0, 10.0, 12.0, 15.0, 20.0):
                    metrics, _ = _evaluate_prior(
                        evaluation_rows,
                        labels_by_id,
                        "val",
                        alpha_width=alpha_width,
                        alpha_height=alpha_height,
                        threshold_width_px=threshold_width,
                        threshold_height_px=threshold_height,
                        inner_width_mm=args.inner_width_mm,
                        inner_height_mm=args.inner_height_mm,
                    )
                    baseline_mean = baseline_val["image_edge_mean_px"]["mean"]
                    candidate_mean = metrics["image_edge_mean_px"]["mean"]
                    candidate_tail = metrics["image_edge_max_px"]["p95"]
                    if (
                        baseline_mean is not None
                        and candidate_mean is not None
                        and candidate_tail is not None
                        and candidate_mean <= baseline_mean * 1.01
                    ):
                        config = {
                            "alpha_width": alpha_width,
                            "alpha_height": alpha_height,
                            "threshold_width_px": threshold_width,
                            "threshold_height_px": threshold_height,
                        }
                        candidates.append((candidate_tail + 0.2 * candidate_mean, config, metrics))
    _, selected, selected_val = min(candidates, key=lambda item: item[0])

    baseline_test, baseline_rows = _evaluate_prior(
        evaluation_rows,
        labels_by_id,
        "test",
        alpha_width=0.0,
        alpha_height=0.0,
        threshold_width_px=1e9,
        threshold_height_px=1e9,
        inner_width_mm=args.inner_width_mm,
        inner_height_mm=args.inner_height_mm,
    )
    candidate_test, candidate_rows = _evaluate_prior(
        evaluation_rows,
        labels_by_id,
        "test",
        inner_width_mm=args.inner_width_mm,
        inner_height_mm=args.inner_height_mm,
        **selected,
    )

    report = {
        "physical_assumption": {
            "inner_width_mm": args.inner_width_mm,
            "inner_height_mm": args.inner_height_mm,
            "outer_width_mm": OUTER_WIDTH_MM,
            "outer_height_mm": OUTER_HEIGHT_MM,
            "canonical_width_px": CANONICAL_WIDTH,
            "canonical_height_px": CANONICAL_HEIGHT,
            "expected_inner_width_px": CANONICAL_WIDTH
            * args.inner_width_mm
            / OUTER_WIDTH_MM,
            "expected_inner_height_px": CANONICAL_HEIGHT
            * args.inner_height_mm
            / OUTER_HEIGHT_MM,
        },
        "manual_labels": _profile_labels(
            manual_rows, args.inner_width_mm, args.inner_height_mm
        ),
        "official_consensus_labels": (
            _profile_labels(official_rows, args.inner_width_mm, args.inner_height_mm)
            if official_rows
            else None
        ),
        "manual_by_source": {
            source: _profile_labels(rows, args.inner_width_mm, args.inner_height_mm)
            for source, rows in sorted(by_source.items())
        },
        "validation_selection": {
            "rule": "minimize worst-edge P95 + 0.2*mean, with mean <= baseline*1.01",
            "selected": selected,
            "baseline": baseline_val,
            "candidate": selected_val,
        },
        "locked_test": {
            "baseline": baseline_test,
            "guarded_physical_prior": candidate_test,
            "paired_bootstrap": _paired_bootstrap(baseline_rows, candidate_rows),
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "inner_physical_prior_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    source_rows: list[dict[str, Any]] = []
    for source, profile in report["manual_by_source"].items():
        source_rows.append(
            {
                "source": source,
                "rows": profile["rows"],
                "width_mean_px": profile["width_px"]["mean"],
                "width_std_px": profile["width_px"]["std"],
                "width_abs_deviation_p95_px": profile["width_abs_deviation_px"]["p95"],
                "height_mean_px": profile["height_px"]["mean"],
                "height_std_px": profile["height_px"]["std"],
                "height_abs_deviation_p95_px": profile["height_abs_deviation_px"]["p95"],
            }
        )
    with (args.output / "inner_physical_prior_by_source.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
