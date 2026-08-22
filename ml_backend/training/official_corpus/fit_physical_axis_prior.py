from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CANONICAL_WIDTH = 630.0
CANONICAL_HEIGHT = 880.0
EXPECTED_WIDTH = 580.0
EXPECTED_HEIGHT = 830.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def array_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def materialize(
    evaluation: list[dict[str, str]],
    labels: dict[str, dict[str, str]],
    split: str,
) -> dict[str, np.ndarray]:
    rows = [
        row
        for row in evaluation
        if row.get("success", "").lower() == "true" and row["split"] == split
    ]
    image_width = np.asarray([float(labels[row["id"]]["width"]) for row in rows])
    image_height = np.asarray([float(labels[row["id"]]["height"]) for row in rows])
    scale = np.stack(
        (
            CANONICAL_WIDTH / image_width,
            CANONICAL_WIDTH / image_width,
            CANONICAL_HEIGHT / image_height,
            CANONICAL_HEIGHT / image_height,
        ),
        axis=1,
    )
    prediction = np.asarray(
        [
            [
                float(row["prediction_left"]),
                float(row["prediction_right"]),
                float(row["prediction_top"]),
                float(row["prediction_bottom"]),
            ]
            for row in rows
        ]
    ) * scale
    target = np.asarray(
        [
            [
                float(row["target_left"]),
                float(row["target_right"]),
                float(row["target_top"]),
                float(row["target_bottom"]),
            ]
            for row in rows
        ]
    ) * scale
    return {
        "prediction": prediction,
        "target": target,
        "source": np.asarray([row["source"] for row in rows], dtype=object),
    }


def apply_axis(
    prediction: np.ndarray,
    *,
    horizontal: bool,
    alpha: float,
    threshold: float,
    first_share: float,
) -> np.ndarray:
    output = prediction.copy()
    first, second = (0, 1) if horizontal else (2, 3)
    expected = EXPECTED_WIDTH if horizontal else EXPECTED_HEIGHT
    current = output[:, second] - output[:, first]
    correction = alpha * (expected - current)
    correction[np.abs(expected - current) <= threshold] = 0.0
    output[:, first] -= first_share * correction
    output[:, second] += (1.0 - first_share) * correction
    return output


def axis_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    horizontal: bool,
) -> dict[str, float | int]:
    indices = (0, 1) if horizontal else (2, 3)
    errors = np.abs(prediction[:, indices] - target[:, indices])
    pair_mean = errors.mean(axis=1)
    pair_max = errors.max(axis=1)
    return {
        "images": int(errors.shape[0]),
        "pair_mean": float(pair_mean.mean()),
        "pair_mean_p95": float(np.percentile(pair_mean, 95)),
        "pair_max_p95": float(np.percentile(pair_max, 95)),
        "pair_max": float(pair_max.max()),
        "first_signed_mean": float((prediction[:, indices[0]] - target[:, indices[0]]).mean()),
        "second_signed_mean": float((prediction[:, indices[1]] - target[:, indices[1]]).mean()),
    }


def fit_axis(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    horizontal: bool,
    sources: np.ndarray | None = None,
) -> tuple[dict[str, float], dict[str, float | int], dict[str, float | int]]:
    baseline = axis_metrics(prediction, target, horizontal=horizontal)
    candidates: list[tuple[float, dict[str, float], dict[str, float | int]]] = []
    for alpha in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0):
        for threshold in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0):
            for first_share in (0.0, 0.25, 0.50, 0.75, 1.0):
                candidate = apply_axis(
                    prediction,
                    horizontal=horizontal,
                    alpha=alpha,
                    threshold=threshold,
                    first_share=first_share,
                )
                metrics = axis_metrics(candidate, target, horizontal=horizontal)
                source_safe = True
                if sources is not None:
                    for source in sorted(set(sources)):
                        selected_rows = sources == source
                        if int(np.count_nonzero(selected_rows)) < 10:
                            continue
                        source_baseline = axis_metrics(
                            prediction[selected_rows],
                            target[selected_rows],
                            horizontal=horizontal,
                        )
                        source_candidate = axis_metrics(
                            candidate[selected_rows],
                            target[selected_rows],
                            horizontal=horizontal,
                        )
                        if (
                            source_candidate["pair_mean"]
                            > source_baseline["pair_mean"] * 1.005
                        ):
                            source_safe = False
                            break
                # Never buy a tail improvement with more than 0.25% mean regression.
                if (
                    source_safe
                    and metrics["pair_mean"] <= baseline["pair_mean"] * 1.0025
                ):
                    score = (
                        metrics["pair_mean"]
                        + 0.35 * metrics["pair_mean_p95"]
                        + 0.20 * metrics["pair_max_p95"]
                    )
                    candidates.append(
                        (
                            float(score),
                            {
                                "alpha": float(alpha),
                                "threshold_px": float(threshold),
                                "first_share": float(first_share),
                            },
                            metrics,
                        )
                    )
    _, parameters, selected = min(candidates, key=lambda item: item[0])
    return parameters, baseline, selected


def combined_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, object]:
    errors = np.abs(prediction - target)
    image_mean = errors.mean(axis=1)
    image_max = errors.max(axis=1)
    return {
        "image_mean": array_summary(image_mean),
        "image_max": array_summary(image_max),
        "by_edge": {
            edge: {
                "absolute": array_summary(errors[:, index]),
                "signed_mean": float((prediction[:, index] - target[:, index]).mean()),
            }
            for index, edge in enumerate(("left", "right", "top", "bottom"))
        },
    }


def paired_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    *,
    repeats: int = 5000,
    seed: int = 20260821,
) -> dict[str, object]:
    baseline_mean = np.abs(baseline - target).mean(axis=1)
    candidate_mean = np.abs(candidate - target).mean(axis=1)
    baseline_max = np.abs(baseline - target).max(axis=1)
    candidate_max = np.abs(candidate - target).max(axis=1)
    rng = np.random.default_rng(seed)
    n = len(target)
    mean_delta: list[float] = []
    max_delta: list[float] = []
    for _ in range(repeats):
        index = rng.integers(0, n, size=n)
        mean_delta.append(float(candidate_mean[index].mean() - baseline_mean[index].mean()))
        max_delta.append(float(candidate_max[index].mean() - baseline_max[index].mean()))
    return {
        "repeats": repeats,
        "candidate_minus_baseline_image_mean_px": {
            "point": float(candidate_mean.mean() - baseline_mean.mean()),
            "ci95": [float(value) for value in np.percentile(mean_delta, [2.5, 97.5])],
        },
        "candidate_minus_baseline_image_max_mean_px": {
            "point": float(candidate_max.mean() - baseline_max.mean()),
            "ci95": [float(value) for value in np.percentile(max_delta, [2.5, 97.5])],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit split-safe pairwise 58x83 mm inner-edge size corrections."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels_rows = read_csv(args.manifest.resolve())
    labels = {row["id"]: row for row in labels_rows}
    evaluation = read_csv(args.evaluation.resolve())
    validation = materialize(evaluation, labels, "val")
    test = materialize(evaluation, labels, "test")

    horizontal, baseline_horizontal, selected_horizontal = fit_axis(
        validation["prediction"],
        validation["target"],
        horizontal=True,
        sources=validation["source"],
    )
    vertical, baseline_vertical, selected_vertical = fit_axis(
        validation["prediction"],
        validation["target"],
        horizontal=False,
        sources=validation["source"],
    )
    candidate_test = apply_axis(
        test["prediction"], horizontal=True, **{
            "alpha": horizontal["alpha"],
            "threshold": horizontal["threshold_px"],
            "first_share": horizontal["first_share"],
        }
    )
    candidate_test = apply_axis(
        candidate_test, horizontal=False, **{
            "alpha": vertical["alpha"],
            "threshold": vertical["threshold_px"],
            "first_share": vertical["first_share"],
        }
    )
    report = {
        "semantics": "58x83 mm is measured inner-edge to inner-edge for every layout",
        "canonical_expected_px": {"width": EXPECTED_WIDTH, "height": EXPECTED_HEIGHT},
        "selection_policy": "parameters selected only on val; test remains locked",
        "validation": {
            "horizontal_parameters": horizontal,
            "vertical_parameters": vertical,
            "horizontal_baseline": baseline_horizontal,
            "horizontal_selected": selected_horizontal,
            "vertical_baseline": baseline_vertical,
            "vertical_selected": selected_vertical,
        },
        "locked_test": {
            "baseline": combined_metrics(test["prediction"], test["target"]),
            "candidate": combined_metrics(candidate_test, test["target"]),
            "paired_bootstrap": paired_bootstrap(
                test["prediction"], candidate_test, test["target"]
            ),
            "by_source": {
                source: {
                    "images": int(np.count_nonzero(test["source"] == source)),
                    "baseline": combined_metrics(
                        test["prediction"][test["source"] == source],
                        test["target"][test["source"] == source],
                    ),
                    "candidate": combined_metrics(
                        candidate_test[test["source"] == source],
                        test["target"][test["source"] == source],
                    ),
                }
                for source in sorted(set(test["source"]))
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
