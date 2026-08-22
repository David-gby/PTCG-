from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ML_ROOT = Path(__file__).resolve().parents[2]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from ptcg_inference import CardFramePipeline, PipelineModels, read_image  # noqa: E402
from training.official_corpus.evaluate_outer_seg_exactness import (  # noqa: E402
    _canonical_metrics,
    _target,
)


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


def _aggregate(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    successes = [row for row in rows if row.get(f"{prefix}_success")]
    return {
        "samples": len(rows),
        "successes": len(successes),
        "success_rate": len(successes) / max(len(rows), 1),
        "image_corner_mae_px": _summary([float(row[f"{prefix}_image_corner_mae_px"]) for row in successes]),
        "canonical_edge_mae_px": _summary([float(row[f"{prefix}_canonical_edge_mae_px"]) for row in successes]),
        "canonical_edge_max_px": _summary([float(row[f"{prefix}_canonical_edge_max_px"]) for row in successes]),
        "max_inward_px": _summary([float(row[f"{prefix}_max_inward_px"]) for row in successes]),
        "max_outward_px": _summary([float(row[f"{prefix}_max_outward_px"]) for row in successes]),
        "area_ratio": _summary([float(row[f"{prefix}_area_ratio"]) for row in successes]),
        "cut_risk_rate": sum(bool(row[f"{prefix}_cut_risk"]) for row in successes) / max(len(successes), 1),
    }


def _add_metrics(row: dict[str, Any], prefix: str, points: Any, target: np.ndarray) -> None:
    if points is None:
        row[f"{prefix}_success"] = False
        return
    array = np.asarray(points, dtype=np.float32)
    row[f"{prefix}_success"] = True
    row[f"{prefix}_image_corner_mae_px"] = float(np.linalg.norm(array - target, axis=1).mean())
    for key, value in _canonical_metrics(array, target).items():
        row[f"{prefix}_{key}"] = value
    row[f"{prefix}_points"] = json.dumps(array.round(3).tolist())


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate detector and final four-side policy together.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--line-refiner", type=Path, default=None)
    parser.add_argument("--line-gate", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="0")
    parser.add_argument("--outward-canonical-px", type=float, default=0.0)
    parser.add_argument("--silhouette-conf", type=float, default=None)
    parser.add_argument("--max-per-source", type=int, default=200)
    parser.add_argument(
        "--disable-physical-edge-refinement",
        action="store_true",
        help="Evaluate the silhouette quad without the legacy full-resolution edge snap.",
    )
    parser.add_argument(
        "--disable-silhouette-fallback",
        action="store_true",
        help="Do not replace the primary mask with legacy-mask/pose consensus.",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    with args.metadata.resolve().open("r", encoding="utf-8-sig", newline="") as stream:
        metadata = {row["sample_id"]: row for row in csv.DictReader(stream)}
    by_source: dict[str, list[Path]] = defaultdict(list)
    for path in sorted((dataset / "images" / args.split).glob("*")):
        if path.is_file():
            by_source[metadata.get(path.stem, {}).get("source", "unknown")].append(path)
    images = [
        path
        for source in sorted(by_source)
        for path in by_source[source][: max(1, args.max_per_source)]
    ]
    defaults = PipelineModels()
    models = PipelineModels(
        outer_seg=args.model.resolve(),
        outer_pose=defaults.outer_pose,
        outer_line_refiner=(
            args.line_refiner.resolve() if args.line_refiner else defaults.outer_line_refiner
        ),
        outer_line_gate=(args.line_gate.resolve() if args.line_gate else defaults.outer_line_gate),
        inner_yolo=defaults.inner_yolo,
        inner_refiner=defaults.inner_refiner,
        inner_refiner_top_left=defaults.inner_refiner_top_left,
        inner_gate=defaults.inner_gate,
    )
    calibration = {
        "outer_detection": {
            "deep_pose": {
                "physical_edge_refinement": {
                    "enabled": not args.disable_physical_edge_refinement,
                },
                "silhouette_refinement": {
                    "fallback": {
                        "enabled": not args.disable_silhouette_fallback,
                    },
                    **(
                        {"conf_threshold": float(args.silhouette_conf)}
                        if args.silhouette_conf is not None
                        else {}
                    ),
                    "corner_calibration": {
                        "enabled": args.outward_canonical_px > 0.0,
                        "outward_canonical_px": args.outward_canonical_px,
                        "canonical_width": 630.0,
                        "canonical_height": 880.0,
                    }
                }
            }
        }
    }
    pipeline = CardFramePipeline(device=args.device, models=models, outer_config=calibration)
    rows: list[dict[str, Any]] = []
    started = time.time()
    for index, image_path in enumerate(images, 1):
        source = metadata.get(image_path.stem, {"source": "unknown"})
        label_path = dataset / "labels" / args.split / f"{image_path.stem}.txt"
        target = _target(image_path, label_path, source)
        image = read_image(image_path)
        outer = pipeline.outer_detector.predict(image, conf=0.25)
        detector_points = outer.get("points")
        final = pipeline._refine_outer(image, outer) if outer.get("success") and detector_points is not None else outer
        row: dict[str, Any] = {
            "sample_id": image_path.stem,
            "source": source.get("source", "unknown"),
            "shadow_type": source.get("shadow_type", ""),
            "sleeve": source.get("sleeve", ""),
            "glare": source.get("glare", ""),
            "line_selected_source": final.get("metrics", {}).get("outer_line_refinement", {}).get("selected_source", ""),
        }
        _add_metrics(row, "detector", detector_points, target)
        _add_metrics(row, "final", final.get("points") if final.get("success") else None, target)
        rows.append(row)
        if index % 50 == 0:
            print(f"evaluated {index}/{len(images)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report: dict[str, Any] = {
        "model": str(args.model.resolve()),
        "line_refiner": str(args.line_refiner.resolve()) if args.line_refiner else str(defaults.outer_line_refiner),
        "line_gate": str(args.line_gate.resolve()) if args.line_gate else str(defaults.outer_line_gate),
        "outward_canonical_px": args.outward_canonical_px,
        "physical_edge_refinement_enabled": not args.disable_physical_edge_refinement,
        "silhouette_fallback_enabled": not args.disable_silhouette_fallback,
        "silhouette_conf_threshold": args.silhouette_conf,
        "split": args.split,
        "elapsed_seconds": time.time() - started,
        "overall": {mode: _aggregate(rows, mode) for mode in ("detector", "final")},
        "by_source": {
            source: {
                mode: _aggregate([row for row in rows if row["source"] == source], mode)
                for mode in ("detector", "final")
            }
            for source in sorted({str(row["source"]) for row in rows})
        },
        "line_sources": {
            source: sum(row["line_selected_source"] == source for row in rows)
            for source in sorted({str(row["line_selected_source"]) for row in rows})
        },
    }
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
