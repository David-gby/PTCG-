from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ML_ROOT = Path(__file__).resolve().parents[2]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from ptcg_inference import CardFramePipeline, PipelineModels, read_image  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate latest outer pipeline on frozen feedback.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outer-seg", type=Path, required=True)
    parser.add_argument("--outer-pose", type=Path, default=None)
    parser.add_argument("--outer-line-refiner", type=Path, default=None)
    parser.add_argument("--outer-line-gate", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", default="train,val,blind")
    parser.add_argument("--device", default="0")
    parser.add_argument("--outward-canonical-px", type=float, default=0.0)
    parser.add_argument("--silhouette-conf", type=float, default=None)
    parser.add_argument("--disable-physical-edge-refinement", action="store_true")
    parser.add_argument("--disable-silhouette-fallback", action="store_true")
    args = parser.parse_args()

    requested = {item.strip() for item in args.splits.split(",") if item.strip()}
    with args.manifest.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = [row for row in csv.DictReader(handle) if row["split"] in requested]

    defaults = PipelineModels()
    models = PipelineModels(
        outer_seg=args.outer_seg.resolve(),
        outer_pose=(args.outer_pose.resolve() if args.outer_pose else defaults.outer_pose),
        outer_line_refiner=(
            args.outer_line_refiner.resolve()
            if args.outer_line_refiner
            else defaults.outer_line_refiner
        ),
        outer_line_gate=(
            args.outer_line_gate.resolve() if args.outer_line_gate else defaults.outer_line_gate
        ),
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
    for index, source in enumerate(source_rows, 1):
        target = np.asarray(json.loads(source["outer_target_points"]), dtype=np.float64)
        image = read_image(Path(source["original_image"]))
        start = time.perf_counter()
        outer = pipeline.outer_detector.predict(image, conf=0.25)
        detector_points = outer.get("points")
        detector_metrics = dict(outer.get("metrics", {}))
        if outer.get("success") and outer.get("points") is not None:
            outer = pipeline._refine_outer(image, outer)
        elapsed = time.perf_counter() - start
        success = bool(outer.get("success") and outer.get("points"))
        row: dict[str, Any] = {
            "sample_id": source["sample_id"],
            "split": source["split"],
            "issue_key": source["issue_key"],
            "success": success,
            "seconds": elapsed,
            "error_code": outer.get("error_code", ""),
        }
        if success:
            points = np.asarray(outer["points"], dtype=np.float64)
            errors = np.linalg.norm(points - target, axis=1)
            metrics = outer.get("metrics", {})
            row.update(
                {
                    "corner_mae_px": float(errors.mean()),
                    "corner_max_px": float(errors.max()),
                    "calibration_applied": bool(metrics.get("calibration_applied")),
                    "physical_refine_applied": bool(metrics.get("physical_edge_refinement_applied")),
                    "line_selected_source": metrics.get("outer_line_refinement", {}).get("selected_source", ""),
                    "raw_points": json.dumps(metrics.get("raw_points", [])),
                    "outer_metrics": json.dumps(metrics, ensure_ascii=False),
                    "prediction_points": json.dumps(points.tolist()),
                }
            )
        if detector_points is not None:
            detector_array = np.asarray(detector_points, dtype=np.float64)
            detector_errors = np.linalg.norm(detector_array - target, axis=1)
            row.update(
                {
                    "detector_corner_mae_px": float(detector_errors.mean()),
                    "detector_corner_max_px": float(detector_errors.max()),
                    "detector_points": json.dumps(detector_array.tolist()),
                    "detector_calibration_applied": bool(detector_metrics.get("calibration_applied")),
                    "detector_physical_refine_applied": bool(
                        detector_metrics.get("physical_edge_refinement_applied")
                    ),
                }
            )
        rows.append(row)
        print(f"evaluated {index}/{len(source_rows)} {source['sample_id']}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (args.output / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report: dict[str, Any] = {
        "outer_seg": str(args.outer_seg.resolve()),
        "outer_pose": (
            str(args.outer_pose.resolve()) if args.outer_pose else str(defaults.outer_pose)
        ),
        "outer_line_refiner": (
            str(args.outer_line_refiner.resolve())
            if args.outer_line_refiner
            else str(defaults.outer_line_refiner)
        ),
        "outward_canonical_px": args.outward_canonical_px,
        "outer_line_gate": (
            str(args.outer_line_gate.resolve())
            if args.outer_line_gate
            else str(defaults.outer_line_gate)
        ),
        "silhouette_conf_threshold": args.silhouette_conf,
        "physical_edge_refinement_enabled": not args.disable_physical_edge_refinement,
        "silhouette_fallback_enabled": not args.disable_silhouette_fallback,
        "splits": {},
    }
    for split in sorted(requested) + ["all"]:
        subset = rows if split == "all" else [row for row in rows if row["split"] == split]
        successes = [row for row in subset if row["success"]]
        report["splits"][split] = {
            "samples": len(subset),
            "successes": len(successes),
            "success_rate": len(successes) / max(len(subset), 1),
            "corner_mae_px": _summary([float(row["corner_mae_px"]) for row in successes]),
            "corner_max_px": _summary([float(row["corner_max_px"]) for row in successes]),
            "seconds": _summary([float(row["seconds"]) for row in subset]),
            "calibration_applied_rate": sum(bool(row.get("calibration_applied")) for row in successes)
            / max(len(successes), 1),
            "physical_refine_applied_rate": sum(bool(row.get("physical_refine_applied")) for row in successes)
            / max(len(successes), 1),
        }
    (args.output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
