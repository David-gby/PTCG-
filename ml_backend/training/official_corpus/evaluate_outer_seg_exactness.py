from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ML_ROOT = Path(__file__).resolve().parents[2]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from card_quality_processor.outer_silhouette import (  # noqa: E402
    extract_silhouette_prediction,
    order_quad,
    polygon_to_quad,
)


CANONICAL = np.asarray(
    [[0.0, 0.0], [630.0, 0.0], [630.0, 880.0], [0.0, 880.0]],
    dtype=np.float32,
)


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


def _metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def _target(image_path: Path, label_path: Path, row: dict[str, str]) -> np.ndarray:
    points = row.get("points")
    if points:
        return order_quad(np.asarray(json.loads(points), dtype=np.float32))
    height, width = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR).shape[:2]
    values = [float(value) for value in label_path.read_text(encoding="utf-8").splitlines()[0].split()]
    polygon = np.asarray(values[1:], dtype=np.float32).reshape(-1, 2)
    polygon *= np.asarray([width, height], dtype=np.float32)
    quad = polygon_to_quad(polygon)
    if quad is None:
        raise ValueError(f"cannot convert target polygon: {label_path}")
    return quad


def _canonical_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    transform = cv2.getPerspectiveTransform(target.astype(np.float32), CANONICAL)
    canonical = cv2.perspectiveTransform(prediction.reshape(1, 4, 2), transform)[0]
    signed = {
        "top": float(np.mean(canonical[[0, 1], 1])),
        "right": float(630.0 - np.mean(canonical[[1, 2], 0])),
        "bottom": float(880.0 - np.mean(canonical[[2, 3], 1])),
        "left": float(np.mean(canonical[[3, 0], 0])),
    }
    absolute = [abs(value) for value in signed.values()]
    inward = [max(value, 0.0) for value in signed.values()]
    outward = [max(-value, 0.0) for value in signed.values()]
    corner = np.linalg.norm(canonical - CANONICAL, axis=1)
    target_area = abs(float(cv2.contourArea(target.reshape(-1, 1, 2))))
    predicted_area = abs(float(cv2.contourArea(prediction.reshape(-1, 1, 2))))
    return {
        "canonical_corner_mae_px": float(np.mean(corner)),
        "canonical_corner_max_px": float(np.max(corner)),
        "canonical_edge_mae_px": float(np.mean(absolute)),
        "canonical_edge_max_px": float(np.max(absolute)),
        "max_inward_px": float(max(inward)),
        "mean_inward_px": float(np.mean(inward)),
        "max_outward_px": float(max(outward)),
        "area_ratio": predicted_area / max(target_area, 1.0),
        "cut_risk": bool(max(inward) > 3.0),
        **{f"signed_{key}_px": value for key, value in signed.items()},
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row["success"]]
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
        "successes": len(successes),
        "success_rate": len(successes) / max(len(rows), 1),
        "cut_risk_count": sum(bool(row["cut_risk"]) for row in successes),
        "cut_risk_rate": sum(bool(row["cut_risk"]) for row in successes) / max(len(successes), 1),
        **{field: _summary([float(row[field]) for row in successes]) for field in fields},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure outer-mask corner and edge exactness.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from ultralytics import YOLO

    dataset = args.dataset.resolve()
    image_dir = dataset / "images" / args.split
    label_dir = dataset / "labels" / args.split
    images = sorted(path for path in image_dir.glob("*") if path.is_file())
    if args.limit > 0:
        images = images[: args.limit]
    metadata = _metadata(args.metadata.resolve())
    model = YOLO(str(args.model.resolve()))
    rows: list[dict[str, Any]] = []

    def prediction_pairs():
        for offset in range(0, len(images), max(1, args.batch)):
            chunk = images[offset : offset + max(1, args.batch)]
            results = model.predict(
                source=[str(path) for path in chunk],
                stream=False,
                imgsz=args.imgsz,
                batch=max(1, args.batch),
                device=args.device,
                conf=0.20,
                iou=0.70,
                retina_masks=False,
                verbose=False,
            )
            yield from zip(chunk, results, strict=True)

    for index, (image_path, raw) in enumerate(prediction_pairs(), 1):
        sample_id = image_path.stem
        source = metadata.get(sample_id, {"source": "unknown"})
        target = _target(image_path, label_dir / f"{sample_id}.txt", source)
        extracted = extract_silhouette_prediction(raw, image_shape=raw.orig_shape)
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "source": source.get("source", "unknown"),
            "shadow_type": source.get("shadow_type", ""),
            "sleeve": source.get("sleeve", ""),
            "glare": source.get("glare", ""),
            "success": extracted is not None,
            "confidence": float(extracted["confidence"]) if extracted else None,
        }
        if extracted is not None:
            points = order_quad(np.asarray(extracted["points"], dtype=np.float32))
            row["image_corner_mae_px"] = float(np.linalg.norm(points - target, axis=1).mean())
            row.update(_canonical_metrics(points, target))
            row["prediction_points"] = json.dumps(points.round(3).tolist())
            row["target_points"] = json.dumps(target.round(3).tolist())
        rows.append(row)
        if index % 100 == 0:
            print(f"evaluated {index}/{len(images)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (args.output / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "model": str(args.model.resolve()),
        "dataset": str(dataset),
        "split": args.split,
        "overall": _aggregate(rows),
        "by_source": {
            source: _aggregate([row for row in rows if row["source"] == source])
            for source in sorted({str(row["source"]) for row in rows})
        },
        "by_shadow": {
            shadow: _aggregate([row for row in rows if row["shadow_type"] == shadow])
            for shadow in sorted({str(row["shadow_type"]) for row in rows if row["shadow_type"]})
        },
    }
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
