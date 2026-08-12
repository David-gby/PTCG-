from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from card_quality_processor.config import load_config  # noqa: E402
from card_quality_processor.io_utils import read_image, write_image, write_json  # noqa: E402
from card_quality_processor.outer_pose_detection import OuterPoseDetector  # noqa: E402
from card_quality_processor.rectification import rectify_card_by_points  # noqa: E402
from card_quality_processor.visualization import draw_outer_pose_result  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CORNER_NAMES = ("tl", "tr", "br", "bl")
GROUP_FIELDS = ("card_type", "glare_level", "background_type", "perspective_level")
ERROR_CASE_FIELDS = [
    "image_id",
    "image_path",
    "success",
    "error_code",
    "confidence",
    "corner_error_px_mean",
    "tl_error_px",
    "tr_error_px",
    "br_error_px",
    "bl_error_px",
    "visualization_path",
    "message",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the PTCG outer-card YOLO Pose model")
    parser.add_argument("--data", type=Path, default=Path("datasets/card_outer_pose/data.yaml"))
    parser.add_argument("--model", type=Path, default=Path("models/outer_pose.pt"))
    parser.add_argument("--output", type=Path, default=Path("reports/outer_pose_eval"))
    parser.add_argument("--config", type=Path, default=None, help="Optional processor YAML configuration")
    parser.add_argument("--conf", type=float, default=None, help="Override inference confidence threshold")
    parser.add_argument("--error-threshold", type=float, default=20.0, help="Mean corner error used to flag cases")
    return parser


def _dataset_root(data_path: Path, raw_root: str | Path) -> Path:
    root = Path(raw_root)
    if root.is_absolute():
        return root
    candidates = [Path.cwd() / root, data_path.parent / root]
    if root.name == data_path.parent.name:
        candidates.insert(0, data_path.parent)
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[0].resolve())


def _read_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        image_id = (row.get("image_id") or "").strip()
        image_path = (row.get("image_path") or "").strip()
        if image_id:
            result[image_id] = row
        if image_path:
            result[Path(image_path).stem] = row
    return result


def _label_path(image_path: Path, image_dir: Path, labels_dir: Path) -> Path:
    return labels_dir / image_path.relative_to(image_dir).with_suffix(".txt")


def _read_ground_truth(label_path: Path, image_shape: tuple[int, ...]) -> np.ndarray | None:
    if not label_path.is_file():
        return None
    height, width = image_shape[:2]
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split()
        if len(fields) < 17:
            continue
        try:
            values = [float(value) for value in fields]
            keypoints = np.asarray(values[5:17], dtype=np.float32).reshape(4, 3)
        except ValueError:
            continue
        if np.any(keypoints[:, 2] <= 0):
            return None
        keypoints[:, 0] *= width
        keypoints[:, 1] *= height
        return keypoints[:, :2]
    return None


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(sum(values) / len(values)) if values else None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    evaluated = [row for row in rows if row.get("corner_errors") is not None]
    all_means = [float(np.mean(row["corner_errors"])) for row in evaluated]
    result: dict[str, Any] = {
        "image_count": count,
        "labeled_image_count": len(evaluated),
        "corner_error_px_mean": _mean(all_means),
        "corner_error_px_median": float(median(all_means)) if all_means else None,
        "keypoint_success_rate": (sum(bool(row["pose_success"]) for row in rows) / count) if count else 0.0,
        "rectification_success_rate": (
            sum(bool(row["rectification_success"]) for row in rows) / count if count else 0.0
        ),
        "aspect_ratio_error": _mean(
            float(row["aspect_ratio_error"])
            for row in rows
            if row.get("aspect_ratio_error") is not None
        ),
    }
    for index, name in enumerate(CORNER_NAMES):
        result[f"{name}_error_px"] = _mean(float(row["corner_errors"][index]) for row in evaluated)
    return result


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for field in GROUP_FIELDS:
        values = sorted({str(row["metadata"].get(field, "")).strip() for row in rows} - {""})
        groups[field] = {
            value: _summarize([row for row in rows if str(row["metadata"].get(field, "")).strip() == value])
            for value in values
        }
    return groups


def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {args.data}")
    raw_data = yaml.safe_load(args.data.read_text(encoding="utf-8")) or {}
    dataset_root = _dataset_root(args.data.resolve(), raw_data.get("path", args.data.parent))
    image_dir = dataset_root / str(raw_data.get("test", "images/test"))
    labels_dir = dataset_root / "labels" / "test"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Test image directory not found: {image_dir}")

    output_dir = args.output
    visual_dir = output_dir / "visual_results"
    visual_dir.mkdir(parents=True, exist_ok=True)
    metadata = _read_metadata(dataset_root / "metadata.csv")
    config = load_config(args.config)
    pose_cfg = config["outer_detection"]["deep_pose"]
    pose_cfg["enabled"] = True
    pose_cfg["model_path"] = str(args.model)
    detector = OuterPoseDetector(model_path=args.model, config=config)
    conf = float(args.conf if args.conf is not None else pose_cfg["conf_threshold"])
    output_size = (int(pose_cfg["output_width"]), int(pose_cfg["output_height"]))

    images = sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    rows: list[dict[str, Any]] = []
    for image_path in images:
        image = read_image(image_path)
        if image is None:
            rows.append(
                {
                    "image_id": image_path.stem,
                    "image_path": str(image_path),
                    "pose_success": False,
                    "rectification_success": False,
                    "confidence": 0.0,
                    "error_code": "OUTER_POSE_NOT_DETECTED",
                    "message": "Could not read image.",
                    "corner_errors": None,
                    "aspect_ratio_error": None,
                    "visualization_path": "",
                    "metadata": metadata.get(image_path.stem, {}),
                }
            )
            continue
        pose = detector.predict(image, conf=conf)
        rectification = (
            rectify_card_by_points(image, pose["points"], output_size, config) if pose["success"] else None
        )
        ground_truth = _read_ground_truth(_label_path(image_path, image_dir, labels_dir), image.shape)
        corner_errors = None
        if ground_truth is not None and pose.get("points") is not None:
            corner_errors = np.linalg.norm(np.asarray(pose["points"], dtype=np.float32) - ground_truth, axis=1).tolist()
        relative_name = "__".join(image_path.relative_to(image_dir).with_suffix("").parts) + ".jpg"
        visual_path = visual_dir / relative_name
        visualization = draw_outer_pose_result(
            image,
            pose.get("points"),
            pose.get("bbox"),
            float(pose.get("confidence", 0.0)),
            pose.get("keypoint_confidence"),
            pose.get("error_code"),
            pose.get("message"),
        )
        write_image(visual_path, visualization)
        rows.append(
            {
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "pose_success": bool(pose["success"]),
                "rectification_success": bool(rectification and rectification["success"]),
                "confidence": float(pose.get("confidence", 0.0)),
                "error_code": pose.get("error_code") or "",
                "message": pose.get("message", ""),
                "corner_errors": corner_errors,
                "aspect_ratio_error": pose.get("metrics", {}).get("aspect_ratio_error"),
                "visualization_path": str(visual_path),
                "metadata": metadata.get(image_path.stem, {}),
            }
        )

    metrics = _summarize(rows)
    metrics["model_path"] = str(args.model)
    metrics["data_path"] = str(args.data)
    metrics["groups"] = _group_metrics(rows)
    return metrics, rows


def _error_case(row: Mapping[str, Any]) -> dict[str, Any]:
    errors = row.get("corner_errors")
    return {
        "image_id": row["image_id"],
        "image_path": row["image_path"],
        "success": row["pose_success"],
        "error_code": row["error_code"],
        "confidence": row["confidence"],
        "corner_error_px_mean": float(np.mean(errors)) if errors is not None else "",
        "tl_error_px": errors[0] if errors is not None else "",
        "tr_error_px": errors[1] if errors is not None else "",
        "br_error_px": errors[2] if errors is not None else "",
        "bl_error_px": errors[3] if errors is not None else "",
        "visualization_path": row["visualization_path"],
        "message": row["message"],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metrics, rows = evaluate(args)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(exc)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "metrics.json", metrics)
    error_rows = [
        row
        for row in rows
        if not row["pose_success"]
        or row.get("corner_errors") is None
        or float(np.mean(row["corner_errors"])) > float(args.error_threshold)
    ]
    with (args.output / "error_cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ERROR_CASE_FIELDS)
        writer.writeheader()
        writer.writerows(_error_case(row) for row in error_rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Metrics: {(args.output / 'metrics.json').resolve()}")
    print(f"Error cases: {(args.output / 'error_cases.csv').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
