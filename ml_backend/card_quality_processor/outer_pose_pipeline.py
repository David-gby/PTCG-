from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

from .config import normalize_config
from .io_utils import read_image, write_image, write_json
from .outer_pose_detection import KEYPOINT_NAMES, OuterPoseDetector
from .rectification import rectify_card_by_points
from .visualization import draw_outer_pose_result


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CSV_FIELDS = [
    "image_id",
    "image_path",
    "success",
    "confidence",
    "bbox_confidence",
    "mean_keypoint_confidence",
    "min_keypoint_confidence",
    "aspect_ratio",
    "aspect_ratio_error",
    "area_ratio",
    "error_code",
    "rectified_path",
    "visualization_path",
    "message",
]


def _empty_pose(message: str) -> dict[str, Any]:
    return {
        "success": False,
        "points": None,
        "bbox": None,
        "confidence": 0.0,
        "keypoint_confidence": {name: 0.0 for name in KEYPOINT_NAMES},
        "method": "deep_outer_pose",
        "error_code": "OUTER_POSE_NOT_DETECTED",
        "message": message,
        "metrics": {
            "bbox_confidence": 0.0,
            "mean_keypoint_confidence": 0.0,
            "min_keypoint_confidence": 0.0,
            "aspect_ratio": 0.0,
            "aspect_ratio_error": 1.0,
            "area_ratio": 0.0,
        },
    }


def process_outer_pose_and_rectify(
    image_path: str | Path,
    output_dir: str | Path,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run deep outer-pose detection and guarded perspective rectification."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    image_id = image_path.stem
    image_output = output_dir / image_id
    image_output.mkdir(parents=True, exist_ok=True)
    for generated_name in (
        "outer_pose_result.jpg",
        "outer_pose_failed.jpg",
        "rectified_card.jpg",
        "outer_pose_result.json",
    ):
        (image_output / generated_name).unlink(missing_ok=True)

    output_paths: dict[str, str | None] = {
        "visualization_path": None,
        "rectified_path": None,
        "result_json_path": str(image_output / "outer_pose_result.json"),
    }
    image = read_image(image_path)
    if image is None:
        pose = _empty_pose(f"Could not read image: {image_path}")
        json_result = {
            "image_path": str(image_path),
            "image_id": image_id,
            "success": False,
            "outer_pose_result": pose,
            "rectification_result": None,
            "output_paths": output_paths,
        }
        write_json(image_output / "outer_pose_result.json", json_result)
        return {
            "image_path": str(image_path),
            "success": False,
            "outer_pose_result": pose,
            "rectification_result": None,
            "output_paths": output_paths,
        }

    cfg = normalize_config(config)
    pose_cfg = cfg["outer_detection"]["deep_pose"]
    detector = OuterPoseDetector(config=cfg)
    pose = detector.predict(image, conf=float(pose_cfg["conf_threshold"]))
    visualization = draw_outer_pose_result(
        image,
        pose.get("points"),
        # The axis-aligned detector bbox is retained in JSON for diagnostics,
        # but the rendered result shows only the final quadrilateral that is
        # actually used for perspective rectification.
        None,
        float(pose.get("confidence", 0.0)),
        pose.get("keypoint_confidence"),
        pose.get("error_code"),
        pose.get("message"),
    )
    visualization_name = "outer_pose_result.jpg" if pose["success"] else "outer_pose_failed.jpg"
    visualization_path = write_image(image_output / visualization_name, visualization)
    output_paths["visualization_path"] = str(visualization_path)

    rectification: dict[str, Any] | None = None
    if pose["success"]:
        output_size = (int(pose_cfg["output_width"]), int(pose_cfg["output_height"]))
        rectification = rectify_card_by_points(image, pose["points"], output_size, cfg)
        if rectification["success"]:
            rectified_path = write_image(image_output / "rectified_card.jpg", rectification["rectified_image"])
            output_paths["rectified_path"] = str(rectified_path)
            rectification["confidence"] = float(pose["confidence"])

    success = bool(pose["success"] and rectification and rectification["success"])
    json_result = {
        "image_path": str(image_path),
        "image_id": image_id,
        "success": success,
        "outer_pose_result": pose,
        "rectification_result": None
        if rectification is None
        else {key: value for key, value in rectification.items() if key != "rectified_image"},
        "output_paths": output_paths,
    }
    write_json(image_output / "outer_pose_result.json", json_result)
    return {
        "image_path": str(image_path),
        "success": success,
        "outer_pose_result": pose,
        "rectification_result": rectification,
        "output_paths": output_paths,
    }


def _csv_row(result: Mapping[str, Any]) -> dict[str, Any]:
    pose = result["outer_pose_result"]
    metrics = pose.get("metrics", {})
    output_paths = result.get("output_paths", {})
    return {
        "image_id": Path(result["image_path"]).stem,
        "image_path": result["image_path"],
        "success": bool(result["success"]),
        "confidence": pose.get("confidence", 0.0),
        "bbox_confidence": metrics.get("bbox_confidence", 0.0),
        "mean_keypoint_confidence": metrics.get("mean_keypoint_confidence", 0.0),
        "min_keypoint_confidence": metrics.get("min_keypoint_confidence", 0.0),
        "aspect_ratio": metrics.get("aspect_ratio", 0.0),
        "aspect_ratio_error": metrics.get("aspect_ratio_error", 1.0),
        "area_ratio": metrics.get("area_ratio", 0.0),
        "error_code": pose.get("error_code") or "",
        "rectified_path": output_paths.get("rectified_path") or "",
        "visualization_path": output_paths.get("visualization_path") or "",
        "message": pose.get("message", ""),
    }


def batch_process_outer_pose_dataset(
    input_dir: str | Path,
    output_dir: str | Path = "data/processed/outer_pose_rectified",
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = (
        sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if input_dir.exists()
        else []
    )
    results = [process_outer_pose_and_rectify(path, output_dir, config) for path in images]
    report_path = output_dir / "batch_outer_pose_report.csv"
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_row(result) for result in results)
    confidences = [float(result["outer_pose_result"].get("confidence", 0.0)) for result in results]
    successful = sum(bool(result["success"]) for result in results)
    return {
        "processed_count": len(results),
        "success_count": successful,
        "failure_count": len(results) - successful,
        "average_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "report_path": str(report_path),
        "output_dir": str(output_dir),
        "results": results,
    }
