from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import cv2
import numpy as np

from feedback import FEEDBACK_SCHEMA_VERSION, normalize_inner_box, normalize_outer_points, read_image, sha256_file


def _load_annotations(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    annotation_dir = root / "annotations"
    if not annotation_dir.is_dir():
        return []
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(annotation_dir.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            rows.append((path, {"_load_error": str(exc)}))
            continue
        rows.append((path, value))
    return rows


def validate_feedback_package(
    feedback_root: str | Path,
    *,
    sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    root = Path(feedback_root).resolve()
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    approved = 0
    eligible = 0
    annotations = _load_annotations(root)
    if sample_ids is not None:
        annotations = [
            (path, row)
            for path, row in annotations
            if str(row.get("sample_id", path.stem)) in sample_ids
        ]
    if not annotations:
        errors.append({"sample_id": "", "message": "No annotation JSON files were found."})

    for path, row in annotations:
        sample_id = str(row.get("sample_id", path.stem))
        if "_load_error" in row:
            errors.append({"sample_id": sample_id, "message": row["_load_error"]})
            continue
        if row.get("schema_version") != FEEDBACK_SCHEMA_VERSION:
            errors.append({"sample_id": sample_id, "message": "Unsupported schema_version."})
            continue
        image_info = row.get("image", {})
        image_path = root / str(image_info.get("path", ""))
        rectified_path = root / str(row.get("rectification", {}).get("image_path", ""))
        if not image_path.is_file():
            errors.append({"sample_id": sample_id, "message": "Original image is missing."})
            continue
        if not rectified_path.is_file():
            errors.append({"sample_id": sample_id, "message": "Rectified image is missing."})
            continue
        if sha256_file(image_path) != image_info.get("sha256"):
            errors.append({"sample_id": sample_id, "message": "Original image SHA-256 mismatch."})
        image = read_image(image_path)
        height, width = image.shape[:2]
        if [width, height] != [int(image_info.get("width", -1)), int(image_info.get("height", -1))]:
            errors.append({"sample_id": sample_id, "message": "Original image dimensions mismatch."})
        outer_correction = row.get("outer_frame", {}).get("correction")
        if outer_correction is not None:
            try:
                normalize_outer_points(outer_correction["points"], width, height)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append({"sample_id": sample_id, "message": f"Invalid outer correction: {exc}"})
        inner_correction = row.get("inner_frame", {}).get("correction")
        if inner_correction is not None:
            try:
                normalize_inner_box(inner_correction["box"], 630, 880)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append({"sample_id": sample_id, "message": f"Invalid inner correction: {exc}"})
        review = row.get("review", {})
        approved += int(bool(review.get("approved_for_training")))
        eligible += int(bool(review.get("training_eligible")))
        if review.get("approved_for_training") and not review.get("training_eligible"):
            warnings.append(
                {"sample_id": sample_id, "message": "Approved sample is not eligible for training."}
            )
        if outer_correction is not None and inner_correction is not None:
            reference = row.get("inner_frame", {}).get("coordinates_reference")
            if reference != "rectified_from_manual_outer_correction":
                errors.append(
                    {
                        "sample_id": sample_id,
                        "message": "Inner correction is not tied to the manually corrected outer rectification.",
                    }
                )

    return {
        "success": not errors,
        "feedback_root": str(root),
        "annotation_count": len(annotations),
        "approved_count": approved,
        "training_eligible_count": eligible,
        "errors": errors,
        "warnings": warnings,
    }


def _format(values: list[float]) -> str:
    return " ".join(f"{value:.8f}" for value in values)


def _copy_image(source: Path, destination_dir: Path, sample_id: str) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower() or ".jpg"
    target = destination_dir / f"{sample_id}{suffix}"
    shutil.copy2(source, target)
    return target


def _write_dataset_yaml(root: Path, *, pose: bool, class_name: str) -> None:
    value: dict[str, Any] = {
        "path": root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: class_name},
    }
    if pose:
        value["kpt_shape"] = [4, 3]
        value["flip_idx"] = [1, 0, 3, 2]
    root.mkdir(parents=True, exist_ok=True)
    lines = [
        f"path: {json.dumps(value['path'], ensure_ascii=False)}",
        f"train: {value['train']}",
        f"val: {value['val']}",
        f"test: {value['test']}",
        "names:",
        f"  0: {class_name}",
    ]
    if pose:
        lines.extend(["kpt_shape: [4, 3]", "flip_idx: [1, 0, 3, 2]"])
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_feedback_to_training(
    feedback_root: str | Path,
    output_root: str | Path,
    *,
    split: str = "train",
    allow_accepted_predictions: bool = False,
    sample_splits: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    feedback_root = Path(feedback_root).resolve()
    output_root = Path(output_root).resolve()
    normalized_splits = None
    if sample_splits is not None:
        normalized_splits = {str(key): str(value) for key, value in sample_splits.items()}
        invalid_splits = sorted(set(normalized_splits.values()) - {"train", "val", "test"})
        if invalid_splits:
            raise ValueError(f"sample_splits contains invalid values: {invalid_splits}")
        if not normalized_splits:
            raise ValueError("sample_splits must not be empty")
    selected_ids = set(normalized_splits) if normalized_splits is not None else None
    validation = validate_feedback_package(feedback_root, sample_ids=selected_ids)
    if not validation["success"]:
        raise ValueError("Feedback package validation failed; inspect validation_report.json.")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    outer_pose_root = output_root / "outer_pose"
    outer_seg_root = output_root / "outer_seg"
    inner_seg_root = output_root / "inner_seg"
    for dataset_root, pose, class_name in (
        (outer_pose_root, True, "card"),
        (outer_seg_root, False, "card"),
        (inner_seg_root, False, "inner_frame"),
    ):
        _write_dataset_yaml(dataset_root, pose=pose, class_name=class_name)

    refiner_rows: list[dict[str, Any]] = []
    converted = {"outer_pose": 0, "outer_seg": 0, "inner_seg": 0, "inner_refiner": 0}
    skipped: list[dict[str, str]] = []

    for annotation_path, row in _load_annotations(feedback_root):
        sample_id = str(row.get("sample_id", annotation_path.stem))
        row_split = normalized_splits.get(sample_id) if normalized_splits is not None else split
        if row_split is None:
            skipped.append({"sample_id": sample_id, "reason": "not_selected"})
            continue
        review = row.get("review", {})
        if not review.get("approved_for_training") or not review.get("training_eligible"):
            skipped.append({"sample_id": sample_id, "reason": "not_training_eligible"})
            continue
        if review.get("status") == "accepted_prediction" and not allow_accepted_predictions:
            skipped.append(
                {"sample_id": sample_id, "reason": "accepted_prediction_requires_explicit_flag"}
            )
            continue
        use_prediction = bool(allow_accepted_predictions and review.get("status") == "accepted_prediction")
        original_path = feedback_root / row["image"]["path"]
        rectified_path = feedback_root / row["rectification"]["image_path"]
        original_width = int(row["image"]["width"])
        original_height = int(row["image"]["height"])

        outer_correction = row.get("outer_frame", {}).get("correction")
        outer_points = None
        if outer_correction is not None:
            outer_points = outer_correction.get("points")
        elif use_prediction:
            outer_points = row.get("outer_frame", {}).get("prediction", {}).get("points")
        if outer_points is not None:
            points = np.asarray(
                normalize_outer_points(outer_points, original_width, original_height), dtype=np.float64
            )
            _copy_image(original_path, outer_pose_root / "images" / row_split, sample_id)
            _copy_image(original_path, outer_seg_root / "images" / row_split, sample_id)
            pose_label_dir = outer_pose_root / "labels" / row_split
            seg_label_dir = outer_seg_root / "labels" / row_split
            pose_label_dir.mkdir(parents=True, exist_ok=True)
            seg_label_dir.mkdir(parents=True, exist_ok=True)
            normalized = points / np.asarray([original_width, original_height], dtype=np.float64)
            x_min, y_min = normalized.min(axis=0)
            x_max, y_max = normalized.max(axis=0)
            bbox = [(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min]
            pose_values: list[float] = [0.0, *bbox]
            for x_value, y_value in normalized:
                pose_values.extend([float(x_value), float(y_value), 2.0])
            (pose_label_dir / f"{sample_id}.txt").write_text(
                _format(pose_values) + "\n", encoding="utf-8"
            )
            seg_values = [0.0, *normalized.reshape(-1).tolist()]
            (seg_label_dir / f"{sample_id}.txt").write_text(
                _format(seg_values) + "\n", encoding="utf-8"
            )
            converted["outer_pose"] += 1
            converted["outer_seg"] += 1

        inner_correction = row.get("inner_frame", {}).get("correction")
        inner_box = None
        if inner_correction is not None:
            inner_box = inner_correction.get("box")
        elif use_prediction:
            inner_box = row.get("inner_frame", {}).get("prediction", {}).get("box")
        if inner_box is not None:
            box = normalize_inner_box(inner_box, 630, 880)
            copied_rectified = _copy_image(rectified_path, inner_seg_root / "images" / row_split, sample_id)
            label_dir = inner_seg_root / "labels" / row_split
            label_dir.mkdir(parents=True, exist_ok=True)
            left = box["left"] / 630.0
            right = box["right"] / 630.0
            top = box["top"] / 880.0
            bottom = box["bottom"] / 880.0
            segmentation = [0.0, left, top, right, top, right, bottom, left, bottom]
            (label_dir / f"{sample_id}.txt").write_text(
                _format(segmentation) + "\n", encoding="utf-8"
            )
            relative_image = copied_rectified.relative_to(output_root).as_posix()
            refiner_rows.append(
                {
                    "id": sample_id,
                    "image": relative_image,
                    "width": 630,
                    "height": 880,
                    "left": f"{left:.8f}",
                    "right": f"{right:.8f}",
                    "top": f"{top:.8f}",
                    "bottom": f"{bottom:.8f}",
                    "split": row_split,
                    "source": "human_feedback",
                }
            )
            converted["inner_seg"] += 1
            converted["inner_refiner"] += 1

    manifest_path = output_root / "inner_refiner_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["id", "image", "width", "height", "left", "right", "top", "bottom", "split", "source"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(refiner_rows)
    summary = {
        "success": True,
        "feedback_root": str(feedback_root),
        "output_root": str(output_root),
        "split": split,
        "sample_splits": normalized_splits,
        "allow_accepted_predictions": bool(allow_accepted_predictions),
        "converted": converted,
        "skipped": skipped,
        "inner_refiner_manifest": str(manifest_path),
    }
    (output_root / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
