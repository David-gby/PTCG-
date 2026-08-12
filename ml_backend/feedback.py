from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping
from uuid import uuid4

import cv2
import numpy as np

from card_quality_processor.config import normalize_config
from card_quality_processor.outer_detection import order_points
from card_quality_processor.rectification import rectify_card_by_points


FEEDBACK_SCHEMA_VERSION = "1.0"
CORNER_ORDER = ["tl", "tr", "br", "bl"]
PACKAGE_ROOT = Path(__file__).resolve().parent


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"Cannot read image file: {path}") from exc
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray, quality: int = 94) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    parameters = [cv2.IMWRITE_JPEG_QUALITY, quality] if suffix in {".jpg", ".jpeg"} else []
    success, encoded = cv2.imencode(suffix, image, parameters)
    if not success:
        raise OSError(f"Cannot encode output image: {path}")
    encoded.tofile(str(path))
    return path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def normalize_outer_points(
    points: Iterable[Iterable[float]],
    image_width: int,
    image_height: int,
) -> list[list[float]]:
    ordered = order_points(np.asarray(points, dtype=np.float32).reshape(4, 2))
    if not np.isfinite(ordered).all():
        raise ValueError("Outer correction must contain four finite points.")
    if (
        np.any(ordered[:, 0] < 0)
        or np.any(ordered[:, 1] < 0)
        or np.any(ordered[:, 0] > image_width - 1)
        or np.any(ordered[:, 1] > image_height - 1)
    ):
        raise ValueError("Outer correction contains points outside the original image.")
    contour = ordered.reshape(-1, 1, 2)
    if not cv2.isContourConvex(np.rint(contour).astype(np.int32)):
        raise ValueError("Outer correction must form a convex quadrilateral.")
    area = abs(float(cv2.contourArea(contour)))
    if area < max(25.0, image_width * image_height * 0.01):
        raise ValueError("Outer correction is too small or degenerate.")
    return np.round(ordered.astype(np.float64), 3).tolist()


def normalize_inner_box(
    box: Mapping[str, float],
    image_width: int = 630,
    image_height: int = 880,
) -> dict[str, float]:
    required = ("left", "top", "right", "bottom")
    if any(key not in box for key in required):
        raise ValueError("Inner correction requires left, top, right, and bottom.")
    result = {key: float(box[key]) for key in required}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Inner correction contains a non-finite coordinate.")
    if not (0 <= result["left"] < result["right"] <= image_width - 1):
        raise ValueError("Inner correction has invalid left/right coordinates.")
    if not (0 <= result["top"] < result["bottom"] <= image_height - 1):
        raise ValueError("Inner correction has invalid top/bottom coordinates.")
    return {key: round(value, 3) for key, value in result.items()}


def box_corners(box: Mapping[str, float]) -> list[list[float]]:
    return [
        [float(box["left"]), float(box["top"])],
        [float(box["right"]), float(box["top"])],
        [float(box["right"]), float(box["bottom"])],
        [float(box["left"]), float(box["bottom"])],
    ]


def line_center_geometry(box: Mapping[str, float]) -> dict[str, Any]:
    centers = {
        "left": round(float(box["left"]), 4),
        "right": round(float(box["right"]), 4),
        "top": round(float(box["top"]), 4),
        "bottom": round(float(box["bottom"]), 4),
    }
    middle_x = 315.0
    middle_y = 440.0
    widths = {
        "left": centers["left"],
        "right": round(629.0 - centers["right"], 4),
        "top": centers["top"],
        "bottom": round(879.0 - centers["bottom"], 4),
    }
    horizontal = widths["left"] + widths["right"]
    vertical = widths["top"] + widths["bottom"]
    pair = None
    if horizontal > 0 and vertical > 0:
        pair = {
            "left": round(100.0 * widths["left"] / horizontal, 4),
            "right": round(100.0 * widths["right"] / horizontal, 4),
            "top": round(100.0 * widths["top"] / vertical, 4),
            "bottom": round(100.0 * widths["bottom"] / vertical, 4),
        }
    return {
        "line_centers_px": centers,
        "line_midpoints_px": {
            "left": [centers["left"], middle_y],
            "right": [centers["right"], middle_y],
            "top": [middle_x, centers["top"]],
            "bottom": [middle_x, centers["bottom"]],
        },
        "coordinate_semantics": "zero_width_red_line_center",
        "centering_measurements": {
            "border_width_px": widths,
            "centering_pair_percent": pair,
            "formula_contract": {
                "input_coordinates": "zero_width_red_line_centers",
                "right_width": "630 - 1 - right_line_center_x",
                "bottom_width": "880 - 1 - bottom_line_center_y",
            },
        },
    }


def _prediction_snapshot(prediction: Mapping[str, Any]) -> dict[str, Any]:
    outer = prediction.get("outer_frame", {})
    inner = prediction.get("inner_frame", {})
    return {
        "pipeline_version": prediction.get("version"),
        "outer_frame": {
            "points": outer.get("points"),
            "bbox": outer.get("bbox"),
            "confidence": outer.get("confidence"),
            "keypoint_confidence": outer.get("keypoint_confidence"),
            "method": outer.get("method"),
            "metrics": outer.get("metrics"),
        },
        "inner_frame": {
            "final_box": inner.get("final_box"),
            "yolo_confidence": inner.get("yolo_confidence"),
            "edge_refinement": inner.get("edge_refinement"),
            "stabilizer": inner.get("stabilizer"),
        },
    }


class FeedbackExporter:
    """Export reviewed samples without performing automatic online training."""

    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root).resolve()
        self.annotation_dir = self.output_root / "annotations"
        self.original_dir = self.output_root / "original_images"
        self.rectified_dir = self.output_root / "rectified_images"
        for directory in (self.output_root, self.annotation_dir, self.original_dir, self.rectified_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        *,
        image_path: str | Path,
        prediction: Mapping[str, Any],
        exif_orientation_handling: str = "not_explicit",
        outer_correction: Iterable[Iterable[float]] | None = None,
        inner_correction: Mapping[str, float] | None = None,
        issue_tags: Iterable[str] = (),
        annotator: str = "",
        review_status: str = "corrected",
        approved_for_training: bool = False,
        card_type: str = "unknown",
        layout: str = "unknown",
        inner_edge_visibility: Mapping[str, bool] | None = None,
        notes: str = "",
        sample_id: str | None = None,
    ) -> dict[str, Any]:
        image_path = Path(image_path).resolve()
        image = read_image(image_path)
        height, width = image.shape[:2]
        image_hash = sha256_file(image_path)

        prediction_outer = prediction.get("outer_frame", {}).get("points")
        if prediction_outer is None and outer_correction is None:
            raise ValueError("Neither prediction nor correction contains outer-frame points.")
        selected_outer_source = "manual_correction" if outer_correction is not None else "model_prediction"
        selected_outer = normalize_outer_points(
            outer_correction if outer_correction is not None else prediction_outer,
            width,
            height,
        )
        normalized_outer_correction = (
            normalize_outer_points(outer_correction, width, height) if outer_correction is not None else None
        )

        rectification = rectify_card_by_points(
            image,
            selected_outer,
            (630, 880),
            normalize_config(),
        )
        if not rectification.get("success") or rectification.get("rectified_image") is None:
            raise ValueError(
                "The selected outer points cannot produce a valid rectified card: "
                + str(rectification.get("message", "unknown error"))
            )
        rectified = rectification["rectified_image"]
        normalized_inner_correction = (
            normalize_inner_box(inner_correction, 630, 880) if inner_correction is not None else None
        )

        if review_status not in {"corrected", "accepted_prediction", "rejected", "no_inner_frame"}:
            raise ValueError("Unsupported review_status.")
        if review_status == "no_inner_frame" and normalized_inner_correction is not None:
            raise ValueError("no_inner_frame feedback cannot contain an inner correction.")

        now = datetime.now(timezone.utc)
        base_id = sample_id or f"{image_hash[:16]}-{now.strftime('%Y%m%dT%H%M%SZ')}"
        final_id = base_id
        while (self.annotation_dir / f"{final_id}.json").exists():
            final_id = f"{base_id}-{uuid4().hex[:8]}"

        source_suffix = image_path.suffix.lower() if image_path.suffix else ".jpg"
        original_target = self.original_dir / f"{final_id}{source_suffix}"
        rectified_target = self.rectified_dir / f"{final_id}.jpg"
        shutil.copy2(image_path, original_target)
        write_image(rectified_target, rectified)

        model_manifest_path = PACKAGE_ROOT / "model_manifest.json"
        model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        prediction_snapshot = _prediction_snapshot(prediction)
        prediction_inner = prediction_snapshot["inner_frame"].get("final_box")
        if prediction_inner is not None:
            prediction_inner = normalize_inner_box(prediction_inner, 630, 880)

        visibility = {edge: True for edge in ("left", "right", "top", "bottom")}
        if inner_edge_visibility:
            visibility.update({key: bool(value) for key, value in inner_edge_visibility.items() if key in visibility})

        has_manual_label = normalized_outer_correction is not None or normalized_inner_correction is not None
        accepted_prediction = review_status == "accepted_prediction"
        training_eligible = bool(
            approved_for_training
            and review_status not in {"rejected", "no_inner_frame"}
            and (has_manual_label or accepted_prediction)
        )
        annotation = {
            "schema_version": FEEDBACK_SCHEMA_VERSION,
            "sample_id": final_id,
            "created_at_utc": now.isoformat(),
            "image": {
                "path": _relative(original_target, self.output_root),
                "sha256": image_hash,
                "width": width,
                "height": height,
                "color_order": "BGR",
                "exif_orientation_handling": str(exif_orientation_handling),
            },
            "model": {
                "pipeline_version": prediction.get("version"),
                "package_version": model_manifest.get("package_version"),
                "models": [
                    {"file": item["file"], "sha256": item["sha256"]}
                    for item in model_manifest.get("models", [])
                ],
            },
            "review": {
                "annotator": annotator,
                "status": review_status,
                "approved_for_training": bool(approved_for_training),
                "training_eligible": training_eligible,
                "issue_tags": sorted({str(tag).strip() for tag in issue_tags if str(tag).strip()}),
                "card_type": card_type,
                "layout": layout,
                "notes": notes,
            },
            "outer_frame": {
                "coordinate_space": "original_image_pixels",
                "corner_order": CORNER_ORDER,
                "prediction": prediction_snapshot["outer_frame"],
                "correction": None
                if normalized_outer_correction is None
                else {"points": normalized_outer_correction},
                "selected_for_rectification": {
                    "source": selected_outer_source,
                    "points": selected_outer,
                },
            },
            "rectification": {
                "image_path": _relative(rectified_target, self.output_root),
                "output_size": [630, 880],
                "homography": _jsonable(rectification.get("homography")),
                "based_on": selected_outer_source,
            },
            "inner_frame": {
                "coordinate_space": "rectified_card_pixels",
                "coordinates_reference": (
                    "rectified_from_manual_outer_correction"
                    if normalized_outer_correction is not None
                    else "rectified_from_model_outer_prediction"
                ),
                "edge_visibility": visibility,
                "prediction": None
                if prediction_inner is None
                else {
                    "box": prediction_inner,
                    **line_center_geometry(prediction_inner),
                    "corners_tl_tr_br_bl": box_corners(prediction_inner),
                    "yolo_confidence": prediction_snapshot["inner_frame"].get("yolo_confidence"),
                    "edge_refinement": prediction_snapshot["inner_frame"].get("edge_refinement"),
                },
                "correction": None
                if normalized_inner_correction is None
                else {
                    "box": normalized_inner_correction,
                    **line_center_geometry(normalized_inner_correction),
                    "corners_tl_tr_br_bl": box_corners(normalized_inner_correction),
                },
            },
        }

        annotation_path = self.annotation_dir / f"{final_id}.json"
        annotation_path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_row = {
            "sample_id": final_id,
            "annotation": _relative(annotation_path, self.output_root),
            "original_image": _relative(original_target, self.output_root),
            "rectified_image": _relative(rectified_target, self.output_root),
            "approved_for_training": bool(approved_for_training),
            "training_eligible": training_eligible,
            "review_status": review_status,
            "issue_tags": annotation["review"]["issue_tags"],
        }
        with (self.output_root / "manifest.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")
        return {
            "success": True,
            "sample_id": final_id,
            "annotation_path": str(annotation_path),
            "manifest_path": str(self.output_root / "manifest.jsonl"),
            "training_eligible": training_eligible,
        }
