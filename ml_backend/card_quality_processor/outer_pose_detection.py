from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .config import DEFAULT_CONFIG
from .outer_detection import _edge_support, order_points
from .outer_edge_refinement import refine_outer_edges
from .outer_silhouette import extract_silhouette_prediction


KEYPOINT_NAMES = ("tl", "tr", "br", "bl")


def _deep_pose_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Accept a full config, outer_detection section, or deep_pose section."""
    merged = deepcopy(DEFAULT_CONFIG["outer_detection"]["deep_pose"])
    if not config:
        return merged
    supplied: Mapping[str, Any] = config
    if isinstance(supplied.get("outer_detection"), Mapping):
        supplied = supplied["outer_detection"]
    if isinstance(supplied.get("deep_pose"), Mapping):
        supplied = supplied["deep_pose"]
    for key, value in supplied.items():
        if key in merged:
            merged[key] = deepcopy(value)
    return merged


def _confidence_values(keypoint_conf: Mapping[str, float] | Sequence[float] | np.ndarray | None) -> np.ndarray:
    if keypoint_conf is None:
        return np.ones(4, dtype=np.float32)
    if isinstance(keypoint_conf, Mapping):
        values = [keypoint_conf.get(name, keypoint_conf.get(name.upper(), 0.0)) for name in KEYPOINT_NAMES]
    else:
        values = list(keypoint_conf)
    result = np.asarray(values, dtype=np.float32).reshape(-1)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError("Four finite keypoint confidence values are required")
    return np.clip(result, 0.0, 1.0)


def _invalid_validation(
    code: str,
    confidence: float,
    ordered_points: np.ndarray | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "success": False,
        "ordered_points": None if ordered_points is None else ordered_points.round(3).tolist(),
        "geometry_valid": False if code == "INVALID_KEYPOINT_GEOMETRY" else bool(metrics and metrics.get("geometry_valid")),
        "confidence": float(confidence),
        "error_code": code,
        "metrics": dict(metrics or {}),
    }


def validate_and_order_outer_keypoints(
    points: Iterable[Iterable[float]],
    keypoint_conf: Mapping[str, float] | Sequence[float] | np.ndarray | None,
    image_shape: Sequence[int],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Order and validate card corners as TL, TR, BR, BL.

    Confidence values are reordered with their points, so a shuffled set of
    predictions can be corrected without assigning confidence to the wrong
    physical corner.
    """
    cfg = _deep_pose_config(config)
    try:
        raw_points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        raw_conf = _confidence_values(keypoint_conf)
        if raw_points.shape != (4, 2) or not np.isfinite(raw_points).all():
            raise ValueError("Four finite points are required")
        ordered = order_points(raw_points)
    except (TypeError, ValueError):
        return _invalid_validation("INVALID_KEYPOINT_GEOMETRY", 0.0)

    # order_points only permutes vertices. Recover the same permutation for
    # the confidence array and guard against ambiguous duplicate vertices.
    distances = np.linalg.norm(ordered[:, None, :] - raw_points[None, :, :], axis=2)
    source_indices = np.argmin(distances, axis=1)
    if len(set(int(index) for index in source_indices)) != 4:
        return _invalid_validation("INVALID_KEYPOINT_GEOMETRY", 0.0, ordered)
    ordered_conf = raw_conf[source_indices]
    mean_conf = float(np.mean(ordered_conf))
    min_conf = float(np.min(ordered_conf))

    if len(image_shape) < 2:
        return _invalid_validation("INVALID_KEYPOINT_GEOMETRY", mean_conf, ordered)
    height, width = int(image_shape[0]), int(image_shape[1])
    if width < 2 or height < 2:
        return _invalid_validation("INVALID_KEYPOINT_GEOMETRY", mean_conf, ordered)

    tl, tr, br, bl = ordered
    in_bounds = bool(
        np.all(ordered[:, 0] >= 0)
        and np.all(ordered[:, 1] >= 0)
        and np.all(ordered[:, 0] <= width - 1)
        and np.all(ordered[:, 1] <= height - 1)
    )
    convex = bool(cv2.isContourConvex(ordered.reshape(-1, 1, 2)))
    spatial_order = bool(
        (tl[1] + tr[1]) < (bl[1] + br[1])
        and (tl[0] + bl[0]) < (tr[0] + br[0])
    )
    area = abs(float(cv2.contourArea(ordered.reshape(-1, 1, 2))))
    area_ratio = area / max(1.0, float(width * height))
    top = float(np.linalg.norm(tr - tl))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))
    right = float(np.linalg.norm(br - tr))
    average_width = (top + bottom) * 0.5
    average_height = (left + right) * 0.5
    shortest_dimension = min(average_width, average_height)
    aspect_ratio = max(average_width, average_height) / max(shortest_dimension, 1e-6)
    target_aspect = max(float(cfg["card_aspect_ratio"]), 1e-6)
    aspect_error = abs(aspect_ratio - target_aspect) / target_aspect
    margins = np.array(
        [
            float(np.min(ordered[:, 0])),
            float(np.min(ordered[:, 1])),
            float(width - 1 - np.max(ordered[:, 0])),
            float(height - 1 - np.max(ordered[:, 1])),
        ],
        dtype=np.float32,
    )
    margin_ratio = float(np.min(margins / np.array([width, height, width, height], dtype=np.float32)))
    geometry_valid = bool(
        in_bounds
        and convex
        and spatial_order
        and shortest_dimension >= 2.0
        and float(cfg["min_area_ratio"]) <= area_ratio <= float(cfg["max_area_ratio"])
        and aspect_error <= float(cfg["aspect_ratio_tolerance"])
        and margin_ratio >= float(cfg.get("border_margin_ratio", 0.0))
    )
    metrics = {
        "mean_keypoint_confidence": mean_conf,
        "min_keypoint_confidence": min_conf,
        "aspect_ratio": float(aspect_ratio),
        "aspect_ratio_error": float(aspect_error),
        "area_ratio": float(area_ratio),
        "min_border_margin_ratio": margin_ratio,
        "in_bounds": in_bounds,
        "convex": convex,
        "spatial_order": spatial_order,
        "geometry_valid": geometry_valid,
        "ordered_keypoint_confidence": [float(value) for value in ordered_conf],
    }
    if not geometry_valid:
        return _invalid_validation("INVALID_KEYPOINT_GEOMETRY", mean_conf, ordered, metrics)
    if min_conf < float(cfg["keypoint_conf_threshold"]) or mean_conf < float(cfg["min_mean_keypoint_conf"]):
        result = _invalid_validation("LOW_CONFIDENCE_OUTER_POSE", mean_conf, ordered, metrics)
        result["geometry_valid"] = True
        return result
    return {
        "success": True,
        "ordered_points": ordered.round(3).tolist(),
        "geometry_valid": True,
        "confidence": mean_conf,
        "error_code": None,
        "metrics": metrics,
    }


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def apply_outer_pose_calibration(
    points: Iterable[Iterable[float]],
    config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Apply model-specific normalized corner offsets to ordered points.

    Offsets are normalized by the predicted card width and height so the
    correction scales with image resolution. Calibration can be disabled in
    config when evaluating a different model checkpoint.
    """
    cfg = _deep_pose_config(config)
    ordered = order_points(points)
    calibration = cfg.get("corner_calibration", {})
    if not isinstance(calibration, Mapping) or not bool(calibration.get("enabled", False)):
        return ordered
    offsets = np.asarray(calibration.get("normalized_offsets", []), dtype=np.float32)
    if offsets.shape != (4, 2) or not np.isfinite(offsets).all():
        return ordered
    tl, tr, br, bl = ordered
    card_width = (float(np.linalg.norm(tr - tl)) + float(np.linalg.norm(br - bl))) * 0.5
    card_height = (float(np.linalg.norm(bl - tl)) + float(np.linalg.norm(br - tr))) * 0.5
    return ordered + offsets * np.asarray([card_width, card_height], dtype=np.float32)


def calculate_outer_pose_edge_support(
    image: np.ndarray,
    points: Iterable[Iterable[float]],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure whether all predicted sides coincide with visible image edges."""
    cfg = _deep_pose_config(config)
    edge_cfg = cfg.get("edge_guard", {})
    try:
        ordered = order_points(points)
    except (TypeError, ValueError):
        return {"edge_support_score": 0.0, "side_edge_support": [0.0] * 4, "min_side_edge_support": 0.0}
    if not isinstance(image, np.ndarray) or image.size == 0:
        return {"edge_support_score": 0.0, "side_edge_support": [0.0] * 4, "min_side_edge_support": 0.0}
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    sigma = float(edge_cfg.get("canny_sigma", 0.33)) if isinstance(edge_cfg, Mapping) else 0.33
    lower = int(max(5, (1.0 - sigma) * median))
    upper = int(min(255, max(lower + 20, (1.0 + sigma) * median)))
    edges = cv2.Canny(gray, lower, upper, L2gradient=True)
    edge_score, side_scores = _edge_support(edges, ordered)
    return {
        "edge_support_score": float(edge_score),
        "side_edge_support": [float(value) for value in side_scores],
        "min_side_edge_support": float(min(side_scores)),
    }


def _prepare_runtime_directories() -> Path:
    runtime_root = Path(
        os.environ.get("PTCG_RUNTIME_DIR")
        or Path(tempfile.gettempdir()) / "ptcg_model_runtime"
    )
    config_root = Path(
        os.environ.get("YOLO_CONFIG_DIR") or runtime_root / "yolo"
    )
    matplotlib_root = Path(
        os.environ.get("MPLCONFIGDIR") or runtime_root / "matplotlib"
    )
    output_root = runtime_root / "ultralytics_runs"
    config_root.mkdir(parents=True, exist_ok=True)
    matplotlib_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_root))
    return output_root


class OuterPoseDetector:
    """Lazy YOLO Pose wrapper for physical card corner detection."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = _deep_pose_config(config)
        project_root = Path(__file__).resolve().parents[1]
        self.model_path = Path(model_path or self.config["model_path"])
        if not self.model_path.is_absolute():
            self.model_path = project_root / self.model_path
        silhouette_cfg = self.config.get("silhouette_refinement", {})
        self._prefer_silhouette = bool(
            model_path is None
            and isinstance(silhouette_cfg, Mapping)
            and silhouette_cfg.get("enabled", False)
        )
        self._model: Any | None = None
        self._silhouette_model: Any | None = None
        self._silhouette_fallback_model: Any | None = None

    @staticmethod
    def _failure(
        code: str,
        message: str,
        *,
        points: list[list[float]] | None = None,
        bbox: list[float] | None = None,
        confidence: float = 0.0,
        keypoint_confidence: Mapping[str, float] | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        empty_metrics = {
            "bbox_confidence": 0.0,
            "mean_keypoint_confidence": 0.0,
            "min_keypoint_confidence": 0.0,
            "aspect_ratio": 0.0,
            "aspect_ratio_error": 1.0,
            "area_ratio": 0.0,
        }
        empty_metrics.update(dict(metrics or {}))
        return {
            "success": False,
            "points": points,
            "bbox": bbox,
            "confidence": float(confidence),
            "keypoint_confidence": dict(keypoint_confidence or {name: 0.0 for name in KEYPOINT_NAMES}),
            "method": "deep_outer_pose",
            "error_code": code,
            "message": message,
            "metrics": empty_metrics,
        }

    def _load_model(self) -> Any:
        if self._model is None:
            _prepare_runtime_directories()
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
        return self._model

    def _load_silhouette_model(self, model_path: Path) -> Any:
        if self._silhouette_model is None:
            _prepare_runtime_directories()
            from ultralytics import YOLO

            self._silhouette_model = YOLO(str(model_path))
        return self._silhouette_model

    def _load_silhouette_fallback_model(self, model_path: Path) -> Any:
        if self._silhouette_fallback_model is None:
            _prepare_runtime_directories()
            from ultralytics import YOLO

            self._silhouette_fallback_model = YOLO(str(model_path))
        return self._silhouette_fallback_model

    def _predict_silhouette(self, image: np.ndarray) -> dict[str, Any] | None:
        refinement = self.config.get("silhouette_refinement", {})
        if not isinstance(refinement, Mapping) or not bool(refinement.get("enabled", False)):
            return None
        model_path = Path(str(refinement.get("model_path", "models/outer_seg.pt")))
        if not model_path.is_absolute() and not model_path.is_file():
            model_path = Path(__file__).resolve().parents[1] / model_path
        if not model_path.is_file():
            return None
        try:
            model = self._load_silhouette_model(model_path)
            predictions = model.predict(
                source=image,
                conf=float(refinement.get("conf_threshold", 0.15)),
                imgsz=int(self.config.get("inference_imgsz", 640)),
                device=self.config.get("device"),
                project=str(_prepare_runtime_directories()),
                name="outer_silhouette",
                exist_ok=True,
                augment=False,
                verbose=False,
            )
        except (ImportError, OSError, RuntimeError, ValueError):
            return None
        if not predictions:
            return None
        extracted = extract_silhouette_prediction(
            predictions[0],
            image_shape=image.shape,
            target_aspect_ratio=float(self.config.get("card_aspect_ratio", 1.397)),
            aspect_ratio_tolerance=float(self.config.get("aspect_ratio_tolerance", 0.25)),
            min_area_ratio=float(self.config.get("min_area_ratio", 0.05)),
            max_area_ratio=float(self.config.get("max_area_ratio", 0.95)),
            area_weight=float(refinement.get("candidate_area_weight", 0.40)),
            aspect_weight=float(refinement.get("candidate_aspect_weight", 0.25)),
        )
        if extracted is None:
            return None
        confidence = float(extracted["confidence"])
        validation = validate_and_order_outer_keypoints(
            extracted["points"],
            [confidence] * 4,
            image.shape,
            self.config,
        )
        points = validation.get("ordered_points")
        validation_metrics = dict(validation.get("metrics", {}))
        if not validation["success"] or points is None:
            code = str(validation.get("error_code") or "INVALID_KEYPOINT_GEOMETRY")
            keypoint_confidence = {name: confidence for name in KEYPOINT_NAMES}
            metrics = {
                "bbox_confidence": confidence,
                "mean_keypoint_confidence": confidence,
                "min_keypoint_confidence": confidence,
                "aspect_ratio": float(validation_metrics.get("aspect_ratio", 0.0)),
                "aspect_ratio_error": float(validation_metrics.get("aspect_ratio_error", 1.0)),
                "area_ratio": float(validation_metrics.get("area_ratio", 0.0)),
                "silhouette_refinement_applied": False,
                "silhouette_detected": True,
            }
            return self._failure(
                code,
                "Detected outer silhouette failed guarded geometry validation.",
                points=points,
                bbox=[float(value) for value in extracted["bbox"]],
                confidence=confidence,
                keypoint_confidence=keypoint_confidence,
                metrics=metrics,
            )
        raw_points = [list(value) for value in points]
        physical_cfg = self.config.get("physical_edge_refinement", {})
        physical_result: dict[str, Any] = {
            "accepted": False,
            "points": raw_points,
            "reason": "disabled",
            "metrics": {},
        }
        corner_confidence_values = np.full(4, confidence, dtype=np.float32)
        physical_applied = False
        if isinstance(physical_cfg, Mapping) and bool(physical_cfg.get("enabled", False)):
            physical_result = refine_outer_edges(image, points, physical_cfg)
            candidate_points = physical_result.get("points")
            if bool(physical_result.get("accepted")) and candidate_points is not None:
                evidence = np.asarray(
                    physical_result.get("metrics", {}).get("corner_confidence", [0.0] * 4),
                    dtype=np.float32,
                ).reshape(-1)
                if evidence.shape == (4,) and np.isfinite(evidence).all():
                    corner_confidence_values = confidence * (0.55 + 0.45 * np.clip(evidence, 0.0, 1.0))
                refined_validation = validate_and_order_outer_keypoints(
                    candidate_points,
                    corner_confidence_values,
                    image.shape,
                    self.config,
                )
                refined_points = refined_validation.get("ordered_points")
                if refined_validation["success"] and refined_points is not None:
                    points = refined_points
                    validation_metrics = dict(refined_validation.get("metrics", {}))
                    physical_applied = True
                else:
                    physical_result["accepted"] = False
                    physical_result["reason"] = "refined_geometry_rejected"
                    corner_confidence_values = np.full(4, confidence, dtype=np.float32)

        edge_metrics = calculate_outer_pose_edge_support(image, points, self.config)
        mean_corner_confidence = float(np.mean(corner_confidence_values))
        min_corner_confidence = float(np.min(corner_confidence_values))
        result_confidence = float(0.35 * confidence + 0.65 * mean_corner_confidence)
        metrics = {
            "bbox_confidence": confidence,
            "mean_keypoint_confidence": mean_corner_confidence,
            "min_keypoint_confidence": min_corner_confidence,
            "aspect_ratio": float(validation_metrics.get("aspect_ratio", 0.0)),
            "aspect_ratio_error": float(validation_metrics.get("aspect_ratio_error", 1.0)),
            "area_ratio": float(validation_metrics.get("area_ratio", 0.0)),
            "calibration_applied": False,
            "silhouette_refinement_applied": True,
            "physical_edge_refinement_applied": physical_applied,
            "physical_edge_refinement": physical_result,
            "raw_points": raw_points,
            "silhouette_candidate_count": int(extracted.get("candidate_count", 1)),
            "silhouette_selected_index": int(extracted.get("selected_index", 0)),
            "silhouette_selection_score": float(extracted.get("selection_score", confidence)),
            **edge_metrics,
        }
        keypoint_confidence = {
            name: float(value) for name, value in zip(KEYPOINT_NAMES, corner_confidence_values)
        }
        return {
            "success": True,
            "points": points,
            "bbox": [float(value) for value in extracted["bbox"]],
            "confidence": result_confidence,
            "keypoint_confidence": keypoint_confidence,
            "method": "deep_outer_pose",
            "error_code": None,
            "message": (
                "Outer card silhouette detected and refined against full-resolution physical edges."
                if physical_applied
                else "Outer card silhouette detected and converted to physical corners."
            ),
            "metrics": metrics,
        }

    def _predict_pose_only(self, image: np.ndarray, conf: float) -> dict[str, Any]:
        """Run the pose branch without recursively invoking silhouette inference."""
        prefer_silhouette = self._prefer_silhouette
        self._prefer_silhouette = False
        try:
            return self.predict(image, conf=conf)
        finally:
            self._prefer_silhouette = prefer_silhouette

    def _replace_suspicious_silhouette(
        self,
        image: np.ndarray,
        primary: dict[str, Any],
        conf: float,
    ) -> dict[str, Any]:
        """Use legacy-segmentation/pose consensus for implausibly small masks.

        The Frame2 model is substantially more accurate on small cards, so a
        large legacy mask alone must never override it.  Replacement occurs
        only when the pose model is geometrically valid and independently
        agrees with the larger legacy silhouette.
        """
        refinement = self.config.get("silhouette_refinement", {})
        fallback = refinement.get("fallback", {}) if isinstance(refinement, Mapping) else {}
        if not isinstance(fallback, Mapping) or not bool(fallback.get("enabled", False)):
            return primary
        if not bool(primary.get("success")):
            return primary
        primary_metrics = primary.get("metrics", {})
        primary_area = float(primary_metrics.get("area_ratio", 0.0))
        if primary_area <= 0.0 or primary_area > float(fallback.get("max_primary_area_ratio", 0.40)):
            return primary

        pose = self._predict_pose_only(image, conf)
        if pose.get("error_code") not in (None, "LOW_CONFIDENCE_OUTER_POSE") or pose.get("points") is None:
            return primary
        pose_bbox_confidence = float(pose.get("metrics", {}).get("bbox_confidence", 0.0))
        if pose_bbox_confidence < float(fallback.get("min_pose_bbox_confidence", 0.75)):
            return primary
        pose_validation = validate_and_order_outer_keypoints(
            pose["points"],
            [1.0] * 4,
            image.shape,
            self.config,
        )
        if not pose_validation["success"]:
            return primary
        pose_metrics = pose_validation.get("metrics", {})
        pose_area = float(pose_metrics.get("area_ratio", 0.0))
        pose_aspect_error = float(pose_metrics.get("aspect_ratio_error", 1.0))
        min_area_gain = float(fallback.get("min_area_gain_ratio", 1.35))
        if (
            pose_area < primary_area * min_area_gain
            or pose_aspect_error > float(fallback.get("max_pose_aspect_error", 0.20))
        ):
            return primary

        legacy_path = Path(str(fallback.get("legacy_model_path", "models/outer_seg_pre_frame2.pt")))
        if not legacy_path.is_absolute() and not legacy_path.is_file():
            legacy_path = Path(__file__).resolve().parents[1] / legacy_path
        if not legacy_path.is_file():
            return primary
        try:
            legacy_model = self._load_silhouette_fallback_model(legacy_path)
            primary_model = self._silhouette_model
            self._silhouette_model = legacy_model
            try:
                legacy = self._predict_silhouette(image)
            finally:
                self._silhouette_model = primary_model
        except (ImportError, OSError, RuntimeError, ValueError):
            return primary
        if legacy is None or not bool(legacy.get("success")):
            return primary
        legacy_metrics = legacy.get("metrics", {})
        legacy_area = float(legacy_metrics.get("area_ratio", 0.0))
        area_tolerance = float(fallback.get("legacy_pose_area_tolerance", 0.20))
        if legacy_area < primary_area * min_area_gain:
            return primary
        if abs(legacy_area - pose_area) / max(pose_area, 1e-6) > area_tolerance:
            return primary

        legacy_metrics["silhouette_fallback_applied"] = True
        legacy_metrics["silhouette_primary_area_ratio"] = primary_area
        legacy_metrics["silhouette_fallback_pose_area_ratio"] = pose_area
        legacy_metrics["silhouette_fallback_pose_bbox_confidence"] = pose_bbox_confidence
        legacy["metrics"] = legacy_metrics
        legacy["message"] = (
            "Suspiciously small primary silhouette replaced by a geometry-valid "
            "legacy-segmentation/pose consensus."
        )
        return legacy

    def predict(self, image: np.ndarray, conf: float = 0.25) -> dict[str, Any]:
        if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in (2, 3):
            return self._failure("OUTER_POSE_NOT_DETECTED", "Input image is empty or invalid.")
        if self._prefer_silhouette:
            silhouette = self._predict_silhouette(image)
            if silhouette is not None:
                return self._replace_suspicious_silhouette(image, silhouette, conf)
        if not self.model_path.is_file():
            return self._failure(
                "OUTER_POSE_MODEL_NOT_FOUND",
                "Outer pose model file not found.",
            )
        try:
            model = self._load_model()
            predictions = model.predict(
                source=image,
                conf=float(conf),
                imgsz=int(self.config.get("inference_imgsz", 640)),
                device=self.config.get("device"),
                project=str(_prepare_runtime_directories()),
                name="outer_pose",
                exist_ok=True,
                augment=False,
                verbose=False,
            )
        except ImportError:
            return self._failure(
                "OUTER_POSE_NOT_DETECTED",
                "ultralytics is not installed. Install it with: pip install ultralytics",
            )
        except Exception as exc:  # Ultralytics backends raise several runtime-specific exceptions.
            return self._failure("OUTER_POSE_NOT_DETECTED", f"Outer pose inference failed: {exc}")

        if not predictions:
            return self._failure("OUTER_POSE_NOT_DETECTED", "No card pose was detected.")
        prediction = predictions[0]
        boxes = getattr(prediction, "boxes", None)
        keypoints = getattr(prediction, "keypoints", None)
        if boxes is None or keypoints is None or len(boxes) == 0:
            return self._failure("OUTER_POSE_NOT_DETECTED", "No card pose was detected.")
        try:
            bbox_confidences = _to_numpy(boxes.conf).astype(np.float32).reshape(-1)
            bbox_values = _to_numpy(boxes.xyxy).astype(np.float32).reshape(-1, 4)
            point_values = _to_numpy(keypoints.xy).astype(np.float32).reshape(-1, 4, 2)
            raw_keypoint_conf = getattr(keypoints, "conf", None)
            if raw_keypoint_conf is None:
                keypoint_confidences = np.repeat(bbox_confidences[:, None], 4, axis=1)
            else:
                keypoint_confidences = _to_numpy(raw_keypoint_conf).astype(np.float32).reshape(-1, 4)
            count = min(len(bbox_confidences), len(bbox_values), len(point_values), len(keypoint_confidences))
            if count == 0:
                raise ValueError("Empty pose tensors")
            scores = 0.40 * bbox_confidences[:count] + 0.60 * np.mean(keypoint_confidences[:count], axis=1)
            selected = int(np.argmax(scores))
        except (AttributeError, TypeError, ValueError) as exc:
            return self._failure("OUTER_POSE_NOT_DETECTED", f"Invalid YOLO pose output: {exc}")

        validation = validate_and_order_outer_keypoints(
            point_values[selected],
            keypoint_confidences[selected],
            image.shape,
            self.config,
        )
        ordered_points = validation.get("ordered_points")
        raw_ordered_points = ordered_points
        bbox = [float(value) for value in bbox_values[selected]]
        bbox_confidence = float(np.clip(bbox_confidences[selected], 0.0, 1.0))
        validation_metrics = dict(validation.get("metrics", {}))
        ordered_conf = validation_metrics.get("ordered_keypoint_confidence")
        if ordered_conf is None:
            ordered_conf = [float(value) for value in keypoint_confidences[selected]]
        calibration_applied = False
        if validation["success"] and ordered_points is not None:
            calibrated_points = apply_outer_pose_calibration(ordered_points, self.config)
            calibration_applied = not np.allclose(calibrated_points, np.asarray(ordered_points), atol=1e-4)
            if calibration_applied:
                validation = validate_and_order_outer_keypoints(
                    calibrated_points,
                    ordered_conf,
                    image.shape,
                    self.config,
                )
                ordered_points = validation.get("ordered_points")
                validation_metrics = dict(validation.get("metrics", {}))
        keypoint_confidence = {
            name: float(np.clip(value, 0.0, 1.0)) for name, value in zip(KEYPOINT_NAMES, ordered_conf)
        }
        mean_keypoint_confidence = float(np.mean(list(keypoint_confidence.values())))
        min_keypoint_confidence = float(np.min(list(keypoint_confidence.values())))
        confidence = float(np.clip(0.40 * bbox_confidence + 0.60 * mean_keypoint_confidence, 0.0, 1.0))
        metrics = {
            "bbox_confidence": bbox_confidence,
            "mean_keypoint_confidence": mean_keypoint_confidence,
            "min_keypoint_confidence": min_keypoint_confidence,
            "aspect_ratio": float(validation_metrics.get("aspect_ratio", 0.0)),
            "aspect_ratio_error": float(validation_metrics.get("aspect_ratio_error", 1.0)),
            "area_ratio": float(validation_metrics.get("area_ratio", 0.0)),
            "calibration_applied": calibration_applied,
            "raw_points": raw_ordered_points,
        }
        if not validation["success"]:
            code = str(validation["error_code"])
            message = (
                "Outer pose keypoint confidence is below the configured threshold."
                if code == "LOW_CONFIDENCE_OUTER_POSE"
                else "Detected outer keypoints failed geometry validation."
            )
            return self._failure(
                code,
                message,
                points=ordered_points,
                bbox=bbox,
                confidence=confidence,
                keypoint_confidence=keypoint_confidence,
                metrics=metrics,
            )
        if confidence < float(self.config["low_confidence_threshold"]):
            return self._failure(
                "LOW_CONFIDENCE_OUTER_POSE",
                "Outer pose confidence is below the configured threshold.",
                points=ordered_points,
                bbox=bbox,
                confidence=confidence,
                keypoint_confidence=keypoint_confidence,
                metrics=metrics,
            )
        edge_metrics = calculate_outer_pose_edge_support(image, ordered_points, self.config)
        metrics.update(edge_metrics)
        edge_guard = self.config.get("edge_guard", {})
        if isinstance(edge_guard, Mapping) and bool(edge_guard.get("enabled", False)):
            if (
                float(edge_metrics["edge_support_score"]) < float(edge_guard.get("min_edge_support", 0.0))
                or float(edge_metrics["min_side_edge_support"])
                < float(edge_guard.get("min_side_edge_support", 0.0))
            ):
                return self._failure(
                    "LOW_CONFIDENCE_OUTER_POSE",
                    "Outer pose does not have enough edge support on every card side.",
                    points=ordered_points,
                    bbox=bbox,
                    confidence=confidence,
                    keypoint_confidence=keypoint_confidence,
                    metrics=metrics,
                )
        return {
            "success": True,
            "points": ordered_points,
            "bbox": bbox,
            "confidence": confidence,
            "keypoint_confidence": keypoint_confidence,
            "method": "deep_outer_pose",
            "error_code": None,
            "message": "Outer card pose detected.",
            "metrics": metrics,
        }
