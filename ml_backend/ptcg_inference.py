from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping


PACKAGE_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PACKAGE_ROOT / "models"
RUNTIME_ROOT = Path(
    os.environ.get("PTCG_RUNTIME_DIR", Path(tempfile.gettempdir()) / "ptcg_model_runtime")
).resolve()
(RUNTIME_ROOT / "yolo" / "Ultralytics").mkdir(parents=True, exist_ok=True)
(RUNTIME_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_ROOT / "yolo"))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_ROOT / "matplotlib"))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

if __package__:
    from .boundary_quality_guard import assess_inner_quality, assess_outer_quality
    from .card_quality_processor.config import DEFAULT_CONFIG, normalize_config
    from .card_quality_processor.outer_line_refiner import (
        load_outer_line_refiner,
        refine_outer_quad_learned,
        select_outer_quad_policy,
    )
    from .card_quality_processor.outer_boundary_contact_recovery import (
        recover_boundary_contact_outer,
    )
    from .card_quality_processor.outer_photometric_rescue import rescue_outer_prediction
    from .card_quality_processor.pre_cropped_card_recovery import (
        confirm_pre_cropped_inner,
        propose_pre_cropped_outer,
    )
    from .card_quality_processor.outer_pose_detection import (
        KEYPOINT_NAMES,
        OuterPoseDetector,
        validate_and_order_outer_keypoints,
    )
    from .card_quality_processor.rectification import rectify_card_by_points
    from .card_quality_processor.visualization import draw_outer_pose_result
    from .inner_frame.calibrate_inner_frame_box import calibrate_inner_frame_box
    from .inner_frame.edge_refiner import EDGE_TO_KEY, EDGES, load_refiner, predict_edge
    from .inner_frame.global_boundary_hypothesis import (
        analyze_inner_edge_hypotheses,
        consensus_rescue_decision,
        guarded_top_specialist_decision,
        guarded_top_tail_fallback_decision,
        score_inner_edge_positions,
    )
    from .inner_frame.physical_inner_prior import guarded_refine_physical_inner_box
    from .inner_frame.stabilize_inner_frame_box import stabilize_inner_frame_box
else:
    from boundary_quality_guard import assess_inner_quality, assess_outer_quality
    from card_quality_processor.config import DEFAULT_CONFIG, normalize_config
    from card_quality_processor.outer_line_refiner import (
        load_outer_line_refiner,
        refine_outer_quad_learned,
        select_outer_quad_policy,
    )
    from card_quality_processor.outer_boundary_contact_recovery import (
        recover_boundary_contact_outer,
    )
    from card_quality_processor.outer_photometric_rescue import rescue_outer_prediction
    from card_quality_processor.pre_cropped_card_recovery import (
        confirm_pre_cropped_inner,
        propose_pre_cropped_outer,
    )
    from card_quality_processor.outer_pose_detection import (
        KEYPOINT_NAMES,
        OuterPoseDetector,
        validate_and_order_outer_keypoints,
    )
    from card_quality_processor.rectification import rectify_card_by_points
    from card_quality_processor.visualization import draw_outer_pose_result
    from inner_frame.calibrate_inner_frame_box import calibrate_inner_frame_box
    from inner_frame.edge_refiner import EDGE_TO_KEY, EDGES, load_refiner, predict_edge
    from inner_frame.global_boundary_hypothesis import (
        analyze_inner_edge_hypotheses,
        consensus_rescue_decision,
        guarded_top_specialist_decision,
        guarded_top_tail_fallback_decision,
        score_inner_edge_positions,
    )
    from inner_frame.physical_inner_prior import guarded_refine_physical_inner_box
    from inner_frame.stabilize_inner_frame_box import stabilize_inner_frame_box


VERSION = "ptcg_outer_inner_pipeline_20260822_precropped_v1"
TOP_LEFT_SPECIALIST_EDGES = frozenset({"left", "top"})
TOP_LEFT_SPECIALIST_MIN_CONFIDENCE = 0.52
TOP_LEFT_SPECIALIST_CONFIDENCE_MARGIN = 0.06
CARD_ASPECT_RATIO = 880.0 / 630.0


def _configure_deterministic_inference() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass


def read_image(path: str | Path) -> np.ndarray:
    """Read a file as an OpenCV BGR image using a Unicode-safe Windows path."""
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


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _resolve_device(value: str | int | None) -> str:
    if value is None:
        return "0" if torch.cuda.is_available() else "cpu"
    text = str(value).strip().lower()
    if text == "cpu" or not torch.cuda.is_available():
        return "cpu"
    return text


def _torch_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value.startswith("cuda:"):
        return torch.device(value)
    return torch.device(f"cuda:{value.split(',', 1)[0]}")


def _internal_box(box: Mapping[str, float]) -> dict[str, float]:
    return {
        "x_left": float(box["left"]),
        "x_right": float(box["right"]),
        "y_top": float(box["top"]),
        "y_bottom": float(box["bottom"]),
    }


def _external_box(box: Mapping[str, float]) -> dict[str, float]:
    return {
        "left": float(box["x_left"]),
        "right": float(box["x_right"]),
        "top": float(box["y_top"]),
        "bottom": float(box["y_bottom"]),
    }


def _clip_box(box: Mapping[str, float], width: int, height: int) -> dict[str, float]:
    left = float(np.clip(box["left"], 0, width - 2))
    right = float(np.clip(box["right"], left + 1, width - 1))
    top = float(np.clip(box["top"], 0, height - 2))
    bottom = float(np.clip(box["bottom"], top + 1, height - 1))
    return {
        "left": round(left, 3),
        "top": round(top, 3),
        "right": round(right, 3),
        "bottom": round(bottom, 3),
    }


def _box_corners(box: Mapping[str, float]) -> list[list[float]]:
    return [
        [float(box["left"]), float(box["top"])],
        [float(box["right"]), float(box["top"])],
        [float(box["right"]), float(box["bottom"])],
        [float(box["left"]), float(box["bottom"])],
    ]


def _line_center_geometry(
    box: Mapping[str, float],
    width: int = 630,
    height: int = 880,
) -> dict[str, Any]:
    centers = {
        "left": round(float(box["left"]), 4),
        "right": round(float(box["right"]), 4),
        "top": round(float(box["top"]), 4),
        "bottom": round(float(box["bottom"]), 4),
    }
    middle_x = round(width / 2.0, 4)
    middle_y = round(height / 2.0, 4)
    widths = {
        "left": centers["left"],
        "right": round((width - 1) - centers["right"], 4),
        "top": centers["top"],
        "bottom": round((height - 1) - centers["bottom"], 4),
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
            "border_fraction_of_card": {
                "left": round(widths["left"] / width, 4),
                "right": round(widths["right"] / width, 4),
                "top": round(widths["top"] / height, 4),
                "bottom": round(widths["bottom"] / height, 4),
            },
            "centering_pair_percent": pair,
            "centering_ratio": None if pair is None else {
                "left_percent": pair["left"],
                "right_percent": pair["right"],
                "top_percent": pair["top"],
                "bottom_percent": pair["bottom"],
            },
            "formula_contract": {
                "input_coordinates": "zero_width_red_line_centers",
                "right_width": "rectified_width - 1 - right_line_center_x",
                "bottom_width": "rectified_height - 1 - bottom_line_center_y",
            },
        },
    }


def _draw_inner_boxes(
    image: np.ndarray,
    base: Mapping[str, float],
    final: Mapping[str, float],
) -> np.ndarray:
    overlay = image.copy()
    for box, color in ((base, (0, 0, 255)), (final, (255, 220, 0))):
        cv2.rectangle(
            overlay,
            (round(box["left"]), round(box["top"])),
            (round(box["right"]), round(box["bottom"])),
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        overlay,
        "base red / final cyan",
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def detect_official_full_frame_alpha(
    image: np.ndarray,
) -> dict[str, Any] | None:
    """Identify a tightly cropped official card PNG from its alpha geometry.

    The route is intentionally strict: a portrait card aspect ratio, opaque
    center and edge midpoints, four transparent corner pixels, a nearly fully
    opaque canvas, and foreground touching every image side are all required.
    Ordinary photographs normalized to RGBA remain fully opaque and therefore
    cannot enter this path.
    """

    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 4
        or image.shape[0] < 100
        or image.shape[1] < 70
    ):
        return None
    height, width = image.shape[:2]
    aspect_ratio = float(height) / max(float(width), 1.0)
    aspect_error = abs(aspect_ratio - CARD_ASPECT_RATIO) / CARD_ASPECT_RATIO
    if aspect_error > 0.035:
        return None

    alpha = image[:, :, 3]
    corners = np.asarray(
        [alpha[0, 0], alpha[0, -1], alpha[-1, -1], alpha[-1, 0]],
        dtype=np.uint8,
    )
    edge_midpoints = np.asarray(
        [
            alpha[0, width // 2],
            alpha[height // 2, -1],
            alpha[-1, width // 2],
            alpha[height // 2, 0],
            alpha[height // 2, width // 2],
        ],
        dtype=np.uint8,
    )
    if not bool(np.all(corners <= 16) and np.all(edge_midpoints >= 250)):
        return None

    foreground = alpha >= 128
    opaque_ratio = float(np.mean(alpha >= 250))
    transparent_ratio = float(np.mean(alpha <= 16))
    partial_alpha_ratio = float(np.mean((alpha > 16) & (alpha < 250)))
    touches_all_sides = bool(
        foreground[0].any()
        and foreground[-1].any()
        and foreground[:, 0].any()
        and foreground[:, -1].any()
    )
    if (
        opaque_ratio < 0.95
        or not (0.00001 <= transparent_ratio <= 0.04)
        or not touches_all_sides
    ):
        return None
    return {
        "kind": "official_full_frame_alpha",
        "confidence": 0.995,
        "aspect_ratio": aspect_ratio,
        "aspect_ratio_error": aspect_error,
        "opaque_ratio": opaque_ratio,
        "transparent_ratio": transparent_ratio,
        "partial_alpha_ratio": partial_alpha_ratio,
        "corner_alpha": [int(value) for value in corners],
        "edge_midpoint_alpha": [int(value) for value in edge_midpoints],
    }


def _official_full_frame_outer(
    image: np.ndarray,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    height, width = image.shape[:2]
    points = [
        [0.0, 0.0],
        [float(width - 1), 0.0],
        [float(width - 1), float(height - 1)],
        [0.0, float(height - 1)],
    ]
    confidence = float(profile.get("confidence", 0.995))
    aspect_ratio = float(height - 1) / max(float(width - 1), 1.0)
    area_ratio = float((width - 1) * (height - 1)) / max(
        float(width * height),
        1.0,
    )
    return {
        "success": True,
        "points": points,
        "bbox": [0.0, 0.0, float(width - 1), float(height - 1)],
        "confidence": confidence,
        "keypoint_confidence": {name: confidence for name in KEYPOINT_NAMES},
        "method": "official_full_frame_alpha",
        "error_code": None,
        "message": (
            "Official transparent full-frame card asset accepted from guarded "
            "alpha geometry."
        ),
        "metrics": {
            "bbox_confidence": confidence,
            "mean_keypoint_confidence": confidence,
            "min_keypoint_confidence": confidence,
            "aspect_ratio": aspect_ratio,
            "aspect_ratio_error": abs(aspect_ratio - CARD_ASPECT_RATIO)
            / CARD_ASPECT_RATIO,
            "area_ratio": area_ratio,
            "min_border_margin_ratio": 0.0,
            "in_bounds": True,
            "convex": True,
            "spatial_order": True,
            "geometry_valid": True,
            "silhouette_refinement_applied": False,
            "physical_edge_refinement_applied": False,
            "outer_line_refinement_applied": False,
            "raw_points": points,
            "input_profile": dict(profile),
            "full_frame_asset_bypass": True,
        },
    }


@dataclass
class PipelineModels:
    outer_seg: Path = MODELS_DIR / "outer_seg.pt"
    outer_pose: Path = MODELS_DIR / "outer_pose.pt"
    outer_line_refiner: Path = MODELS_DIR / "outer_line_refiner_v1.pt"
    outer_line_gate: Path = PACKAGE_ROOT / "card_quality_processor" / "outer_line_gate.json"
    inner_yolo: Path = MODELS_DIR / "inner_frame_yolo_v3_base_candidate.pt"
    inner_refiner: Path = MODELS_DIR / "inner_frame_edge_refiner_v5_candidate.pt"
    inner_refiner_top_left: Path | None = field(
        default_factory=lambda: (
            MODELS_DIR / "inner_frame_edge_refiner_top_left.pt"
            if (MODELS_DIR / "inner_frame_edge_refiner_top_left.pt").is_file()
            else None
        )
    )
    inner_refiner_top: Path | None = field(
        default_factory=lambda: (
            MODELS_DIR / "inner_frame_edge_refiner_top_v6.pt"
            if (MODELS_DIR / "inner_frame_edge_refiner_top_v6.pt").is_file()
            else None
        )
    )
    inner_top_gate: Path = PACKAGE_ROOT / "inner_frame" / "top_specialist_gate.json"
    inner_gate: Path = PACKAGE_ROOT / "inner_frame" / "gate.json"
    inner_physical_prior: Path = (
        PACKAGE_ROOT / "inner_frame" / "physical_inner_prior.json"
    )

    def validate(self) -> None:
        missing = [
            str(path)
            for path in vars(self).values()
            if path is not None and not Path(path).is_file()
        ]
        if missing:
            raise FileNotFoundError("Missing required model/config files: " + ", ".join(missing))


class CardFramePipeline:
    """End-to-end outer-card rectification and inner-frame detection.

    `infer_image` accepts an OpenCV BGR ndarray. Outer coordinates belong to
    the original image; inner coordinates belong to the 630x880 rectified card.
    """

    def __init__(
        self,
        device: str | int | None = None,
        models: PipelineModels | None = None,
        outer_config: Mapping[str, Any] | None = None,
    ) -> None:
        _configure_deterministic_inference()
        self.device = _resolve_device(device)
        self.torch_device = _torch_device(self.device)
        self.models = models or PipelineModels()
        self.models.validate()

        # Resolve user overrides first, then bind the explicitly supplied model
        # paths.  The previous shallow top-level merge could replace the whole
        # ``outer_detection`` section and silently restore DEFAULT_CONFIG's
        # outer_seg path, so candidate/deployed weights were never loaded when
        # any outer_config override was also present.
        config = normalize_config(outer_config) if outer_config else deepcopy(DEFAULT_CONFIG)
        config["outer_detection"]["deep_pose"]["device"] = self.device
        config["outer_detection"]["deep_pose"]["model_path"] = str(self.models.outer_pose)
        config["outer_detection"]["deep_pose"]["silhouette_refinement"]["model_path"] = str(
            self.models.outer_seg
        )
        self.outer_config = config
        self.outer_detector = OuterPoseDetector(config=config)

        self._outer_line_refiner: Any | None = None
        self._outer_line_refiner_config: dict[str, Any] | None = None
        self._outer_line_gate = json.loads(
            self.models.outer_line_gate.read_text(encoding="utf-8")
        )
        self._inner_yolo: YOLO | None = None
        self._inner_refiner: Any | None = None
        self._inner_refiner_config: dict[str, Any] | None = None
        self._inner_refiner_top_left: Any | None = None
        self._inner_refiner_top_left_config: dict[str, Any] | None = None
        self._inner_refiner_top: Any | None = None
        self._inner_refiner_top_config: dict[str, Any] | None = None
        self._gate = json.loads(self.models.inner_gate.read_text(encoding="utf-8"))
        self._top_gate = json.loads(self.models.inner_top_gate.read_text(encoding="utf-8"))
        self._physical_inner_prior = json.loads(
            self.models.inner_physical_prior.read_text(encoding="utf-8")
        )

    def _load_outer_line_model(self) -> None:
        if self._outer_line_refiner is None:
            self._outer_line_refiner, self._outer_line_refiner_config = (
                load_outer_line_refiner(
                    self.models.outer_line_refiner,
                    self.torch_device,
                )
            )

    def _refine_outer(self, image: np.ndarray, outer: dict[str, Any]) -> dict[str, Any]:
        """Run the frozen four-side policy, falling back safely on any anomaly."""

        current_points = outer.get("points")
        metrics = dict(outer.get("metrics", {}))
        raw_points = metrics.get("raw_points") or current_points
        if current_points is None or raw_points is None:
            return outer
        try:
            self._load_outer_line_model()
            assert self._outer_line_refiner is not None
            assert self._outer_line_refiner_config is not None
            learned = refine_outer_quad_learned(
                image,
                raw_points,
                self._outer_line_refiner,
                self._outer_line_refiner_config,
                device=self.torch_device,
                config=self._outer_line_gate.get("learned_gate", {}),
            )
            selection = select_outer_quad_policy(
                raw_points,
                current_points,
                learned,
                self._outer_line_gate,
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            metrics["outer_line_refinement"] = {
                "applied": False,
                "selected_source": "legacy_exception_fallback",
                "error": f"{type(exc).__name__}: {exc}",
            }
            outer["metrics"] = metrics
            return outer

        confidence = outer.get("keypoint_confidence")
        candidates = (
            (
                selection["points"],
                selection["selected_source"],
                float(selection["margin_canonical_px"]),
            ),
            (
                selection["selected_points_before_margin"],
                f"{selection['selected_source']}_without_margin",
                0.0,
            ),
            (raw_points, "raw_silhouette_validation_fallback", 0.0),
            (current_points, "legacy_validation_fallback", 0.0),
        )
        accepted_points: list[list[float]] | None = None
        accepted_source = "legacy_validation_fallback"
        applied_margin = 0.0
        validation_error = None
        for candidate_points, candidate_source, candidate_margin in candidates:
            validation = validate_and_order_outer_keypoints(
                candidate_points,
                confidence,
                image.shape,
                self.outer_config,
            )
            ordered = validation.get("ordered_points")
            if validation.get("success") and ordered is not None:
                accepted_points = ordered
                accepted_source = candidate_source
                applied_margin = candidate_margin
                break
            validation_error = validation.get("error_code")
        if accepted_points is None:
            metrics["outer_line_refinement"] = {
                "applied": False,
                "selected_source": "legacy_validation_fallback",
                "error": validation_error or "INVALID_KEYPOINT_GEOMETRY",
            }
            outer["metrics"] = metrics
            return outer

        points_array = np.asarray(accepted_points, dtype=np.float32)
        outer["points"] = points_array.round(3).tolist()
        outer["bbox"] = [
            float(points_array[:, 0].min()),
            float(points_array[:, 1].min()),
            float(points_array[:, 0].max()),
            float(points_array[:, 1].max()),
        ]
        metrics["outer_line_refinement_applied"] = accepted_source.startswith(
            "learned_four_side_refiner"
        )
        metrics["outer_line_refinement"] = {
            "applied": True,
            "policy_version": self._outer_line_gate.get("version"),
            "selected_source": accepted_source,
            "margin_canonical_px": applied_margin,
            "learned_allowed": selection["learned_allowed"],
            "production_learned_allowed": selection["production_learned_allowed"],
            "safe_legacy_refinement": selection["safe_legacy_refinement"],
            "legacy_fallback_geometry": selection["legacy_fallback_geometry"],
            "learned_reason": learned.get("reason"),
            "learned_metrics": selection["learned_metrics"],
        }
        outer["metrics"] = metrics
        outer["message"] = (
            "Outer card silhouette refined by the guarded high-resolution "
            "four-side line policy."
        )
        return outer

    def _load_inner_models(self) -> None:
        if self._inner_yolo is None:
            self._inner_yolo = YOLO(str(self.models.inner_yolo))
        if self._inner_refiner is None:
            self._inner_refiner, self._inner_refiner_config = load_refiner(
                self.models.inner_refiner,
                self.torch_device,
            )
        if self.models.inner_refiner_top_left is not None and self._inner_refiner_top_left is None:
            self._inner_refiner_top_left, self._inner_refiner_top_left_config = load_refiner(
                self.models.inner_refiner_top_left,
                self.torch_device,
            )
        if self.models.inner_refiner_top is not None and self._inner_refiner_top is None:
            self._inner_refiner_top, self._inner_refiner_top_config = load_refiner(
                self.models.inner_refiner_top,
                self.torch_device,
            )

    def _infer_inner(self, rectified: np.ndarray) -> dict[str, Any]:
        self._load_inner_models()
        assert self._inner_yolo is not None
        assert self._inner_refiner is not None
        assert self._inner_refiner_config is not None

        height, width = rectified.shape[:2]
        prediction = self._inner_yolo.predict(
            source=rectified,
            imgsz=960,
            conf=0.05,
            device=self.device,
            retina_masks=False,
            augment=False,
            verbose=False,
        )[0]
        if prediction.masks is None or prediction.boxes is None or len(prediction.boxes) == 0:
            return {
                "success": False,
                "error_code": "INNER_FRAME_NOT_DETECTED",
                "message": "No inner-frame segmentation mask was detected.",
            }
        confidences = prediction.boxes.conf.detach().cpu().numpy()
        selected = int(np.argmax(confidences))
        polygon = np.asarray(prediction.masks.xy[selected], dtype=np.float64)
        if len(polygon) < 3:
            return {
                "success": False,
                "error_code": "INVALID_INNER_FRAME_MASK",
                "message": "The selected inner-frame mask polygon is invalid.",
            }
        raw = {
            "left": float(polygon[:, 0].min()),
            "right": float(polygon[:, 0].max()),
            "top": float(polygon[:, 1].min()),
            "bottom": float(polygon[:, 1].max()),
        }
        stabilized = stabilize_inner_frame_box(rectified, _internal_box(raw))
        base = _external_box(calibrate_inner_frame_box(stabilized.box, width, height))
        base_internal = _internal_box(base)
        global_hypotheses = analyze_inner_edge_hypotheses(rectified, base_internal)
        final = dict(base)
        details: dict[str, dict[str, float | bool | str | None]] = {}
        for edge in EDGES:
            key = EDGE_TO_KEY[edge]
            stable_prediction = predict_edge(
                self._inner_refiner,
                rectified,
                edge,
                base_internal[key],
                base_internal,
                device=self.torch_device,
                band_half=int(self._inner_refiner_config.get("band_half", 32)),
                patch_width=int(self._inner_refiner_config.get("patch_width", 96)),
                patch_height=int(self._inner_refiner_config.get("patch_height", 192)),
            )
            specialist_prediction = None
            if (
                edge in TOP_LEFT_SPECIALIST_EDGES
                and self._inner_refiner_top_left is not None
                and self._inner_refiner_top_left_config is not None
            ):
                specialist_prediction = predict_edge(
                    self._inner_refiner_top_left,
                    rectified,
                    edge,
                    base_internal[key],
                    base_internal,
                    device=self.torch_device,
                    band_half=int(self._inner_refiner_top_left_config.get("band_half", 32)),
                    patch_width=int(self._inner_refiner_top_left_config.get("patch_width", 96)),
                    patch_height=int(self._inner_refiner_top_left_config.get("patch_height", 192)),
                )
            top_prediction = None
            if (
                edge == "top"
                and self._inner_refiner_top is not None
                and self._inner_refiner_top_config is not None
            ):
                top_prediction = predict_edge(
                    self._inner_refiner_top,
                    rectified,
                    edge,
                    base_internal[key],
                    base_internal,
                    device=self.torch_device,
                    band_half=int(self._inner_refiner_top_config.get("band_half", 32)),
                    patch_width=int(self._inner_refiner_top_config.get("patch_width", 96)),
                    patch_height=int(self._inner_refiner_top_config.get("patch_height", 224)),
                )
            use_top_left_specialist = bool(
                specialist_prediction is not None
                and specialist_prediction.confidence >= TOP_LEFT_SPECIALIST_MIN_CONFIDENCE
                and specialist_prediction.confidence - stable_prediction.confidence
                >= TOP_LEFT_SPECIALIST_CONFIDENCE_MARGIN
            )
            edge_prediction = (
                specialist_prediction if use_top_left_specialist else stable_prediction
            )
            assert edge_prediction is not None
            settings = self._gate["edges"][edge]
            accepted = bool(
                edge_prediction.confidence >= float(settings["confidence"])
                and abs(edge_prediction.offset) <= float(settings["max_move"])
            )
            applied = float(settings["blend"]) * edge_prediction.offset if accepted else 0.0
            top_route: dict[str, Any] = {
                "accepted": False,
                "position": base[edge] + applied,
                "reason": "top_specialist_unavailable" if edge == "top" else "not_top_edge",
            }
            top_evidence = None
            if edge == "top" and top_prediction is not None:
                stable_position = base[edge] + applied
                evidence = score_inner_edge_positions(
                    rectified,
                    base_internal,
                    "top",
                    [stable_position, top_prediction.refined],
                    segments=int(self._top_gate.get("segments", 8)),
                )
                top_evidence = {
                    "stable": evidence[0].to_dict(),
                    "specialist": evidence[1].to_dict(),
                }
                top_route = guarded_top_specialist_decision(
                    base_position=base[edge],
                    stable_position=stable_position,
                    specialist_position=top_prediction.refined,
                    specialist_confidence=top_prediction.confidence,
                    specialist_tta_disagreement=top_prediction.tta_disagreement,
                    stable_evidence=evidence[0],
                    specialist_evidence=evidence[1],
                    config=self._top_gate["gate"],
                )
                if bool(top_route["accepted"]):
                    applied = float(top_route["position"]) - base[edge]
                    accepted = True
            rescue = consensus_rescue_decision(
                edge,
                base_position=base[edge],
                stable_position=stable_prediction.refined,
                stable_confidence=stable_prediction.confidence,
                specialist_position=(
                    specialist_prediction.refined
                    if specialist_prediction is not None
                    else None
                ),
                specialist_confidence=(
                    specialist_prediction.confidence
                    if specialist_prediction is not None
                    else None
                ),
                hypothesis=global_hypotheses[edge],
            )
            if bool(rescue["accepted"]):
                applied = float(rescue["position"]) - base[edge]
                accepted = True
            top_tail_guard: dict[str, Any] = {
                "fallback": False,
                "position": base[edge] + applied,
                "reason": "not_top_edge" if edge != "top" else "tail_guard_unavailable",
            }
            if edge == "top" and top_evidence is not None:
                selected_position = base[edge] + applied
                selected_evidence = score_inner_edge_positions(
                    rectified,
                    base_internal,
                    "top",
                    [selected_position],
                    segments=int(self._top_gate.get("segments", 8)),
                )[0]
                top_tail_guard = guarded_top_tail_fallback_decision(
                    base_position=base[edge],
                    selected_position=selected_position,
                    selected_confidence=edge_prediction.confidence,
                    selected_evidence=selected_evidence,
                    config=self._top_gate["tail_guard"],
                )
                if bool(top_tail_guard["fallback"]):
                    applied = 0.0
                    accepted = False
            final[edge] = base[edge] + applied
            details[edge] = {
                "proposed_offset": round(edge_prediction.offset, 4),
                "applied_offset": round(applied, 4),
                "confidence": round(edge_prediction.confidence, 4),
                "entropy": round(edge_prediction.entropy, 4),
                "peak_mass": round(edge_prediction.peak_mass, 4),
                "accepted": accepted,
                "global_consensus_rescue": bool(rescue["accepted"]),
                "global_consensus_reason": str(rescue["reason"]),
                "refiner_variant": (
                    "top_left_specialist" if use_top_left_specialist else "stable"
                ),
                "stable_confidence": round(stable_prediction.confidence, 4),
                "specialist_confidence": (
                    round(specialist_prediction.confidence, 4)
                    if specialist_prediction is not None
                    else None
                ),
                "specialist_confidence_margin": TOP_LEFT_SPECIALIST_CONFIDENCE_MARGIN,
                "top_v6_router_accepted": bool(top_route["accepted"]),
                "top_v6_router_reason": str(top_route["reason"]),
                "top_v6_router": top_route if edge == "top" else None,
                "top_v6_evidence": top_evidence,
                "top_v6_confidence": (
                    round(top_prediction.confidence, 4)
                    if top_prediction is not None
                    else None
                ),
                "top_tail_guard_fallback": bool(top_tail_guard["fallback"]),
                "top_tail_guard_reason": str(top_tail_guard["reason"]),
                "top_tail_guard": top_tail_guard if edge == "top" else None,
            }
        def physical_evidence_provider(
            edge: str,
            positions: list[float] | tuple[float, ...],
            candidate_box: Mapping[str, float],
        ) -> list[dict[str, Any]]:
            return [
                item.to_dict()
                for item in score_inner_edge_positions(
                    rectified,
                    _internal_box(candidate_box),
                    edge,
                    list(positions),
                    segments=8,
                )
            ]

        physical_prior = guarded_refine_physical_inner_box(
            final,
            width,
            height,
            self._physical_inner_prior,
            evidence_provider=physical_evidence_provider,
        )
        final = _clip_box(physical_prior["box"], width, height)
        result = {
            "success": True,
            "coordinate_space": "rectified_card_pixels",
            "image_size": [width, height],
            "yolo_confidence": round(float(confidences[selected]), 6),
            "raw_box": {key: round(value, 3) for key, value in raw.items()},
            "base_box": {key: round(value, 3) for key, value in base.items()},
            "final_box": final,
            **_line_center_geometry(final, width, height),
            "corners_tl_tr_br_bl": _box_corners(final),
            "stabilizer": {
                "status": stabilized.status,
                "reason": stabilized.reason,
                "evidence": stabilized.evidence,
            },
            "edge_refinement": details,
            "global_edge_hypotheses": {
                edge: hypothesis.to_dict()
                for edge, hypothesis in global_hypotheses.items()
            },
            "physical_inner_prior": physical_prior,
            "_overlay": _draw_inner_boxes(rectified, base, final),
        }
        result["quality_assessment"] = assess_inner_quality(result)
        return result

    def infer_image(self, image_bgr: np.ndarray) -> dict[str, Any]:
        if not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
            return {
                "success": False,
                "stage": "input",
                "error_code": "INVALID_IMAGE",
                "message": "Input must be a non-empty OpenCV BGR ndarray.",
            }
        input_profile = detect_official_full_frame_alpha(image_bgr)
        if image_bgr.ndim != 3 or image_bgr.shape[2] not in (3, 4):
            return {
                "success": False,
                "stage": "input",
                "error_code": "INVALID_IMAGE",
                "message": "Input must be an OpenCV BGR or BGRA image.",
            }
        if image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

        if input_profile is not None:
            outer = _official_full_frame_outer(image_bgr, input_profile)
        else:
            outer = self.outer_detector.predict(image_bgr, conf=0.25)
            if not outer.get("success") and outer.get("points") is not None:
                outer = recover_boundary_contact_outer(
                    image_bgr,
                    outer,
                    self.outer_config,
                )
            if outer.get("success") and outer.get("points") is not None:
                outer, refinement_image, rescue_metrics = rescue_outer_prediction(
                    self.outer_detector.predict,
                    image_bgr,
                    outer,
                    conf=0.25,
                )
                metrics = dict(outer.get("metrics", {}))
                metrics["photometric_rescue"] = rescue_metrics
                outer["metrics"] = metrics
                outer = self._refine_outer(refinement_image, outer)
            if not outer.get("success"):
                provisional_outer = propose_pre_cropped_outer(image_bgr, outer)
                if provisional_outer is not None:
                    outer = provisional_outer
        outer["quality_assessment"] = assess_outer_quality(outer)
        outer_public = _jsonable(outer)
        if not outer.get("success") or not outer.get("points"):
            return {
                "success": False,
                "version": VERSION,
                "stage": "outer_frame",
                "error_code": outer.get("error_code") or "OUTER_FRAME_NOT_DETECTED",
                "message": outer.get("message", "Outer-frame detection failed."),
                "outer_frame": outer_public,
            }

        output_size = (
            int(self.outer_config["outer_detection"]["deep_pose"]["output_width"]),
            int(self.outer_config["outer_detection"]["deep_pose"]["output_height"]),
        )
        rectification = rectify_card_by_points(
            image_bgr,
            outer["points"],
            output_size,
            self.outer_config,
        )
        if not rectification.get("success") or rectification.get("rectified_image") is None:
            return {
                "success": False,
                "version": VERSION,
                "stage": "rectification",
                "error_code": rectification.get("error_code") or "RECTIFICATION_FAILED",
                "message": rectification.get("message", "Perspective rectification failed."),
                "outer_frame": outer_public,
            }

        rectification["confidence"] = float(outer.get("confidence", 0.0))
        rectified = rectification["rectified_image"]
        inner = self._infer_inner(rectified)
        if not inner.get("success"):
            return {
                "success": False,
                "version": VERSION,
                "stage": "inner_frame",
                "error_code": inner.get("error_code"),
                "message": inner.get("message"),
                "outer_frame": {
                    **outer_public,
                    "coordinate_space": "original_image_pixels",
                    "corner_order": list(KEYPOINT_NAMES),
                },
                "rectification": {
                    key: _jsonable(value)
                    for key, value in rectification.items()
                    if key != "rectified_image"
                },
                "_rectified_image": rectified,
            }

        recovery_metrics = outer.get("metrics", {}).get(
            "pre_cropped_card_recovery"
        )
        if isinstance(recovery_metrics, Mapping) and bool(
            recovery_metrics.get("provisional", False)
        ):
            confirmation = confirm_pre_cropped_inner(inner)
            metrics = dict(outer.get("metrics", {}))
            metrics["pre_cropped_card_recovery"] = {
                **dict(recovery_metrics),
                **confirmation,
                "provisional": False,
            }
            outer["metrics"] = metrics
            outer_public = _jsonable(outer)
            if not bool(confirmation.get("confirmed", False)):
                return {
                    "success": False,
                    "version": VERSION,
                    "stage": "outer_frame",
                    "error_code": "PRE_CROPPED_CARD_NOT_CONFIRMED",
                    "message": (
                        "The full-frame outer hypothesis was rejected because "
                        "the detected inner frame did not match the global "
                        "58x83 mm inner-edge geometry."
                    ),
                    "outer_frame": outer_public,
                    "rectification": {
                        key: _jsonable(value)
                        for key, value in rectification.items()
                        if key != "rectified_image"
                    },
                    "inner_frame": {
                        key: _jsonable(value)
                        for key, value in inner.items()
                        if not key.startswith("_")
                    },
                    "_rectified_image": rectified,
                    "_inner_overlay": inner["_overlay"],
                }
            outer["message"] = (
                "Pre-cropped full-frame card confirmed by the independently "
                "detected 58x83 mm printed inner-line inner edge."
            )
            outer["quality_assessment"] = assess_outer_quality(outer)
            outer_public = _jsonable(outer)

        outer_overlay = draw_outer_pose_result(
            image_bgr,
            outer.get("points"),
            None,
            float(outer.get("confidence", 0.0)),
            outer.get("keypoint_confidence"),
            outer.get("error_code"),
            outer.get("message"),
        )
        return {
            "success": True,
            "version": VERSION,
            "stage": "complete",
            "outer_frame": {
                **outer_public,
                "coordinate_space": "original_image_pixels",
                "corner_order": list(KEYPOINT_NAMES),
            },
            "rectification": {
                key: _jsonable(value)
                for key, value in rectification.items()
                if key != "rectified_image"
            },
            "inner_frame": {
                key: _jsonable(value)
                for key, value in inner.items()
                if not key.startswith("_")
            },
            "_outer_overlay": outer_overlay,
            "_rectified_image": rectified,
            "_inner_overlay": inner["_overlay"],
        }

    def infer_file(self, image_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        image_path = Path(image_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            encoded = np.fromfile(str(image_path), dtype=np.uint8)
        except OSError as exc:
            raise ValueError(f"Cannot read image file: {image_path}") from exc
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"Cannot decode image: {image_path}")
        result = self.infer_image(image)

        for key, filename in (
            ("_outer_overlay", "outer_frame_overlay.jpg"),
            ("_rectified_image", "rectified_card.jpg"),
            ("_inner_overlay", "inner_frame_overlay.jpg"),
        ):
            value = result.get(key)
            if isinstance(value, np.ndarray):
                write_image(output_dir / filename, value)

        public = {key: _jsonable(value) for key, value in result.items() if not key.startswith("_")}
        public["input"] = {
            "path": str(image_path),
            "color_order": "BGRA" if image.shape[2] == 4 else "BGR",
            "image_size": [int(image.shape[1]), int(image.shape[0])],
            "input_profile": detect_official_full_frame_alpha(image),
        }
        public["output_files"] = {
            "result_json": str(output_dir / "result.json"),
            "outer_overlay": str(output_dir / "outer_frame_overlay.jpg")
            if (output_dir / "outer_frame_overlay.jpg").exists()
            else None,
            "rectified_card": str(output_dir / "rectified_card.jpg")
            if (output_dir / "rectified_card.jpg").exists()
            else None,
            "inner_overlay": str(output_dir / "inner_frame_overlay.jpg")
            if (output_dir / "inner_frame_overlay.jpg").exists()
            else None,
        }
        (output_dir / "result.json").write_text(
            json.dumps(public, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return public


def infer_image(
    image_bgr: np.ndarray,
    device: str | int | None = None,
) -> dict[str, Any]:
    """Convenience API for one image; reuse CardFramePipeline for batches."""
    return CardFramePipeline(device=device).infer_image(image_bgr)
