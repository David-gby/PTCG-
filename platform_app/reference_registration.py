# -*- coding: utf-8 -*-
"""Automatic registration between an official reference image and a warped photo.

This module assumes that the photographed card has already been:

1. detected by its physical outer frame;
2. perspective-corrected into a rectangular standard view.

The goal is to estimate the displacement of the printed content relative to the
selected official reference image. No manually annotated logo rectangle is
required. The implementation automatically combines:

- SIFT or AKAZE feature matching;
- mutual Lowe-ratio filtering;
- pure-translation RANSAC;
- gradient phase correlation;
- ECC translation refinement;
- residual rotation and scale diagnostics.

Positive dx means that the photographed printed content is shifted to the
right relative to the reference image. Positive dy means a downward shift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2
    import numpy as np
except ImportError:  # The platform can validate uploads before ML runtime setup.
    cv2 = None
    np = None


class RegistrationInputError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def validate_pair(*, capture_png: bytes | None, reference_png: bytes | None) -> None:
    if not capture_png:
        raise RegistrationInputError("CAPTURE_IMAGE_REQUIRED", "请上传实拍图。")
    if not reference_png:
        raise RegistrationInputError("REFERENCE_IMAGE_REQUIRED", "请上传标准图。")


def _require_vision_runtime() -> None:
    if cv2 is None or np is None:
        raise RegistrationInputError(
            "ML_RUNTIME_DEPENDENCY_MISSING",
            "服务器缺少 OpenCV 或 NumPy，无法执行参考图配准。",
        )


@lru_cache(maxsize=1)
def _pipeline() -> Any:
    from ml_backend.ptcg_inference import CardFramePipeline

    return CardFramePipeline(device=None)


def _decode_and_rectify(payload: bytes, code_prefix: str) -> tuple[np.ndarray, np.ndarray, dict]:
    _require_vision_runtime()
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RegistrationInputError(f"{code_prefix}_IMAGE_UNREADABLE", "图片无法读取。")
    result = _pipeline().infer_image(image)
    rectified = result.get("_rectified_image")
    if not isinstance(rectified, np.ndarray) or rectified.size == 0:
        code = result.get("error_code") or f"{code_prefix}_RECTIFICATION_FAILED"
        raise RegistrationInputError(str(code), str(result.get("message") or "图片透视校正失败。"))
    return image, rectified, result


def run_reference_registration(*, capture_png: bytes, reference_png: bytes) -> tuple[dict, dict]:
    """Register a user-uploaded standard image against a user-uploaded card photo."""
    validate_pair(capture_png=capture_png, reference_png=reference_png)
    _require_vision_runtime()
    _, capture_rectified, capture_result = _decode_and_rectify(capture_png, "CAPTURE")
    _, reference_rectified, reference_result = _decode_and_rectify(reference_png, "REFERENCE")
    registration, debug = detect_automatic_reference_registration(
        reference_rectified,
        capture_rectified,
        reference_id="user_upload",
    )
    registration["measurement_mode"] = "reference_registration"
    registration["reference"] = {
        "source": "user_upload",
        "image_size": {"width": int(reference_rectified.shape[1]), "height": int(reference_rectified.shape[0])},
    }
    registration["source_size"] = {
        "width": int(capture_result.get("image_size", {}).get("width", capture_rectified.shape[1])),
        "height": int(capture_result.get("image_size", {}).get("height", capture_rectified.shape[0])),
    }
    registration["model_version"] = str(capture_result.get("version") or "unknown")
    registration["outer_corners"] = (capture_result.get("outer_frame") or {}).get("points")
    assets = {
        "capture_rectified": capture_rectified,
        "reference_rectified": reference_rectified,
        "registration_overlay": make_reference_registration_overlay(capture_rectified, registration, debug),
        "capture_model_result": capture_result,
        "reference_model_result": reference_result,
    }
    return registration, assets


DEFAULT_OPTIONS: Dict[str, Any] = {
    "border_ignore_ratio": 0.045,
    "max_features": 3000,
    "ratio_test": 0.76,
    "max_matches_for_ransac": 500,
    "ransac_threshold_px": 3.5,
    "min_good_matches": 12,
    "min_inliers": 8,
    "min_inlier_ratio": 0.35,
    "min_spatial_coverage": 0.10,
    "max_residual_median_px": 3.0,
    "phase_min_response": 0.035,
    "ecc_min_score": 0.20,
    "method_agreement_px": 5.0,
    "max_rotation_deg": 2.0,
    "max_scale_error": 0.035,
    "default_tolerance_px_x": 2.0,
    "default_tolerance_px_y": 2.0,
    "max_visualized_matches": 36,
}


@dataclass
class FeatureBackend:
    detector: Any
    norm_type: int
    name: str


def merge_options(overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    options = dict(DEFAULT_OPTIONS)
    if overrides:
        for key, value in overrides.items():
            if key in options and value is not None:
                options[key] = value
    return options


def read_reference_image(path: str | Path) -> np.ndarray:
    """Read a reference image and composite transparency over white."""
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(f"Reference image could not be read: {path}")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 3:
        return image

    if image.shape[2] != 4:
        raise ValueError(
            f"Unsupported reference image channel count: {image.shape[2]}"
        )

    bgr = image[:, :, :3].astype(np.float32)
    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    white = np.full_like(bgr, 255.0)
    composed = bgr * alpha + white * (1.0 - alpha)
    return np.clip(composed, 0, 255).astype(np.uint8)


def normalize_reference_to_capture(
    reference_image: np.ndarray,
    capture_image: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Rotate when needed and resize the reference to the capture dimensions."""
    capture_h, capture_w = capture_image.shape[:2]
    ref_h, ref_w = reference_image.shape[:2]

    capture_ratio = capture_h / max(float(capture_w), 1.0)
    same_ratio_error = abs(
        ref_h / max(float(ref_w), 1.0) - capture_ratio
    )
    rotated_ratio_error = abs(
        ref_w / max(float(ref_h), 1.0) - capture_ratio
    )

    rotated = False
    normalized = reference_image

    if rotated_ratio_error + 0.02 < same_ratio_error:
        normalized = cv2.rotate(reference_image, cv2.ROTATE_90_CLOCKWISE)
        rotated = True

    before_resize_h, before_resize_w = normalized.shape[:2]
    resized = (
        before_resize_w != capture_w
        or before_resize_h != capture_h
    )

    if resized:
        interpolation = (
            cv2.INTER_AREA
            if before_resize_w > capture_w or before_resize_h > capture_h
            else cv2.INTER_CUBIC
        )
        normalized = cv2.resize(
            normalized,
            (capture_w, capture_h),
            interpolation=interpolation,
        )

    info = {
        "original_width": int(ref_w),
        "original_height": int(ref_h),
        "rotated_90_clockwise": bool(rotated),
        "resized": bool(resized),
        "normalized_width": int(capture_w),
        "normalized_height": int(capture_h),
    }
    return normalized, info


def preprocess_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    return clahe.apply(gray)


def gradient_image(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    magnitude = cv2.GaussianBlur(magnitude, (3, 3), 0)

    low = float(np.percentile(magnitude, 5))
    high = float(np.percentile(magnitude, 99))
    normalized = (magnitude - low) / max(high - low, 1e-6)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def make_content_mask(
    image: np.ndarray,
    border_ignore_ratio: float,
) -> np.ndarray:
    """Build a mask that excludes the physical card edge and extreme glare."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    inset_x = max(4, int(round(w * float(border_ignore_ratio))))
    inset_y = max(4, int(round(h * float(border_ignore_ratio))))

    if w - 2 * inset_x <= 20 or h - 2 * inset_y <= 20:
        inset_x = max(2, int(w * 0.02))
        inset_y = max(2, int(h * 0.02))

    mask[
        inset_y : max(inset_y + 1, h - inset_y),
        inset_x : max(inset_x + 1, w - inset_x),
    ] = 255

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    glare = (
        (hsv[:, :, 2] >= 250)
        & (hsv[:, :, 1] <= 28)
    )
    crushed_black = gray <= 3

    invalid = (glare | crushed_black).astype(np.uint8) * 255
    invalid = cv2.dilate(
        invalid,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    )
    mask[invalid > 0] = 0

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    return mask


def create_feature_backend(max_features: int) -> FeatureBackend:
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(
            nfeatures=int(max_features),
            contrastThreshold=0.022,
            edgeThreshold=12,
            sigma=1.4,
        )
        return FeatureBackend(
            detector=detector,
            norm_type=cv2.NORM_L2,
            name="SIFT",
        )

    detector = cv2.AKAZE_create(
        descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
        threshold=0.0008,
    )
    return FeatureBackend(
        detector=detector,
        norm_type=cv2.NORM_HAMMING,
        name="AKAZE",
    )


def _ratio_filter(
    pairs: Sequence[Sequence[cv2.DMatch]],
    ratio: float,
) -> List[cv2.DMatch]:
    good: List[cv2.DMatch] = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        first, second = pair[0], pair[1]
        if first.distance < float(ratio) * second.distance:
            good.append(first)
    return good


def mutual_ratio_matches(
    descriptors_reference: np.ndarray,
    descriptors_capture: np.ndarray,
    norm_type: int,
    ratio: float,
) -> List[cv2.DMatch]:
    matcher = cv2.BFMatcher(norm_type, crossCheck=False)

    forward_pairs = matcher.knnMatch(
        descriptors_reference,
        descriptors_capture,
        k=2,
    )
    reverse_pairs = matcher.knnMatch(
        descriptors_capture,
        descriptors_reference,
        k=2,
    )

    forward = _ratio_filter(forward_pairs, ratio)
    reverse = _ratio_filter(reverse_pairs, ratio)
    reverse_pairs_set = {
        (match.trainIdx, match.queryIdx)
        for match in reverse
    }

    mutual = [
        match
        for match in forward
        if (match.queryIdx, match.trainIdx) in reverse_pairs_set
    ]
    mutual.sort(key=lambda item: float(item.distance))
    return mutual


def estimate_translation_ransac(
    reference_points: np.ndarray,
    capture_points: np.ndarray,
    threshold_px: float,
) -> Dict[str, Any]:
    """Estimate pure translation using robust displacement clustering."""
    if len(reference_points) == 0:
        return {
            "success": False,
            "error_code": "NO_MATCH_POINTS",
        }

    displacements = (
        capture_points.astype(np.float32)
        - reference_points.astype(np.float32)
    )

    best_mask = None
    best_count = -1
    best_median_residual = float("inf")

    for hypothesis in displacements:
        residuals = np.linalg.norm(
            displacements - hypothesis,
            axis=1,
        )
        mask = residuals <= float(threshold_px)
        count = int(np.sum(mask))

        if count == 0:
            continue

        median_residual = float(
            np.median(residuals[mask])
        )

        if (
            count > best_count
            or (
                count == best_count
                and median_residual < best_median_residual
            )
        ):
            best_mask = mask
            best_count = count
            best_median_residual = median_residual

    if best_mask is None or best_count <= 0:
        return {
            "success": False,
            "error_code": "TRANSLATION_RANSAC_FAILED",
        }

    for _ in range(3):
        center = np.median(
            displacements[best_mask],
            axis=0,
        )
        residuals = np.linalg.norm(
            displacements - center,
            axis=1,
        )
        refined_mask = residuals <= float(threshold_px)
        if np.array_equal(refined_mask, best_mask):
            break
        best_mask = refined_mask

    inlier_displacements = displacements[best_mask]
    center = np.median(inlier_displacements, axis=0)
    residuals = np.linalg.norm(
        inlier_displacements - center,
        axis=1,
    )

    dx_values = inlier_displacements[:, 0]
    dy_values = inlier_displacements[:, 1]

    return {
        "success": True,
        "dx": float(center[0]),
        "dy": float(center[1]),
        "inlier_mask": best_mask,
        "inlier_count": int(np.sum(best_mask)),
        "inlier_ratio": float(np.mean(best_mask)),
        "residual_median_px": float(np.median(residuals)),
        "residual_p90_px": float(np.percentile(residuals, 90)),
        "dx_mad_px": float(
            np.median(np.abs(dx_values - np.median(dx_values)))
        ),
        "dy_mad_px": float(
            np.median(np.abs(dy_values - np.median(dy_values)))
        ),
    }


def spatial_coverage_score(
    points: np.ndarray,
    width: int,
    height: int,
    columns: int = 6,
    rows: int = 8,
) -> float:
    if len(points) == 0:
        return 0.0

    occupied = set()
    for x, y in points:
        column = min(
            columns - 1,
            max(0, int(float(x) / max(width, 1) * columns)),
        )
        row = min(
            rows - 1,
            max(0, int(float(y) / max(height, 1) * rows)),
        )
        occupied.add((column, row))

    return float(len(occupied) / float(columns * rows))


def estimate_affine_diagnostic(
    reference_points: np.ndarray,
    capture_points: np.ndarray,
) -> Dict[str, Any]:
    if len(reference_points) < 4:
        return {
            "success": False,
            "rotation_deg": None,
            "scale": None,
        }

    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        reference_points.reshape(-1, 1, 2),
        capture_points.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.5,
        maxIters=3000,
        confidence=0.995,
        refineIters=20,
    )

    if matrix is None:
        return {
            "success": False,
            "rotation_deg": None,
            "scale": None,
        }

    a = float(matrix[0, 0])
    b = float(matrix[1, 0])
    scale = math.sqrt(a * a + b * b)
    rotation_deg = math.degrees(math.atan2(b, a))

    return {
        "success": True,
        "matrix": matrix.tolist(),
        "rotation_deg": float(rotation_deg),
        "scale": float(scale),
        "inlier_count": (
            int(np.sum(inlier_mask))
            if inlier_mask is not None
            else None
        ),
    }


def phase_translation(
    reference_gradient: np.ndarray,
    capture_gradient: np.ndarray,
    border_ignore_ratio: float,
) -> Dict[str, Any]:
    h, w = reference_gradient.shape[:2]
    inset_x = max(4, int(round(w * border_ignore_ratio)))
    inset_y = max(4, int(round(h * border_ignore_ratio)))

    reference_crop = reference_gradient[
        inset_y : h - inset_y,
        inset_x : w - inset_x,
    ]
    capture_crop = capture_gradient[
        inset_y : h - inset_y,
        inset_x : w - inset_x,
    ]

    if min(reference_crop.shape[:2]) < 32:
        return {
            "success": False,
            "error_code": "PHASE_CROP_TOO_SMALL",
        }

    window = cv2.createHanningWindow(
        (reference_crop.shape[1], reference_crop.shape[0]),
        cv2.CV_32F,
    )

    shift, response = cv2.phaseCorrelate(
        reference_crop.astype(np.float32) * window,
        capture_crop.astype(np.float32) * window,
    )

    return {
        "success": bool(np.isfinite(shift[0]) and np.isfinite(shift[1])),
        "dx": float(shift[0]),
        "dy": float(shift[1]),
        "response": float(response),
    }


def ecc_translation(
    reference_gradient: np.ndarray,
    capture_gradient: np.ndarray,
    initial_dx: float,
    initial_dy: float,
    content_mask: np.ndarray,
) -> Dict[str, Any]:
    warp_matrix = np.array(
        [
            [1.0, 0.0, float(initial_dx)],
            [0.0, 1.0, float(initial_dy)],
        ],
        dtype=np.float32,
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        100,
        1e-6,
    )

    try:
        score, refined = cv2.findTransformECC(
            reference_gradient.astype(np.float32),
            capture_gradient.astype(np.float32),
            warp_matrix,
            cv2.MOTION_TRANSLATION,
            criteria,
            inputMask=content_mask,
            gaussFiltSize=5,
        )
    except cv2.error as error:
        return {
            "success": False,
            "error_code": "ECC_FAILED",
            "message": str(error),
        }

    return {
        "success": True,
        "dx": float(refined[0, 2]),
        "dy": float(refined[1, 2]),
        "score": float(score),
    }


def _method_confidence(method: Dict[str, Any], kind: str) -> float:
    if not method.get("success"):
        return 0.0

    if kind == "feature":
        inlier_ratio = float(method.get("inlier_ratio", 0.0))
        coverage = float(method.get("spatial_coverage", 0.0))
        residual = float(method.get("residual_median_px", 99.0))
        residual_score = max(0.0, 1.0 - residual / 4.0)
        count_score = min(1.0, float(method.get("inlier_count", 0)) / 60.0)
        return float(
            0.38 * min(1.0, inlier_ratio / 0.75)
            + 0.24 * min(1.0, coverage / 0.35)
            + 0.23 * residual_score
            + 0.15 * count_score
        )

    if kind == "phase":
        response = float(method.get("response", 0.0))
        return float(np.clip(response / 0.45, 0.0, 1.0))

    if kind == "ecc":
        score = float(method.get("score", 0.0))
        return float(np.clip((score - 0.1) / 0.8, 0.0, 1.0))

    return 0.0


def fuse_translation_methods(
    feature_result: Dict[str, Any],
    phase_result: Dict[str, Any],
    ecc_result: Dict[str, Any],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    methods: List[Dict[str, Any]] = []

    for kind, result in (
        ("feature", feature_result),
        ("phase", phase_result),
        ("ecc", ecc_result),
    ):
        if not result.get("success"):
            continue

        dx = result.get("dx")
        dy = result.get("dy")
        if dx is None or dy is None:
            continue
        if not np.isfinite(float(dx)) or not np.isfinite(float(dy)):
            continue

        confidence = _method_confidence(result, kind)
        if kind == "phase" and float(result.get("response", 0.0)) < float(
            options["phase_min_response"]
        ):
            continue
        if kind == "ecc" and float(result.get("score", 0.0)) < float(
            options["ecc_min_score"]
        ):
            continue

        methods.append(
            {
                "name": kind,
                "dx": float(dx),
                "dy": float(dy),
                "confidence": float(confidence),
            }
        )

    if not methods:
        return {
            "success": False,
            "error_code": "REGISTRATION_METHODS_FAILED",
            "used_methods": [],
        }

    coords = np.array(
        [[item["dx"], item["dy"]] for item in methods],
        dtype=np.float32,
    )
    robust_center = np.median(coords, axis=0)
    distances = np.linalg.norm(coords - robust_center, axis=1)
    agreement_px = float(options["method_agreement_px"])
    keep = distances <= agreement_px

    kept_methods = [
        item
        for item, keep_value in zip(methods, keep)
        if bool(keep_value)
    ]

    feature_available = any(
        item["name"] == "feature"
        for item in kept_methods
    )

    if not feature_available and len(kept_methods) < 2:
        return {
            "success": False,
            "error_code": "REGISTRATION_INCONSISTENT",
            "used_methods": kept_methods,
            "all_methods": methods,
        }

    weights = np.array(
        [max(0.05, item["confidence"]) for item in kept_methods],
        dtype=np.float64,
    )
    kept_coords = np.array(
        [[item["dx"], item["dy"]] for item in kept_methods],
        dtype=np.float64,
    )
    fused = np.sum(kept_coords * weights[:, None], axis=0) / np.sum(weights)

    if len(kept_methods) >= 2:
        pairwise = []
        for i in range(len(kept_methods)):
            for j in range(i + 1, len(kept_methods)):
                pairwise.append(
                    float(
                        np.linalg.norm(
                            kept_coords[i] - kept_coords[j]
                        )
                    )
                )
        mean_disagreement = float(np.mean(pairwise)) if pairwise else 0.0
    else:
        mean_disagreement = 0.0

    agreement_score = max(
        0.0,
        1.0 - mean_disagreement / max(agreement_px, 1e-6),
    )
    method_confidence = float(
        np.average(
            [item["confidence"] for item in kept_methods],
            weights=weights,
        )
    )
    final_confidence = float(
        np.clip(
            0.72 * method_confidence
            + 0.28 * agreement_score,
            0.0,
            1.0,
        )
    )

    return {
        "success": True,
        "dx": float(fused[0]),
        "dy": float(fused[1]),
        "confidence": final_confidence,
        "agreement_score": float(agreement_score),
        "mean_method_disagreement_px": float(mean_disagreement),
        "used_methods": kept_methods,
        "all_methods": methods,
    }


def classify_bias(value: float, tolerance: float, negative: str, positive: str) -> str:
    if value < -float(tolerance):
        return negative
    if value > float(tolerance):
        return positive
    return "center"


def detect_automatic_reference_registration(
    reference_image: np.ndarray,
    capture_image: np.ndarray,
    *,
    reference_id: str,
    calibration_offset_px: Optional[Dict[str, float]] = None,
    tolerance_px: Optional[Dict[str, float]] = None,
    option_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Estimate printed-content displacement without manual logo annotation."""
    options = merge_options(option_overrides)

    if reference_image is None or reference_image.size == 0:
        return (
            {
                "success": False,
                "error_code": "REFERENCE_IMAGE_EMPTY",
                "message": "The selected reference image is empty.",
                "confidence": 0.0,
            },
            {},
        )

    if capture_image is None or capture_image.size == 0:
        return (
            {
                "success": False,
                "error_code": "CAPTURE_IMAGE_EMPTY",
                "message": "The perspective-corrected capture image is empty.",
                "confidence": 0.0,
            },
            {},
        )

    normalized_reference, normalization = normalize_reference_to_capture(
        reference_image,
        capture_image,
    )

    h, w = capture_image.shape[:2]
    reference_gray = preprocess_gray(normalized_reference)
    capture_gray = preprocess_gray(capture_image)
    reference_gradient = gradient_image(reference_gray)
    capture_gradient = gradient_image(capture_gray)

    reference_mask = make_content_mask(
        normalized_reference,
        float(options["border_ignore_ratio"]),
    )
    capture_mask = make_content_mask(
        capture_image,
        float(options["border_ignore_ratio"]),
    )
    common_mask = cv2.bitwise_and(reference_mask, capture_mask)

    backend = create_feature_backend(int(options["max_features"]))
    keypoints_reference, descriptors_reference = backend.detector.detectAndCompute(
        reference_gray,
        common_mask,
    )
    keypoints_capture, descriptors_capture = backend.detector.detectAndCompute(
        capture_gray,
        common_mask,
    )

    feature_result: Dict[str, Any] = {
        "success": False,
        "backend": backend.name,
        "reference_feature_count": len(keypoints_reference or []),
        "capture_feature_count": len(keypoints_capture or []),
    }
    reference_points = np.zeros((0, 2), dtype=np.float32)
    capture_points = np.zeros((0, 2), dtype=np.float32)
    inlier_mask = np.zeros((0,), dtype=bool)
    mutual_matches: List[cv2.DMatch] = []

    if descriptors_reference is not None and descriptors_capture is not None:
        mutual_matches = mutual_ratio_matches(
            descriptors_reference,
            descriptors_capture,
            backend.norm_type,
            float(options["ratio_test"]),
        )
        mutual_matches = mutual_matches[
            : int(options["max_matches_for_ransac"])
        ]

        if mutual_matches:
            reference_points = np.array(
                [
                    keypoints_reference[match.queryIdx].pt
                    for match in mutual_matches
                ],
                dtype=np.float32,
            )
            capture_points = np.array(
                [
                    keypoints_capture[match.trainIdx].pt
                    for match in mutual_matches
                ],
                dtype=np.float32,
            )

            translation = estimate_translation_ransac(
                reference_points,
                capture_points,
                float(options["ransac_threshold_px"]),
            )

            if translation.get("success"):
                inlier_mask = np.asarray(
                    translation["inlier_mask"],
                    dtype=bool,
                )
                inlier_reference_points = reference_points[inlier_mask]
                inlier_capture_points = capture_points[inlier_mask]
                coverage = spatial_coverage_score(
                    inlier_reference_points,
                    w,
                    h,
                )
                affine = estimate_affine_diagnostic(
                    inlier_reference_points,
                    inlier_capture_points,
                )

                feature_result.update(translation)
                feature_result.update(
                    {
                        "good_match_count": int(len(mutual_matches)),
                        "spatial_coverage": float(coverage),
                        "affine_diagnostic": affine,
                    }
                )

                feature_result["success"] = bool(
                    len(mutual_matches) >= int(options["min_good_matches"])
                    and int(translation["inlier_count"]) >= int(options["min_inliers"])
                    and float(translation["inlier_ratio"]) >= float(options["min_inlier_ratio"])
                    and float(coverage) >= float(options["min_spatial_coverage"])
                    and float(translation["residual_median_px"]) <= float(options["max_residual_median_px"])
                )
            else:
                feature_result.update(translation)
                feature_result["good_match_count"] = int(len(mutual_matches))
        else:
            feature_result.update(
                {
                    "error_code": "NOT_ENOUGH_FEATURE_MATCHES",
                    "good_match_count": 0,
                }
            )
    else:
        feature_result["error_code"] = "FEATURE_DESCRIPTORS_NOT_FOUND"

    phase_result = phase_translation(
        reference_gradient,
        capture_gradient,
        float(options["border_ignore_ratio"]),
    )

    initial_dx = 0.0
    initial_dy = 0.0
    if feature_result.get("success"):
        initial_dx = float(feature_result["dx"])
        initial_dy = float(feature_result["dy"])
    elif phase_result.get("success"):
        initial_dx = float(phase_result["dx"])
        initial_dy = float(phase_result["dy"])

    ecc_result = ecc_translation(
        reference_gradient,
        capture_gradient,
        initial_dx,
        initial_dy,
        common_mask,
    )

    fused = fuse_translation_methods(
        feature_result,
        phase_result,
        ecc_result,
        options,
    )

    if not fused.get("success"):
        result = {
            "success": False,
            "error_code": fused.get(
                "error_code",
                "REGISTRATION_FAILED",
            ),
            "message": (
                "The official reference image and the perspective-corrected "
                "photo could not be registered reliably."
            ),
            "reference_id": reference_id,
            "measurement_mode": "reference_registration",
            "confidence": 0.0,
            "image_size": {
                "width": int(w),
                "height": int(h),
            },
            "reference_normalization": normalization,
            "methods": {
                "feature": _serialize_method(feature_result),
                "phase": _serialize_method(phase_result),
                "ecc": _serialize_method(ecc_result),
                "fusion": _serialize_method(fused),
            },
        }
        debug = {
            "normalized_reference": normalized_reference,
            "reference_points": reference_points,
            "capture_points": capture_points,
            "inlier_mask": inlier_mask,
            "content_mask": common_mask,
        }
        return result, debug

    raw_dx = float(fused["dx"])
    raw_dy = float(fused["dy"])

    calibration = calibration_offset_px or {}
    calibration_x = float(calibration.get("x", 0.0))
    calibration_y = float(calibration.get("y", 0.0))

    corrected_dx = raw_dx - calibration_x
    corrected_dy = raw_dy - calibration_y

    tolerance = tolerance_px or {}
    tolerance_x = float(
        tolerance.get(
            "x",
            options["default_tolerance_px_x"],
        )
    )
    tolerance_y = float(
        tolerance.get(
            "y",
            options["default_tolerance_px_y"],
        )
    )

    affine = feature_result.get("affine_diagnostic") or {}
    rotation_deg = affine.get("rotation_deg")
    scale = affine.get("scale")

    geometry_warning = False
    warnings: List[str] = []

    if rotation_deg is not None and abs(float(rotation_deg)) > float(
        options["max_rotation_deg"]
    ):
        geometry_warning = True
        warnings.append("RESIDUAL_ROTATION_HIGH")

    if scale is not None and abs(float(scale) - 1.0) > float(
        options["max_scale_error"]
    ):
        geometry_warning = True
        warnings.append("RESIDUAL_SCALE_ERROR_HIGH")

    confidence = float(fused.get("confidence", 0.0))
    if geometry_warning:
        confidence *= 0.78

    result = {
        "success": True,
        "error_code": None,
        "message": (
            "Automatic official-reference registration completed successfully."
        ),
        "measurement_mode": "reference_registration",
        "reference_id": reference_id,
        "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 4),
        "image_size": {
            "width": int(w),
            "height": int(h),
        },
        "offset": {
            "raw_dx_px": round(raw_dx, 4),
            "raw_dy_px": round(raw_dy, 4),
            "calibration_dx_px": round(calibration_x, 4),
            "calibration_dy_px": round(calibration_y, 4),
            "dx_px": round(corrected_dx, 4),
            "dy_px": round(corrected_dy, 4),
            "dx_percent": round(corrected_dx / max(float(w), 1.0) * 100.0, 5),
            "dy_percent": round(corrected_dy / max(float(h), 1.0) * 100.0, 5),
            "horizontal_bias": classify_bias(
                corrected_dx,
                tolerance_x,
                "left",
                "right",
            ),
            "vertical_bias": classify_bias(
                corrected_dy,
                tolerance_y,
                "up",
                "down",
            ),
            "within_tolerance": bool(
                abs(corrected_dx) <= tolerance_x
                and abs(corrected_dy) <= tolerance_y
            ),
            "tolerance_px": {
                "x": round(tolerance_x, 4),
                "y": round(tolerance_y, 4),
            },
        },
        "reference_normalization": normalization,
        "registration": {
            "primary_method": (
                "feature_translation_ransac"
                if feature_result.get("success")
                else "phase_ecc_fusion"
            ),
            "used_methods": fused.get("used_methods", []),
            "method_agreement_score": round(
                float(fused.get("agreement_score", 0.0)),
                4,
            ),
            "mean_method_disagreement_px": round(
                float(fused.get("mean_method_disagreement_px", 0.0)),
                4,
            ),
            "feature_backend": backend.name,
            "reference_feature_count": int(
                feature_result.get("reference_feature_count", 0)
            ),
            "capture_feature_count": int(
                feature_result.get("capture_feature_count", 0)
            ),
            "good_matches": int(
                feature_result.get("good_match_count", 0)
            ),
            "inlier_matches": int(
                feature_result.get("inlier_count", 0)
            ),
            "inlier_ratio": round(
                float(feature_result.get("inlier_ratio", 0.0)),
                4,
            ),
            "spatial_coverage": round(
                float(feature_result.get("spatial_coverage", 0.0)),
                4,
            ),
            "residual_median_px": round(
                float(feature_result.get("residual_median_px", 0.0)),
                4,
            ),
            "rotation_deg": (
                round(float(rotation_deg), 5)
                if rotation_deg is not None
                else None
            ),
            "scale": (
                round(float(scale), 6)
                if scale is not None
                else None
            ),
        },
        "methods": {
            "feature": _serialize_method(feature_result),
            "phase": _serialize_method(phase_result),
            "ecc": _serialize_method(ecc_result),
        },
        "warnings": warnings,
    }

    debug = {
        "normalized_reference": normalized_reference,
        "reference_points": reference_points,
        "capture_points": capture_points,
        "inlier_mask": inlier_mask,
        "content_mask": common_mask,
    }
    return result, debug


def _serialize_method(method: Dict[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in method.items():
        if key == "inlier_mask":
            continue
        if isinstance(value, np.ndarray):
            output[key] = value.tolist()
        elif isinstance(value, np.integer):
            output[key] = int(value)
        elif isinstance(value, np.floating):
            output[key] = float(value)
        else:
            output[key] = value
    return output


def make_reference_registration_overlay(
    capture_image: np.ndarray,
    result: Dict[str, Any],
    debug: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Visualize automatically matched content points and the final offset."""
    overlay = capture_image.copy()
    h, w = overlay.shape[:2]

    if not result.get("success"):
        cv2.rectangle(overlay, (0, 0), (w - 1, 58), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            "REFERENCE REGISTRATION FAILED",
            (14, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return overlay

    debug = debug or {}
    reference_points = np.asarray(
        debug.get("reference_points", np.zeros((0, 2), dtype=np.float32)),
        dtype=np.float32,
    )
    capture_points = np.asarray(
        debug.get("capture_points", np.zeros((0, 2), dtype=np.float32)),
        dtype=np.float32,
    )
    inlier_mask = np.asarray(
        debug.get("inlier_mask", np.zeros((0,), dtype=bool)),
        dtype=bool,
    )

    if (
        len(reference_points) == len(capture_points)
        and len(inlier_mask) == len(reference_points)
        and len(reference_points) > 0
    ):
        indexes = np.flatnonzero(inlier_mask)
        max_matches = int(DEFAULT_OPTIONS["max_visualized_matches"])

        if len(indexes) > max_matches:
            selection = np.linspace(
                0,
                len(indexes) - 1,
                max_matches,
            ).astype(int)
            indexes = indexes[selection]

        for index in indexes:
            ref_point = tuple(
                int(round(value))
                for value in reference_points[index]
            )
            capture_point = tuple(
                int(round(value))
                for value in capture_points[index]
            )

            cv2.circle(
                overlay,
                ref_point,
                3,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.circle(
                overlay,
                capture_point,
                3,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.arrowedLine(
                overlay,
                ref_point,
                capture_point,
                (0, 215, 255),
                1,
                cv2.LINE_AA,
                tipLength=0.25,
            )

    offset = result.get("offset") or {}
    dx = float(offset.get("dx_px", 0.0))
    dy = float(offset.get("dy_px", 0.0))
    confidence = float(result.get("confidence", 0.0))
    reference_id = str(result.get("reference_id", "unknown"))

    cv2.rectangle(
        overlay,
        (0, 0),
        (w - 1, 82),
        (12, 12, 12),
        -1,
    )
    cv2.putText(
        overlay,
        f"REFERENCE: {reference_id}",
        (14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.63,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"dx={dx:+.2f}px  dy={dy:+.2f}px  confidence={confidence:.3f}",
        (14, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )

    center = (w // 2, h // 2)
    shifted_center = (
        int(round(center[0] + dx)),
        int(round(center[1] + dy)),
    )
    cv2.drawMarker(
        overlay,
        center,
        (0, 255, 0),
        cv2.MARKER_CROSS,
        18,
        2,
        cv2.LINE_AA,
    )
    cv2.drawMarker(
        overlay,
        shifted_center,
        (0, 0, 255),
        cv2.MARKER_CROSS,
        18,
        2,
        cv2.LINE_AA,
    )
    cv2.arrowedLine(
        overlay,
        center,
        shifted_center,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
        tipLength=0.22,
    )

    return overlay
