from __future__ import annotations

import io

import numpy as np
from PIL import Image

from ml_backend.card_quality_processor.outer_pose_detection import OuterPoseDetector
from ml_backend.ptcg_inference import (
    _inverse_map_rotated_outer_points,
    _inner_orientation_score,
    _outer_quad_is_landscape,
    decode_input_image,
)


def test_landscape_quad_detection_is_rotation_invariant() -> None:
    landscape = [[10, 10], [310, 15], [305, 205], [12, 200]]
    portrait = [[10, 10], [200, 12], [205, 312], [8, 310]]
    assert _outer_quad_is_landscape(landscape)
    assert not _outer_quad_is_landscape(portrait)


def test_orientation_score_prefers_normal_supported_inner_result() -> None:
    normal = {
        "success": True,
        "yolo_confidence": 0.9,
        "quality_assessment": {"severity": "normal", "review_recommended": False},
        "edge_refinement": {
            edge: {"confidence": 0.8}
            for edge in ("left", "right", "top", "bottom")
        },
    }
    upside_down = {
        "success": True,
        "yolo_confidence": 0.8,
        "quality_assessment": {"severity": "high", "review_recommended": True},
        "edge_refinement": {
            edge: {"confidence": 0.35}
            for edge in ("left", "right", "top", "bottom")
        },
    }
    assert _inner_orientation_score(normal) > _inner_orientation_score(upside_down) + 5.0


def test_jpeg_exif_orientation_is_applied_during_decode() -> None:
    raw = np.zeros((40, 80, 3), dtype=np.uint8)
    raw[:, :20] = (255, 0, 0)
    image = Image.fromarray(raw, mode="RGB")
    exif = Image.Exif()
    exif[274] = 6  # display as 90 degrees clockwise
    stream = io.BytesIO()
    image.save(stream, format="JPEG", exif=exif)

    decoded = decode_input_image(stream.getvalue())

    assert decoded.shape[:2] == (80, 40)


def test_quarter_turn_outer_points_map_back_without_pixel_drift() -> None:
    original = np.asarray(
        [[10, 20], [100, 20], [100, 80], [10, 80]],
        dtype=np.float32,
    )
    clockwise = np.column_stack((99.0 - original[:, 1], original[:, 0]))
    counterclockwise = np.column_stack((original[:, 1], 199.0 - original[:, 0]))

    mapped_clockwise = _inverse_map_rotated_outer_points(
        clockwise,
        rotation="cw90",
        original_shape=(100, 200, 3),
    )
    mapped_counterclockwise = _inverse_map_rotated_outer_points(
        counterclockwise,
        rotation="ccw90",
        original_shape=(100, 200, 3),
    )

    assert np.allclose(mapped_clockwise, original)
    assert np.allclose(mapped_counterclockwise, original)


def test_invalid_primary_uses_geometry_valid_legacy_expert() -> None:
    detector = OuterPoseDetector.__new__(OuterPoseDetector)
    detector.config = {
        "silhouette_refinement": {
            "invalid_primary_recovery": {
                "enabled": True,
                "min_confidence": 0.68,
                "max_aspect_error": 0.14,
                "min_area_ratio": 0.04,
                "max_area_ratio": 0.92,
            }
        }
    }
    legacy = {
        "success": True,
        "confidence": 0.91,
        "points": [[1, 1], [10, 1], [10, 15], [1, 15]],
        "metrics": {"aspect_ratio_error": 0.02, "area_ratio": 0.35},
    }
    detector._predict_legacy_silhouette = lambda image: dict(legacy)
    detector._predict_pose_only = lambda image, conf: {"success": False}
    primary = {
        "success": False,
        "confidence": 0.97,
        "error_code": "INVALID_KEYPOINT_GEOMETRY",
        "metrics": {"aspect_ratio_error": 0.28, "area_ratio": 0.48},
    }

    recovered = detector._recover_invalid_silhouette(
        np.zeros((20, 20, 3), dtype=np.uint8),
        primary,
        0.25,
    )

    assert recovered["success"]
    audit = recovered["metrics"]["invalid_primary_recovery"]
    assert audit["selected_source"] == "legacy_silhouette"
    assert audit["primary_error_code"] == "INVALID_KEYPOINT_GEOMETRY"
