from __future__ import annotations

from typing import Any, Mapping


RECOVERY_VERSION = "pre_cropped_card_recovery_20260822_v1"
CARD_HEIGHT_OVER_WIDTH = 88.0 / 63.0
KEYPOINT_NAMES = ("tl", "tr", "br", "bl")


def propose_pre_cropped_outer(
    image: Any,
    failed_prediction: Mapping[str, Any],
    *,
    max_aspect_error: float = 0.04,
) -> dict[str, Any] | None:
    """Propose the image boundary as a *provisional* outer card frame.

    This route is intentionally not a successful detection by itself.  It is
    used only after the normal outer detector has failed, for a portrait image
    whose dimensions are already close to the physical 63:88 card ratio.  The
    caller must rectify the proposal, run the inner detector, and call
    :func:`confirm_pre_cropped_inner` before returning a complete result.
    """

    if (
        getattr(image, "size", 0) == 0
        or len(getattr(image, "shape", ())) != 3
        or image.shape[0] < 280
        or image.shape[1] < 190
        or bool(failed_prediction.get("success", False))
    ):
        return None
    height, width = image.shape[:2]
    aspect_ratio = float(height) / max(float(width), 1.0)
    aspect_error = abs(aspect_ratio - CARD_HEIGHT_OVER_WIDTH) / CARD_HEIGHT_OVER_WIDTH
    if aspect_error > float(max_aspect_error):
        return None

    # Four exact image corners are only a hypothesis at this point.  Inner
    # geometry supplies the independent semantic confirmation later.
    points = [
        [0.0, 0.0],
        [float(width - 1), 0.0],
        [float(width - 1), float(height - 1)],
        [0.0, float(height - 1)],
    ]
    source_confidence = float(failed_prediction.get("confidence", 0.0) or 0.0)
    provisional_confidence = float(min(max(max(0.72, source_confidence), 0.0), 0.92))
    source_metrics = failed_prediction.get("metrics", {})
    source_metrics = source_metrics if isinstance(source_metrics, Mapping) else {}
    return {
        "success": True,
        "points": points,
        "bbox": [0.0, 0.0, float(width - 1), float(height - 1)],
        "confidence": provisional_confidence,
        "keypoint_confidence": {
            name: provisional_confidence for name in KEYPOINT_NAMES
        },
        "method": "provisional_pre_cropped_full_frame",
        "error_code": None,
        "message": (
            "Card-like full-frame input accepted provisionally; final acceptance "
            "requires an independently detected 58x83 mm inner frame."
        ),
        "metrics": {
            "bbox_confidence": provisional_confidence,
            "mean_keypoint_confidence": provisional_confidence,
            "min_keypoint_confidence": provisional_confidence,
            "aspect_ratio": aspect_ratio,
            "aspect_ratio_error": aspect_error,
            "area_ratio": float((width - 1) * (height - 1)) / float(width * height),
            "min_border_margin_ratio": 0.0,
            "in_bounds": True,
            "convex": True,
            "spatial_order": True,
            "geometry_valid": True,
            "raw_points": points,
            "pre_cropped_card_recovery": {
                "version": RECOVERY_VERSION,
                "provisional": True,
                "confirmed": False,
                "reason": "outer_failed_but_full_frame_aspect_matches_63x88",
                "source_error_code": failed_prediction.get("error_code"),
                "source_message": failed_prediction.get("message"),
                "source_confidence": source_confidence,
                "source_area_ratio": source_metrics.get("area_ratio"),
                "input_size": [int(width), int(height)],
                "input_aspect_ratio": aspect_ratio,
                "relative_aspect_error": aspect_error,
            },
        },
    }


def confirm_pre_cropped_inner(
    inner: Mapping[str, Any],
    *,
    rectified_width: int = 630,
    rectified_height: int = 880,
    inner_width_mm: float = 58.0,
    inner_height_mm: float = 83.0,
    outer_width_mm: float = 63.0,
    outer_height_mm: float = 88.0,
    max_width_residual_px: float = 32.0,
    max_height_residual_px: float = 32.0,
    min_inner_confidence: float = 0.20,
) -> dict[str, Any]:
    """Confirm a provisional full-frame outer candidate from inner geometry."""

    result: dict[str, Any] = {
        "version": RECOVERY_VERSION,
        "confirmed": False,
        "reason": "inner_detection_failed",
    }
    if not isinstance(inner, Mapping) or not bool(inner.get("success", False)):
        return result
    box = inner.get("final_box")
    if not isinstance(box, Mapping):
        result["reason"] = "inner_box_missing"
        return result
    try:
        observed_width = float(box["right"]) - float(box["left"])
        observed_height = float(box["bottom"]) - float(box["top"])
        confidence = float(inner.get("yolo_confidence", 0.0) or 0.0)
    except (KeyError, TypeError, ValueError):
        result["reason"] = "inner_box_invalid"
        return result

    expected_width = float(rectified_width) * float(inner_width_mm) / float(outer_width_mm)
    expected_height = float(rectified_height) * float(inner_height_mm) / float(outer_height_mm)
    width_residual = observed_width - expected_width
    height_residual = observed_height - expected_height
    result.update(
        {
            "inner_confidence": confidence,
            "measurement_semantics": "printed_inner_line_inner_edge",
            "expected_inner_size_px": {
                "width": round(expected_width, 4),
                "height": round(expected_height, 4),
            },
            "observed_inner_size_px": {
                "width": round(observed_width, 4),
                "height": round(observed_height, 4),
            },
            "residual_px": {
                "width": round(width_residual, 4),
                "height": round(height_residual, 4),
            },
            "limits_px": {
                "width": float(max_width_residual_px),
                "height": float(max_height_residual_px),
            },
        }
    )
    if confidence < float(min_inner_confidence):
        result["reason"] = "inner_confidence_too_low"
        return result
    if abs(width_residual) > float(max_width_residual_px):
        result["reason"] = "inner_width_inconsistent_with_58mm"
        return result
    if abs(height_residual) > float(max_height_residual_px):
        result["reason"] = "inner_height_inconsistent_with_83mm"
        return result
    result["confirmed"] = True
    result["reason"] = "inner_detection_confirms_pre_cropped_card"
    return result
