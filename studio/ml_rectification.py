"""Rectification bridge that matches the supplied ML pipeline exactly."""

from __future__ import annotations

import math
from pathlib import Path
import sys

from .errors import StudioError
from .images import rectify_png as pillow_rectify_png


_ROOT = Path(__file__).resolve().parents[1]
_ML_ROOT = _ROOT / "ml_backend"


def _sideways_order(corners: list[list[float]]) -> list[list[float]]:
    points = [[float(point[0]), float(point[1])] for point in corners]
    top = math.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1])
    bottom = math.hypot(points[2][0] - points[3][0], points[2][1] - points[3][1])
    left = math.hypot(points[3][0] - points[0][0], points[3][1] - points[0][1])
    right = math.hypot(points[2][0] - points[1][0], points[2][1] - points[1][1])
    if top + bottom > left + right:
        return [points[3], points[0], points[1], points[2]]
    return points


def _diagnostic_fallback(
    normalized_png: bytes,
    corners: list[list[float]],
    width: int,
    height: int,
) -> bytes:
    # Used only by non-ML unit-test environments. It mirrors the ML pipeline's
    # sideways source ordering, but production CUDA validation must exercise
    # the OpenCV path below.
    return pillow_rectify_png(normalized_png, _sideways_order(corners), width, height)


def rectify_ml_png(
    normalized_png: bytes,
    corners: list[list[float]],
    width: int,
    height: int,
) -> bytes:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return _diagnostic_fallback(normalized_png, corners, width, height)

    if str(_ML_ROOT) not in sys.path:
        sys.path.insert(0, str(_ML_ROOT))
    try:
        from card_quality_processor.rectification import rectify_card_by_points

        image = cv2.imdecode(np.frombuffer(normalized_png, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise StudioError(422, "IMAGE_DECODE_FAILED", "Cannot decode normalized image for ML rectification.")
        result = rectify_card_by_points(image, corners, (width, height), None)
        rectified = result.get("rectified_image") if isinstance(result, dict) else None
        if not result.get("success") or rectified is None:
            raise StudioError(
                422,
                str(result.get("error_code") or "RECTIFICATION_FAILED"),
                str(result.get("message") or "ML rectification failed."),
            )
        success, encoded = cv2.imencode(".png", rectified)
        if not success:
            raise StudioError(500, "IMAGE_ENCODE_FAILED", "Cannot encode ML rectified PNG.")
        return bytes(encoded)
    except StudioError:
        raise
    except Exception as exc:
        raise StudioError(
            500,
            "ML_RECTIFICATION_ERROR",
            f"ML rectification failed: {type(exc).__name__}: {exc}",
        ) from exc


__all__ = ["rectify_ml_png"]
