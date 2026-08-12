from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from .config import normalize_config
from .io_utils import write_image
from .outer_detection import order_points


def _failure(output_size: tuple[int, int], code: str, message: str) -> dict[str, Any]:
    return {
        "success": False,
        "rectified_image": None,
        "homography": None,
        "output_size": [int(output_size[0]), int(output_size[1])],
        "confidence": 0.0,
        "error_code": code,
        "message": message,
    }


def rectify_card(
    image: np.ndarray,
    outer_points: Iterable[Iterable[float]],
    output_size: tuple[int, int] = (630, 880),
    config: Mapping[str, Any] | None = None,
    save_debug: bool = False,
    debug_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = normalize_config(config)["rectification"]
    if output_size == (630, 880) and config is not None:
        output_size = (int(cfg["output_width"]), int(cfg["output_height"]))
    width, height = int(output_size[0]), int(output_size[1])
    if not isinstance(image, np.ndarray) or image.size == 0 or width < 2 or height < 2:
        return _failure((width, height), "INVALID_OUTER_POINTS", "Image or output size is invalid.")
    try:
        source = order_points(outer_points)
    except (TypeError, ValueError):
        return _failure((width, height), "INVALID_OUTER_POINTS", "Four valid outer points are required.")
    image_height, image_width = image.shape[:2]
    if (
        source[:, 0].min() < -1
        or source[:, 1].min() < -1
        or source[:, 0].max() > image_width
        or source[:, 1].max() > image_height
        or abs(cv2.contourArea(source.reshape(-1, 1, 2))) < 25
    ):
        return _failure((width, height), "INVALID_OUTER_POINTS", "Outer points are outside the image or degenerate.")

    top = np.linalg.norm(source[1] - source[0])
    bottom = np.linalg.norm(source[2] - source[3])
    left = np.linalg.norm(source[3] - source[0])
    right = np.linalg.norm(source[2] - source[1])
    # If the photographed card lies sideways, rotate the source mapping so the
    # physical long side still becomes the vertical side of the standard card.
    if (top + bottom) > (left + right):
        source = np.array([source[3], source[0], source[1], source[2]], dtype=np.float32)

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    interpolation = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }.get(str(cfg.get("interpolation", "linear")).lower(), cv2.INTER_LINEAR)
    border_mode = {
        "constant": cv2.BORDER_CONSTANT,
        "replicate": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT,
    }.get(str(cfg.get("border_mode", "constant")).lower(), cv2.BORDER_CONSTANT)
    try:
        homography = cv2.getPerspectiveTransform(source.astype(np.float32), destination)
        if not np.isfinite(homography).all() or abs(float(np.linalg.det(homography))) < 1e-10:
            return _failure((width, height), "RECTIFICATION_FAILED", "Perspective transform is singular.")
        rectified = cv2.warpPerspective(
            image,
            homography,
            (width, height),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=(0, 0, 0),
        )
    except cv2.error as exc:
        return _failure((width, height), "RECTIFICATION_FAILED", f"OpenCV rectification failed: {exc}")
    if rectified.size == 0:
        return _failure((width, height), "RECTIFICATION_FAILED", "Rectified image is empty.")

    if save_debug:
        target = Path(debug_dir) if debug_dir is not None else Path("debug")
        target.mkdir(parents=True, exist_ok=True)
        write_image(target / "rectified_card.jpg", rectified)
        mapping = image.copy()
        cv2.polylines(mapping, [np.rint(source).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 4, cv2.LINE_AA)
        write_image(target / "rectification_mapping.jpg", mapping)

    return {
        "success": True,
        "rectified_image": rectified,
        "homography": homography.tolist(),
        "output_size": [width, height],
        "confidence": 1.0,
        "error_code": None,
        "message": "Perspective rectification completed.",
    }


def rectify_card_by_points(
    image: np.ndarray,
    outer_points: Iterable[Iterable[float]],
    output_size: tuple[int, int] = (630, 880),
    config: Mapping[str, Any] | None = None,
    save_debug: bool = False,
    debug_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Rectify a card from TL, TR, BR, BL outer keypoints.

    This public name is used by the deep-pose branch while ``rectify_card``
    remains available for the existing OpenCV workflow.
    """
    return rectify_card(image, outer_points, output_size, config, save_debug, debug_dir)
