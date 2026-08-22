from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np

from .outer_detection import order_points


RISK_VERSION = "outer_gray_parallel_edge_risk_20260812_v1"
SIDE_NAMES = ("top", "right", "bottom", "left")


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        raise ValueError("Cannot normalize a zero-length vector")
    return np.asarray(vector, dtype=np.float32) / norm


def _sample(image: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    points = np.asarray(coordinates, dtype=np.float32)
    leading = points.shape[:-1]
    sampled = cv2.remap(
        image,
        points[..., 0].reshape(1, -1),
        points[..., 1].reshape(1, -1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    if image.ndim == 2:
        return sampled.reshape(leading)
    return sampled.reshape((*leading, image.shape[2]))


def _peak_pair(profile: np.ndarray, step: float) -> dict[str, float | bool]:
    values = np.asarray(profile, dtype=np.float32)
    if values.size < 7:
        return {
            "ambiguous": False,
            "primary_strength": 0.0,
            "secondary_strength": 0.0,
            "secondary_ratio": 0.0,
            "separation_px": 0.0,
        }
    smoothed = cv2.GaussianBlur(values.reshape(1, -1), (0, 0), 1.0).reshape(-1)
    local = [
        index
        for index in range(1, len(smoothed) - 1)
        if smoothed[index] >= smoothed[index - 1]
        and smoothed[index] >= smoothed[index + 1]
    ]
    if not local:
        local = [int(np.argmax(smoothed))]
    local.sort(key=lambda index: float(smoothed[index]), reverse=True)
    primary = local[0]
    secondary = None
    for index in local[1:]:
        separation = abs(index - primary) * step
        if 3.0 <= separation <= 24.0:
            secondary = index
            break
    primary_strength = float(smoothed[primary])
    secondary_strength = float(smoothed[secondary]) if secondary is not None else 0.0
    ratio = secondary_strength / max(primary_strength, 1e-6)
    separation = abs(secondary - primary) * step if secondary is not None else 0.0
    return {
        "ambiguous": bool(secondary is not None and ratio >= 0.74),
        "primary_strength": primary_strength,
        "secondary_strength": secondary_strength,
        "secondary_ratio": float(ratio),
        "separation_px": float(separation),
    }


def assess_outer_shadow_risk(
    image: np.ndarray,
    points: Iterable[Iterable[float]] | None,
) -> dict[str, Any]:
    """Flag neutral borders with multiple strong parallel long-edge candidates.

    This is deliberately an uncertainty detector, not an edge snapper.  Gray
    card material and cast shadows can produce similarly strong local gradients;
    changing coordinates from one view would be unsafe.  The caller can use
    this signal to request independent photometric views and only accept a
    geometry change when those views agree.
    """

    if points is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            "version": RISK_VERSION,
            "available": False,
            "high_risk": False,
            "reason": "missing_image_or_points",
        }
    try:
        quad = order_points(np.asarray(points, dtype=np.float32))
    except (TypeError, ValueError):
        return {
            "version": RISK_VERSION,
            "available": False,
            "high_risk": False,
            "reason": "invalid_points",
        }

    original_height, original_width = image.shape[:2]
    scale = min(1.0, 1400.0 / max(original_height, original_width))
    work = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image.copy()
    )
    quad = quad * scale
    if work.ndim == 2:
        work = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    fine = cv2.GaussianBlur(gray, (0, 0), 0.8)
    grad_x = cv2.Scharr(fine, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(fine, cv2.CV_32F, 0, 1)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)

    lengths = [float(np.linalg.norm(np.roll(quad, -1, axis=0)[i] - quad[i])) for i in range(4)]
    short_dimension = max(min(lengths), 20.0)
    search_radius = float(np.clip(short_dimension * 0.045, 9.0, 32.0))
    step = max(0.75, short_dimension / 700.0)
    offsets = np.arange(-search_radius, search_radius + 0.5 * step, step, dtype=np.float32)
    center = quad.mean(axis=0)
    side_metrics: list[dict[str, Any]] = []
    neutral_sides = 0
    ambiguous_sides = 0

    for index, (start, end) in enumerate(zip(quad, np.roll(quad, -1, axis=0))):
        tangent = _normalize(end - start)
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
        midpoint = 0.5 * (start + end)
        if float(np.dot(normal, midpoint - center)) < 0.0:
            normal = -normal
        fractions = np.linspace(0.14, 0.86, 160, dtype=np.float32)
        line = start[None, :] + fractions[:, None] * (end - start)[None, :]
        band = line[None, :, :] + offsets[:, None, None] * normal[None, None, :]
        gx = _sample(grad_x, band)
        gy = _sample(grad_y, band)
        profile = np.quantile(np.abs(gx * normal[0] + gy * normal[1]), 0.45, axis=1)
        peaks = _peak_pair(profile, step)

        inside_gap = max(2.0, short_dimension * 0.012)
        inside_points = line - inside_gap * normal[None, :]
        inside_hsv = _sample(hsv, inside_points)
        inside_lab = _sample(lab, inside_points)
        saturation = float(np.median(inside_hsv[:, 1]))
        chroma = np.sqrt(
            (inside_lab[:, 1] - 128.0) ** 2 + (inside_lab[:, 2] - 128.0) ** 2
        )
        chroma_median = float(np.median(chroma))
        neutral = bool(saturation <= 45.0 and chroma_median <= 20.0)
        ambiguous = bool(peaks["ambiguous"])
        neutral_sides += neutral
        ambiguous_sides += neutral and ambiguous
        side_metrics.append(
            {
                "name": SIDE_NAMES[index],
                "neutral": neutral,
                "inside_saturation_median": saturation,
                "inside_chroma_median": chroma_median,
                **peaks,
            }
        )

    high_risk = bool(neutral_sides >= 2 and ambiguous_sides >= 2)
    risk_score = float(
        np.clip(
            0.25 * neutral_sides
            + 0.35 * ambiguous_sides
            + 0.25
            * max((float(side["secondary_ratio"]) for side in side_metrics), default=0.0),
            0.0,
            2.0,
        )
        / 2.0
    )
    return {
        "version": RISK_VERSION,
        "available": True,
        "high_risk": high_risk,
        "reason": "neutral_parallel_edge_ambiguity" if high_risk else "risk_not_met",
        "risk_score": risk_score,
        "neutral_side_count": int(neutral_sides),
        "ambiguous_neutral_side_count": int(ambiguous_sides),
        "sides": side_metrics,
    }
