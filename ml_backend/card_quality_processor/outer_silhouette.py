from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    total = points.sum(axis=1)
    delta = np.diff(points, axis=1).reshape(-1)
    return np.asarray(
        [points[np.argmin(total)], points[np.argmin(delta)], points[np.argmax(total)], points[np.argmax(delta)]],
        dtype=np.float32,
    )


def polygon_to_quad(polygon: np.ndarray) -> np.ndarray | None:
    """Convert a card silhouette polygon to TL/TR/BR/BL physical corners."""
    polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(polygon) < 4:
        return None
    hull = cv2.convexHull(polygon.reshape(-1, 1, 2)).reshape(-1, 2)
    perimeter = float(cv2.arcLength(hull.reshape(-1, 1, 2), True))
    if perimeter <= 0:
        return None
    hull_area = abs(float(cv2.contourArea(hull.reshape(-1, 1, 2))))
    candidates: list[tuple[float, np.ndarray]] = []
    for ratio in np.linspace(0.002, 0.08, 80):
        approx = cv2.approxPolyDP(hull.reshape(-1, 1, 2), float(ratio) * perimeter, True).reshape(-1, 2)
        if len(approx) != 4 or not cv2.isContourConvex(approx.reshape(-1, 1, 2)):
            continue
        area = abs(float(cv2.contourArea(approx.reshape(-1, 1, 2))))
        candidates.append((abs(area - hull_area), approx.astype(np.float32)))
    if candidates:
        return order_quad(min(candidates, key=lambda item: item[0])[1])
    rectangle = cv2.boxPoints(cv2.minAreaRect(hull.reshape(-1, 1, 2))).astype(np.float32)
    return order_quad(rectangle)


def extract_silhouette_prediction(
    prediction: Any,
    *,
    image_shape: Any | None = None,
    target_aspect_ratio: float = 1.397,
    aspect_ratio_tolerance: float = 0.25,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.95,
    area_weight: float = 0.40,
    aspect_weight: float = 0.25,
) -> dict[str, Any] | None:
    """Extract the most plausible physical-card mask.

    A segmentation model can return both the card and a high-confidence inner
    artwork rectangle.  Confidence-only selection is therefore unsafe.  When
    the original image shape is available, candidates are also ranked by
    physical-card geometry and area; callers without shape information retain
    the legacy confidence-only behavior.
    """
    boxes = getattr(prediction, "boxes", None)
    masks = getattr(prediction, "masks", None)
    if boxes is None or masks is None or len(boxes) == 0:
        return None
    try:
        confidences = _to_numpy(boxes.conf).astype(np.float32).reshape(-1)
        bounding_boxes = _to_numpy(boxes.xyxy).astype(np.float32).reshape(-1, 4)
        polygons = masks.xy
        count = min(len(confidences), len(bounding_boxes), len(polygons))
        if count == 0:
            return None

        resolved_shape = image_shape if image_shape is not None else getattr(prediction, "orig_shape", None)
        height = width = 0
        if resolved_shape is not None and len(resolved_shape) >= 2:
            height, width = int(resolved_shape[0]), int(resolved_shape[1])
        has_shape = height > 1 and width > 1
        target_aspect = max(float(target_aspect_ratio), 1e-6)

        candidates: list[dict[str, Any]] = []
        for index in range(count):
            points = polygon_to_quad(np.asarray(polygons[index], dtype=np.float32))
            if points is None:
                continue
            confidence = float(np.clip(confidences[index], 0.0, 1.0))
            candidate: dict[str, Any] = {
                "index": index,
                "points": points,
                "confidence": confidence,
                "bbox": bounding_boxes[index],
                "geometry_valid": True,
                "area_ratio": None,
                "aspect_ratio": None,
                "aspect_ratio_error": None,
                "selection_score": confidence,
            }
            if has_shape:
                area = abs(float(cv2.contourArea(points.reshape(-1, 1, 2))))
                area_ratio = area / max(1.0, float(width * height))
                top = float(np.linalg.norm(points[1] - points[0]))
                bottom = float(np.linalg.norm(points[2] - points[3]))
                left = float(np.linalg.norm(points[3] - points[0]))
                right = float(np.linalg.norm(points[2] - points[1]))
                average_width = (top + bottom) * 0.5
                average_height = (left + right) * 0.5
                aspect_ratio = max(average_width, average_height) / max(
                    min(average_width, average_height), 1e-6
                )
                aspect_error = abs(aspect_ratio - target_aspect) / target_aspect
                geometry_valid = bool(
                    float(min_area_ratio) <= area_ratio <= float(max_area_ratio)
                    and aspect_error <= float(aspect_ratio_tolerance)
                )
                candidate.update(
                    {
                        "geometry_valid": geometry_valid,
                        "area_ratio": float(area_ratio),
                        "aspect_ratio": float(aspect_ratio),
                        "aspect_ratio_error": float(aspect_error),
                        "selection_score": float(
                            confidence
                            + float(area_weight) * area_ratio
                            - float(aspect_weight) * aspect_error
                        ),
                    }
                )
            candidates.append(candidate)

        if not candidates:
            return None
        valid_candidates = [candidate for candidate in candidates if candidate["geometry_valid"]]
        pool = valid_candidates or candidates
        selected_candidate = max(pool, key=lambda candidate: candidate["selection_score"])
        return {
            "points": selected_candidate["points"],
            "bbox": selected_candidate["bbox"],
            "confidence": selected_candidate["confidence"],
            "selected_index": int(selected_candidate["index"]),
            "candidate_count": len(candidates),
            "selection_score": float(selected_candidate["selection_score"]),
            "candidate_metrics": [
                {
                    key: value
                    for key, value in candidate.items()
                    if key not in {"points", "bbox"}
                }
                for candidate in candidates
            ],
        }
    except (AttributeError, TypeError, ValueError):
        return None
