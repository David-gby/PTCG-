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


def _candidate_edge_support(edges: np.ndarray, points: np.ndarray) -> tuple[float, list[float]]:
    """Return four-side edge support without depending on the legacy detector.

    This is intentionally evaluated only when the segmentation model returns
    multiple plausible masks.  It prevents a large, smooth background region
    from winning solely because the legacy ranking rewarded mask area.
    """
    radius = max(2, int(round(min(edges.shape[:2]) * 0.003)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    support = cv2.dilate(edges, kernel)
    side_scores: list[float] = []
    for start, end in zip(points, np.roll(points, -1, axis=0)):
        length = max(2, int(round(np.linalg.norm(end - start))))
        count = min(500, max(50, length))
        samples = np.linspace(start, end, count)
        xs = np.clip(np.rint(samples[:, 0]).astype(int), 0, edges.shape[1] - 1)
        ys = np.clip(np.rint(samples[:, 1]).astype(int), 0, edges.shape[0] - 1)
        side_scores.append(float(np.mean(support[ys, xs] > 0)))
    mean_score = float(np.mean(side_scores))
    weakest = float(min(side_scores))
    return float(np.clip(0.70 * mean_score + 0.30 * weakest, 0.0, 1.0)), side_scores


def _prepare_candidate_edges(image: np.ndarray, max_dimension: int = 1600) -> tuple[np.ndarray, float] | None:
    if not isinstance(image, np.ndarray) or image.size == 0:
        return None
    height, width = image.shape[:2]
    scale = min(1.0, float(max_dimension) / max(height, width))
    working = image
    if scale < 1.0:
        working = cv2.resize(
            image,
            (max(2, int(round(width * scale))), max(2, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    gray = working if working.ndim == 2 else cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    lower = int(max(5, 0.67 * median))
    upper = int(min(255, max(lower + 20, 1.33 * median)))
    return cv2.Canny(gray, lower, upper, L2gradient=True), scale


def _border_contacts(
    points: np.ndarray,
    *,
    width: int,
    height: int,
    margin_ratio: float,
) -> list[str]:
    x_margin = max(2.0, float(width) * float(margin_ratio))
    y_margin = max(2.0, float(height) * float(margin_ratio))
    contacts: list[str] = []
    if float(np.min(points[:, 0])) <= x_margin:
        contacts.append("left")
    if float(np.min(points[:, 1])) <= y_margin:
        contacts.append("top")
    if float(width - 1 - np.max(points[:, 0])) <= x_margin:
        contacts.append("right")
    if float(height - 1 - np.max(points[:, 1])) <= y_margin:
        contacts.append("bottom")
    return contacts


def extract_silhouette_prediction(
    prediction: Any,
    *,
    image_shape: Any | None = None,
    image: np.ndarray | None = None,
    target_aspect_ratio: float = 1.397,
    aspect_ratio_tolerance: float = 0.25,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.95,
    area_weight: float = 0.40,
    aspect_weight: float = 0.25,
    edge_weight: float = 0.18,
    border_contact_penalty: float = 0.07,
    border_margin_ratio: float = 0.006,
    full_frame_exempt_area_ratio: float = 0.86,
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
                "base_selection_score": confidence,
                "selection_score": confidence,
                "edge_support_score": None,
                "side_edge_support": None,
                "border_contacts": [],
                "border_contact_penalty": 0.0,
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
                        "base_selection_score": float(
                            confidence
                            + float(area_weight) * area_ratio
                            - float(aspect_weight) * aspect_error
                        ),
                    }
                )
                candidate["selection_score"] = candidate["base_selection_score"]
            candidates.append(candidate)

        if not candidates:
            return None

        # The expensive image evidence is needed only for an actual ambiguity.
        # Single-mask behavior therefore remains identical to the deployed
        # pipeline, while multi-mask scenes gain protection against large,
        # border-touching background regions (for example saturated mats).
        if has_shape and len(candidates) > 1:
            image_aspect = max(width, height) / max(1.0, float(min(width, height)))
            image_aspect_error = abs(image_aspect - target_aspect) / target_aspect
            for candidate in candidates:
                points = np.asarray(candidate["points"], dtype=np.float32)
                contacts = _border_contacts(
                    points,
                    width=width,
                    height=height,
                    margin_ratio=border_margin_ratio,
                )
                candidate["border_contacts"] = contacts
                area_ratio = float(candidate.get("area_ratio") or 0.0)
                full_frame_exempt = bool(
                    area_ratio >= float(full_frame_exempt_area_ratio)
                    and image_aspect_error <= 0.08
                    and len(contacts) >= 3
                )
                candidate["full_frame_border_exempt"] = full_frame_exempt

            # Preserve the proven legacy card-vs-artwork ranking when all
            # candidates are safely inside the photograph.  The new evidence
            # changes ranking only when at least one non-full-frame candidate
            # suspiciously touches the source-image boundary.
            border_ambiguity = any(
                candidate.get("border_contacts")
                and not bool(candidate.get("full_frame_border_exempt"))
                for candidate in candidates
            )
            prepared_edges = (
                _prepare_candidate_edges(image)
                if border_ambiguity and image is not None
                else None
            )
            for candidate in candidates:
                contacts = list(candidate.get("border_contacts") or [])
                full_frame_exempt = bool(candidate.get("full_frame_border_exempt"))
                penalty = 0.0 if full_frame_exempt else float(border_contact_penalty) * len(contacts)
                candidate["border_contact_penalty"] = float(penalty)

                edge_score = 0.0
                side_scores = [0.0] * 4
                if prepared_edges is not None:
                    edges, scale = prepared_edges
                    edge_score, side_scores = _candidate_edge_support(edges, points * float(scale))
                    candidate["edge_support_score"] = float(edge_score)
                    candidate["side_edge_support"] = [float(value) for value in side_scores]
                candidate["selection_score"] = float(
                    candidate["base_selection_score"]
                    + (float(edge_weight) * edge_score - penalty if border_ambiguity else 0.0)
                )

        valid_candidates = [candidate for candidate in candidates if candidate["geometry_valid"]]
        pool = valid_candidates or candidates
        selected_candidate = max(pool, key=lambda candidate: candidate["selection_score"])
        ranked = sorted(pool, key=lambda candidate: candidate["selection_score"], reverse=True)
        score_margin = (
            float(ranked[0]["selection_score"] - ranked[1]["selection_score"])
            if len(ranked) > 1
            else None
        )
        return {
            "points": selected_candidate["points"],
            "bbox": selected_candidate["bbox"],
            "confidence": selected_candidate["confidence"],
            "selected_index": int(selected_candidate["index"]),
            "candidate_count": len(candidates),
            "selection_score": float(selected_candidate["selection_score"]),
            "selection_audit": {
                "policy": "outer_multicandidate_edge_border_v1",
                "score_margin": score_margin,
                "edge_evidence_used": bool(
                    any(candidate.get("edge_support_score") is not None for candidate in candidates)
                ),
                "selected_border_contacts": list(selected_candidate.get("border_contacts") or []),
            },
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
