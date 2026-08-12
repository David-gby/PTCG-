from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from .outer_detection import order_points


SIDE_NAMES = ("top", "right", "bottom", "left")


def _sample(image: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float32)
    leading_shape = coordinates.shape[:-1]
    sampled = cv2.remap(
        image,
        coordinates[..., 0].reshape(1, -1),
        coordinates[..., 1].reshape(1, -1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    if image.ndim == 2:
        return sampled.reshape(leading_shape)
    return sampled.reshape((*leading_shape, image.shape[2]))


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        raise ValueError("Cannot normalize a zero-length vector")
    return np.asarray(vector, dtype=np.float32) / length


def _line_intersection(first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]) -> np.ndarray | None:
    point_a, direction_a = first
    point_b, direction_b = second
    matrix = np.column_stack((direction_a, -direction_b))
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-5:
        return None
    parameters = np.linalg.solve(matrix, point_b - point_a)
    return (point_a + parameters[0] * direction_a).astype(np.float32)


def _quad_dimensions(points: np.ndarray) -> tuple[float, float]:
    tl, tr, br, bl = points
    width = 0.5 * (float(np.linalg.norm(tr - tl)) + float(np.linalg.norm(br - bl)))
    height = 0.5 * (float(np.linalg.norm(bl - tl)) + float(np.linalg.norm(br - tr)))
    return width, height


def _shared_directions(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate opposing-side directions without trusting a single corner.

    A silhouette outlier usually corrupts two adjacent sides. Averaging the two
    opposing directions halves that angular error and gives the local search a
    stable orientation prior.
    """
    tl, tr, br, bl = points
    horizontal = _normalize(_normalize(tr - tl) + _normalize(br - bl))
    vertical = _normalize(_normalize(br - tr) + _normalize(bl - tl))
    return horizontal, vertical


def _line_profile_score(
    lab: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    coarse_grad_x: np.ndarray,
    coarse_grad_y: np.ndarray,
    line_points: np.ndarray,
    normal: np.ndarray,
    strip_gap: float,
    exterior_color: np.ndarray,
    outside_spread_weight: float,
    exterior_distance_weight: float,
) -> tuple[float, dict[str, float]]:
    inside = _sample(lab, line_points - strip_gap * normal[None, :])
    outside = _sample(lab, line_points + strip_gap * normal[None, :])
    color_delta = np.linalg.norm(outside - inside, axis=1)

    gx = _sample(grad_x, line_points)
    gy = _sample(grad_y, line_points)
    fine = np.abs(gx * normal[0] + gy * normal[1])
    coarse_x = _sample(coarse_grad_x, line_points)
    coarse_y = _sample(coarse_grad_y, line_points)
    coarse = np.abs(coarse_x * normal[0] + coarse_y * normal[1])

    # A physical printed/card edge is both continuous and locally sharp. The
    # lower quantile penalizes candidates supported by only a short texture
    # fragment; fine-vs-coarse response suppresses broad cast-shadow borders.
    gradient_median = float(np.quantile(fine, 0.50))
    gradient_floor = float(np.quantile(fine, 0.28))
    sharp_response = np.maximum(fine - 0.72 * coarse, 0.0)
    sharpness = float(np.quantile(sharp_response, 0.45))
    contrast = float(np.quantile(color_delta, 0.45))

    outside_centered = outside - np.median(outside, axis=0, keepdims=True)
    outside_spread = float(np.quantile(np.linalg.norm(outside_centered, axis=1), 0.50))
    exterior_distance = float(np.linalg.norm(np.median(outside, axis=0) - exterior_color))
    score = 0.040 * gradient_median + 0.025 * gradient_floor + 0.035 * sharpness + contrast
    score -= outside_spread_weight * outside_spread
    # Internal artwork/border lines can be sharper than the card boundary, but
    # the strip immediately outside them still looks like the card. Compare it
    # with a farther, coarse-mask exterior sample to reject those inner lines.
    score -= exterior_distance_weight * exterior_distance
    return float(score), {
        "contrast": contrast,
        "gradient_median": gradient_median,
        "gradient_floor": gradient_floor,
        "sharpness": sharpness,
        "outside_spread": outside_spread,
        "exterior_distance": exterior_distance,
    }


def refine_outer_edges(
    image: np.ndarray,
    points: Iterable[Iterable[float]],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Refine a coarse card quadrilateral against full-resolution image edges.

    The neural mask remains the coarse locator. Four narrow, high-resolution
    line searches then find continuous sharp card/background transitions and
    intersect the fitted lines. A conservative geometry/movement gate rejects
    refinements that lack enough original-image evidence.
    """
    cfg = dict(config or {})
    try:
        ordered = order_points(points).astype(np.float32)
    except (TypeError, ValueError):
        return {"accepted": False, "points": None, "reason": "invalid_points", "metrics": {}}
    if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in (2, 3):
        return {"accepted": False, "points": ordered.tolist(), "reason": "invalid_image", "metrics": {}}

    original_height, original_width = image.shape[:2]
    max_dimension = max(640, int(cfg.get("max_dimension", 1800)))
    scale = min(1.0, max_dimension / max(original_height, original_width))
    if scale < 1.0:
        work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        work = image.copy()
    work_points = ordered * scale
    card_width, card_height = _quad_dimensions(work_points)
    short_dimension = min(card_width, card_height)
    if short_dimension < 40:
        return {"accepted": False, "points": ordered.tolist(), "reason": "card_too_small", "metrics": {}}

    if work.ndim == 2:
        work = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    fine_blur = cv2.GaussianBlur(gray, (0, 0), 0.8)
    coarse_blur = cv2.GaussianBlur(gray, (0, 0), 2.8)
    grad_x = cv2.Scharr(fine_blur, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(fine_blur, cv2.CV_32F, 0, 1)
    coarse_grad_x = cv2.Scharr(coarse_blur, cv2.CV_32F, 1, 0)
    coarse_grad_y = cv2.Scharr(coarse_blur, cv2.CV_32F, 0, 1)

    horizontal, vertical = _shared_directions(work_points)
    tangents = (horizontal, vertical, -horizontal, -vertical)
    center = work_points.mean(axis=0)
    search_ratio = float(cfg.get("search_ratio", 0.075))
    high_resolution_threshold = int(cfg.get("high_resolution_threshold", 3000))
    if max(original_height, original_width) < high_resolution_threshold:
        # At the smaller source scale the coarse mask is already close, while
        # a wide band is more likely to include strong internal print borders.
        search_ratio = min(search_ratio, float(cfg.get("standard_resolution_search_ratio", 0.022)))
    search_radius = max(5.0, short_dimension * search_ratio)
    offset_step = max(1.0, short_dimension / 700.0)
    offset_direction = str(cfg.get("offset_direction", "both")).strip().lower()
    if offset_direction == "inward":
        offsets = np.arange(-search_radius, offset_step * 0.5, offset_step, dtype=np.float32)
    elif offset_direction == "outward":
        offsets = np.arange(0.0, search_radius + offset_step * 0.5, offset_step, dtype=np.float32)
    else:
        offsets = np.arange(
            -search_radius,
            search_radius + offset_step * 0.5,
            offset_step,
            dtype=np.float32,
        )
    max_angle = max(0.0, float(cfg.get("max_angle_degrees", 3.0)))
    angle_step = max(0.5, float(cfg.get("angle_step_degrees", 1.0)))
    angles = np.arange(-max_angle, max_angle + angle_step * 0.5, angle_step, dtype=np.float32)
    strip_gap = max(2.0, short_dimension * float(cfg.get("strip_gap_ratio", 0.005)))
    sample_start = float(cfg.get("sample_start", 0.14))
    sample_end = float(cfg.get("sample_end", 0.86))
    position_penalty = float(cfg.get("position_penalty", 2.2))
    outside_spread_weight = float(cfg.get("outside_spread_weight", 0.10))
    exterior_distance_weight = float(cfg.get("exterior_distance_weight", 0.22))
    side_lines: list[tuple[np.ndarray, np.ndarray]] = []
    side_diagnostics: list[dict[str, Any]] = []

    for side_index, (start, end, base_tangent) in enumerate(
        zip(work_points, np.roll(work_points, -1, axis=0), tangents)
    ):
        midpoint = 0.5 * (start + end)
        side_length = float(np.linalg.norm(end - start))
        reference_length = card_width if side_index in (0, 2) else card_height
        sample_count = int(np.clip(reference_length / 3.0, 128, 480))
        fractions = np.linspace(sample_start, sample_end, sample_count, dtype=np.float32)
        base_tangent = _normalize(base_tangent)
        base_normal = np.asarray([-base_tangent[1], base_tangent[0]], dtype=np.float32)
        if float(np.dot(base_normal, midpoint - center)) < 0:
            base_normal = -base_normal
        reference_along = (fractions - 0.5)[:, None] * reference_length * base_tangent[None, :]
        exterior_points = (
            midpoint[None, :]
            + reference_along
            + (search_radius + 2.0 * strip_gap) * base_normal[None, :]
        )
        exterior_color = np.median(_sample(lab, exterior_points), axis=0)

        raw_candidates: list[tuple[float, float, float, np.ndarray, np.ndarray, dict[str, float]]] = []
        baseline_evidence = -float("inf")
        for angle in angles:
            radians = math.radians(float(angle))
            rotation = np.asarray(
                [[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]],
                dtype=np.float32,
            )
            tangent = _normalize(rotation @ base_tangent)
            normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
            if float(np.dot(normal, midpoint - center)) < 0:
                normal = -normal
            along = (fractions - 0.5)[:, None] * reference_length * tangent[None, :]
            for offset in offsets:
                line_center = midpoint + float(offset) * normal
                line_points = line_center[None, :] + along
                evidence_score, evidence = _line_profile_score(
                    lab,
                    grad_x,
                    grad_y,
                    coarse_grad_x,
                    coarse_grad_y,
                    line_points,
                    normal,
                    strip_gap,
                    exterior_color,
                    outside_spread_weight,
                    exterior_distance_weight,
                )
                raw_candidates.append(
                    (evidence_score, float(offset), float(angle), line_center, tangent, evidence)
                )
                if abs(float(offset)) <= offset_step * 0.55 and abs(float(angle)) <= angle_step * 0.55:
                    baseline_evidence = max(baseline_evidence, evidence_score)

        evidence_values = np.asarray([item[0] for item in raw_candidates], dtype=np.float32)
        evidence_scale = max(float(np.quantile(evidence_values, 0.55)), baseline_evidence, 5.0)
        candidates: list[tuple[float, float, float, np.ndarray, np.ndarray, dict[str, float]]] = []
        for evidence_score, offset, angle, line_center, tangent, evidence in raw_candidates:
            normalized_offset = abs(offset) / max(search_radius, 1.0)
            score = evidence_score - position_penalty * evidence_scale * normalized_offset**1.35
            score -= 0.12 * position_penalty * evidence_scale * abs(angle) / max(max_angle, 1.0)
            candidates.append((score, evidence_score, offset, line_center, tangent, evidence))
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected_score, selected_evidence, selected_offset, line_center, tangent, evidence = candidates[0]
        baseline_score = baseline_evidence
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
        if float(np.dot(normal, midpoint - center)) < 0:
            normal = -normal

        # Sub-pixel/local robust fitting: at each along-side sample, keep the
        # strongest normal-gradient location close to the globally selected line.
        local_radius = max(2, int(round(short_dimension * 0.0035)))
        local_offsets = np.arange(-local_radius, local_radius + 1, dtype=np.float32)
        base = line_center[None, :] + (fractions - 0.5)[:, None] * reference_length * tangent[None, :]
        band = base[None, :, :] + local_offsets[:, None, None] * normal[None, None, :]
        gx = _sample(grad_x, band)
        gy = _sample(grad_y, band)
        strengths = np.abs(gx * normal[0] + gy * normal[1])
        best_indices = np.argmax(strengths, axis=0)
        edge_points = base + local_offsets[best_indices, None] * normal[None, :]
        best_strengths = strengths[best_indices, np.arange(sample_count)]
        keep = best_strengths >= np.quantile(best_strengths, 0.38)
        fitted = False
        if int(np.count_nonzero(keep)) >= 40:
            vx, vy, x0, y0 = cv2.fitLine(edge_points[keep], cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
            fitted_tangent = _normalize(np.asarray([vx, vy], dtype=np.float32))
            if float(np.dot(fitted_tangent, tangent)) < 0:
                fitted_tangent = -fitted_tangent
            angular_change = math.degrees(math.acos(float(np.clip(np.dot(fitted_tangent, tangent), -1.0, 1.0))))
            if angular_change <= 1.75:
                line_center = np.asarray([x0, y0], dtype=np.float32)
                tangent = fitted_tangent
                fitted = True
        side_lines.append((line_center, tangent))
        second_score = candidates[min(4, len(candidates) - 1)][0]
        side_diagnostics.append(
            {
                "name": SIDE_NAMES[side_index],
                "offset_px": float(selected_offset / scale),
                "offset_ratio": float(selected_offset / max(short_dimension, 1.0)),
                "score": float(selected_score),
                "baseline_score": float(baseline_score),
                "score_gain": float(selected_score - baseline_score),
                "peak_margin": float(selected_score - second_score),
                "fitted": fitted,
                **{key: float(value) for key, value in evidence.items()},
            }
        )

    top, right, bottom, left = side_lines
    corners = (
        _line_intersection(top, left),
        _line_intersection(top, right),
        _line_intersection(bottom, right),
        _line_intersection(bottom, left),
    )
    if any(corner is None for corner in corners):
        return {
            "accepted": False,
            "points": ordered.tolist(),
            "reason": "parallel_lines",
            "metrics": {"sides": side_diagnostics},
        }
    refined_work = np.asarray(corners, dtype=np.float32)
    refined = refined_work / scale
    convex = bool(cv2.isContourConvex(np.rint(refined).astype(np.int32).reshape(-1, 1, 2)))
    in_bounds = bool(
        np.all(refined[:, 0] >= 0)
        and np.all(refined[:, 1] >= 0)
        and np.all(refined[:, 0] <= original_width - 1)
        and np.all(refined[:, 1] <= original_height - 1)
    )
    movement = np.linalg.norm(refined - ordered, axis=1)
    original_short = short_dimension / scale
    max_movement_ratio = float(np.max(movement) / max(original_short, 1.0))
    original_area = abs(float(cv2.contourArea(ordered.reshape(-1, 1, 2))))
    refined_area = abs(float(cv2.contourArea(refined.reshape(-1, 1, 2))))
    area_change_ratio = abs(refined_area - original_area) / max(original_area, 1.0)
    mean_gain = float(np.mean([side["score_gain"] for side in side_diagnostics]))
    minimum_evidence = float(min(side["score"] for side in side_diagnostics))
    max_side_offset_ratio = float(max(abs(side["offset_ratio"]) for side in side_diagnostics))
    min_side_peak_margin = float(min(side["peak_margin"] for side in side_diagnostics))
    side_confidence = [
        float(
            np.clip(
                (1.0 - math.exp(-max(float(side["score"]), 0.0) / 45.0))
                * math.exp(-float(side["exterior_distance"]) / 180.0),
                0.0,
                1.0,
            )
        )
        for side in side_diagnostics
    ]
    corner_confidence = [
        float(math.sqrt(side_confidence[(index - 1) % 4] * side_confidence[index]))
        for index in range(4)
    ]
    accepted = bool(
        convex
        and in_bounds
        and max_movement_ratio <= float(cfg.get("max_corner_movement_ratio", 0.13))
        and area_change_ratio <= float(cfg.get("max_area_change_ratio", 0.18))
        and minimum_evidence >= float(cfg.get("min_side_score", 2.0))
        and mean_gain >= float(cfg.get("min_mean_score_gain", -0.25))
        and mean_gain <= float(cfg.get("max_mean_score_gain", float("inf")))
        and max_side_offset_ratio <= float(cfg.get("max_side_offset_ratio", float("inf")))
        and min_side_peak_margin >= float(cfg.get("min_side_peak_margin", -float("inf")))
    )
    if accepted:
        reason = None
    elif mean_gain > float(cfg.get("max_mean_score_gain", float("inf"))):
        reason = "excessive_score_gain"
    elif max_side_offset_ratio > float(cfg.get("max_side_offset_ratio", float("inf"))):
        reason = "excessive_side_offset"
    elif min_side_peak_margin < float(cfg.get("min_side_peak_margin", -float("inf"))):
        reason = "ambiguous_side_peak"
    else:
        reason = "refinement_guard_rejected"
    return {
        "accepted": accepted,
        "points": (refined if accepted else ordered).round(3).tolist(),
        "candidate_points": refined.round(3).tolist(),
        "reason": reason,
        "metrics": {
            "sides": side_diagnostics,
            "mean_corner_movement_px": float(np.mean(movement)),
            "max_corner_movement_px": float(np.max(movement)),
            "max_corner_movement_ratio": max_movement_ratio,
            "area_change_ratio": float(area_change_ratio),
            "mean_score_gain": mean_gain,
            "min_side_score": minimum_evidence,
            "max_side_offset_ratio": max_side_offset_ratio,
            "min_side_peak_margin": min_side_peak_margin,
            "effective_search_ratio": search_ratio,
            "side_confidence": side_confidence,
            "corner_confidence": corner_confidence,
        },
    }
