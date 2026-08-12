from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from .config import normalize_config
from .io_utils import write_image


def order_points(points: Iterable[Iterable[float]]) -> np.ndarray:
    """Order a quadrilateral as TL, TR, BR, BL using cyclic geometry.

    The vertices are first ordered around their centroid.  Every cyclic start
    and direction is then evaluated against top/bottom and left/right spatial
    relationships.  This is more stable under perspective than assigning all
    corners independently from x+y and x-y extrema.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        raise ValueError("Exactly four finite 2D points are required")
    if len(np.unique(np.round(pts, 4), axis=0)) != 4:
        raise ValueError("Outer points must be unique")

    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    cyclic = pts[np.argsort(angles)]
    diagonal = max(float(np.ptp(pts[:, 0]) + np.ptp(pts[:, 1])), 1.0)

    best: np.ndarray | None = None
    best_score = -float("inf")
    for direction in (cyclic, cyclic[::-1]):
        for offset in range(4):
            candidate = np.roll(direction, -offset, axis=0)
            tl, tr, br, bl = candidate
            top_y = (tl[1] + tr[1]) * 0.5
            bottom_y = (bl[1] + br[1]) * 0.5
            left_x = (tl[0] + bl[0]) * 0.5
            right_x = (tr[0] + br[0]) * 0.5
            score = (bottom_y - top_y + right_x - left_x) / diagonal
            score += 0.20 * np.sign(tr[0] - tl[0])
            score += 0.20 * np.sign(br[1] - tr[1])
            score += 0.20 * np.sign(br[0] - bl[0])
            score += 0.20 * np.sign(bl[1] - tl[1])
            if score > best_score:
                best_score = float(score)
                best = candidate.copy()
    if best is None or abs(cv2.contourArea(best.reshape(-1, 1, 2))) < 1.0:
        raise ValueError("Degenerate quadrilateral")
    return best.astype(np.float32)


def _side_lengths(points: np.ndarray) -> tuple[float, float, float, float]:
    tl, tr, br, bl = points
    return (
        float(np.linalg.norm(tr - tl)),
        float(np.linalg.norm(br - bl)),
        float(np.linalg.norm(bl - tl)),
        float(np.linalg.norm(br - tr)),
    )


def _point_line_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    vector = end - start
    denom = max(float(np.linalg.norm(vector)), 1e-6)
    offsets = points - start
    cross = vector[0] * offsets[..., 1] - vector[1] * offsets[..., 0]
    return np.abs(cross / denom)


def _edge_support(edges: np.ndarray, points: np.ndarray) -> tuple[float, list[float]]:
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


def _straightness(contour: np.ndarray | None, points: np.ndarray) -> float:
    if contour is None or len(contour) < 8:
        return 0.75
    samples = contour.reshape(-1, 2).astype(np.float32)
    distances = np.vstack(
        [_point_line_distances(samples, a, b) for a, b in zip(points, np.roll(points, -1, axis=0))]
    )
    nearest = distances.min(axis=0)
    tolerance = max(2.0, 0.008 * max(np.ptp(points[:, 0]), np.ptp(points[:, 1])))
    return float(np.clip(np.mean(nearest <= tolerance), 0.0, 1.0))


def _geometry_metrics(points: np.ndarray, image_shape: tuple[int, int], cfg: Mapping[str, Any]) -> dict[str, Any]:
    height, width = image_shape
    area = abs(float(cv2.contourArea(points.reshape(-1, 1, 2))))
    area_ratio = area / max(1.0, float(width * height))
    top, bottom, left, right = _side_lengths(points)
    avg_width = max(1e-6, (top + bottom) * 0.5)
    avg_height = max(1e-6, (left + right) * 0.5)
    aspect_ratio = max(avg_height, avg_width) / min(avg_height, avg_width)
    target = float(cfg["card_aspect_ratio"])
    aspect_error = abs(aspect_ratio - target) / target
    tolerance = max(float(cfg["aspect_ratio_tolerance"]), 1e-6)
    aspect_score = float(np.clip(1.0 - aspect_error / tolerance, 0.0, 1.0))

    min_area = float(cfg["min_area_ratio"])
    max_area = float(cfg["max_area_ratio"])
    if area_ratio < min_area or area_ratio > max_area:
        area_score = 0.0
    else:
        # Broad plateau: real cards may be either loosely framed or nearly fill the photo.
        ramp = min(1.0, (area_ratio - min_area) / max(0.08, 0.18 - min_area))
        upper = min(1.0, (max_area - area_ratio) / 0.10)
        area_score = float(np.clip(min(ramp, upper), 0.0, 1.0))

    centroid = points.mean(axis=0)
    center_distance = float(
        np.linalg.norm((centroid - np.array([width / 2.0, height / 2.0])) / np.array([width / 2.0, height / 2.0]))
    )
    center_score = float(np.clip(1.0 - 0.70 * center_distance, 0.0, 1.0))

    vectors = np.roll(points, -1, axis=0) - points
    corner_scores: list[float] = []
    angles: list[float] = []
    for index in range(4):
        before = -vectors[index - 1]
        after = vectors[index]
        cosine = float(np.dot(before, after) / max(np.linalg.norm(before) * np.linalg.norm(after), 1e-6))
        angle = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
        angles.append(angle)
        corner_scores.append(float(np.clip(1.0 - abs(angle - 90.0) / 70.0, 0.0, 1.0)))
    convex = bool(cv2.isContourConvex(points.astype(np.int32).reshape(-1, 1, 2)))
    side_balance = min(top, bottom) / max(top, bottom, 1e-6) * min(left, right) / max(left, right, 1e-6)
    perspective_score = float(np.clip(math.sqrt(max(side_balance, 0.0)), 0.0, 1.0))
    geometry_score = float((0.55 * np.mean(corner_scores) + 0.25 * perspective_score + 0.20 * float(convex)))

    margins = np.array(
        [points[:, 0].min(), points[:, 1].min(), width - 1 - points[:, 0].max(), height - 1 - points[:, 1].max()]
    )
    min_margin_ratio = float(np.min(margins / np.array([width, height, width, height], dtype=np.float32)))
    border_threshold = float(cfg.get("border_margin_ratio", 0.008))
    border_margin_score = float(np.clip(min_margin_ratio / max(border_threshold * 2.5, 1e-6), 0.0, 1.0))
    image_is_portrait = height >= width
    candidate_is_portrait = avg_height >= avg_width
    orientation_aligned = bool(image_is_portrait == candidate_is_portrait or max(height, width) / max(1, min(height, width)) < 1.12)
    geometry_valid = bool(
        convex
        and area_ratio >= min_area
        and area_ratio <= max_area
        and aspect_error <= tolerance * 1.8
        and min(angles) >= 25.0
        and max(angles) <= 155.0
    )
    return {
        "area_ratio": float(area_ratio),
        "area_score": area_score,
        "aspect_ratio": float(aspect_ratio),
        "aspect_ratio_error": float(aspect_error),
        "aspect_score": aspect_score,
        "center_score": center_score,
        "convex_score": float(convex),
        "angle_score": float(np.mean(corner_scores)),
        "angles": [float(v) for v in angles],
        "border_margin_score": border_margin_score,
        "min_border_margin_ratio": min_margin_ratio,
        "perspective_score": perspective_score,
        "orientation_aligned": orientation_aligned,
        "geometry_score": float(np.clip(geometry_score, 0.0, 1.0)),
        "geometry_valid": geometry_valid,
    }


def assess_outer_box_quality(
    image: np.ndarray,
    points: Iterable[Iterable[float]],
    edges: np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = normalize_config(config)["outer_detection"]
    ordered = order_points(points)
    if edges.shape[:2] != image.shape[:2]:
        edges = cv2.resize(edges, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    geometry = _geometry_metrics(ordered, image.shape[:2], cfg)
    edge_score, side_scores = _edge_support(edges, ordered)
    confidence = float(np.clip(0.48 * edge_score + 0.32 * geometry["aspect_score"] + 0.20 * geometry["geometry_score"], 0.0, 1.0))
    return {
        "edge_score": edge_score,
        "side_edge_scores": side_scores,
        "aspect_ratio": geometry["aspect_ratio"],
        "aspect_ratio_error": geometry["aspect_ratio_error"],
        "area_ratio": geometry["area_ratio"],
        "perspective_score": geometry["perspective_score"],
        "geometry_valid": geometry["geometry_valid"],
        "confidence": confidence,
    }


def _candidate_score(
    points: np.ndarray,
    contour: np.ndarray | None,
    method: str,
    edges: np.ndarray,
    cfg: Mapping[str, Any],
) -> dict[str, Any] | None:
    try:
        ordered = order_points(points)
    except ValueError:
        return None
    geometry = _geometry_metrics(ordered, edges.shape[:2], cfg)
    if geometry["area_ratio"] < float(cfg["min_area_ratio"]) * 0.75:
        return None
    if geometry["area_ratio"] > float(cfg["max_area_ratio"]) * 1.03:
        return None
    if geometry["aspect_ratio_error"] > float(cfg["aspect_ratio_tolerance"]) * 2.4:
        return None
    edge_score, side_edges = _edge_support(edges, ordered)
    straightness = _straightness(contour, ordered)
    weights = cfg["scoring"]
    final_score = (
        float(weights["edge_weight"]) * edge_score
        + float(weights["aspect_weight"]) * geometry["aspect_score"]
        + float(weights["area_weight"]) * geometry["area_score"]
        + float(weights["geometry_weight"]) * geometry["geometry_score"]
        + float(weights["center_weight"]) * geometry["center_score"]
        + float(weights["straightness_weight"]) * straightness
    )
    # A weak individual side is a common sign that an internal artwork box was selected.
    final_score *= 0.82 + 0.18 * min(side_edges)
    if not geometry["orientation_aligned"]:
        # Internal artwork/text panels commonly have the opposite orientation
        # from the physical card.  Keep them visible as debug candidates but
        # make a plausible full-card contour rank ahead of them.
        final_score *= 0.72
    if method == "min_area_rect":
        final_score -= 0.025
    return {
        "points": ordered,
        "method": method,
        "contour": contour,
        "edge_score": edge_score,
        "side_edge_scores": side_edges,
        "straightness_score": straightness,
        "final_score": float(np.clip(final_score, 0.0, 1.0)),
        **geometry,
    }


def _contour_candidates(edges: np.ndarray, cfg: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(edges.shape[0] * edges.shape[1])
    candidates: list[dict[str, Any]] = []
    min_area = float(cfg["min_area_ratio"]) * image_area * 0.65
    for contour in contours:
        area = abs(float(cv2.contourArea(contour)))
        if area < min_area:
            continue
        hull = cv2.convexHull(contour)
        perimeter = float(cv2.arcLength(hull, True))
        if perimeter <= 0:
            continue
        found_quad = False
        for epsilon_factor in (0.012, 0.018, 0.025, 0.035, 0.05):
            approx = cv2.approxPolyDP(hull, epsilon_factor * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                scored = _candidate_score(approx.reshape(4, 2), contour, "contour", edges, cfg)
                if scored is not None:
                    candidates.append(scored)
                    found_quad = True
        if not found_quad:
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            scored = _candidate_score(box, contour, "min_area_rect", edges, cfg)
            if scored is not None:
                candidates.append(scored)
    return candidates, contours


def _fit_line_ransac(points: np.ndarray, threshold: float, iterations: int = 100) -> np.ndarray | None:
    if len(points) < 4:
        return None
    rng = np.random.default_rng(20260712)
    best_mask: np.ndarray | None = None
    best_count = 0
    for _ in range(iterations):
        indices = rng.choice(len(points), 2, replace=False)
        start, end = points[indices]
        if np.linalg.norm(end - start) < 5:
            continue
        distances = _point_line_distances(points, start, end)
        mask = distances <= threshold
        count = int(mask.sum())
        if count > best_count:
            best_count, best_mask = count, mask
    if best_mask is None or best_count < 4:
        return None
    vx, vy, x0, y0 = cv2.fitLine(points[best_mask].astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    return np.array([float(vx), float(vy), float(x0), float(y0)], dtype=np.float64)


def _line_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    v1, p1 = first[:2], first[2:]
    v2, p2 = second[:2], second[2:]
    matrix = np.column_stack((v1, -v2))
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-4:
        return None
    parameters = np.linalg.solve(matrix, p2 - p1)
    return (p1 + parameters[0] * v1).astype(np.float32)


def _hough_candidate(edges: np.ndarray, cfg: Mapping[str, Any]) -> tuple[dict[str, Any] | None, np.ndarray | None]:
    hcfg = cfg["hough"]
    min_length = int(round(min(edges.shape[:2]) * float(hcfg["min_line_length_ratio"])))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=int(hcfg["threshold"]),
        minLineLength=max(20, min_length),
        maxLineGap=int(hcfg["max_line_gap"]),
    )
    if lines is None or len(lines) < 4:
        return None, lines
    segments = lines.reshape(-1, 4).astype(np.float32)
    horizontal: list[np.ndarray] = []
    vertical: list[np.ndarray] = []
    for segment in segments:
        x1, y1, x2, y2 = segment
        target = horizontal if abs(x2 - x1) >= abs(y2 - y1) else vertical
        target.extend((np.array([x1, y1]), np.array([x2, y2])))
    if len(horizontal) < 8 or len(vertical) < 8:
        return None, lines
    hp = np.asarray(horizontal, dtype=np.float32)
    vp = np.asarray(vertical, dtype=np.float32)
    hmid = float(np.median(hp[:, 1]))
    vmid = float(np.median(vp[:, 0]))
    threshold = max(2.5, min(edges.shape[:2]) * 0.004)
    top = _fit_line_ransac(hp[hp[:, 1] <= hmid], threshold)
    bottom = _fit_line_ransac(hp[hp[:, 1] > hmid], threshold)
    left = _fit_line_ransac(vp[vp[:, 0] <= vmid], threshold)
    right = _fit_line_ransac(vp[vp[:, 0] > vmid], threshold)
    if any(line is None for line in (top, bottom, left, right)):
        return None, lines
    intersections = [
        _line_intersection(top, left),
        _line_intersection(top, right),
        _line_intersection(bottom, right),
        _line_intersection(bottom, left),
    ]
    if any(point is None for point in intersections):
        return None, lines
    points = np.asarray(intersections, dtype=np.float32)
    margin = 0.08 * max(edges.shape[:2])
    if (
        points[:, 0].min() < -margin
        or points[:, 1].min() < -margin
        or points[:, 0].max() > edges.shape[1] - 1 + margin
        or points[:, 1].max() > edges.shape[0] - 1 + margin
    ):
        return None, lines
    return _candidate_score(points, None, "hough", edges, cfg), lines


def _deduplicate(candidates: list[dict[str, Any]], diagonal: float) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["final_score"], reverse=True):
        if any(
            float(np.mean(np.linalg.norm(candidate["points"] - existing["points"], axis=1))) < diagonal * 0.012
            for existing in unique
        ):
            continue
        unique.append(candidate)
    return unique


def _candidate_public(candidate: dict[str, Any], scale: float) -> dict[str, Any]:
    keys = (
        "method",
        "edge_score",
        "side_edge_scores",
        "straightness_score",
        "final_score",
        "area_ratio",
        "area_score",
        "aspect_ratio",
        "aspect_ratio_error",
        "aspect_score",
        "center_score",
        "geometry_score",
        "border_margin_score",
        "min_border_margin_ratio",
        "perspective_score",
        "orientation_aligned",
        "geometry_valid",
    )
    result = {key: candidate[key] for key in keys}
    result["points"] = (candidate["points"] / scale).round(3).tolist()
    return result


def _save_debug_images(
    debug_dir: Path,
    images: Mapping[str, np.ndarray],
    contours: list[np.ndarray],
    candidates: list[dict[str, Any]],
    hough_lines: np.ndarray | None,
    base_image: np.ndarray,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        write_image(debug_dir / name, image)
    contour_view = base_image.copy()
    cv2.drawContours(contour_view, contours, -1, (0, 180, 255), 1, cv2.LINE_AA)
    write_image(debug_dir / "outer_contours.jpg", contour_view)
    candidate_view = base_image.copy()
    for index, candidate in enumerate(candidates[:30]):
        points = np.rint(candidate["points"]).astype(np.int32).reshape(-1, 1, 2)
        color = (0, 255, 0) if index == 0 else (255, 150, 0)
        cv2.polylines(candidate_view, [points], True, color, 2, cv2.LINE_AA)
        anchor = tuple(points[0, 0])
        cv2.putText(candidate_view, f"{index}:{candidate['final_score']:.2f}", anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    write_image(debug_dir / "outer_candidates.jpg", candidate_view)
    hough_view = base_image.copy()
    if hough_lines is not None:
        for x1, y1, x2, y2 in hough_lines.reshape(-1, 4):
            cv2.line(hough_view, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 1, cv2.LINE_AA)
    write_image(debug_dir / "outer_hough_lines.jpg", hough_view)


def detect_outer_box(
    image: np.ndarray,
    config: Mapping[str, Any] | None = None,
    save_debug: bool = False,
    debug_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Detect the physical PTCG card boundary with confidence guardrails."""
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
        return {
            "success": False,
            "points": None,
            "confidence": 0.0,
            "method": None,
            "error_code": "NO_CARD_CANDIDATE",
            "message": "Input image is empty or invalid.",
            "metrics": {},
            "candidates": [],
        }
    cfg = normalize_config(config)["outer_detection"]
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    original_height, original_width = image.shape[:2]
    max_dimension = int(cfg.get("process_max_dimension", 1400))
    scale = min(1.0, max_dimension / max(original_height, original_width)) if max_dimension > 0 else 1.0
    work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else image.copy()

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    grid = tuple(int(v) for v in cfg["clahe_tile_grid_size"])
    clahe = cv2.createCLAHE(clipLimit=float(cfg["clahe_clip_limit"]), tileGridSize=grid).apply(gray)
    bilateral = cv2.bilateralFilter(clahe, 7, 35, 35)
    blur = cv2.GaussianBlur(bilateral, (5, 5), 0)
    median = float(np.median(blur))
    sigma = float(cfg["canny_sigma"])
    lower = int(max(5, (1.0 - sigma) * median))
    upper = int(min(255, max(lower + 20, (1.0 + sigma) * median)))
    canny = cv2.Canny(blur, lower, upper, L2gradient=True)

    grad_x = cv2.Scharr(blur, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(blur, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(grad_x, grad_y)
    threshold = float(np.percentile(magnitude, 82.0))
    gradient = np.where(magnitude >= max(threshold, 1.0), 255, 0).astype(np.uint8)
    gradient = cv2.morphologyEx(gradient, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    combined = cv2.bitwise_or(canny, gradient)
    kernel_size = max(3, int(cfg["morph_kernel_size"]) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, np.ones((3, 3), np.uint8), iterations=1)

    candidates, contours = _contour_candidates(closed, cfg)
    hough_candidate = None
    hough_lines = None
    if bool(cfg["hough"].get("enabled", True)):
        hough_candidate, hough_lines = _hough_candidate(closed, cfg)
        if hough_candidate is not None:
            candidates.append(hough_candidate)
    diagonal = float(math.hypot(work.shape[1], work.shape[0]))
    candidates = _deduplicate(candidates, diagonal)
    if candidates and hough_candidate is not None:
        best_contour = next((item for item in candidates if item["method"] in {"contour", "min_area_rect"}), None)
        if best_contour is not None:
            distance = float(np.mean(np.linalg.norm(best_contour["points"] - hough_candidate["points"], axis=1)))
            if distance < diagonal * 0.045:
                fused_points = 0.72 * best_contour["points"] + 0.28 * hough_candidate["points"]
                fused = _candidate_score(fused_points, best_contour.get("contour"), "fusion", closed, cfg)
                if fused is not None:
                    candidates.append(fused)
                    candidates = _deduplicate(candidates, diagonal)
    candidates.sort(key=lambda item: item["final_score"], reverse=True)

    if save_debug:
        target = Path(debug_dir) if debug_dir is not None else Path("debug")
        _save_debug_images(
            target,
            {
                "outer_gray.jpg": gray,
                "outer_clahe.jpg": clahe,
                "outer_blur.jpg": blur,
                "outer_edges_canny.jpg": canny,
                "outer_edges_gradient.jpg": gradient,
                "outer_edges_closed.jpg": closed,
            },
            contours,
            candidates,
            hough_lines,
            work,
        )

    base_metrics = {
        "image_width": int(original_width),
        "image_height": int(original_height),
        "candidate_count": len(candidates),
        "selected_area_ratio": 0.0,
        "aspect_ratio": 0.0,
        "aspect_ratio_error": 1.0,
        "edge_score": 0.0,
        "center_score": 0.0,
        "geometry_score": 0.0,
        "straightness_score": 0.0,
        "final_score": 0.0,
    }
    public_candidates = [_candidate_public(candidate, scale) for candidate in candidates[:50]]
    if not candidates:
        return {
            "success": False,
            "points": None,
            "confidence": 0.0,
            "method": None,
            "error_code": "NO_CARD_CANDIDATE",
            "message": "No plausible card-shaped quadrilateral was found.",
            "metrics": base_metrics,
            "candidates": public_candidates,
        }

    selected = candidates[0]
    confidence = float(selected["final_score"])
    base_metrics.update(
        {
            "selected_area_ratio": float(selected["area_ratio"]),
            "aspect_ratio": float(selected["aspect_ratio"]),
            "aspect_ratio_error": float(selected["aspect_ratio_error"]),
            "edge_score": float(selected["edge_score"]),
            "center_score": float(selected["center_score"]),
            "geometry_score": float(selected["geometry_score"]),
            "straightness_score": float(selected["straightness_score"]),
            "final_score": confidence,
        }
    )
    error_code: str | None = None
    message = "Outer card boundary detected."
    if selected["min_border_margin_ratio"] <= float(cfg.get("border_margin_ratio", 0.008)):
        error_code = "CARD_PARTIALLY_OUT_OF_FRAME"
        message = "The selected card boundary touches the image border and may be cropped."
    elif not bool(selected["geometry_valid"]):
        error_code = "GEOMETRY_INVALID"
        message = "The best candidate failed quadrilateral geometry checks."
    elif confidence < float(cfg["low_confidence_threshold"]):
        error_code = "LOW_CONFIDENCE_OUTER_BOX"
        message = "The best outer-boundary candidate is below the confidence threshold."
    elif selected["method"] == "hough" and confidence < max(float(cfg["low_confidence_threshold"]), 0.68):
        error_code = "LOW_CONFIDENCE_OUTER_BOX"
        message = "The Hough-only fallback is below its stricter confidence threshold."
    elif min(selected["side_edge_scores"]) < 0.18:
        error_code = "LOW_CONFIDENCE_OUTER_BOX"
        message = "At least one side has insufficient edge support."

    points = (selected["points"] / scale).round(3).tolist() if error_code is None else None
    return {
        "success": error_code is None,
        "points": points,
        "confidence": confidence,
        "method": selected["method"],
        "error_code": error_code,
        "message": message,
        "metrics": base_metrics,
        "candidates": public_candidates,
    }
