from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import torch

try:
    from ..inner_frame.edge_refiner import EdgeRefinerV5, patch_to_tensor
except ImportError:
    from inner_frame.edge_refiner import EdgeRefinerV5, patch_to_tensor

from .outer_detection import order_points


SIDE_NAMES = ("top", "right", "bottom", "left")
CANONICAL_SIDE_LENGTHS = (630.0, 880.0, 630.0, 880.0)
CANONICAL_QUAD = np.asarray(
    [[0.0, 0.0], [630.0, 0.0], [630.0, 880.0], [0.0, 880.0]],
    dtype=np.float32,
)


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        raise ValueError("Cannot normalize a zero-length vector")
    return np.asarray(vector, dtype=np.float32) / length


def _outward_normal(
    start: np.ndarray,
    end: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    tangent = _normalize(end - start)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    midpoint = 0.5 * (start + end)
    if float(np.dot(normal, midpoint - center)) < 0.0:
        normal = -normal
    return normal


def _line_intersection(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
) -> np.ndarray | None:
    point_a, direction_a = first
    point_b, direction_b = second
    matrix = np.column_stack((direction_a, -direction_b))
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-6:
        return None
    parameters = np.linalg.solve(matrix, point_b - point_a)
    return (point_a + float(parameters[0]) * direction_a).astype(np.float32)


def make_outer_side_patch(
    image: np.ndarray,
    quad: Iterable[Iterable[float]],
    side_index: int,
    *,
    center_fraction: float,
    span_fraction: float,
    band_canonical_px: float = 32.0,
    patch_width: int = 96,
    patch_height: int = 224,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rectify a local side strip with outside on the left and inside on the right."""

    ordered = order_points(quad).astype(np.float32)
    start = ordered[side_index]
    end = ordered[(side_index + 1) % 4]
    center = ordered.mean(axis=0)
    outward = _outward_normal(start, end, center)
    side_vector = end - start
    side_length = float(np.linalg.norm(side_vector))
    canonical_length = CANONICAL_SIDE_LENGTHS[side_index]
    source_per_canonical = side_length / canonical_length
    band_source_px = max(3.0, float(band_canonical_px) * source_per_canonical)

    half_span = 0.5 * float(span_fraction)
    start_fraction = float(np.clip(center_fraction - half_span, 0.02, 0.98))
    end_fraction = float(np.clip(center_fraction + half_span, 0.02, 0.98))
    along = np.linspace(
        start_fraction,
        end_fraction,
        patch_height,
        dtype=np.float32,
    )
    across = np.linspace(
        band_source_px,
        -band_source_px,
        patch_width,
        dtype=np.float32,
    )
    base = start[None, :] + along[:, None] * side_vector[None, :]
    coordinates = base[:, None, :] + across[None, :, None] * outward[None, None, :]
    patch = cv2.remap(
        image,
        coordinates[..., 0],
        coordinates[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    metadata = {
        "side_index": int(side_index),
        "side_name": SIDE_NAMES[side_index],
        "center_fraction": float(center_fraction),
        "span_fraction": float(span_fraction),
        "base_point": (
            start + float(center_fraction) * side_vector
        ).astype(np.float32),
        "outward_normal": outward.astype(np.float32),
        "source_per_canonical": float(source_per_canonical),
        "band_source_px": float(band_source_px),
    }
    return np.ascontiguousarray(patch), metadata


def canonical_position_to_source_offset(
    position: float,
    *,
    band_source_px: float,
    patch_width: int,
) -> float:
    """Return signed source-pixel displacement from coarse side; positive is outward."""

    fraction = float(position) / max(float(patch_width - 1), 1.0)
    return float(band_source_px) * (1.0 - 2.0 * fraction)


def source_offset_to_canonical_position(
    offset_source_px: float,
    *,
    band_source_px: float,
    patch_width: int,
) -> float:
    fraction = 0.5 * (1.0 - float(offset_source_px) / max(float(band_source_px), 1e-6))
    return float(np.clip(fraction, 0.0, 1.0) * float(patch_width - 1))


def load_outer_line_refiner(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[EdgeRefinerV5, dict[str, Any]]:
    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    config = dict(payload.get("config", {}))
    input_channels = int(config.get("input_channels", 7))
    model = EdgeRefinerV5(input_channels=input_channels)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, config


def expand_outer_quad(
    quad: Iterable[Iterable[float]],
    margin_canonical_px: float,
) -> np.ndarray:
    """Expand a quadrilateral by a small margin in rectified-card coordinates."""

    ordered = order_points(quad).astype(np.float32)
    margin = float(margin_canonical_px)
    expanded = np.asarray(
        [
            [-margin, -margin],
            [630.0 + margin, -margin],
            [630.0 + margin, 880.0 + margin],
            [-margin, 880.0 + margin],
        ],
        dtype=np.float32,
    )
    inverse = cv2.getPerspectiveTransform(CANONICAL_QUAD, ordered)
    return cv2.perspectiveTransform(expanded.reshape(1, 4, 2), inverse)[0]


def outer_quad_relative_geometry(
    candidate: Iterable[Iterable[float]],
    raw: Iterable[Iterable[float]],
) -> dict[str, float]:
    """Measure whether a legacy candidate cuts inward relative to the raw silhouette."""

    ordered_candidate = order_points(candidate).astype(np.float32)
    ordered_raw = order_points(raw).astype(np.float32)
    transform = cv2.getPerspectiveTransform(ordered_raw, CANONICAL_QUAD)
    canonical = cv2.perspectiveTransform(
        ordered_candidate.reshape(1, 4, 2),
        transform,
    )[0]
    signed_inward = (
        float(np.mean(canonical[[0, 1], 1])),
        float(630.0 - np.mean(canonical[[1, 2], 0])),
        float(880.0 - np.mean(canonical[[2, 3], 1])),
        float(np.mean(canonical[[3, 0], 0])),
    )
    positive_inward = [max(value, 0.0) for value in signed_inward]
    raw_area = abs(float(cv2.contourArea(ordered_raw.reshape(-1, 1, 2))))
    candidate_area = abs(
        float(cv2.contourArea(ordered_candidate.reshape(-1, 1, 2)))
    )
    return {
        "max_inward_relative_to_raw_px": max(positive_inward),
        "mean_inward_relative_to_raw_px": float(np.mean(positive_inward)),
        "candidate_to_raw_area_ratio": candidate_area / max(raw_area, 1.0),
    }


def select_outer_quad_policy(
    raw_quad: Iterable[Iterable[float]],
    legacy_quad: Iterable[Iterable[float]],
    learned_result: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen learned gate, asymmetric legacy fallback, and safety margin."""

    raw = order_points(raw_quad).astype(np.float32)
    legacy = order_points(legacy_quad).astype(np.float32)
    learned_gate = dict(policy.get("learned_gate", {}))
    fallback_gate = dict(policy.get("legacy_fallback_gate", {}))
    metrics = dict(learned_result.get("metrics", {}))
    candidate: np.ndarray | None = None
    try:
        value = learned_result.get("candidate_points")
        if value is not None:
            candidate = order_points(value).astype(np.float32)
            if candidate.shape != (4, 2) or not np.isfinite(candidate).all():
                candidate = None
    except (TypeError, ValueError):
        candidate = None

    learned_allowed = bool(
        candidate is not None
        and float(metrics.get("min_side_confidence", -1.0))
        >= float(learned_gate.get("min_side_confidence", 0.10))
        and float(metrics.get("max_angle_change_degrees", float("inf")))
        <= float(learned_gate.get("max_angle_change_degrees", 2.5))
        and float(metrics.get("max_canonical_offset_px", float("inf")))
        <= float(learned_gate.get("max_canonical_offset_px", 16.0))
        and float(metrics.get("max_corner_movement_ratio", float("inf")))
        <= float(learned_gate.get("max_corner_movement_ratio", 0.04))
        and float(metrics.get("area_change_ratio", float("inf")))
        <= float(learned_gate.get("max_area_change_ratio", 0.025))
    )
    fallback_geometry = outer_quad_relative_geometry(legacy, raw)
    safe_legacy_refinement = bool(
        fallback_geometry["max_inward_relative_to_raw_px"]
        <= float(fallback_gate.get("max_inward_relative_to_raw_px", 5.0))
        and fallback_geometry["mean_inward_relative_to_raw_px"]
        <= float(fallback_gate.get("max_mean_inward_relative_to_raw_px", 2.0))
        and fallback_geometry["candidate_to_raw_area_ratio"]
        >= float(fallback_gate.get("min_candidate_to_raw_area_ratio", 0.998))
    )
    if learned_allowed and candidate is not None:
        selected = candidate
        selected_source = "learned_four_side_refiner"
    elif safe_legacy_refinement:
        selected = legacy
        selected_source = "safe_legacy_refinement"
    else:
        selected = raw
        selected_source = "raw_silhouette"

    margin = float(policy.get("margin_canonical_px", 1.0))
    final = expand_outer_quad(selected, margin)
    return {
        "points": final.round(4).tolist(),
        "selected_points_before_margin": selected.round(4).tolist(),
        "selected_source": selected_source,
        "margin_canonical_px": margin,
        "learned_allowed": learned_allowed,
        "safe_legacy_refinement": safe_legacy_refinement,
        "legacy_fallback_geometry": fallback_geometry,
        "learned_metrics": metrics,
    }


def _distribution_statistics(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = logits.softmax(dim=1)
    coordinates = torch.arange(
        logits.shape[1],
        device=logits.device,
        dtype=probabilities.dtype,
    )
    expected = (probabilities * coordinates[None]).sum(dim=1)
    entropy = (
        -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=1)
        / math.log(logits.shape[1])
    )
    peak_indices = probabilities.argmax(dim=1)
    peak_mass: list[float] = []
    for row, peak in zip(probabilities, peak_indices):
        index = int(peak.item())
        peak_mass.append(float(row[max(0, index - 2) : min(len(row), index + 3)].sum().item()))
    return (
        expected.detach().cpu().numpy(),
        entropy.detach().cpu().numpy(),
        np.asarray(peak_mass, dtype=np.float32),
    )


@torch.inference_mode()
def refine_outer_quad_learned(
    image: np.ndarray,
    coarse_quad: Iterable[Iterable[float]],
    model: EdgeRefinerV5,
    model_config: Mapping[str, Any],
    *,
    device: torch.device,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Refine four sides independently using batched high-resolution strip inference."""

    cfg = dict(config or {})
    try:
        ordered = order_points(coarse_quad).astype(np.float32)
    except (TypeError, ValueError):
        return {"accepted": False, "points": None, "reason": "invalid_points", "metrics": {}}
    if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim != 3:
        return {
            "accepted": False,
            "points": ordered.tolist(),
            "reason": "invalid_image",
            "metrics": {},
        }

    band_canonical_px = float(
        cfg.get("band_canonical_px", model_config.get("band_canonical_px", 32.0))
    )
    patch_width = int(cfg.get("patch_width", model_config.get("patch_width", 96)))
    patch_height = int(cfg.get("patch_height", model_config.get("patch_height", 224)))
    centers = tuple(
        float(value)
        for value in cfg.get("center_fractions", (0.18, 0.34, 0.50, 0.66, 0.82))
    )
    span_fraction = float(cfg.get("span_fraction", 0.30))
    input_channels = int(getattr(model, "input_channels", model_config.get("input_channels", 7)))

    patches: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    for side_index in range(4):
        for center_fraction in centers:
            patch, patch_metadata = make_outer_side_patch(
                image,
                ordered,
                side_index,
                center_fraction=center_fraction,
                span_fraction=span_fraction,
                band_canonical_px=band_canonical_px,
                patch_width=patch_width,
                patch_height=patch_height,
            )
            patches.append(patch_to_tensor(patch, input_channels=input_channels))
            metadata.append(patch_metadata)

    batch = torch.stack(patches, dim=0).to(device, non_blocking=True)
    flipped = torch.flip(batch, dims=(2,))
    logits_original = model(batch)
    logits_flipped = model(flipped)
    logits = 0.5 * (logits_original + logits_flipped)
    positions, entropies, peak_masses = _distribution_statistics(logits)
    first_positions, _, _ = _distribution_statistics(logits_original)
    second_positions, _, _ = _distribution_statistics(logits_flipped)
    disagreements = np.abs(first_positions - second_positions)

    side_lines: list[tuple[np.ndarray, np.ndarray]] = []
    side_metrics: list[dict[str, Any]] = []
    all_canonical_offsets: list[float] = []
    all_confidences: list[float] = []
    for side_index, side_name in enumerate(SIDE_NAMES):
        begin = side_index * len(centers)
        finish = begin + len(centers)
        points: list[np.ndarray] = []
        weights: list[float] = []
        local_offsets: list[float] = []
        local_confidences: list[float] = []
        for local_index in range(begin, finish):
            item = metadata[local_index]
            offset_source = canonical_position_to_source_offset(
                float(positions[local_index]),
                band_source_px=float(item["band_source_px"]),
                patch_width=patch_width,
            )
            canonical_offset = offset_source / max(float(item["source_per_canonical"]), 1e-6)
            confidence = float(
                np.clip(
                    0.58 * (1.0 - float(entropies[local_index]))
                    + 0.42 * float(peak_masses[local_index])
                    - min(0.35, float(disagreements[local_index]) / 10.0),
                    0.0,
                    1.0,
                )
            )
            refined_point = (
                np.asarray(item["base_point"], dtype=np.float32)
                + offset_source * np.asarray(item["outward_normal"], dtype=np.float32)
            )
            points.append(refined_point)
            weights.append(max(confidence, 0.05))
            local_offsets.append(float(canonical_offset))
            local_confidences.append(confidence)

        point_array = np.asarray(points, dtype=np.float32)
        median_offset = float(np.median(local_offsets))
        mad = float(np.median(np.abs(np.asarray(local_offsets) - median_offset)))
        keep = np.abs(np.asarray(local_offsets) - median_offset) <= max(2.5, 2.8 * mad)
        if int(np.count_nonzero(keep)) < 3:
            keep = np.ones(len(points), dtype=bool)
        vx, vy, x0, y0 = cv2.fitLine(
            point_array[keep],
            cv2.DIST_HUBER,
            0,
            0.01,
            0.01,
        ).reshape(-1)
        direction = _normalize(np.asarray([vx, vy], dtype=np.float32))
        coarse_direction = _normalize(
            ordered[(side_index + 1) % 4] - ordered[side_index]
        )
        if float(np.dot(direction, coarse_direction)) < 0.0:
            direction = -direction
        angle_change = math.degrees(
            math.acos(float(np.clip(np.dot(direction, coarse_direction), -1.0, 1.0)))
        )
        line = (np.asarray([x0, y0], dtype=np.float32), direction)
        side_lines.append(line)
        side_confidence = float(np.mean(np.asarray(local_confidences)[keep]))
        all_canonical_offsets.extend(local_offsets)
        all_confidences.extend(local_confidences)
        side_metrics.append(
            {
                "name": side_name,
                "canonical_offsets_px": local_offsets,
                "median_canonical_offset_px": median_offset,
                "offset_mad_px": mad,
                "confidence": side_confidence,
                "angle_change_degrees": float(angle_change),
                "kept_points": int(np.count_nonzero(keep)),
            }
        )

    top, right, bottom, left = side_lines
    intersections = (
        _line_intersection(top, left),
        _line_intersection(top, right),
        _line_intersection(bottom, right),
        _line_intersection(bottom, left),
    )
    if any(point is None for point in intersections):
        return {
            "accepted": False,
            "points": ordered.tolist(),
            "reason": "parallel_lines",
            "metrics": {"sides": side_metrics},
        }
    candidate = order_points(np.asarray(intersections, dtype=np.float32))
    height, width = image.shape[:2]
    convex = bool(
        cv2.isContourConvex(np.rint(candidate).astype(np.int32).reshape(-1, 1, 2))
    )
    in_bounds = bool(
        np.all(candidate[:, 0] >= 0.0)
        and np.all(candidate[:, 1] >= 0.0)
        and np.all(candidate[:, 0] <= width - 1)
        and np.all(candidate[:, 1] <= height - 1)
    )
    movement = np.linalg.norm(candidate - ordered, axis=1)
    side_lengths = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
    short_side = max(float(np.min(side_lengths)), 1.0)
    movement_ratio = float(np.max(movement) / short_side)
    original_area = abs(float(cv2.contourArea(ordered.reshape(-1, 1, 2))))
    candidate_area = abs(float(cv2.contourArea(candidate.reshape(-1, 1, 2))))
    area_change_ratio = abs(candidate_area - original_area) / max(original_area, 1.0)
    minimum_side_confidence = min(float(item["confidence"]) for item in side_metrics)
    maximum_angle_change = max(float(item["angle_change_degrees"]) for item in side_metrics)
    maximum_offset = max(abs(float(value)) for value in all_canonical_offsets)
    accepted = bool(
        convex
        and in_bounds
        and movement_ratio <= float(cfg.get("max_corner_movement_ratio", 0.055))
        and area_change_ratio <= float(cfg.get("max_area_change_ratio", 0.075))
        and minimum_side_confidence >= float(cfg.get("min_side_confidence", 0.18))
        and maximum_angle_change <= float(cfg.get("max_angle_change_degrees", 2.5))
        and maximum_offset <= float(cfg.get("max_canonical_offset_px", 26.0))
    )
    return {
        "accepted": accepted,
        "points": (candidate if accepted else ordered).round(3).tolist(),
        "candidate_points": candidate.round(3).tolist(),
        "reason": None if accepted else "learned_refinement_guard_rejected",
        "metrics": {
            "sides": side_metrics,
            "mean_corner_movement_px": float(np.mean(movement)),
            "max_corner_movement_px": float(np.max(movement)),
            "max_corner_movement_ratio": movement_ratio,
            "area_change_ratio": float(area_change_ratio),
            "min_side_confidence": minimum_side_confidence,
            "mean_patch_confidence": float(np.mean(all_confidences)),
            "max_angle_change_degrees": maximum_angle_change,
            "max_canonical_offset_px": float(maximum_offset),
        },
    }
