from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .outer_detection import order_points
from .outer_pose_detection import validate_and_order_outer_keypoints


RECOVERY_VERSION = "outer_boundary_contact_recovery_20260816_v1"
SIDE_NAMES = ("top", "right", "bottom", "left")


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


def _trimmed_mean(values: np.ndarray, trim_fraction: float = 0.15) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float32).reshape(-1))
    cut = int(round(float(trim_fraction) * len(ordered)))
    if cut > 0 and len(ordered) > 2 * cut:
        ordered = ordered[cut:-cut]
    return float(np.mean(ordered)) if len(ordered) else 0.0


def _sample_lab(
    lab: np.ndarray,
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = lab.shape[:2]
    x = np.asarray(coordinates[:, 0], dtype=np.float32)
    y = np.asarray(coordinates[:, 1], dtype=np.float32)
    valid = (x >= 0.0) & (y >= 0.0) & (x <= width - 1.0) & (y <= height - 1.0)
    sampled = cv2.remap(
        lab,
        x.reshape(1, -1),
        y.reshape(1, -1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )[0]
    return sampled, valid


def _side_contact_threshold(side_index: int, image_shape: Sequence[int]) -> float:
    height, width = int(image_shape[0]), int(image_shape[1])
    dimension = height if side_index in (0, 2) else width
    return max(2.0, 0.0025 * float(dimension))


def _side_border_distances(
    ordered: np.ndarray,
    image_shape: Sequence[int],
) -> tuple[float, float, float, float]:
    height, width = int(image_shape[0]), int(image_shape[1])
    return (
        float(max(ordered[0, 1], ordered[1, 1])),
        float(max(width - 1.0 - ordered[1, 0], width - 1.0 - ordered[2, 0])),
        float(max(height - 1.0 - ordered[2, 1], height - 1.0 - ordered[3, 1])),
        float(max(ordered[3, 0], ordered[0, 0])),
    )


def _find_inward_edge(
    lab: np.ndarray,
    ordered: np.ndarray,
    side_index: int,
    *,
    max_search_ratio: float = 0.09,
) -> dict[str, Any]:
    start = ordered[side_index]
    end = ordered[(side_index + 1) % 4]
    side_vector = end - start
    side_length = float(np.linalg.norm(side_vector))
    center = ordered.mean(axis=0)
    inward = -_outward_normal(start, end, center)
    side_lengths = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
    short_side = max(float(np.min(side_lengths)), 1.0)
    gap = float(np.clip(round(short_side / 350.0), 2.0, 6.0))
    max_search = int(round(max(12.0, max_search_ratio * short_side)))
    along = np.linspace(0.12, 0.88, 420, dtype=np.float32)
    base = start[None, :] + along[:, None] * side_vector[None, :]

    candidates: list[dict[str, float]] = []
    for offset in range(int(gap) + 1, max_search + 1):
        center_line = base + float(offset) * inward[None, :]
        outside, outside_valid = _sample_lab(lab, center_line - gap * inward[None, :])
        inside, inside_valid = _sample_lab(lab, center_line + gap * inward[None, :])
        valid = outside_valid & inside_valid
        coverage = float(np.mean(valid))
        if coverage < 0.85:
            continue
        delta = np.linalg.norm(inside[valid] - outside[valid], axis=1)
        candidates.append(
            {
                "offset_px": float(offset),
                "score": _trimmed_mean(delta),
                "median_delta": float(np.median(delta)),
                "support_ratio": float(np.mean(delta >= 10.0)),
                "coverage": coverage,
            }
        )

    if not candidates:
        return {"accepted": False, "reason": "no_valid_scan", "side": SIDE_NAMES[side_index]}

    selected = max(candidates, key=lambda item: item["score"])
    scores = np.asarray([item["score"] for item in candidates], dtype=np.float32)
    prominence = float(selected["score"] - np.median(scores))
    accepted = bool(
        selected["score"] >= 14.0
        and selected["median_delta"] >= 9.0
        and selected["support_ratio"] >= 0.58
        and prominence >= 5.0
        and selected["offset_px"] < float(max_search - 1)
    )
    return {
        "accepted": accepted,
        "reason": None if accepted else "insufficient_long_edge_evidence",
        "side": SIDE_NAMES[side_index],
        "inward_normal": inward.round(6).tolist(),
        "side_length_px": side_length,
        "gap_px": gap,
        "max_search_px": max_search,
        "peak_prominence": prominence,
        **selected,
    }


def recover_boundary_contact_outer(
    image: np.ndarray,
    prediction: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Recover a high-confidence silhouette clipped to an image boundary.

    This does not relax production geometry validation. It only replaces a
    side when both of its endpoints touch the image boundary and a strong,
    long physical edge is independently observed farther inside the image.
    The recovered quadrilateral must then pass the unchanged validator.
    """

    original = dict(prediction)
    metrics = dict(original.get("metrics", {}))
    audit: dict[str, Any] = {
        "version": RECOVERY_VERSION,
        "triggered": False,
        "accepted": False,
        "reason": "not_eligible",
        "sides": [],
    }
    if (
        not isinstance(image, np.ndarray)
        or image.size == 0
        or image.ndim != 3
        or str(original.get("error_code") or "") != "INVALID_KEYPOINT_GEOMETRY"
        or original.get("points") is None
        or float(original.get("confidence") or 0.0) < 0.75
        or not bool(metrics.get("silhouette_detected", False))
        or float(metrics.get("aspect_ratio_error", 1.0)) > 0.12
        or not (0.30 <= float(metrics.get("area_ratio", 0.0)) <= 0.95)
    ):
        metrics["boundary_contact_recovery"] = audit
        original["metrics"] = metrics
        return original

    try:
        ordered = order_points(original["points"]).astype(np.float32)
    except (TypeError, ValueError):
        metrics["boundary_contact_recovery"] = audit
        original["metrics"] = metrics
        return original

    distances = _side_border_distances(ordered, image.shape)
    touching = [
        index
        for index, distance in enumerate(distances)
        if distance <= _side_contact_threshold(index, image.shape)
    ]
    audit["triggered"] = bool(touching)
    audit["touching_sides"] = [SIDE_NAMES[index] for index in touching]
    if not touching or len(touching) > 2:
        audit["reason"] = "unsupported_boundary_contact_count"
        metrics["boundary_contact_recovery"] = audit
        original["metrics"] = metrics
        return original

    short_side = max(
        float(np.min(np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1))),
        1.0,
    )
    sigma = float(np.clip(short_side / 500.0, 1.2, 3.0))
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
    recovered = ordered.copy()
    accepted_sides = 0
    for side_index in touching:
        evidence = _find_inward_edge(lab, ordered, side_index)
        audit["sides"].append(evidence)
        if not bool(evidence.get("accepted")):
            continue
        inward = np.asarray(evidence["inward_normal"], dtype=np.float32)
        offset = float(evidence["offset_px"])
        recovered[side_index] += offset * inward
        recovered[(side_index + 1) % 4] += offset * inward
        accepted_sides += 1

    if accepted_sides != len(touching):
        audit["reason"] = "not_all_touching_sides_recovered"
        metrics["boundary_contact_recovery"] = audit
        original["metrics"] = metrics
        return original

    validation = validate_and_order_outer_keypoints(
        recovered,
        original.get("keypoint_confidence"),
        image.shape,
        config,
    )
    accepted_points = validation.get("ordered_points")
    audit["validation"] = {
        "success": bool(validation.get("success")),
        "error_code": validation.get("error_code"),
        "metrics": dict(validation.get("metrics", {})),
    }
    if not validation.get("success") or accepted_points is None:
        audit["reason"] = "recovered_geometry_rejected"
        metrics["boundary_contact_recovery"] = audit
        original["metrics"] = metrics
        return original

    accepted_array = np.asarray(accepted_points, dtype=np.float32)
    audit["accepted"] = True
    audit["reason"] = "long_edge_relocalized_and_revalidated"
    audit["original_points"] = ordered.round(3).tolist()
    audit["recovered_points"] = accepted_array.round(3).tolist()
    metrics.update(dict(validation.get("metrics", {})))
    metrics["raw_points"] = accepted_array.round(3).tolist()
    metrics["boundary_contact_recovery"] = audit
    return {
        **original,
        "success": True,
        "points": accepted_array.round(3).tolist(),
        "bbox": [
            float(accepted_array[:, 0].min()),
            float(accepted_array[:, 1].min()),
            float(accepted_array[:, 0].max()),
            float(accepted_array[:, 1].max()),
        ],
        "error_code": None,
        "message": (
            "Boundary-clipped outer silhouette recovered from a strong "
            "full-resolution physical card edge."
        ),
        "metrics": metrics,
    }

