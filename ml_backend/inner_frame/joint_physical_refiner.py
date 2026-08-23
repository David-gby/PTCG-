from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np

from .global_boundary_hypothesis import _feature_maps, _robust_z, _segmented_profile
from .physical_inner_prior import assess_physical_inner_box


EDGES = ("left", "right", "top", "bottom")


def _sample(values: np.ndarray, position: float) -> float:
    if values.size == 0:
        return 0.0
    position = float(np.clip(position, 0.0, float(values.size - 1)))
    low = int(np.floor(position))
    high = min(low + 1, values.size - 1)
    fraction = position - low
    return float(values[low] * (1.0 - fraction) + values[high] * fraction)


def _support(matrix: np.ndarray, position: float, threshold: float) -> tuple[int, float]:
    values = np.asarray([_sample(row, position) for row in matrix], dtype=np.float32)
    return int(np.count_nonzero(values >= threshold)), float(np.quantile(values, 0.30))


def _axis_profiles(
    feature: np.ndarray,
    box: Mapping[str, float],
    *,
    horizontal: bool,
    segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    left = int(round(float(box["left"])))
    right = int(round(float(box["right"])))
    top = int(round(float(box["top"])))
    bottom = int(round(float(box["bottom"])))
    if horizontal:
        pad = max(8, int(round((bottom - top) * 0.08)))
        return _segmented_profile(
            feature,
            vertical=True,
            long_low=top + pad,
            long_high=bottom - pad,
            segments=segments,
        )
    pad = max(8, int(round((right - left) * 0.08)))
    return _segmented_profile(
        feature,
        vertical=False,
        long_low=left + pad,
        long_high=right - pad,
        segments=segments,
    )


def _search_axis(
    profile: np.ndarray,
    segment_profiles: np.ndarray,
    *,
    current_first: float,
    current_second: float,
    expected_span: float,
    scale: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    center = 0.5 * (float(current_first) + float(current_second))
    radius = float(config.get("center_search_radius_px", 5.0)) * scale
    step = max(float(config.get("center_step_px", 0.25)) * scale, 0.25)
    movement_penalty = float(config.get("center_movement_penalty", 0.035))
    support_threshold = float(config.get("segment_support_threshold", 0.35))
    min_support = int(config.get("min_supporting_segments", 4))
    band_z = _robust_z(profile)

    offsets = np.arange(-radius, radius + 0.5 * step, step, dtype=np.float32)
    candidates: list[dict[str, Any]] = []
    for offset in offsets:
        candidate_center = center + float(offset)
        first = candidate_center - 0.5 * expected_span
        second = candidate_center + 0.5 * expected_span
        if first < 1.0 or second > profile.size - 2.0:
            continue
        first_score = _sample(profile, first)
        second_score = _sample(profile, second)
        first_z = _sample(band_z, first)
        second_z = _sample(band_z, second)
        first_support, first_q30 = _support(segment_profiles, first, support_threshold)
        second_support, second_q30 = _support(segment_profiles, second, support_threshold)
        # Long-side consensus receives more weight than a single sharp local
        # gradient. This suppresses logos, title bars and isolated artwork.
        visual = (
            first_score
            + second_score
            + 0.20 * (first_q30 + second_q30)
            + 0.08 * (first_support + second_support)
        )
        canonical_move = float(offset) / max(scale, 1e-6)
        objective = visual - movement_penalty * canonical_move * canonical_move
        candidates.append(
            {
                "center": candidate_center,
                "first": first,
                "second": second,
                "objective": objective,
                "visual_score": visual,
                "first_score": first_score,
                "second_score": second_score,
                "first_robust_z": first_z,
                "second_robust_z": second_z,
                "first_support": first_support,
                "second_support": second_support,
            }
        )

    if not candidates:
        return {"accepted": False, "reason": "no_valid_joint_candidate"}
    selected = max(candidates, key=lambda value: float(value["objective"]))
    supported = bool(
        int(selected["first_support"]) >= min_support
        and int(selected["second_support"]) >= min_support
    )
    require_support = bool(config.get("require_visual_support", False))
    accepted = bool(supported or not require_support)
    return {
        "accepted": accepted,
        "reason": (
            "paired_long_edge_consensus"
            if supported
            else "physical_span_locked_visual_support_low"
        ),
        "visual_support_passed": supported,
        "selected": selected,
        "candidate_count": len(candidates),
        "current_center": center,
        "expected_span": expected_span,
    }


def refine_trusted_inner_box(
    refinement_image: np.ndarray,
    box: Mapping[str, float],
    canonical_width: int,
    canonical_height: int,
    physical_config: Mapping[str, Any],
    *,
    trusted_outer: bool,
) -> dict[str, Any]:
    """Jointly localize opposite inner edges in a trusted outer-card plane.

    The learned pipeline supplies the semantic centre.  This final stage
    operates on a higher-resolution rectification and treats opposite edges as
    a pair whose separation is the confirmed 58 x 83 mm printed-line inner-edge
    geometry.  Validation showed that unconstrained gradient-based centre
    movement can lock onto card artwork, so production keeps the learned centre
    fixed while using long-edge evidence as an audit signal.  The stage is
    deliberately disabled when the outer coordinate system is not independently
    trusted.
    """

    current = {edge: float(box[edge]) for edge in EDGES}
    before = assess_physical_inner_box(
        current, canonical_width, canonical_height, physical_config
    )
    config = physical_config.get("trusted_joint", {})
    if not trusted_outer or not bool(config.get("enabled", True)):
        return {
            "box": current,
            "applied": False,
            "reason": "outer_geometry_not_trusted" if not trusted_outer else "disabled",
            "before": before,
            "after": before,
            "axes": {},
        }
    if (
        not isinstance(refinement_image, np.ndarray)
        or refinement_image.ndim != 3
        or refinement_image.shape[2] != 3
    ):
        return {
            "box": current,
            "applied": False,
            "reason": "invalid_refinement_image",
            "before": before,
            "after": before,
            "axes": {},
        }

    image_height, image_width = refinement_image.shape[:2]
    scale_x = float(image_width) / max(float(canonical_width), 1.0)
    scale_y = float(image_height) / max(float(canonical_height), 1.0)
    scaled = {
        "left": current["left"] * scale_x,
        "right": current["right"] * scale_x,
        "top": current["top"] * scale_y,
        "bottom": current["bottom"] * scale_y,
    }
    segments = int(config.get("segments", 10))
    feature_x, feature_y = _feature_maps(refinement_image)
    x_profile, x_segments = _axis_profiles(
        feature_x, scaled, horizontal=True, segments=segments
    )
    y_profile, y_segments = _axis_profiles(
        feature_y, scaled, horizontal=False, segments=segments
    )
    expected_width = (
        float(image_width)
        * float(physical_config.get("inner_width_mm", 58.0))
        / float(physical_config.get("outer_width_mm", 63.0))
    )
    expected_height = (
        float(image_height)
        * float(physical_config.get("inner_height_mm", 83.0))
        / float(physical_config.get("outer_height_mm", 88.0))
    )
    horizontal = _search_axis(
        x_profile,
        x_segments,
        current_first=scaled["left"],
        current_second=scaled["right"],
        expected_span=expected_width,
        scale=scale_x,
        config=config,
    )
    vertical = _search_axis(
        y_profile,
        y_segments,
        current_first=scaled["top"],
        current_second=scaled["bottom"],
        expected_span=expected_height,
        scale=scale_y,
        config=config,
    )

    output = dict(current)
    applied_axes: list[str] = []
    if bool(horizontal.get("accepted", False)):
        selected = horizontal["selected"]
        output["left"] = float(selected["first"]) / scale_x
        output["right"] = float(selected["second"]) / scale_x
        applied_axes.append("horizontal")
    if bool(vertical.get("accepted", False)):
        selected = vertical["selected"]
        output["top"] = float(selected["first"]) / scale_y
        output["bottom"] = float(selected["second"]) / scale_y
        applied_axes.append("vertical")

    after = assess_physical_inner_box(
        output, canonical_width, canonical_height, physical_config
    )
    return {
        "box": output,
        "applied": bool(applied_axes),
        "applied_axes": applied_axes,
        "reason": "trusted_outer_highres_joint_refinement" if applied_axes else "no_supported_axis",
        "before": before,
        "after": after,
        "refinement_size": [int(image_width), int(image_height)],
        "canonical_size": [int(canonical_width), int(canonical_height)],
        "axes": {"horizontal": horizontal, "vertical": vertical},
    }
