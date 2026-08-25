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


def _edge_anchor_search(
    profile: np.ndarray,
    segment_profiles: np.ndarray,
    *,
    current: float,
    scale: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Search one physical edge independently of its opposite edge.

    A valid printed boundary must be supported by many disjoint segments of the
    long side.  This deliberately gives less weight to one locally strong peak,
    which is the common failure mode around logos, title ornaments and artwork.
    """

    radius = float(config.get("anchor_search_radius_px", 12.0)) * scale
    step = max(float(config.get("anchor_step_px", 0.5)) * scale, 0.5)
    movement_penalty = float(config.get("anchor_movement_penalty", 0.012))
    support_threshold = float(config.get("segment_support_threshold", 0.35))
    min_support = int(config.get("single_anchor_min_supporting_segments", 6))
    min_robust_z = float(config.get("single_anchor_min_robust_z", 1.6))
    min_q30 = float(config.get("single_anchor_min_segment_q30", 0.10))
    profile_z = _robust_z(profile)
    offsets = np.arange(-radius, radius + 0.5 * step, step, dtype=np.float32)
    candidates: list[dict[str, Any]] = []
    for offset in offsets:
        position = float(current) + float(offset)
        if position < 1.0 or position > profile.size - 2.0:
            continue
        aggregate = _sample(profile, position)
        robust_z = _sample(profile_z, position)
        support, q30 = _support(segment_profiles, position, support_threshold)
        canonical_move = float(offset) / max(scale, 1e-6)
        # Support across separate long-side segments is more reliable than a
        # single gradient magnitude.  The small movement term only breaks ties.
        objective = (
            aggregate
            + 0.24 * q30
            + 0.10 * support
            + 0.10 * robust_z
            - movement_penalty * canonical_move * canonical_move
        )
        candidates.append(
            {
                "position": position,
                "objective": objective,
                "aggregate_score": aggregate,
                "robust_z": robust_z,
                "supporting_segments": support,
                "segment_q30": q30,
                "movement_from_model_px": canonical_move,
            }
        )
    if not candidates:
        return {"trusted": False, "reason": "no_valid_anchor_candidate"}
    selected = max(candidates, key=lambda value: float(value["objective"]))
    trusted = bool(
        int(selected["supporting_segments"]) >= min_support
        and float(selected["robust_z"]) >= min_robust_z
        and float(selected["segment_q30"]) >= min_q30
    )
    trust_score = (
        0.40 * float(selected["robust_z"])
        + 0.35 * float(selected["segment_q30"])
        + 0.25 * float(selected["supporting_segments"])
    )
    return {
        "trusted": trusted,
        "reason": "long_edge_anchor_supported" if trusted else "anchor_support_below_gate",
        "selected": selected,
        "trust_score": trust_score,
        "candidate_count": len(candidates),
        "thresholds": {
            "min_supporting_segments": min_support,
            "min_robust_z": min_robust_z,
            "min_segment_q30": min_q30,
        },
    }


def _select_axis_from_anchors(
    profile: np.ndarray,
    segment_profiles: np.ndarray,
    *,
    current_first: float,
    current_second: float,
    expected_span: float,
    scale: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Choose paired, one-sided, or conservative centre-locked geometry."""

    first = _edge_anchor_search(
        profile,
        segment_profiles,
        current=current_first,
        scale=scale,
        config=config,
    )
    second = _edge_anchor_search(
        profile,
        segment_profiles,
        current=current_second,
        scale=scale,
        config=config,
    )
    def gate(anchor: Mapping[str, Any], prefix: str) -> bool:
        selected = anchor.get("selected", {})
        return bool(
            int(selected.get("supporting_segments", 0))
            >= int(config.get(f"{prefix}_min_supporting_segments", 10 if prefix == "single_anchor" else 7))
            and float(selected.get("robust_z", 0.0))
            >= float(config.get(f"{prefix}_min_robust_z", 4.0 if prefix == "single_anchor" else 2.5))
            and float(selected.get("segment_q30", 0.0))
            >= float(config.get(f"{prefix}_min_segment_q30", 0.5 if prefix == "single_anchor" else 0.25))
            and abs(float(selected.get("movement_from_model_px", 1e9)))
            <= float(config.get(f"{prefix}_max_anchor_move_px", 4.0))
        )

    first_paired = gate(first, "paired_anchor")
    second_paired = gate(second, "paired_anchor")
    first_single = gate(first, "single_anchor")
    second_single = gate(second, "single_anchor")
    first_weak = gate(first, "weak_anchor")
    second_weak = gate(second, "weak_anchor")
    first_position = float(first.get("selected", {}).get("position", current_first))
    second_position = float(second.get("selected", {}).get("position", current_second))
    max_pair_error = float(config.get("paired_anchor_max_span_error_px", 4.0)) * scale
    max_anchor_move = float(config.get("single_anchor_max_inferred_move_px", 48.0)) * scale
    trust_margin = float(config.get("single_anchor_min_trust_margin", 1.0))
    allow_single = bool(config.get("single_anchor_enabled", True))
    span_error = (second_position - first_position) - expected_span

    selected_first: float | None = None
    selected_second: float | None = None
    decision = ""
    anchor_edge: str | None = None
    inferred_edge: str | None = None

    if first_paired and second_paired and abs(span_error) <= max_pair_error:
        center = 0.5 * (first_position + second_position)
        selected_first = center - 0.5 * expected_span
        selected_second = center + 0.5 * expected_span
        decision = "paired_anchor_consensus"
    elif allow_single and first_single and not second_weak:
        inferred = first_position + expected_span
        score_margin = float(first.get("trust_score", 0.0)) - float(second.get("trust_score", 0.0))
        if abs(inferred - current_second) <= max_anchor_move and score_margin >= trust_margin:
            selected_first, selected_second = first_position, inferred
            decision, anchor_edge, inferred_edge = "single_first_anchor", "first", "second"
    elif allow_single and second_single and not first_weak:
        inferred = second_position - expected_span
        score_margin = float(second.get("trust_score", 0.0)) - float(first.get("trust_score", 0.0))
        if abs(inferred - current_first) <= max_anchor_move and score_margin >= trust_margin:
            selected_first, selected_second = inferred, second_position
            decision, anchor_edge, inferred_edge = "single_second_anchor", "second", "first"
    if selected_first is None or selected_second is None:
        if not bool(config.get("fallback_center_lock", True)):
            return {
                "accepted": False,
                "reason": "anchor_ambiguity_without_safe_fallback",
                "first_anchor": first,
                "second_anchor": second,
                "independent_span_error_px": span_error / max(scale, 1e-6),
            }
        center = 0.5 * (float(current_first) + float(current_second))
        selected_first = center - 0.5 * expected_span
        selected_second = center + 0.5 * expected_span
        decision = "learned_center_physical_span_fallback"

    if selected_first < 1.0 or selected_second > profile.size - 2.0:
        half_span = 0.5 * expected_span
        low_center = 1.0 + half_span
        high_center = float(profile.size - 2.0) - half_span
        if low_center > high_center:
            return {
                "accepted": False,
                "reason": "physical_span_larger_than_image",
                "first_anchor": first,
                "second_anchor": second,
                "independent_span_error_px": span_error / max(scale, 1e-6),
            }
        bounded_center = float(
            np.clip(0.5 * (float(current_first) + float(current_second)), low_center, high_center)
        )
        selected_first = bounded_center - half_span
        selected_second = bounded_center + half_span
        decision = "bounded_learned_center_physical_span_fallback"
        anchor_edge = None
        inferred_edge = None
    return {
        "accepted": True,
        "reason": decision,
        "selected": {"first": selected_first, "second": selected_second},
        "anchor_edge": anchor_edge,
        "inferred_edge": inferred_edge,
        "first_anchor": first,
        "second_anchor": second,
        "gates": {
            "first_paired": first_paired,
            "second_paired": second_paired,
            "first_single": first_single,
            "second_single": second_single,
            "first_weak": first_weak,
            "second_weak": second_weak,
        },
        "independent_span_error_px": span_error / max(scale, 1e-6),
        "expected_span": expected_span,
    }


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
    horizontal = _select_axis_from_anchors(
        x_profile,
        x_segments,
        current_first=scaled["left"],
        current_second=scaled["right"],
        expected_span=expected_width,
        scale=scale_x,
        config=config,
    )
    vertical = _select_axis_from_anchors(
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
