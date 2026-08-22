from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence


EDGES = ("left", "right", "top", "bottom")


@dataclass(frozen=True)
class PhysicalPriorEvidence:
    edge: str
    position: float
    aggregate_score: float
    robust_z: float
    supporting_segments: int
    total_segments: int

    @classmethod
    def from_mapping(cls, edge: str, value: Mapping[str, Any]) -> "PhysicalPriorEvidence":
        return cls(
            edge=edge,
            position=float(value.get("position", 0.0)),
            aggregate_score=float(value.get("aggregate_score", 0.0)),
            robust_z=float(value.get("robust_z", 0.0)),
            supporting_segments=int(value.get("supporting_segments", 0)),
            total_segments=int(value.get("total_segments", 0)),
        )


EvidenceProvider = Callable[
    [str, Sequence[float], Mapping[str, float]],
    Sequence[Mapping[str, Any]],
]


def _copy_box(box: Mapping[str, float]) -> dict[str, float]:
    return {edge: float(box[edge]) for edge in EDGES}


def _scaled_config(
    config: Mapping[str, Any],
    axis: str,
    scale: float,
) -> dict[str, float]:
    axis_config = config.get(axis, {})
    if not isinstance(axis_config, Mapping):
        axis_config = {}
    return {
        "threshold_px": float(axis_config.get("threshold_px", 12.0)) * scale,
        "max_residual_px": float(axis_config.get("max_residual_px", 32.0)) * scale,
        "alpha": float(axis_config.get("alpha", 0.10)),
        "first_share": float(axis_config.get("first_share", 0.50)),
        "max_visual_score_drop": float(axis_config.get("max_visual_score_drop", 0.10)),
        "max_support_drop": float(axis_config.get("max_support_drop", 1.0)),
        "min_candidate_support": float(axis_config.get("min_candidate_support", 4.0)),
    }


def assess_physical_inner_box(
    box: Mapping[str, float],
    width: int,
    height: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_width = float(width) * float(config.get("inner_width_mm", 58.0)) / float(
        config.get("outer_width_mm", 63.0)
    )
    expected_height = float(height) * float(config.get("inner_height_mm", 83.0)) / float(
        config.get("outer_height_mm", 88.0)
    )
    current_width = float(box["right"]) - float(box["left"])
    current_height = float(box["bottom"]) - float(box["top"])
    width_residual = current_width - expected_width
    height_residual = current_height - expected_height
    horizontal = _scaled_config(config, "horizontal", float(width) / 630.0)
    vertical = _scaled_config(config, "vertical", float(height) / 880.0)
    severe = bool(
        abs(width_residual) > horizontal["max_residual_px"]
        or abs(height_residual) > vertical["max_residual_px"]
    )
    moderate = bool(
        abs(width_residual) > horizontal["threshold_px"]
        or abs(height_residual) > vertical["threshold_px"]
    )
    return {
        "version": str(config.get("version", "physical_inner_prior_v1")),
        "measurement_semantics": "printed_inner_line_inner_edge",
        "applies_to_all_layouts": bool(config.get("applies_to_all_layouts", True)),
        "expected_size_mm": {
            "width": float(config.get("inner_width_mm", 58.0)),
            "height": float(config.get("inner_height_mm", 83.0)),
        },
        "expected_size_px": {
            "width": round(expected_width, 4),
            "height": round(expected_height, 4),
        },
        "observed_size_px": {
            "width": round(current_width, 4),
            "height": round(current_height, 4),
        },
        "residual_px": {
            "width": round(width_residual, 4),
            "height": round(height_residual, 4),
        },
        "residual_mm": {
            "width": round(width_residual * float(config.get("outer_width_mm", 63.0)) / width, 5),
            "height": round(height_residual * float(config.get("outer_height_mm", 88.0)) / height, 5),
        },
        "risk": "severe" if severe else "moderate" if moderate else "normal",
        "review_recommended": severe,
    }


def _candidate_axis(
    box: Mapping[str, float],
    *,
    first: str,
    second: str,
    expected: float,
    settings: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    current = _copy_box(box)
    current_size = current[second] - current[first]
    correction = float(settings["alpha"]) * (expected - current_size)
    candidate = dict(current)
    candidate[first] -= float(settings["first_share"]) * correction
    candidate[second] += (1.0 - float(settings["first_share"])) * correction
    return candidate, {
        "current_size_px": current_size,
        "expected_size_px": expected,
        "residual_before_px": current_size - expected,
        "residual_after_px": candidate[second] - candidate[first] - expected,
        "correction_px": correction,
    }


def _visual_gate(
    current: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    edges: tuple[str, str],
    settings: Mapping[str, float],
    evidence_provider: EvidenceProvider | None,
) -> dict[str, Any]:
    moved = [edge for edge in edges if abs(candidate[edge] - current[edge]) >= 0.05]
    if not moved:
        return {"accepted": False, "reason": "no_coordinate_change", "edges": {}}
    if evidence_provider is None:
        return {
            "accepted": False,
            "reason": "visual_evidence_unavailable",
            "edges": {},
        }
    evidence: dict[str, Any] = {}
    accepted = True
    for edge in moved:
        values = evidence_provider(edge, [current[edge], candidate[edge]], candidate)
        if len(values) != 2:
            accepted = False
            evidence[edge] = {"accepted": False, "reason": "invalid_evidence_count"}
            continue
        current_evidence = PhysicalPriorEvidence.from_mapping(edge, values[0])
        candidate_evidence = PhysicalPriorEvidence.from_mapping(edge, values[1])
        score_drop = current_evidence.aggregate_score - candidate_evidence.aggregate_score
        support_drop = current_evidence.supporting_segments - candidate_evidence.supporting_segments
        edge_accepted = bool(
            score_drop <= float(settings["max_visual_score_drop"])
            and support_drop <= int(settings["max_support_drop"])
            and candidate_evidence.supporting_segments
            >= int(settings["min_candidate_support"])
        )
        accepted = accepted and edge_accepted
        evidence[edge] = {
            "accepted": edge_accepted,
            "score_drop": round(score_drop, 5),
            "support_drop": support_drop,
            "current": asdict(current_evidence),
            "candidate": asdict(candidate_evidence),
        }
    return {
        "accepted": accepted,
        "reason": "visual_consensus" if accepted else "visual_gate_rejected",
        "edges": evidence,
    }


def guarded_refine_physical_inner_box(
    box: Mapping[str, float],
    width: int,
    height: int,
    config: Mapping[str, Any],
    *,
    evidence_provider: EvidenceProvider | None = None,
) -> dict[str, Any]:
    current = _copy_box(box)
    before = assess_physical_inner_box(current, width, height, config)
    if not bool(config.get("enabled", True)):
        return {
            "box": current,
            "applied": False,
            "reason": "disabled",
            "before": before,
            "after": before,
            "axes": {},
        }

    output = dict(current)
    expected_width = float(before["expected_size_px"]["width"])
    expected_height = float(before["expected_size_px"]["height"])
    axis_specs = (
        (
            "horizontal",
            "left",
            "right",
            expected_width,
            _scaled_config(config, "horizontal", float(width) / 630.0),
        ),
        (
            "vertical",
            "top",
            "bottom",
            expected_height,
            _scaled_config(config, "vertical", float(height) / 880.0),
        ),
    )
    audit: dict[str, Any] = {}
    applied_axes: list[str] = []
    for axis, first, second, expected, settings in axis_specs:
        current_size = output[second] - output[first]
        residual = current_size - expected
        if abs(residual) <= settings["threshold_px"]:
            audit[axis] = {
                "applied": False,
                "reason": "within_deadband",
                "residual_px": round(residual, 4),
            }
            continue
        if abs(residual) > settings["max_residual_px"]:
            audit[axis] = {
                "applied": False,
                "reason": "residual_too_large_outer_rectification_suspected",
                "residual_px": round(residual, 4),
            }
            continue
        candidate, geometry = _candidate_axis(
            output,
            first=first,
            second=second,
            expected=expected,
            settings=settings,
        )
        visual = _visual_gate(
            output,
            candidate,
            edges=(first, second),
            settings=settings,
            evidence_provider=evidence_provider,
        )
        axis_applied = bool(visual["accepted"])
        if axis_applied:
            output = candidate
            applied_axes.append(axis)
        audit[axis] = {
            "applied": axis_applied,
            "reason": visual["reason"],
            "geometry": {key: round(float(value), 4) for key, value in geometry.items()},
            "visual": visual,
        }
    after = assess_physical_inner_box(output, width, height, config)
    return {
        "box": output,
        "applied": bool(applied_axes),
        "applied_axes": applied_axes,
        "reason": "evidence_guarded_soft_constraint" if applied_axes else "no_safe_correction",
        "before": before,
        "after": after,
        "axes": audit,
    }
