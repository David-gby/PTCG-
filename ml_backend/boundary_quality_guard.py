from __future__ import annotations

from typing import Any, Mapping


def assess_outer_quality(outer: Mapping[str, Any]) -> dict[str, Any]:
    metrics = outer.get("metrics", {}) if isinstance(outer, Mapping) else {}
    refinement = metrics.get("outer_line_refinement", {}) if isinstance(metrics, Mapping) else {}
    photometric_rescue = (
        metrics.get("photometric_rescue", {}) if isinstance(metrics, Mapping) else {}
    )
    photometric_rescue = (
        photometric_rescue if isinstance(photometric_rescue, Mapping) else {}
    )
    selected = str(refinement.get("selected_source") or "")
    learned_allowed = bool(refinement.get("learned_allowed", False))
    learned = refinement.get("learned_metrics", {})
    learned = learned if isinstance(learned, Mapping) else {}
    reasons: list[str] = []
    severity = "normal"

    # On the frozen feedback benchmark every safe-legacy selection was a large
    # error (12.4--72.7 px corner MAE).  Keep the output for compatibility, but
    # make the uncertainty explicit so it is routed to review.
    if selected.startswith("safe_legacy_refinement") or selected.startswith(
        "production_safe_legacy_fallback"
    ):
        reasons.append("legacy_fallback_after_learned_rejection")
        severity = "high"
    elif (
        selected.startswith("raw_silhouette")
        or selected.startswith("production_raw_fallback")
    ) and not learned_allowed:
        reasons.append("raw_silhouette_after_learned_rejection")
        severity = "high"

    max_offset = float(learned.get("max_canonical_offset_px", 0.0) or 0.0)
    min_confidence = float(learned.get("min_side_confidence", 1.0) or 0.0)
    area_ratio = float(metrics.get("area_ratio", 1.0) or 0.0)
    if max_offset > 16.0:
        reasons.append("large_learned_boundary_disagreement")
        severity = "high"
    if min_confidence < 0.10:
        reasons.append("low_learned_side_confidence")
        severity = "high"
    # The frozen production-feedback set contains a separate failure mode in
    # which the upstream mask locks onto a small interior rectangle.  Its
    # learned four-side refinement looks locally confident, so disagreement
    # gates alone cannot detect it.  A card occupying <20% of the uploaded
    # image was isolated to the two worst small-mask cases on that benchmark.
    # Route these cases to review instead of silently returning a precise but
    # semantically wrong quadrilateral.
    if area_ratio < 0.20:
        reasons.append("implausibly_small_card_area")
        severity = "high"
    # A photometric retry is allowed to replace coordinates only when the two
    # independently normalized views agree.  If the views disagree, retain the
    # identity geometry and surface the uncertainty instead of guessing.
    if bool(photometric_rescue.get("requires_review", False)):
        reasons.append("photometric_view_disagreement")
        severity = "high"

    shadow_risk = photometric_rescue.get("shadow_boundary_risk", {})
    shadow_risk = shadow_risk if isinstance(shadow_risk, Mapping) else {}

    return {
        "review_recommended": severity == "high",
        "severity": severity,
        "reasons": reasons,
        "selected_source": selected or None,
        "photometric_rescue_decision": photometric_rescue.get("decision"),
        "shadow_boundary_risk": bool(shadow_risk.get("high_risk", False)),
        "shadow_boundary_risk_score": shadow_risk.get("risk_score"),
        "policy": "outer_boundary_quality_guard_20260812_v4",
    }


def assess_inner_quality(inner: Mapping[str, Any]) -> dict[str, Any]:
    refinement = inner.get("edge_refinement", {}) if isinstance(inner, Mapping) else {}
    global_hypotheses = inner.get("global_edge_hypotheses", {}) if isinstance(inner, Mapping) else {}
    reasons: list[str] = []
    edges: list[str] = []
    for edge in ("left", "top", "right", "bottom"):
        details = refinement.get(edge, {}) if isinstance(refinement, Mapping) else {}
        if not isinstance(details, Mapping):
            continue
        proposed = abs(float(details.get("proposed_offset", 0.0) or 0.0))
        confidence = float(details.get("confidence", 0.0) or 0.0)
        accepted = bool(details.get("accepted", False))
        hypothesis = global_hypotheses.get(edge, {}) if isinstance(global_hypotheses, Mapping) else {}
        ambiguous = bool(hypothesis.get("ambiguous", False)) if isinstance(hypothesis, Mapping) else False
        rescued = bool(details.get("global_consensus_rescue", False))
        # A classical image can contain many legitimate parallel print lines.
        # It becomes actionable only when the selected learned refiner points
        # to the same distant line.  This keeps review volume low while still
        # catching the gate-blocked large corrections seen in production.
        if edge in ("left", "top") and ambiguous and not rescued:
            base = float(hypothesis.get("base_position", 0.0) or 0.0)
            alternative = float(hypothesis.get("alternative_position", base) or base)
            learned_target = base + float(details.get("proposed_offset", 0.0) or 0.0)
            if (
                proposed >= 5.0
                and confidence >= 0.30
                and abs(learned_target - alternative) <= 4.0
            ):
                reasons.append(f"{edge}:learned_and_long_edge_disagree_with_base")
                edges.append(edge)
        if proposed > 8.0 and confidence >= 0.45 and not accepted and not rescued:
            reasons.append(f"{edge}:large_refiner_disagreement")
            edges.append(edge)
        # Historical replay isolated a high-precision bottom failure signature:
        # the coarse/base line stays too low while the learned strip refiner
        # proposes an inward correction of at least 2 px, but its confidence is
        # just below the acceptance gate.  Do not guess a coordinate; route the
        # image for review until the new feedback batch can train this case.
        if (
            edge == "bottom"
            and not accepted
            and not rescued
            and float(details.get("proposed_offset", 0.0) or 0.0) <= -2.0
        ):
            reasons.append("bottom:rejected_inward_correction_may_leave_line_too_low")
            edges.append(edge)

    unique_edges = list(dict.fromkeys(edges))
    return {
        "review_recommended": bool(unique_edges),
        "severity": "high" if unique_edges else "normal",
        "edges": unique_edges,
        "reasons": reasons,
        "policy": "inner_boundary_quality_guard_20260812_v2",
    }
