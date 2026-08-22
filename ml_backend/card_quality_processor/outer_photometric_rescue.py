from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np

from .outer_detection import order_points
from .outer_shadow_risk import assess_outer_shadow_risk


RESCUE_VERSION = "outer_photometric_two_vote_20260812_v1"


def should_trigger_photometric_rescue(
    prediction: dict[str, Any],
    image: np.ndarray | None = None,
) -> bool:
    """Narrow, audit-derived trigger for expensive multi-view inference."""

    metrics = prediction.get("metrics", {})
    physical = metrics.get("physical_edge_refinement", {})
    physical_metrics = physical.get("metrics", {})
    reason = str(physical.get("reason") or "")
    shadow_risk = (
        assess_outer_shadow_risk(image, prediction.get("points"))
        if image is not None
        else {"high_risk": False}
    )
    return bool(
        float(prediction.get("confidence") or 0.0) < 0.82
        or (
            reason == "ambiguous_side_peak"
            and float(physical_metrics.get("min_side_peak_margin", float("inf"))) < 2.0
        )
        or (
            reason == "refinement_guard_rejected"
            and float(physical_metrics.get("max_corner_movement_ratio", 0.0)) > 0.05
        )
        or bool(shadow_risk.get("high_risk", False))
    )


def photometric_views(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(luminance)
    clahe_image = cv2.cvtColor(
        cv2.merge((clahe, channel_a, channel_b)), cv2.COLOR_LAB2BGR
    )

    median = float(np.median(luminance)) / 255.0
    median = float(np.clip(median, 0.08, 0.92))
    gamma = float(np.clip(np.log(0.50) / np.log(median), 0.55, 1.80))
    lookup = np.asarray(
        [np.clip(((value / 255.0) ** gamma) * 255.0, 0, 255) for value in range(256)],
        dtype=np.uint8,
    )
    gamma_image = cv2.LUT(image, lookup)
    return [("identity", image), ("clahe_l", clahe_image), ("adaptive_gamma", gamma_image)]


def rescue_outer_prediction(
    predict: Callable[..., dict[str, Any]],
    image: np.ndarray,
    identity_prediction: dict[str, Any],
    *,
    conf: float = 0.25,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    """Use normalized views only when they form a strict 2-vote consensus.

    A disagreement without two normalized votes is surfaced as review evidence;
    it never replaces the production identity coordinates.
    """

    shadow_risk = assess_outer_shadow_risk(image, identity_prediction.get("points"))
    if not should_trigger_photometric_rescue(identity_prediction, image):
        return identity_prediction, image, {
            "version": RESCUE_VERSION,
            "triggered": False,
            "selected_view": "identity_no_rescue",
            "decision": "not_triggered",
            "requires_review": False,
            "max_pair_disagreement_ratio": 0.0,
            "shadow_boundary_risk": shadow_risk,
        }

    candidates: list[dict[str, Any]] = []
    for view_name, view in photometric_views(image):
        prediction = (
            identity_prediction
            if view_name == "identity"
            else predict(view, conf=conf)
        )
        if not prediction.get("success") or not prediction.get("points"):
            continue
        candidates.append(
            {
                "view_name": view_name,
                "view": view,
                "prediction": prediction,
                "points": order_points(
                    np.asarray(prediction["points"], dtype=np.float32)
                ),
                "confidence": float(prediction.get("confidence") or 0.0),
            }
        )

    by_name = {candidate["view_name"]: index for index, candidate in enumerate(candidates)}
    required = ("identity", "clahe_l", "adaptive_gamma")
    if any(name not in by_name for name in required):
        return identity_prediction, image, {
            "version": RESCUE_VERSION,
            "triggered": True,
            "selected_view": "identity",
            "decision": "identity_kept_missing_view",
            "requires_review": True,
            "candidate_count": len(candidates),
            "max_pair_disagreement_ratio": None,
            "shadow_boundary_risk": shadow_risk,
        }

    diagonal = max(float(np.hypot(image.shape[1], image.shape[0])), 1.0)
    distances = np.zeros((len(candidates), len(candidates)), dtype=np.float32)
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            value = float(
                np.linalg.norm(
                    candidates[left]["points"] - candidates[right]["points"], axis=1
                ).mean()
                / diagonal
            )
            distances[left, right] = value
            distances[right, left] = value

    identity_index = by_name["identity"]
    clahe_index = by_name["clahe_l"]
    gamma_index = by_name["adaptive_gamma"]
    normalized_pair = float(distances[clahe_index, gamma_index])
    identity_to_normalized = min(
        float(distances[identity_index, clahe_index]),
        float(distances[identity_index, gamma_index]),
    )
    max_disagreement = float(distances.max())
    strong_two_vote = bool(
        normalized_pair <= 0.006 and identity_to_normalized >= 0.012
    )
    if strong_two_vote:
        normalized_candidates = (candidates[clahe_index], candidates[gamma_index])
        selected = max(normalized_candidates, key=lambda candidate: candidate["confidence"])
        decision = "consensus_selected"
        requires_review = False
    else:
        selected = candidates[identity_index]
        decision = (
            "identity_kept_all_views_agree"
            if max_disagreement < 0.012
            else "identity_kept_no_two_vote_consensus"
        )
        requires_review = max_disagreement >= 0.012

    return selected["prediction"], selected["view"], {
        "version": RESCUE_VERSION,
        "triggered": True,
        "selected_view": selected["view_name"],
        "decision": decision,
        "requires_review": requires_review,
        "candidate_count": len(candidates),
        "normalized_pair_disagreement_ratio": normalized_pair,
        "identity_to_normalized_disagreement_ratio": identity_to_normalized,
        "max_pair_disagreement_ratio": max_disagreement,
        "view_confidences": {
            candidate["view_name"]: candidate["confidence"] for candidate in candidates
        },
        "shadow_boundary_risk": shadow_risk,
    }
