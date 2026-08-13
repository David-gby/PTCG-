from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import cv2
import numpy as np


EDGES = ("left", "right", "top", "bottom")
EDGE_TO_KEY = {
    "left": "x_left",
    "right": "x_right",
    "top": "y_top",
    "bottom": "y_bottom",
}


@dataclass(frozen=True)
class GlobalEdgeHypothesis:
    edge: str
    base_position: float
    alternative_position: float
    displacement_px: float
    base_score: float
    alternative_score: float
    score_margin: float
    alternative_robust_z: float
    supporting_segments: int
    total_segments: int
    ambiguous: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    median = float(np.median(values))
    spread = float(np.median(np.abs(values - median)) * 1.4826)
    if spread < 1e-4:
        spread = float(np.std(values) + 1e-4)
    return (values - median) / max(spread, 1e-4)


def _smooth(values: np.ndarray, size: int = 5) -> np.ndarray:
    if values.size < size:
        return values.astype(np.float32)
    kernel = np.ones(size, dtype=np.float32) / float(size)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def _feature_maps(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.65)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    scharr_x = np.abs(cv2.Scharr(gray, cv2.CV_32F, 1, 0))
    scharr_y = np.abs(cv2.Scharr(gray, cv2.CV_32F, 0, 1))

    # A printed boundary is normally both sharp and a material/colour change.
    # The colour term is deliberately local so broad cast shadows score lower.
    lab_left = np.roll(lab, 2, axis=1)
    lab_right = np.roll(lab, -2, axis=1)
    lab_up = np.roll(lab, 2, axis=0)
    lab_down = np.roll(lab, -2, axis=0)
    color_x = np.linalg.norm(lab_right - lab_left, axis=2)
    color_y = np.linalg.norm(lab_down - lab_up, axis=2)
    return scharr_x + 8.0 * color_x, scharr_y + 8.0 * color_y


def _segmented_profile(
    feature: np.ndarray,
    *,
    vertical: bool,
    long_low: int,
    long_high: int,
    segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    long_low = max(0, int(long_low))
    long_limit = feature.shape[0] if vertical else feature.shape[1]
    long_high = min(long_limit, int(long_high))
    if long_high - long_low < segments * 4:
        long_low, long_high = 0, long_limit

    boundaries = np.linspace(long_low, long_high, segments + 1).astype(int)
    profiles: list[np.ndarray] = []
    for index in range(segments):
        begin, end = int(boundaries[index]), int(boundaries[index + 1])
        if vertical:
            region = feature[begin:end, :]
            profile = np.quantile(region, 0.48, axis=0)
        else:
            region = feature[:, begin:end]
            profile = np.quantile(region, 0.48, axis=1)
        profiles.append(_smooth(np.asarray(profile, dtype=np.float32)))
    matrix = np.stack(profiles, axis=0)
    normalized = np.stack([_robust_z(row) for row in matrix], axis=0)

    # Median pooling requires support over much of the side; the lower-quantile
    # term suppresses logos and header ornaments that occupy only one segment.
    aggregate = np.median(normalized, axis=0) + 0.35 * np.quantile(
        normalized, 0.30, axis=0
    )
    return aggregate.astype(np.float32), normalized.astype(np.float32)


def _band(edge: str, width: int, height: int) -> tuple[int, int]:
    if edge == "left":
        return int(round(width * 0.008)), int(round(width * 0.090))
    if edge == "right":
        return int(round(width * 0.910)), int(round(width * 0.992))
    if edge == "top":
        return int(round(height * 0.004)), int(round(height * 0.090))
    return int(round(height * 0.900)), int(round(height * 0.996))


def analyze_inner_edge_hypotheses(
    image: np.ndarray,
    box: Mapping[str, float],
    *,
    segments: int = 6,
    min_displacement_px: float = 8.0,
    min_robust_z: float = 2.6,
    min_score_margin: float = 1.5,
    min_supporting_segments: int = 5,
) -> dict[str, GlobalEdgeHypothesis]:
    """Find strong, long-edge alternatives without changing the prediction.

    The routine is an ambiguity detector, not a free-running edge snapper.  It
    deliberately reports alternatives first; a caller may only move an edge
    when independent learned refiners agree with the same alternative.
    """

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a non-empty BGR image")
    height, width = image.shape[:2]
    feature_x, feature_y = _feature_maps(image)
    top = int(round(float(box["y_top"])))
    bottom = int(round(float(box["y_bottom"])))
    left = int(round(float(box["x_left"])))
    right = int(round(float(box["x_right"])))
    long_y_pad = max(4, int(round((bottom - top) * 0.08)))
    long_x_pad = max(4, int(round((right - left) * 0.08)))
    x_profile, x_segments = _segmented_profile(
        feature_x,
        vertical=True,
        long_low=top + long_y_pad,
        long_high=bottom - long_y_pad,
        segments=segments,
    )
    y_profile, y_segments = _segmented_profile(
        feature_y,
        vertical=False,
        long_low=left + long_x_pad,
        long_high=right - long_x_pad,
        segments=segments,
    )

    output: dict[str, GlobalEdgeHypothesis] = {}
    for edge in EDGES:
        key = EDGE_TO_KEY[edge]
        base = float(box[key])
        profile = x_profile if edge in ("left", "right") else y_profile
        segment_profile = x_segments if edge in ("left", "right") else y_segments
        low, high = _band(edge, width, height)
        low = int(np.clip(low, 0, profile.size - 2))
        high = int(np.clip(high, low + 1, profile.size - 1))
        positions = np.arange(low, high + 1)
        base_index = int(np.clip(round(base), low, high))
        exclusion = max(4, int(round((width if edge in ("left", "right") else height) * 0.004)))
        eligible = np.abs(positions - base_index) >= exclusion
        band_scores = profile[low : high + 1].copy()
        if np.any(eligible):
            masked = np.where(eligible, band_scores, -np.inf)
            alternative = int(low + int(np.argmax(masked)))
        else:
            alternative = base_index
        local_base = profile[max(low, base_index - 2) : min(high + 1, base_index + 3)]
        base_score = float(np.max(local_base)) if local_base.size else float(profile[base_index])
        alternative_score = float(profile[alternative])
        band_z = _robust_z(band_scores)
        alternative_z = float(band_z[alternative - low])
        support = int(np.count_nonzero(segment_profile[:, alternative] >= 0.65))
        displacement = float(alternative - base)
        score_margin = float(alternative_score - base_score)
        ambiguous = bool(
            abs(displacement) >= min_displacement_px
            and alternative_z >= min_robust_z
            and score_margin >= min_score_margin
            and support >= min_supporting_segments
        )
        output[edge] = GlobalEdgeHypothesis(
            edge=edge,
            base_position=round(base, 3),
            alternative_position=float(alternative),
            displacement_px=round(displacement, 3),
            base_score=round(base_score, 4),
            alternative_score=round(alternative_score, 4),
            score_margin=round(score_margin, 4),
            alternative_robust_z=round(alternative_z, 4),
            supporting_segments=support,
            total_segments=segments,
            ambiguous=ambiguous,
        )
    return output


def consensus_rescue_decision(
    edge: str,
    *,
    base_position: float,
    stable_position: float,
    stable_confidence: float,
    specialist_position: float | None,
    specialist_confidence: float | None,
    hypothesis: GlobalEdgeHypothesis,
    max_move_px: float = 24.0,
    min_confidence: float = 0.36,
    max_model_disagreement_px: float = 3.5,
    max_hypothesis_distance_px: float = 3.0,
) -> dict[str, Any]:
    """Gate a large correction using two learned views plus line evidence."""

    result: dict[str, Any] = {
        "accepted": False,
        "position": float(base_position),
        "reason": "consensus_not_met",
    }
    if edge not in ("left", "top") or not hypothesis.ambiguous:
        result["reason"] = "no_global_ambiguity"
        return result
    if specialist_position is None or specialist_confidence is None:
        result["reason"] = "specialist_unavailable"
        return result
    if min(float(stable_confidence), float(specialist_confidence)) < min_confidence:
        result["reason"] = "learned_confidence_too_low"
        return result
    if abs(float(stable_position) - float(specialist_position)) > max_model_disagreement_px:
        result["reason"] = "learned_models_disagree"
        return result
    consensus = 0.5 * (float(stable_position) + float(specialist_position))
    if abs(consensus - hypothesis.alternative_position) > max_hypothesis_distance_px:
        result["reason"] = "line_evidence_disagrees"
        return result
    if abs(consensus - float(base_position)) > max_move_px:
        result["reason"] = "movement_too_large"
        return result
    result.update(
        {
            "accepted": True,
            "position": round(consensus, 3),
            "reason": "three_way_consensus",
        }
    )
    return result
