from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


DEFAULT_RATIOS = {
    "left": 0.035,
    "right": 0.965,
    "top": 0.028,
    "bottom": 0.965,
}
MIN_EDGE_EVIDENCE = 0.35


@dataclass(frozen=True)
class StabilizeResult:
    box: dict[str, float]
    status: str
    reason: str
    confidence: float
    offsets: dict[str, float]
    pre_stabilize_box: dict[str, float]
    evidence: dict[str, float]


def _copy_box(box: dict[str, Any]) -> dict[str, float]:
    return {
        "x_left": float(box["x_left"]),
        "x_right": float(box["x_right"]),
        "y_top": float(box["y_top"]),
        "y_bottom": float(box["y_bottom"]),
    }


def _smooth(values: np.ndarray, size: int = 7) -> np.ndarray:
    if values.size < size:
        return values.astype(np.float32)
    kernel = np.ones(size, dtype=np.float32) / float(size)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def _line_profiles(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    sobel_x = np.abs(cv2.Sobel(clahe, cv2.CV_32F, 1, 0, ksize=3))
    sobel_y = np.abs(cv2.Sobel(clahe, cv2.CV_32F, 0, 1, ksize=3))
    canny = cv2.Canny(clahe, 50, 150).astype(np.float32) / 255.0
    x_profile = sobel_x.mean(axis=0) + canny.mean(axis=0) * 35.0
    y_profile = sobel_y.mean(axis=1) + canny.mean(axis=1) * 35.0
    return _smooth(x_profile), _smooth(y_profile)


def _candidate_from_band(
    profile: np.ndarray,
    low: int,
    high: int,
    prior: int,
) -> tuple[int, float]:
    high = max(low + 1, min(high, profile.size - 1))
    low = max(0, min(low, high - 1))
    band = profile[low : high + 1]
    if band.size == 0:
        return prior, 0.0

    # Prefer strong lines, but avoid jumping to a decorative line far from the
    # normal inner-frame location when scores are similar.
    scores = band.astype(np.float32).copy()
    positions = np.arange(low, high + 1, dtype=np.float32)
    distance = np.abs(positions - float(prior)) / max(1.0, float(high - low))
    scores *= 1.0 - np.clip(distance * 0.35, 0.0, 0.35)

    best_index = int(np.argmax(scores))
    best_position = int(low + best_index)
    best_score = float(scores[best_index])
    median = float(np.median(scores))
    spread = float(np.median(np.abs(scores - median)) * 1.4826 + 1e-6)
    if best_score <= 1e-6:
        return prior, 0.0
    evidence = (best_score - median) / max(best_score, spread, 1e-6)
    return best_position, float(np.clip(evidence, 0.0, 1.0))


def _needs_correction(value: float, low: float, high: float) -> bool:
    return value < low or value > high


def _clip_box(box: dict[str, float], width: int, height: int) -> dict[str, float]:
    x_left = float(np.clip(box["x_left"], 0, width - 2))
    x_right = float(np.clip(box["x_right"], x_left + 1, width - 1))
    y_top = float(np.clip(box["y_top"], 0, height - 2))
    y_bottom = float(np.clip(box["y_bottom"], y_top + 1, height - 1))
    return {
        "x_left": x_left,
        "x_right": x_right,
        "y_top": y_top,
        "y_bottom": y_bottom,
    }


def stabilize_inner_frame_box(
    bgr: np.ndarray,
    box: dict[str, Any],
    *,
    prior_ratios: dict[str, float] | None = None,
) -> StabilizeResult:
    """Keep YOLO inner-frame boxes away from physical-card outer edges.

    The segmentation model is useful for coarse localization, but on real photos
    the strongest horizontal edge is often the physical card border. This
    stabilizer only intervenes when an edge is outside the normal inner-frame
    band, then snaps it back to an edge candidate inside that band. If no strong
    candidate exists, it falls back to the conservative ratio prior.
    """

    height, width = bgr.shape[:2]
    ratios = dict(DEFAULT_RATIOS)
    if prior_ratios:
        ratios.update(prior_ratios)

    pre = _clip_box(_copy_box(box), width, height)
    stabilized = dict(pre)
    x_profile, y_profile = _line_profiles(bgr)

    priors = {
        "x_left": int(round(width * ratios["left"])),
        "x_right": int(round(width * ratios["right"])),
        "y_top": int(round(height * ratios["top"])),
        "y_bottom": int(round(height * ratios["bottom"])),
    }

    bands = {
        "x_left": (int(round(width * 0.028)), int(round(width * 0.075))),
        "x_right": (int(round(width * 0.925)), int(round(width * 0.985))),
        "y_top": (int(round(height * 0.020)), int(round(height * 0.070))),
        "y_bottom": (int(round(height * 0.925)), int(round(height * 0.982))),
    }
    allowed = {
        # Real rectified cards can have a valid printed frame at about 2.0-2.6%
        # of image width.  Values below 1.5% remain likely physical-border
        # snaps; the transition band is handled adaptively below.
        "x_left": (width * 0.015, width * 0.080),
        "x_right": (width * 0.920, width * 0.988),
        # Some valid card layouts place the printed inner frame very close to
        # the top edge (about 1.0-1.7% of image height).  The previous 1.8%
        # lower bound incorrectly pulled those already-correct predictions down
        # to a decorative line.  Values below 0.8% are still treated as likely
        # physical-card-border snaps.
        "y_top": (height * 0.008, height * 0.075),
        "y_bottom": (height * 0.915, height * 0.985),
    }

    candidates: dict[str, tuple[int, float]] = {
        "x_left": _candidate_from_band(x_profile, *bands["x_left"], priors["x_left"]),
        "x_right": _candidate_from_band(x_profile, *bands["x_right"], priors["x_right"]),
        "y_top": _candidate_from_band(y_profile, *bands["y_top"], priors["y_top"]),
        "y_bottom": _candidate_from_band(y_profile, *bands["y_bottom"], priors["y_bottom"]),
    }

    # Between 0.8% and 1.8% image height, a top edge can be either a valid
    # narrow-margin layout or a mild outer-edge snap.  Only move it when the
    # nearby in-band candidate is close and visibly stronger than the line at
    # the model prediction.  This preserves unusual layouts while still
    # correcting weak, slightly-too-high predictions.
    raw_top = float(pre["y_top"])
    top_candidate = float(candidates["y_top"][0])
    raw_top_index = int(np.clip(round(raw_top), 0, y_profile.size - 1))
    raw_top_strength = float(
        np.max(y_profile[max(0, raw_top_index - 2) : min(y_profile.size, raw_top_index + 3)])
    )
    candidate_top_strength = float(y_profile[int(top_candidate)])
    raw_left = float(pre["x_left"])
    left_candidate = float(candidates["x_left"][0])
    raw_right = float(pre["x_right"])
    right_candidate = float(candidates["x_right"][0])
    raw_bottom = float(pre["y_bottom"])
    ambiguous_left_should_snap = (
        width * 0.015 <= raw_left < width * 0.026
        and abs(left_candidate - raw_left) <= width * 0.012
    )
    ambiguous_right_should_snap = raw_right >= width * 0.978
    ambiguous_bottom_should_snap = (
        raw_bottom >= height * 0.983
        and candidates["y_bottom"][1] < MIN_EDGE_EVIDENCE
    )
    ambiguous_top_should_snap = (
        height * 0.008 <= raw_top < height * 0.018
        and abs(top_candidate - raw_top) <= height * 0.025
        and raw_top_strength < candidate_top_strength * 0.85
    )

    changed: list[str] = []
    for key, source_key in (
        ("x_left", "left"),
        ("x_right", "right"),
        ("y_top", "top"),
        ("y_bottom", "bottom"),
    ):
        low, high = allowed[key]
        current = float(pre[key])
        needs_correction = _needs_correction(current, low, high)
        if key == "x_left" and ambiguous_left_should_snap:
            needs_correction = True
        if key == "x_right" and ambiguous_right_should_snap:
            needs_correction = True
        if key == "y_top" and ambiguous_top_should_snap:
            needs_correction = True
        if key == "y_bottom" and ambiguous_bottom_should_snap:
            needs_correction = True
        if not needs_correction:
            continue
        candidate, evidence = candidates[key]
        replacement = float(candidate if evidence >= MIN_EDGE_EVIDENCE else priors[key])
        # A strong line can itself be the physical right border.  When both the
        # model and candidate remain in the outermost 2%, prefer the conservative
        # printed-frame prior instead of snapping from one outer line to another.
        if key == "x_right" and right_candidate >= width * 0.980:
            replacement = float(priors[key])
        stabilized[key] = replacement
        if abs(replacement - current) >= 1.0:
            changed.append(source_key)

    stabilized = _clip_box(stabilized, width, height)

    offsets = {
        "left": round(stabilized["x_left"] - pre["x_left"], 2),
        "right": round(stabilized["x_right"] - pre["x_right"], 2),
        "top": round(stabilized["y_top"] - pre["y_top"], 2),
        "bottom": round(stabilized["y_bottom"] - pre["y_bottom"], 2),
    }
    evidence = {
        "left": round(candidates["x_left"][1], 4),
        "right": round(candidates["x_right"][1], 4),
        "top": round(candidates["y_top"][1], 4),
        "bottom": round(candidates["y_bottom"][1], 4),
    }
    changed = sorted(set(changed), key=("left", "right", "top", "bottom").index)
    status = "stabilized" if changed else "unchanged"
    reason = "outer_edge_guard:" + "|".join(changed) if changed else "box_within_inner_frame_band"
    confidence = float(np.mean([evidence[name] for name in ("left", "right", "top", "bottom")]))
    return StabilizeResult(
        box={key: round(float(value), 2) for key, value in stabilized.items()},
        status=status,
        reason=reason,
        confidence=round(float(np.clip(confidence, 0.0, 0.99)), 4),
        offsets=offsets,
        pre_stabilize_box={key: round(float(value), 2) for key, value in pre.items()},
        evidence=evidence,
    )
