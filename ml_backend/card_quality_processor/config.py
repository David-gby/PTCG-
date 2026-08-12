from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "outer_detection": {
        "enabled": True,
        "method": "opencv",
        "card_aspect_ratio": 1.397,
        "aspect_ratio_tolerance": 0.20,
        "min_area_ratio": 0.08,
        "max_area_ratio": 0.95,
        "low_confidence_threshold": 0.55,
        "canny_sigma": 0.33,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": [8, 8],
        "morph_kernel_size": 5,
        "process_max_dimension": 1400,
        "border_margin_ratio": 0.008,
        "hough": {
            "enabled": True,
            "threshold": 80,
            "min_line_length_ratio": 0.20,
            "max_line_gap": 30,
        },
        "scoring": {
            "edge_weight": 0.25,
            "aspect_weight": 0.25,
            "area_weight": 0.20,
            "geometry_weight": 0.15,
            "center_weight": 0.10,
            "straightness_weight": 0.05,
        },
        "deep_pose": {
            "enabled": False,
            "device": None,
            "model_path": "models/outer_pose.pt",
            "conf_threshold": 0.25,
            "inference_imgsz": 640,
            "keypoint_conf_threshold": 0.35,
            "min_mean_keypoint_conf": 0.50,
            "low_confidence_threshold": 0.55,
            "card_aspect_ratio": 1.397,
            "aspect_ratio_tolerance": 0.25,
            "min_area_ratio": 0.05,
            "max_area_ratio": 0.95,
            "border_margin_ratio": 0.002,
            "output_width": 630,
            "output_height": 880,
            "silhouette_refinement": {
                "enabled": True,
                "model_path": "models/outer_seg.pt",
                "conf_threshold": 0.15,
                "candidate_area_weight": 0.40,
                "candidate_aspect_weight": 0.25,
                "fallback": {
                    "enabled": True,
                    "max_primary_area_ratio": 0.40,
                    "min_area_gain_ratio": 1.35,
                    "min_pose_bbox_confidence": 0.75,
                    "max_pose_aspect_error": 0.20,
                    "legacy_model_path": "models/outer_seg_pre_frame2.pt",
                    "legacy_pose_area_tolerance": 0.20,
                },
            },
            "physical_edge_refinement": {
                "enabled": True,
                "max_dimension": 1800,
                "search_ratio": 0.075,
                "standard_resolution_search_ratio": 0.022,
                "high_resolution_threshold": 3000,
                "max_angle_degrees": 3.0,
                "angle_step_degrees": 1.0,
                "strip_gap_ratio": 0.005,
                "position_penalty": 2.2,
                "max_corner_movement_ratio": 0.04,
                "max_area_change_ratio": 0.06,
                "min_side_score": 2.0,
                "min_mean_score_gain": 20.0,
                "max_mean_score_gain": 95.0,
                "max_side_offset_ratio": 0.03,
                "min_side_peak_margin": 2.0,
            },
            "corner_calibration": {
                "enabled": False,
                "normalized_offsets": [],
            },
            "edge_guard": {
                "enabled": False,
                "canny_sigma": 0.33,
                "min_edge_support": 0.05,
                "min_side_edge_support": 0.01,
            },
        },
    },
    "rectification": {
        "output_width": 630,
        "output_height": 880,
        "interpolation": "linear",
        "border_mode": "constant",
    },
}


def _deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def normalize_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a complete config while accepting either full or section-only input."""
    merged = deepcopy(DEFAULT_CONFIG)
    if config:
        supplied = dict(config)
        if "outer_detection" not in supplied and "rectification" not in supplied:
            supplied = {"outer_detection": supplied}
        _deep_update(merged, supplied)
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        default_path = Path(__file__).resolve().parents[1] / "configs" / "default_config.yaml"
        path = default_path if default_path.exists() else None
    if path is None:
        return normalize_config()
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    return normalize_config(raw)
