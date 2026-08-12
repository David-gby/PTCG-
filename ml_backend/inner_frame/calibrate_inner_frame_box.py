from __future__ import annotations

from typing import Any

import numpy as np


NORMALIZED_OFFSETS = {
    "left": -0.00002980389222941731,
    "right": -0.00294543132978723,
    "top": -0.00015817437330429795,
    "bottom": -0.001742115432865096,
}
RIGHT_MIN_RATIO_FOR_INWARD_CALIBRATION = 0.95


def calibrate_inner_frame_box(box: dict[str, Any], width: int, height: int) -> dict[str, float]:
    """Apply validation-fitted half-shrinkage calibration after geometry stabilization."""
    right_offset = (
        NORMALIZED_OFFSETS["right"]
        if float(box["x_right"]) / max(1.0, float(width)) >= RIGHT_MIN_RATIO_FOR_INWARD_CALIBRATION
        else 0.0
    )
    output = {
        "x_left": float(box["x_left"]) + NORMALIZED_OFFSETS["left"] * width,
        "x_right": float(box["x_right"]) + right_offset * width,
        "y_top": float(box["y_top"]) + NORMALIZED_OFFSETS["top"] * height,
        "y_bottom": float(box["y_bottom"]) + NORMALIZED_OFFSETS["bottom"] * height,
    }
    output["x_left"] = float(np.clip(output["x_left"], 0, width - 2))
    output["x_right"] = float(np.clip(output["x_right"], output["x_left"] + 1, width - 1))
    output["y_top"] = float(np.clip(output["y_top"], 0, height - 2))
    output["y_bottom"] = float(np.clip(output["y_bottom"], output["y_top"] + 1, height - 1))
    return {key: round(value, 2) for key, value in output.items()}
