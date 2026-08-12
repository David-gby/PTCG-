"""Canonical centerline and centering calculations for rectified card space."""

from __future__ import annotations

from typing import Any, Mapping


def center_geometry_from_centers(
    centers: Mapping[str, float],
    width: int,
    height: int,
) -> dict[str, Any]:
    value = {side: round(float(centers[side]), 4) for side in ("left", "right", "top", "bottom")}
    if not (
        0.0 <= value["left"] < value["right"] <= width - 1
        and 0.0 <= value["top"] < value["bottom"] <= height - 1
    ):
        raise ValueError("Inner centerlines are outside the rectified card or incorrectly ordered.")
    # The rendered red centerlines span the full rectified card.  Their
    # representative midpoint therefore belongs to the card canvas centre,
    # not to the centre of the detected inner box.
    middle_x = round(width / 2.0, 4)
    middle_y = round(height / 2.0, 4)
    return {
        "line_centers_px": value,
        "line_midpoints_px": {
            "left": [value["left"], middle_y],
            "right": [value["right"], middle_y],
            "top": [middle_x, value["top"]],
            "bottom": [middle_x, value["bottom"]],
        },
        "coordinate_semantics": "zero_width_red_line_center",
    }


def centers_from_lines(lines: Any) -> dict[str, float] | None:
    if not isinstance(lines, dict):
        return None
    try:
        return {
            "left": round(sum(float(point[0]) for point in lines["left"]) / 2.0, 4),
            "right": round(sum(float(point[0]) for point in lines["right"]) / 2.0, 4),
            "top": round(sum(float(point[1]) for point in lines["top"]) / 2.0, 4),
            "bottom": round(sum(float(point[1]) for point in lines["bottom"]) / 2.0, 4),
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def centering_measurements(
    centers: Mapping[str, float],
    width: int,
    height: int,
) -> dict[str, Any]:
    geometry = center_geometry_from_centers(centers, width, height)
    value = geometry["line_centers_px"]
    widths = {
        "left": value["left"],
        "right": round((width - 1) - value["right"], 4),
        "top": value["top"],
        "bottom": round((height - 1) - value["bottom"], 4),
    }
    horizontal = widths["left"] + widths["right"]
    vertical = widths["top"] + widths["bottom"]
    if horizontal <= 0 or vertical <= 0:
        raise ValueError("Inner centerlines leave no measurable outer border.")
    fractions = {
        "left": widths["left"] / width,
        "right": widths["right"] / width,
        "top": widths["top"] / height,
        "bottom": widths["bottom"] / height,
    }
    pair = {
        "left": 100.0 * widths["left"] / horizontal,
        "right": 100.0 * widths["right"] / horizontal,
        "top": 100.0 * widths["top"] / vertical,
        "bottom": 100.0 * widths["bottom"] / vertical,
    }
    rounded = lambda values: {key: round(float(item), 4) for key, item in values.items()}
    pair = rounded(pair)
    return {
        **geometry,
        "border_width_px": rounded(widths),
        "border_fraction_of_card": rounded(fractions),
        "centering_pair_percent": pair,
        "centering_ratio": {
            "left_percent": pair["left"],
            "right_percent": pair["right"],
            "top_percent": pair["top"],
            "bottom_percent": pair["bottom"],
        },
        "formula_contract": {
            "coordinate_origin": "physical_outer_top_left",
            "input_coordinates": "zero_width_red_line_centers",
            "right_width": "rectified_width - 1 - right_line_center_x",
            "bottom_width": "rectified_height - 1 - bottom_line_center_y",
            "horizontal_denominator": "left_width + right_width",
            "vertical_denominator": "top_width + bottom_width",
        },
    }


__all__ = ["center_geometry_from_centers", "centers_from_lines", "centering_measurements"]
