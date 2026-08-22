from __future__ import annotations

import json
import unittest
from pathlib import Path

from ml_backend.boundary_quality_guard import assess_inner_quality
from ml_backend.inner_frame.physical_inner_prior import (
    assess_physical_inner_box,
    guarded_refine_physical_inner_box,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "ml_backend" / "inner_frame" / "physical_inner_prior.json").read_text(
        encoding="utf-8"
    )
)


def strong_evidence(
    edge: str,
    positions: list[float] | tuple[float, ...],
    _box: dict[str, float],
) -> list[dict[str, float | int | str]]:
    return [
        {
            "edge": edge,
            "position": float(position),
            "aggregate_score": 2.0,
            "robust_z": 3.2,
            "supporting_segments": 7,
            "total_segments": 8,
        }
        for position in positions
    ]


class PhysicalInnerPriorTests(unittest.TestCase):
    def test_canonical_size_is_580_by_830_inner_edge_pixels(self) -> None:
        assessment = assess_physical_inner_box(
            {"left": 25.0, "right": 605.0, "top": 25.0, "bottom": 855.0},
            630,
            880,
            CONFIG,
        )
        self.assertEqual(assessment["measurement_semantics"], "printed_inner_line_inner_edge")
        self.assertEqual(assessment["expected_size_px"], {"width": 580.0, "height": 830.0})
        self.assertEqual(assessment["risk"], "normal")

    def test_moderate_size_outlier_is_only_softly_corrected_with_visual_support(self) -> None:
        result = guarded_refine_physical_inner_box(
            {"left": 20.0, "right": 615.0, "top": 20.0, "bottom": 860.0},
            630,
            880,
            CONFIG,
            evidence_provider=strong_evidence,
        )
        self.assertTrue(result["applied"])
        self.assertEqual(result["applied_axes"], ["horizontal"])
        self.assertAlmostEqual(result["box"]["left"], 20.5625)
        self.assertAlmostEqual(result["box"]["right"], 613.3125)
        self.assertAlmostEqual(result["box"]["top"], 20.0)
        self.assertAlmostEqual(result["box"]["bottom"], 860.0)

    def test_rule_does_not_move_without_independent_visual_evidence(self) -> None:
        box = {"left": 20.0, "right": 615.0, "top": 20.0, "bottom": 860.0}
        result = guarded_refine_physical_inner_box(box, 630, 880, CONFIG)
        self.assertFalse(result["applied"])
        self.assertEqual(result["box"], box)

    def test_severe_residual_is_reviewed_instead_of_hidden(self) -> None:
        box = {"left": 0.0, "right": 620.0, "top": 0.0, "bottom": 870.0}
        physical = guarded_refine_physical_inner_box(
            box,
            630,
            880,
            CONFIG,
            evidence_provider=strong_evidence,
        )
        quality = assess_inner_quality(
            {
                "edge_refinement": {},
                "global_edge_hypotheses": {},
                "physical_inner_prior": physical,
            }
        )
        self.assertFalse(physical["applied"])
        self.assertTrue(quality["review_recommended"])
        self.assertEqual(quality["physical_prior_risk"], "severe")


if __name__ == "__main__":
    unittest.main()
