from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from ml_backend.inner_frame.joint_physical_refiner import (
    _select_axis_from_anchors,
    refine_trusted_inner_box,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "ml_backend" / "inner_frame" / "physical_inner_prior.json").read_text(
        encoding="utf-8"
    )
)


class JointPhysicalRefinerTests(unittest.TestCase):
    def test_one_clear_edge_infers_occluded_opposite_edge(self) -> None:
        profile = np.zeros(1260, dtype=np.float32)
        segments = np.zeros((10, 1260), dtype=np.float32)
        profile[54] = 6.0
        segments[:, 54] = 3.0
        result = _select_axis_from_anchors(
            profile,
            segments,
            current_first=50.0,
            current_second=1210.0,
            expected_span=1160.0,
            scale=2.0,
            config=CONFIG["trusted_joint"],
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["reason"], "single_first_anchor")
        self.assertEqual(result["anchor_edge"], "first")
        self.assertEqual(result["inferred_edge"], "second")
        self.assertAlmostEqual(result["selected"]["first"], 54.0)
        self.assertAlmostEqual(result["selected"]["second"], 1214.0)

    def test_no_visual_anchor_keeps_learned_center_but_locks_span(self) -> None:
        profile = np.zeros(1260, dtype=np.float32)
        segments = np.zeros((10, 1260), dtype=np.float32)
        result = _select_axis_from_anchors(
            profile,
            segments,
            current_first=44.0,
            current_second=1216.0,
            expected_span=1160.0,
            scale=2.0,
            config=CONFIG["trusted_joint"],
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["reason"], "learned_center_physical_span_fallback")
        self.assertAlmostEqual(result["selected"]["first"], 50.0)
        self.assertAlmostEqual(result["selected"]["second"], 1210.0)

    def test_extreme_model_center_is_bounded_without_breaking_physical_span(self) -> None:
        profile = np.zeros(880, dtype=np.float32)
        segments = np.zeros((10, 880), dtype=np.float32)
        result = _select_axis_from_anchors(
            profile,
            segments,
            current_first=-30.0,
            current_second=820.0,
            expected_span=830.0,
            scale=1.0,
            config=CONFIG["trusted_joint"],
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["reason"], "bounded_learned_center_physical_span_fallback")
        self.assertAlmostEqual(result["selected"]["first"], 1.0)
        self.assertAlmostEqual(result["selected"]["second"], 831.0)

    def test_trusted_outer_locks_opposite_edges_to_physical_span(self) -> None:
        image = np.full((1760, 1260, 3), 210, dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (1210, 1710), (20, 20, 20), 5)
        result = refine_trusted_inner_box(
            image,
            {"left": 22.0, "right": 607.0, "top": 22.0, "bottom": 858.0},
            630,
            880,
            CONFIG,
            trusted_outer=True,
        )
        self.assertTrue(result["applied"])
        self.assertAlmostEqual(result["box"]["right"] - result["box"]["left"], 580.0, places=4)
        self.assertAlmostEqual(result["box"]["bottom"] - result["box"]["top"], 830.0, places=4)
        self.assertLess(abs((result["box"]["left"] + result["box"]["right"]) / 2 - 315.0), 2.0)
        self.assertLess(abs((result["box"]["top"] + result["box"]["bottom"]) / 2 - 440.0), 2.0)

    def test_untrusted_outer_is_never_hard_constrained(self) -> None:
        image = np.full((1760, 1260, 3), 210, dtype=np.uint8)
        box = {"left": 22.0, "right": 607.0, "top": 22.0, "bottom": 858.0}
        result = refine_trusted_inner_box(
            image, box, 630, 880, CONFIG, trusted_outer=False
        )
        self.assertFalse(result["applied"])
        self.assertEqual(result["box"], box)


if __name__ == "__main__":
    unittest.main()
