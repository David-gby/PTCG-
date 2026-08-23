from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from ml_backend.inner_frame.joint_physical_refiner import refine_trusted_inner_box


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "ml_backend" / "inner_frame" / "physical_inner_prior.json").read_text(
        encoding="utf-8"
    )
)


class JointPhysicalRefinerTests(unittest.TestCase):
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
