from __future__ import annotations

import unittest

import cv2
import numpy as np

from platform_app.reference_registration import detect_automatic_reference_registration


class ReferenceRegistrationInnerTests(unittest.TestCase):
    def test_successful_registration_emits_physical_inner_frame(self) -> None:
        rng = np.random.default_rng(7)
        reference = rng.integers(0, 256, (880, 630, 3), dtype=np.uint8)
        capture = cv2.warpAffine(
            reference,
            np.float32([[1.0, 0.0, 1.25], [0.0, 1.0, -0.75]]),
            (630, 880),
            borderMode=cv2.BORDER_REFLECT,
        )
        result, _ = detect_automatic_reference_registration(
            reference, capture, reference_id="synthetic"
        )
        self.assertTrue(result["success"])
        inner = result["reference_fused_inner_frame"]
        box = inner["final_box"]
        self.assertAlmostEqual(box["right"] - box["left"], 580.0, places=3)
        self.assertAlmostEqual(box["bottom"] - box["top"], 830.0, places=3)
        self.assertLess(abs(inner["center"]["x"] - 316.25), 0.15)
        self.assertLess(abs(inner["center"]["y"] - 439.25), 0.15)


if __name__ == "__main__":
    unittest.main()
