from __future__ import annotations

import unittest

from ml_backend.card_quality_processor.pre_cropped_card_recovery import (
    confirm_pre_cropped_inner,
    propose_pre_cropped_outer,
)


class FakeImage:
    def __init__(self, height: int, width: int) -> None:
        self.shape = (height, width, 3)
        self.size = height * width * 3


class PreCroppedCardRecoveryTests(unittest.TestCase):
    def test_proposes_full_frame_only_after_outer_failure(self) -> None:
        image = FakeImage(880, 630)
        proposal = propose_pre_cropped_outer(
            image,
            {"success": False, "error_code": "OUTER_FRAME_NOT_DETECTED"},
        )
        self.assertIsNotNone(proposal)
        self.assertEqual(
            proposal["points"],
            [[0.0, 0.0], [629.0, 0.0], [629.0, 879.0], [0.0, 879.0]],
        )
        self.assertTrue(
            proposal["metrics"]["pre_cropped_card_recovery"]["provisional"]
        )

    def test_rejects_non_card_aspect_ratio(self) -> None:
        image = FakeImage(880, 880)
        self.assertIsNone(
            propose_pre_cropped_outer(
                image,
                {"success": False, "error_code": "OUTER_FRAME_NOT_DETECTED"},
            )
        )

    def test_does_not_override_successful_outer_detection(self) -> None:
        image = FakeImage(880, 630)
        self.assertIsNone(propose_pre_cropped_outer(image, {"success": True}))

    def test_confirms_expected_58_by_83_inner_edge(self) -> None:
        confirmation = confirm_pre_cropped_inner(
            {
                "success": True,
                "yolo_confidence": 0.91,
                "final_box": {
                    "left": 25.0,
                    "right": 605.0,
                    "top": 25.0,
                    "bottom": 855.0,
                },
            }
        )
        self.assertTrue(confirmation["confirmed"])
        self.assertEqual(confirmation["expected_inner_size_px"], {"width": 580.0, "height": 830.0})

    def test_rejects_inner_geometry_that_cannot_be_a_card(self) -> None:
        confirmation = confirm_pre_cropped_inner(
            {
                "success": True,
                "yolo_confidence": 0.91,
                "final_box": {
                    "left": 80.0,
                    "right": 550.0,
                    "top": 120.0,
                    "bottom": 760.0,
                },
            }
        )
        self.assertFalse(confirmation["confirmed"])
        self.assertEqual(confirmation["reason"], "inner_width_inconsistent_with_58mm")


if __name__ == "__main__":
    unittest.main()
