from __future__ import annotations

import unittest

import cv2
import numpy as np

from ml_backend.card_quality_processor.config import DEFAULT_CONFIG
from ml_backend.card_quality_processor.outer_boundary_contact_recovery import (
    recover_boundary_contact_outer,
)


def prediction(points: list[list[float]]) -> dict[str, object]:
    confidence = 0.96
    return {
        "success": False,
        "points": points,
        "bbox": [0.0, 70.0, 950.0, 1327.0],
        "confidence": confidence,
        "keypoint_confidence": {
            "tl": confidence,
            "tr": confidence,
            "br": confidence,
            "bl": confidence,
        },
        "error_code": "INVALID_KEYPOINT_GEOMETRY",
        "metrics": {
            "silhouette_detected": True,
            "aspect_ratio_error": 0.06,
            "area_ratio": 0.84,
        },
    }


class BoundaryContactRecoveryTests(unittest.TestCase):
    def test_recovers_a_clipped_side_from_long_physical_edge(self) -> None:
        image = np.full((1400, 1000, 3), (18, 20, 22), dtype=np.uint8)
        cv2.rectangle(image, (50, 70), (950, 1327), (38, 214, 244), -1)
        cv2.rectangle(image, (95, 112), (910, 1284), (65, 105, 190), -1)
        result = recover_boundary_contact_outer(
            image,
            prediction(
                [[0.0, 70.0], [950.0, 70.0], [950.0, 1327.0], [0.0, 1327.0]]
            ),
            DEFAULT_CONFIG,
        )

        self.assertTrue(result["success"])
        recovered = np.asarray(result["points"], dtype=np.float32)
        self.assertTrue(np.allclose(recovered[[0, 3], 0], 50.0, atol=3.0))
        audit = result["metrics"]["boundary_contact_recovery"]
        self.assertTrue(audit["accepted"])
        self.assertEqual(audit["touching_sides"], ["left"])

    def test_keeps_failure_when_no_independent_edge_exists(self) -> None:
        image = np.full((1400, 1000, 3), (128, 128, 128), dtype=np.uint8)
        result = recover_boundary_contact_outer(
            image,
            prediction(
                [[0.0, 70.0], [950.0, 70.0], [950.0, 1327.0], [0.0, 1327.0]]
            ),
            DEFAULT_CONFIG,
        )

        self.assertFalse(result["success"])
        audit = result["metrics"]["boundary_contact_recovery"]
        self.assertTrue(audit["triggered"])
        self.assertFalse(audit["accepted"])

    def test_does_not_modify_non_target_failures(self) -> None:
        image = np.full((1400, 1000, 3), (18, 20, 22), dtype=np.uint8)
        candidate = prediction(
            [[0.0, 70.0], [950.0, 70.0], [950.0, 1327.0], [0.0, 1327.0]]
        )
        candidate["error_code"] = "LOW_CONFIDENCE_OUTER_POSE"
        result = recover_boundary_contact_outer(image, candidate, DEFAULT_CONFIG)
        self.assertFalse(result["success"])
        self.assertFalse(result["metrics"]["boundary_contact_recovery"]["triggered"])


if __name__ == "__main__":
    unittest.main()
