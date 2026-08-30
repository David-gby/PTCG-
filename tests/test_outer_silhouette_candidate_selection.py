from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ml_backend.card_quality_processor.outer_silhouette import extract_silhouette_prediction


@dataclass
class _Boxes:
    conf: np.ndarray
    xyxy: np.ndarray

    def __len__(self) -> int:
        return len(self.conf)


@dataclass
class _Masks:
    xy: list[np.ndarray]


@dataclass
class _Prediction:
    boxes: _Boxes
    masks: _Masks
    orig_shape: tuple[int, int]


def _prediction(polygons: list[np.ndarray], confidences: list[float], shape: tuple[int, int]) -> _Prediction:
    boxes = []
    for points in polygons:
        boxes.append(
            [
                float(np.min(points[:, 0])),
                float(np.min(points[:, 1])),
                float(np.max(points[:, 0])),
                float(np.max(points[:, 1])),
            ]
        )
    return _Prediction(
        boxes=_Boxes(
            conf=np.asarray(confidences, dtype=np.float32),
            xyxy=np.asarray(boxes, dtype=np.float32),
        ),
        masks=_Masks(xy=polygons),
        orig_shape=shape,
    )


def test_border_touching_background_does_not_beat_card_candidate() -> None:
    height, width = 1600, 1200
    image = np.full((height, width, 3), (0, 220, 245), dtype=np.uint8)
    card = np.asarray([[275, 435], [800, 435], [795, 1153], [275, 1153]], dtype=np.float32)
    background = np.asarray([[0, 425], [800, 425], [800, 1599], [0, 1599]], dtype=np.float32)
    cv2.polylines(image, [card.astype(np.int32)], True, (20, 20, 20), 8, cv2.LINE_AA)

    result = extract_silhouette_prediction(
        _prediction([card, background], [0.92, 0.81], (height, width)),
        image_shape=image.shape,
        image=image,
    )

    assert result is not None
    assert result["selected_index"] == 0
    assert result["selection_audit"]["policy"] == "outer_multicandidate_edge_border_v1"
    assert result["selection_audit"]["selected_border_contacts"] == []
    background_metrics = result["candidate_metrics"][1]
    assert set(background_metrics["border_contacts"]) == {"left", "bottom"}
    assert background_metrics["border_contact_penalty"] > 0.0
    assert result["selection_audit"]["score_margin"] > 0.05


def test_near_full_frame_card_is_exempt_from_border_penalty() -> None:
    height, width = 880, 630
    image = np.full((height, width, 3), 210, dtype=np.uint8)
    card = np.asarray([[1, 1], [628, 1], [628, 878], [1, 878]], dtype=np.float32)
    artwork = np.asarray([[45, 65], [585, 65], [585, 815], [45, 815]], dtype=np.float32)
    cv2.polylines(image, [card.astype(np.int32)], True, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.polylines(image, [artwork.astype(np.int32)], True, (80, 80, 80), 2, cv2.LINE_AA)

    result = extract_silhouette_prediction(
        _prediction([card, artwork], [0.88, 0.92], (height, width)),
        image_shape=image.shape,
        image=image,
        max_area_ratio=1.0,
    )

    assert result is not None
    assert result["selected_index"] == 0
    card_metrics = result["candidate_metrics"][0]
    assert len(card_metrics["border_contacts"]) == 4
    assert card_metrics["border_contact_penalty"] == 0.0


def test_single_candidate_keeps_legacy_score() -> None:
    height, width = 1000, 800
    card = np.asarray([[180, 120], [620, 120], [620, 735], [180, 735]], dtype=np.float32)
    result = extract_silhouette_prediction(
        _prediction([card], [0.9], (height, width)),
        image_shape=(height, width, 3),
    )

    assert result is not None
    metrics = result["candidate_metrics"][0]
    assert metrics["selection_score"] == metrics["base_selection_score"]
    assert result["selection_audit"]["edge_evidence_used"] is False
