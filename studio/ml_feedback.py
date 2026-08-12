"""Bridge accepted Studio labels into the ML handoff feedback schema."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


_ROOT = Path(__file__).resolve().parents[1]
_ML_ROOT = _ROOT / "ml_backend"


def export_reviewed_feedback(
    *,
    project: dict[str, Any],
    sample: dict[str, Any],
    label: dict[str, Any],
    outer: list[list[float]],
    inner: dict[str, float],
    normalized_path: Path,
    prediction: dict[str, Any],
    feedback_root: Path,
    feedback_id: str,
    issue_tags: list[str],
    review_status: str,
) -> dict[str, Any]:
    if str(_ML_ROOT) not in sys.path:
        sys.path.insert(0, str(_ML_ROOT))
    from feedback import FeedbackExporter

    return FeedbackExporter(feedback_root).export(
        image_path=normalized_path,
        prediction=prediction,
        exif_orientation_handling="studio_exif_normalized_png",
        outer_correction=outer,
        inner_correction=inner,
        issue_tags=issue_tags,
        annotator=str(label["annotation"].get("labeler") or ""),
        review_status=review_status,
        approved_for_training=True,
        card_type=str(label["classification"].get("card_type") or "unknown"),
        layout=str(label["classification"].get("layout_id") or "unknown"),
        notes=str(label["assessment"].get("notes") or ""),
        sample_id=feedback_id,
    )


__all__ = ["export_reviewed_feedback"]
