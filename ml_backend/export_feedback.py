from __future__ import annotations

import argparse
import json
from pathlib import Path

from feedback import FeedbackExporter


def _outer_points(value: str | None) -> list[list[float]] | None:
    if not value:
        return None
    points = []
    for point in value.split(";"):
        x_value, y_value = point.split(",", 1)
        points.append([float(x_value), float(y_value)])
    if len(points) != 4:
        raise argparse.ArgumentTypeError("--outer requires four x,y points separated by semicolons")
    return points


def _inner_box(value: str | None) -> dict[str, float] | None:
    if not value:
        return None
    values = [float(item) for item in value.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("--inner requires left,top,right,bottom")
    return dict(zip(("left", "top", "right", "bottom"), values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a human-reviewed PTCG feedback sample")
    parser.add_argument("--image", required=True, help="Original photographed image")
    parser.add_argument("--prediction", required=True, help="Unified result.json produced by run_pipeline.py")
    parser.add_argument("--output", required=True, help="Feedback package directory")
    parser.add_argument("--outer", default=None, help="TLx,TLy;TRx,TRy;BRx,BRy;BLx,BLy")
    parser.add_argument("--inner", default=None, help="left,top,right,bottom in corrected rectified pixels")
    parser.add_argument("--issue-tags", default="", help="Comma-separated issue tags")
    parser.add_argument("--annotator", default="")
    parser.add_argument(
        "--status",
        default="corrected",
        choices=("corrected", "accepted_prediction", "rejected", "no_inner_frame"),
    )
    parser.add_argument("--approve", action="store_true", help="Approve the reviewed label for training")
    parser.add_argument("--card-type", default="unknown")
    parser.add_argument("--layout", default="unknown")
    parser.add_argument("--notes", default="")
    parser.add_argument("--sample-id", default=None)
    args = parser.parse_args()

    prediction = json.loads(Path(args.prediction).read_text(encoding="utf-8"))
    result = FeedbackExporter(args.output).export(
        image_path=args.image,
        prediction=prediction,
        outer_correction=_outer_points(args.outer),
        inner_correction=_inner_box(args.inner),
        issue_tags=[item.strip() for item in args.issue_tags.split(",") if item.strip()],
        annotator=args.annotator,
        review_status=args.status,
        approved_for_training=args.approve,
        card_type=args.card_type,
        layout=args.layout,
        notes=args.notes,
        sample_id=args.sample_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
