from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .outer_pipeline import batch_process_outer_dataset, process_outer_and_rectify
from .outer_pose_pipeline import batch_process_outer_pose_dataset, process_outer_pose_and_rectify


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="card_quality_processor", description="PTCG card image quality tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("detect-outer", help="Detect and rectify one real-photo card")
    single.add_argument("--image", required=True, type=Path, help="Input image path")
    single.add_argument("--output", required=True, type=Path, help="Output root directory")
    single.add_argument("--config", type=Path, help="Optional YAML configuration")

    batch = subparsers.add_parser("batch-outer", help="Detect and rectify a directory of real-photo cards")
    batch.add_argument("--input", required=True, type=Path, help="Input directory")
    batch.add_argument("--output", default=Path("data/processed/outer_rectified"), type=Path, help="Output root directory")
    batch.add_argument("--config", type=Path, help="Optional YAML configuration")

    pose_single = subparsers.add_parser(
        "detect-outer-pose",
        help="Detect physical card corners with silhouette/Pose and rectify one image",
    )
    pose_single.add_argument("--image", required=True, type=Path, help="Input image path")
    pose_single.add_argument("--output", required=True, type=Path, help="Output root directory")
    pose_single.add_argument("--config", type=Path, help="Optional YAML configuration")

    pose_batch = subparsers.add_parser(
        "batch-outer-pose",
        help="Detect physical card corners with silhouette/Pose for an image directory",
    )
    pose_batch.add_argument("--input", required=True, type=Path, help="Input directory")
    pose_batch.add_argument(
        "--output",
        default=Path("data/processed/outer_pose_rectified"),
        type=Path,
        help="Output root directory",
    )
    pose_batch.add_argument("--config", type=Path, help="Optional YAML configuration")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "detect-outer":
        result = process_outer_and_rectify(args.image, args.output, config)
        outer = result["outer_result"]
        print(f"success: {result['success']}")
        print(f"confidence: {float(outer['confidence']):.4f}")
        print(f"method: {outer.get('method') or '-'}")
        print(f"error_code: {outer.get('error_code') or '-'}")
        print(f"output_dir: {args.output.resolve()}")
        print(f"result_json: {result['output_paths']['result_json_path']}")
        return 0 if result["success"] else 2

    if args.command == "batch-outer":
        summary = batch_process_outer_dataset(args.input, args.output, config)
        print(f"processed_count: {summary['processed_count']}")
        print(f"success_count: {summary['success_count']}")
        print(f"failure_count: {summary['failure_count']}")
        print(f"average_confidence: {summary['average_confidence']:.4f}")
        print(f"batch_outer_report: {Path(summary['report_path']).resolve()}")
        print(f"output_dir: {Path(summary['output_dir']).resolve()}")
        return 0

    # Invoking a pose-specific command is an explicit opt-in even though the
    # default configuration keeps the deep branch disabled.
    config["outer_detection"]["deep_pose"]["enabled"] = True
    if args.command == "detect-outer-pose":
        result = process_outer_pose_and_rectify(args.image, args.output, config)
        pose = result["outer_pose_result"]
        print(f"success: {result['success']}")
        print(f"confidence: {float(pose['confidence']):.4f}")
        print(f"method: {pose.get('method') or '-'}")
        print(f"points: {pose.get('points')}")
        print(f"error_code: {pose.get('error_code') or '-'}")
        if pose.get("error_code") == "OUTER_POSE_MODEL_NOT_FOUND":
            print("Outer pose model not found. Please train the model first:")
            print("python scripts/train_outer_pose.py --data datasets/card_outer_pose/data.yaml")
        print(f"output_dir: {args.output.resolve()}")
        print(f"result_json: {result['output_paths']['result_json_path']}")
        return 0 if result["success"] else 2

    summary = batch_process_outer_pose_dataset(args.input, args.output, config)
    print(f"processed_count: {summary['processed_count']}")
    print(f"success_count: {summary['success_count']}")
    print(f"failure_count: {summary['failure_count']}")
    print(f"average_confidence: {summary['average_confidence']:.4f}")
    if any(
        result["outer_pose_result"].get("error_code") == "OUTER_POSE_MODEL_NOT_FOUND"
        for result in summary["results"]
    ):
        print("Outer pose model not found. Please train the model first:")
        print("python scripts/train_outer_pose.py --data datasets/card_outer_pose/data.yaml")
    print(f"batch_outer_pose_report: {Path(summary['report_path']).resolve()}")
    print(f"output_dir: {Path(summary['output_dir']).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
