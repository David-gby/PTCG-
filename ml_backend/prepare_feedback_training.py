from __future__ import annotations

import argparse
import json

from feedback_dataset import convert_feedback_to_training, validate_feedback_package


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate or convert reviewed PTCG feedback")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--feedback", required=True)

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--feedback", required=True)
    convert_parser.add_argument("--output", required=True)
    convert_parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    convert_parser.add_argument("--allow-accepted-predictions", action="store_true")
    args = parser.parse_args()

    if args.command == "validate":
        result = validate_feedback_package(args.feedback)
    else:
        result = convert_feedback_to_training(
            args.feedback,
            args.output,
            split=args.split,
            allow_accepted_predictions=args.allow_accepted_predictions,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
