from __future__ import annotations

import argparse
import json
from pathlib import Path

from ptcg_inference import CardFramePipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="PTCG outer-frame and inner-frame inference")
    parser.add_argument("--image", required=True, help="Raw photographed card image")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--device", default=None, help="cpu, 0, 1, or cuda:0; defaults to CUDA when available")
    args = parser.parse_args()

    pipeline = CardFramePipeline(device=args.device)
    result = pipeline.infer_file(Path(args.image), Path(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
