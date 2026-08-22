from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml_backend.ptcg_inference import CardFramePipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the guarded pre-cropped-card fallback end to end."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    pipeline = CardFramePipeline(device=args.device)

    # Exercise the exact fallback contract rather than relying on the current
    # outer model to fail on this particular image/model combination.
    pipeline.outer_detector.predict = lambda *_args, **_kwargs: {
        "success": False,
        "points": None,
        "confidence": 0.0,
        "error_code": "OUTER_FRAME_NOT_DETECTED",
        "message": "Forced failure for guarded pre-cropped smoke test.",
        "metrics": {},
    }
    result = pipeline.infer_file(args.input, args.output)
    recovery = (
        result.get("outer_frame", {})
        .get("metrics", {})
        .get("pre_cropped_card_recovery", {})
    )
    summary = {
        "success": result.get("success"),
        "version": result.get("version"),
        "stage": result.get("stage"),
        "error_code": result.get("error_code"),
        "pre_cropped_confirmed": recovery.get("confirmed"),
        "confirmation_reason": recovery.get("reason"),
        "expected_inner_size_px": recovery.get("expected_inner_size_px"),
        "observed_inner_size_px": recovery.get("observed_inner_size_px"),
        "residual_px": recovery.get("residual_px"),
        "output_files": result.get("output_files"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not result.get("success") or not recovery.get("confirmed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
