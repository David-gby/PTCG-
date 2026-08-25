from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from ptcg_inference import PipelineModels  # noqa: E402
from training.official_corpus.build_inner_consensus import (  # noqa: E402
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    EDGES,
    _decode_entry,
    _make_inner_engine,
    _views,
)


@lru_cache(maxsize=4)
def _archive(path: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(path)


def _read_image(row: dict[str, str], manifest: Path) -> np.ndarray:
    if row.get("archive") and row.get("entry_name"):
        archive_path = Path(row["archive"])
        if not archive_path.is_absolute():
            archive_path = (manifest.parent / archive_path).resolve()
        return _decode_entry(_archive(str(archive_path)), row["entry_name"])
    path = Path(row["image"])
    if not path.is_absolute():
        path = (manifest.parent / path).resolve()
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(path)
    return image


def _target(row: dict[str, str]) -> dict[str, float]:
    width, height = float(row["width"]), float(row["height"])
    return {
        "left": float(row["left"]) * width,
        "right": float(row["right"]) * width,
        "top": float(row["top"]) * height,
        "bottom": float(row["bottom"]) * height,
    }


def _rank(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def _limit_official(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0:
        return rows
    output = [row for row in rows if not row["source"].startswith("official_")]
    for split in sorted({row["split"] for row in rows}):
        official = [
            row
            for row in rows
            if row["split"] == split and row["source"].startswith("official_")
        ]
        official.sort(key=lambda row: _rank(row.get("group_id", "") + "\x1f" + row["id"]))
        output.extend(official[:limit])
    return output


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row.get("success")]
    return {
        "images": len(rows),
        "successes": len(successes),
        "success_rate": len(successes) / max(len(rows), 1),
        "edge_mae_px": _summary([float(row["edge_mae_px"]) for row in successes]),
        "edge_max_px": _summary([float(row["edge_max_px"]) for row in successes]),
        "view_max_range_px": _summary(
            [float(row["view_max_range_px"]) for row in successes if row.get("view_max_range_px") is not None]
        ),
        "review_rate": sum(bool(row.get("review_recommended")) for row in successes) / max(len(successes), 1),
        "by_edge": {
            edge: {
                "absolute": _summary([float(row[f"error_{edge}_px"]) for row in successes]),
                "signed_mean_px": statistics.fmean(float(row[f"signed_{edge}_px"]) for row in successes)
                if successes else None,
            }
            for edge in EDGES
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the complete inner pipeline on manual and official consensus labels.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refiner", type=Path, required=True)
    parser.add_argument("--horizontal", type=Path, default=None)
    parser.add_argument("--top-left", type=Path, default=None)
    parser.add_argument("--top", type=Path, default=None)
    parser.add_argument("--top-gate", type=Path, default=None)
    parser.add_argument("--physical-prior", type=Path, default=None)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--official-multiview", action="store_true")
    parser.add_argument(
        "--trusted-outer",
        action="store_true",
        help="Evaluate the strong 58x83 joint optimizer for independently trusted outer crops.",
    )
    parser.add_argument("--max-official-per-split", type=int, default=400)
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    requested = {value.strip() for value in args.splits.split(",") if value.strip()}
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        source_rows = [row for row in csv.DictReader(stream) if row["split"] in requested]
    source_rows = _limit_official(source_rows, args.max_official_per_split)
    defaults = PipelineModels()
    models = PipelineModels(
        outer_seg=defaults.outer_seg,
        outer_pose=defaults.outer_pose,
        outer_line_refiner=defaults.outer_line_refiner,
        outer_line_gate=defaults.outer_line_gate,
        inner_yolo=defaults.inner_yolo,
        inner_refiner=args.refiner.resolve(),
        inner_refiner_horizontal=(
            args.horizontal.resolve() if args.horizontal else defaults.inner_refiner_horizontal
        ),
        inner_refiner_top_left=args.top_left.resolve() if args.top_left else defaults.inner_refiner_top_left,
        inner_refiner_top=args.top.resolve() if args.top else defaults.inner_refiner_top,
        inner_top_gate=args.top_gate.resolve() if args.top_gate else defaults.inner_top_gate,
        inner_gate=defaults.inner_gate,
        inner_physical_prior=(
            args.physical_prior.resolve() if args.physical_prior else defaults.inner_physical_prior
        ),
    )
    engine = _make_inner_engine(args.device, models)
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows, 1):
        row: dict[str, Any] = {
            "id": source["id"],
            "split": source["split"],
            "source": source["source"],
            "entry_name": source.get("entry_name", ""),
        }
        try:
            image = _read_image(source, manifest)
            target = _target(source)
            views = _views(image) if args.official_multiview and source["source"].startswith("official_") else [("clean", image)]
            predictions: list[dict[str, float]] = []
            reviews: list[bool] = []
            physical_audits: list[dict[str, Any]] = []
            for _, view in views:
                refinement_view = None
                if args.trusted_outer:
                    refinement_view = cv2.resize(
                        view,
                        (view.shape[1] * 2, view.shape[0] * 2),
                        interpolation=cv2.INTER_CUBIC,
                    )
                result = engine._infer_inner(  # noqa: SLF001
                    view,
                    refinement_image=refinement_view,
                    trusted_outer=args.trusted_outer,
                )
                if not result.get("success"):
                    raise RuntimeError(str(result.get("error_code") or "inner_failed"))
                predictions.append({edge: float(result["final_box"][edge]) for edge in EDGES})
                reviews.append(bool(result.get("quality_assessment", {}).get("review_recommended")))
                physical_audits.append(dict(result.get("physical_inner_prior", {})))
            prediction = {edge: float(np.median([box[edge] for box in predictions])) for edge in EDGES}
            errors = {edge: abs(prediction[edge] - target[edge]) for edge in EDGES}
            signed = {edge: prediction[edge] - target[edge] for edge in EDGES}
            ranges = {edge: float(np.ptp([box[edge] for box in predictions])) for edge in EDGES}
            row.update(
                {
                    "success": True,
                    "edge_mae_px": statistics.fmean(errors.values()),
                    "edge_max_px": max(errors.values()),
                    "view_max_range_px": max(ranges.values()) if len(predictions) > 1 else None,
                    "review_recommended": any(reviews),
                    "physical_prior_applied": any(
                        bool(audit.get("applied")) for audit in physical_audits
                    ),
                    "physical_prior_audit_json": json.dumps(
                        physical_audits, ensure_ascii=False, separators=(",", ":")
                    ),
                    **{f"prediction_{edge}": prediction[edge] for edge in EDGES},
                    **{f"target_{edge}": target[edge] for edge in EDGES},
                    **{f"error_{edge}_px": errors[edge] for edge in EDGES},
                    **{f"signed_{edge}_px": signed[edge] for edge in EDGES},
                    **{f"range_{edge}_px": ranges[edge] for edge in EDGES},
                }
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            row.update({"success": False, "error": f"{type(exc).__name__}:{exc}"})
        rows.append(row)
        if index % 50 == 0:
            print(f"evaluated {index}/{len(source_rows)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "refiner": str(args.refiner.resolve()),
        "horizontal": (
            str(args.horizontal.resolve())
            if args.horizontal
            else str(defaults.inner_refiner_horizontal)
        ),
        "top_left": str(args.top_left.resolve()) if args.top_left else str(defaults.inner_refiner_top_left),
        "top": str(args.top.resolve()) if args.top else str(defaults.inner_refiner_top),
        "top_gate": str(args.top_gate.resolve()) if args.top_gate else str(defaults.inner_top_gate),
        "physical_prior": (
            str(args.physical_prior.resolve())
            if args.physical_prior
            else str(defaults.inner_physical_prior)
        ),
        "manifest": str(manifest),
        "splits": sorted(requested),
        "trusted_outer": bool(args.trusted_outer),
        "overall": _aggregate(rows),
        "by_source": {
            source: _aggregate([row for row in rows if row["source"] == source])
            for source in sorted({str(row["source"]) for row in rows})
        },
    }
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
