from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from ptcg_inference import CardFramePipeline, PipelineModels  # noqa: E402


EDGES = ("left", "right", "top", "bottom")
CANONICAL_WIDTH = 630
CANONICAL_HEIGHT = 880


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _stable_rank(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def _select_balanced(
    rows: Iterable[dict[str, str]],
    *,
    max_per_group: int,
    limit: int,
) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("extension", "").lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        try:
            if abs(float(row.get("aspect_error") or 1.0)) > 0.018:
                continue
            if int(float(row.get("min_dimension") or 0)) < 450:
                continue
        except ValueError:
            continue
        grouped[(row["split"], row["group_id"])].append(row)

    selected: list[dict[str, str]] = []
    for key in sorted(grouped, key=lambda value: _stable_rank("\x1f".join(value))):
        candidates = sorted(grouped[key], key=lambda row: _stable_rank(row["entry_name"]))
        selected.extend(candidates[:max_per_group])
    selected.sort(key=lambda row: (_stable_rank(row["group_id"]), _stable_rank(row["entry_name"])))
    return selected[:limit] if limit > 0 else selected


def _decode_entry(archive: zipfile.ZipFile, entry_name: str) -> np.ndarray:
    payload = archive.read(entry_name)
    raw = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("OpenCV could not decode archive entry")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        bgr = image[:, :, :3].astype(np.float32)
        # A neutral bright backing preserves printed edge contrast without
        # teaching the refiner a transparent checkerboard artefact.
        image = np.clip(bgr * alpha + 242.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
    return cv2.resize(image, (CANONICAL_WIDTH, CANONICAL_HEIGHT), interpolation=cv2.INTER_AREA)


def _gamma(image: np.ndarray, exponent: float) -> np.ndarray:
    values = np.arange(256, dtype=np.float32) / 255.0
    table = np.clip(np.power(values, exponent) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


def _views(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("clean", image),
        ("dark", _gamma(image, 1.28)),
        ("bright", _gamma(image, 0.78)),
    ]


def _make_inner_engine(device: str, models: PipelineModels | None = None) -> CardFramePipeline:
    """Construct only the inner half of CardFramePipeline.

    CardFramePipeline normally also initializes the much larger outer stack.
    Official assets are already perfectly rectified, so loading those models
    wastes GPU memory and makes the corpus pass slower.
    """

    engine = object.__new__(CardFramePipeline)
    engine.device = int(device) if device.isdigit() else device
    if torch.cuda.is_available() and str(device).lower() != "cpu":
        engine.torch_device = torch.device(f"cuda:{int(device)}" if device.isdigit() else device)
    else:
        engine.device = "cpu"
        engine.torch_device = torch.device("cpu")
    engine.models = models or PipelineModels()
    engine._inner_yolo = None
    engine._inner_refiner = None
    engine._inner_refiner_config = None
    engine._inner_refiner_horizontal = None
    engine._inner_refiner_horizontal_config = None
    engine._inner_refiner_top_left = None
    engine._inner_refiner_top_left_config = None
    engine._inner_refiner_top = None
    engine._inner_refiner_top_config = None
    engine._gate = json.loads(engine.models.inner_gate.read_text(encoding="utf-8"))
    engine._top_gate = json.loads(engine.models.inner_top_gate.read_text(encoding="utf-8"))
    engine._physical_inner_prior = json.loads(
        engine.models.inner_physical_prior.read_text(encoding="utf-8")
    )
    return engine


def _plausible(box: dict[str, float]) -> bool:
    margins = {
        "left": box["left"],
        "right": CANONICAL_WIDTH - box["right"],
        "top": box["top"],
        "bottom": CANONICAL_HEIGHT - box["bottom"],
    }
    return bool(
        5.0 <= margins["left"] <= 70.0
        and 5.0 <= margins["right"] <= 70.0
        and 3.0 <= margins["top"] <= 85.0
        and 3.0 <= margins["bottom"] <= 90.0
        and box["right"] - box["left"] >= CANONICAL_WIDTH * 0.80
        and box["bottom"] - box["top"] >= CANONICAL_HEIGHT * 0.80
    )


def _predict_consensus(engine: CardFramePipeline, image: np.ndarray) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    for view_name, view in _views(image):
        result = engine._infer_inner(view)  # noqa: SLF001
        if not result.get("success"):
            return {
                "accepted_precluster": False,
                "reason": f"{view_name}:{result.get('error_code', 'inner_failed')}",
                "views": predictions,
            }
        predictions.append(
            {
                "name": view_name,
                "confidence": float(result.get("yolo_confidence", 0.0)),
                "review": bool(result.get("quality_assessment", {}).get("review_recommended")),
                "box": {edge: float(result["final_box"][edge]) for edge in EDGES},
                "base": {edge: float(result["base_box"][edge]) for edge in EDGES},
            }
        )

    arrays = {edge: np.asarray([view["box"][edge] for view in predictions]) for edge in EDGES}
    box = {edge: float(np.median(arrays[edge])) for edge in EDGES}
    ranges = {edge: float(np.ptp(arrays[edge])) for edge in EDGES}
    mads = {
        edge: float(np.median(np.abs(arrays[edge] - np.median(arrays[edge]))))
        for edge in EDGES
    }
    min_confidence = min(float(view["confidence"]) for view in predictions)
    review_count = sum(bool(view["review"]) for view in predictions)
    accepted = bool(
        min_confidence >= 0.45
        and review_count == 0
        and ranges["left"] <= 2.25
        and ranges["right"] <= 2.25
        and ranges["top"] <= 3.0
        and ranges["bottom"] <= 3.0
        and _plausible(box)
    )
    reasons: list[str] = []
    if min_confidence < 0.45:
        reasons.append("low_yolo_confidence")
    if review_count:
        reasons.append("production_review_guard")
    if any(ranges[edge] > (2.25 if edge in ("left", "right") else 3.0) for edge in EDGES):
        reasons.append("photometric_instability")
    if not _plausible(box):
        reasons.append("implausible_geometry")
    return {
        "accepted_precluster": accepted,
        "reason": "accepted" if accepted else "|".join(reasons),
        "box": box,
        "ranges": ranges,
        "mads": mads,
        "min_confidence": min_confidence,
        "review_count": review_count,
        "views": predictions,
    }


def _margin_vector(item: dict[str, Any]) -> np.ndarray:
    box = item["box"]
    return np.asarray(
        [
            box["left"],
            CANONICAL_WIDTH - box["right"],
            box["top"],
            CANONICAL_HEIGHT - box["bottom"],
        ],
        dtype=np.float32,
    )


def _clusters(items: list[dict[str, Any]], radius_px: float) -> list[list[int]]:
    """Small deterministic complete-link-like clustering for a single set."""

    ordered = sorted(range(len(items)), key=lambda index: _stable_rank(items[index]["entry_name"]))
    clusters: list[list[int]] = []
    for index in ordered:
        vector = _margin_vector(items[index])
        best_cluster = None
        best_distance = float("inf")
        for cluster_index, members in enumerate(clusters):
            center = np.median(np.stack([_margin_vector(items[m]) for m in members]), axis=0)
            distance = float(np.max(np.abs(vector - center)))
            if distance <= radius_px and distance < best_distance:
                best_cluster = cluster_index
                best_distance = distance
        if best_cluster is None:
            clusters.append([index])
        else:
            clusters[best_cluster].append(index)
    return clusters


def _apply_layout_consensus(
    predictions: list[dict[str, Any]],
    *,
    radius_px: float,
    min_cluster_size: int,
) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in predictions:
        if item.get("accepted_precluster"):
            groups[(item["split"], item["group_id"])].append(item)

    for group_items in groups.values():
        for cluster_number, member_indexes in enumerate(_clusters(group_items, radius_px), 1):
            members = [group_items[index] for index in member_indexes]
            vectors = np.stack([_margin_vector(member) for member in members])
            center = np.median(vectors, axis=0)
            cluster_size = len(members)
            for member, vector in zip(members, vectors, strict=True):
                distance = float(np.max(np.abs(vector - center)))
                strong_pair = cluster_size == 2 and max(member["ranges"].values()) <= 1.0
                accepted = bool(
                    (cluster_size >= min_cluster_size or strong_pair)
                    and distance <= radius_px
                )
                member["cluster_id"] = cluster_number
                member["cluster_size"] = cluster_size
                member["cluster_distance_px"] = distance
                member["accepted"] = accepted
                if accepted:
                    # The digital template is shared within a layout. Blending
                    # the image consensus with the set-layout median reduces
                    # self-training noise while preserving distinct layouts.
                    blended_margins = 0.55 * vector + 0.45 * center
                    member["label_box"] = {
                        "left": float(blended_margins[0]),
                        "right": float(CANONICAL_WIDTH - blended_margins[1]),
                        "top": float(blended_margins[2]),
                        "bottom": float(CANONICAL_HEIGHT - blended_margins[3]),
                    }
                else:
                    member["reason"] = "layout_cluster_too_small_or_distant"

    for item in predictions:
        if not item.get("accepted_precluster"):
            item["accepted"] = False
        elif "accepted" not in item:
            item["accepted"] = False
            item["reason"] = "no_layout_cluster"


def _write_manifest(path: Path, predictions: list[dict[str, Any]], archive: Path) -> None:
    fields = [
        "id", "split", "source", "source_split", "image", "archive", "entry_name",
        "group_id", "locale", "era", "category", "set_name", "width", "height",
        "left", "right", "top", "bottom", "confidence", "cluster_id",
        "cluster_size", "cluster_distance_px",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in predictions:
            if not item.get("accepted"):
                continue
            box = item["label_box"]
            identifier = "official_" + hashlib.sha256(
                item["entry_name"].encode("utf-8", errors="surrogatepass")
            ).hexdigest()[:20]
            writer.writerow(
                {
                    "id": identifier,
                    "split": item["split"],
                    "source": "official_consensus_v1",
                    "source_split": item["split"],
                    "image": "",
                    "archive": str(archive),
                    "entry_name": item["entry_name"],
                    "group_id": item["group_id"],
                    "locale": item.get("locale", ""),
                    "era": item.get("era", ""),
                    "category": item.get("category", ""),
                    "set_name": item.get("set_name", ""),
                    "width": CANONICAL_WIDTH,
                    "height": CANONICAL_HEIGHT,
                    "left": box["left"] / CANONICAL_WIDTH,
                    "right": box["right"] / CANONICAL_WIDTH,
                    "top": box["top"] / CANONICAL_HEIGHT,
                    "bottom": box["bottom"] / CANONICAL_HEIGHT,
                    "confidence": item["min_confidence"],
                    "cluster_id": item.get("cluster_id"),
                    "cluster_size": item.get("cluster_size"),
                    "cluster_distance_px": item.get("cluster_distance_px"),
                }
            )


def _summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [item for item in predictions if item.get("accepted")]
    preaccepted = [item for item in predictions if item.get("accepted_precluster")]
    by_split: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        split_rows = [item for item in predictions if item["split"] == split]
        split_accepted = [item for item in split_rows if item.get("accepted")]
        by_split[split] = {
            "processed": len(split_rows),
            "precluster_accepted": sum(bool(item.get("accepted_precluster")) for item in split_rows),
            "accepted": len(split_accepted),
            "acceptance_rate": len(split_accepted) / len(split_rows) if split_rows else 0.0,
        }
    margins = [_margin_vector(item) for item in accepted]
    return {
        "processed": len(predictions),
        "precluster_accepted": len(preaccepted),
        "accepted": len(accepted),
        "acceptance_rate": len(accepted) / len(predictions) if predictions else 0.0,
        "by_split": by_split,
        "accepted_margin_px": {
            edge: {
                "mean": statistics.fmean(float(vector[index]) for vector in margins),
                "p05": float(np.percentile([vector[index] for vector in margins], 5)),
                "p95": float(np.percentile([vector[index] for vector in margins], 95)),
            }
            for index, edge in enumerate(EDGES)
        } if margins else {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build high-confidence, set-isolated inner-frame pseudo labels from official cards."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-per-group", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cluster-radius", type=float, default=3.5)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()

    archive_path = args.archive.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = _select_balanced(
        _read_rows(args.splits.resolve()),
        max_per_group=max(1, args.max_per_group),
        limit=max(0, args.limit),
    )
    engine = _make_inner_engine(args.device)
    predictions: list[dict[str, Any]] = []
    started = time.time()
    with zipfile.ZipFile(archive_path) as archive:
        for index, source in enumerate(selected, 1):
            item: dict[str, Any] = {
                key: source.get(key, "")
                for key in ("entry_name", "split", "group_id", "locale", "era", "category", "set_name")
            }
            try:
                image = _decode_entry(archive, source["entry_name"])
                item.update(_predict_consensus(engine, image))
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                item.update(
                    {
                        "accepted_precluster": False,
                        "accepted": False,
                        "reason": f"{type(exc).__name__}:{exc}",
                    }
                )
            predictions.append(item)
            if index % args.checkpoint_every == 0 or index == len(selected):
                partial = {
                    "processed": index,
                    "selected": len(selected),
                    "elapsed_seconds": time.time() - started,
                    "precluster_accepted": sum(bool(row.get("accepted_precluster")) for row in predictions),
                }
                (output / "progress.json").write_text(
                    json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(json.dumps(partial, ensure_ascii=False), flush=True)

    _apply_layout_consensus(
        predictions,
        radius_px=args.cluster_radius,
        min_cluster_size=max(2, args.min_cluster_size),
    )
    _write_manifest(output / "inner_official_consensus_manifest.csv", predictions, archive_path)
    report = {
        "archive": str(archive_path),
        "splits": str(args.splits.resolve()),
        "model_yolo": str(engine.models.inner_yolo),
        "model_refiner": str(engine.models.inner_refiner),
        "model_top_left": str(engine.models.inner_refiner_top_left),
        "max_per_group": args.max_per_group,
        "cluster_radius_px": args.cluster_radius,
        "min_cluster_size": args.min_cluster_size,
        "elapsed_seconds": time.time() - started,
        **_summary(predictions),
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Keep detailed diagnostics compressed enough for auditing without writing
    # thousands of decoded official images to the nearly-full system drive.
    (output / "predictions.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in predictions),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
