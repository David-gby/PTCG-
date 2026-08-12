from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
RUNTIME_ROOT = Path(
    os.environ.get("PTCG_RUNTIME_DIR", Path(tempfile.gettempdir()) / "ptcg_model_runtime")
)
(RUNTIME_ROOT / "yolo" / "Ultralytics").mkdir(parents=True, exist_ok=True)
(RUNTIME_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_ROOT / "yolo"))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_ROOT / "matplotlib"))
sys.path.insert(0, str(ROOT))

from inner_frame.calibrate_inner_frame_box import calibrate_inner_frame_box  # noqa: E402
from inner_frame.edge_refiner import EDGE_TO_KEY, EDGES, load_refiner, predict_edge  # noqa: E402
from inner_frame.stabilize_inner_frame_box import stabilize_inner_frame_box  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def read_manifest(path: Path, splits: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] in splits]
    for row in rows:
        image_path = Path(row["image"])
        if not image_path.is_absolute():
            row["image"] = str((path.parent / image_path).resolve())
    return rows


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def gt_box(row: dict[str, str]) -> dict[str, float]:
    width, height = float(row["width"]), float(row["height"])
    return {
        "left": float(row["left"]) * width,
        "right": float(row["right"]) * width,
        "top": float(row["top"]) * height,
        "bottom": float(row["bottom"]) * height,
    }


def prediction_box(result) -> tuple[dict[str, float] | None, float]:
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return None, 0.0
    confidences = result.boxes.conf.detach().cpu().numpy()
    index = int(np.argmax(confidences))
    polygons = result.masks.xy
    if index >= len(polygons) or len(polygons[index]) < 3:
        return None, float(confidences[index])
    points = np.asarray(polygons[index], dtype=np.float64)
    return {
        "left": float(points[:, 0].min()),
        "right": float(points[:, 0].max()),
        "top": float(points[:, 1].min()),
        "bottom": float(points[:, 1].max()),
    }, float(confidences[index])


def to_internal(box: dict[str, float]) -> dict[str, float]:
    return {
        "x_left": float(box["left"]),
        "x_right": float(box["right"]),
        "y_top": float(box["top"]),
        "y_bottom": float(box["bottom"]),
    }


def from_internal(box: dict[str, float]) -> dict[str, float]:
    return {
        "left": float(box["x_left"]),
        "right": float(box["x_right"]),
        "top": float(box["y_top"]),
        "bottom": float(box["y_bottom"]),
    }


def clip_box(box: dict[str, float], width: int, height: int) -> dict[str, float]:
    left = float(np.clip(box["left"], 0, width - 2))
    right = float(np.clip(box["right"], left + 1, width - 1))
    top = float(np.clip(box["top"], 0, height - 2))
    bottom = float(np.clip(box["bottom"], top + 1, height - 1))
    return {"left": left, "right": right, "top": top, "bottom": bottom}


def box_iou(a: dict[str, float], b: dict[str, float]) -> float:
    left, right = max(a["left"], b["left"]), min(a["right"], b["right"])
    top, bottom = max(a["top"], b["top"]), min(a["bottom"], b["bottom"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a["right"] - a["left"]) * max(0.0, a["bottom"] - a["top"])
    area_b = max(0.0, b["right"] - b["left"]) * max(0.0, b["bottom"] - b["top"])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def outer_snap(pred: dict[str, float], gt: dict[str, float], width: int, height: int) -> bool:
    margins_pred = {
        "left": pred["left"],
        "right": width - pred["right"],
        "top": pred["top"],
        "bottom": height - pred["bottom"],
    }
    margins_gt = {
        "left": gt["left"],
        "right": width - gt["right"],
        "top": gt["top"],
        "bottom": height - gt["bottom"],
    }
    dimensions = {"left": width, "right": width, "top": height, "bottom": height}
    return any(
        margins_pred[edge] <= 0.015 * dimensions[edge]
        and margins_gt[edge] >= 0.025 * dimensions[edge]
        and abs(pred[edge] - gt[edge]) >= 8.0
        for edge in EDGES
    )


def summarize(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    detected = [row for row in rows if row.get("detected")]
    ious = [float(row[f"{mode}_iou"]) for row in detected]
    edge_errors = {
        edge: [float(row[f"{mode}_{edge}_error"]) for row in detected]
        for edge in EDGES
    }
    all_edge_errors = [value for edge in EDGES for value in edge_errors[edge]]
    per_image_mae = [
        statistics.fmean(float(row[f"{mode}_{edge}_error"]) for edge in EDGES)
        for row in detected
    ]
    return {
        "images": len(rows),
        "detected": len(detected),
        "detection_rate": len(detected) / len(rows) if rows else 0.0,
        "mean_iou": statistics.fmean(ious) if ious else None,
        "min_iou": min(ious) if ious else None,
        "p05_iou": float(np.percentile(ious, 5)) if ious else None,
        "edge_mae": statistics.fmean(all_edge_errors) if all_edge_errors else None,
        "edge_p95": float(np.percentile(all_edge_errors, 95)) if all_edge_errors else None,
        "image_mae_p95": float(np.percentile(per_image_mae, 95)) if per_image_mae else None,
        "outer_snap_count": sum(bool(row[f"{mode}_outer_snap"]) for row in detected),
        **{
            f"{edge}_mae": statistics.fmean(edge_errors[edge]) if edge_errors[edge] else None
            for edge in EDGES
        },
        **{
            f"{edge}_p95": float(np.percentile(edge_errors[edge], 95)) if edge_errors[edge] else None
            for edge in EDGES
        },
    }


def add_mode_metrics(
    row: dict[str, Any],
    mode: str,
    box: dict[str, float],
    gt: dict[str, float],
    width: int,
    height: int,
) -> None:
    row[f"{mode}_iou"] = box_iou(box, gt)
    row[f"{mode}_outer_snap"] = outer_snap(box, gt, width, height)
    for edge in EDGES:
        row[f"{mode}_{edge}"] = box[edge]
        row[f"{mode}_{edge}_error"] = abs(box[edge] - gt[edge])


def refine_candidate(
    refiner,
    config: dict[str, Any],
    image: np.ndarray,
    base: dict[str, float],
    device: torch.device,
) -> tuple[dict[str, float], dict[str, Any]]:
    internal = to_internal(base)
    candidate = dict(base)
    details: dict[str, Any] = {}
    for edge in EDGES:
        key = EDGE_TO_KEY[edge]
        prediction = predict_edge(
            refiner,
            image,
            edge,
            internal[key],
            internal,
            device=device,
            band_half=int(config.get("band_half", 32)),
            patch_width=int(config.get("patch_width", 96)),
            patch_height=int(config.get("patch_height", 256)),
        )
        candidate[edge] = prediction.refined
        details[f"refiner_{edge}_offset"] = prediction.offset
        details[f"refiner_{edge}_confidence"] = prediction.confidence
        details[f"refiner_{edge}_entropy"] = prediction.entropy
        details[f"refiner_{edge}_peak_mass"] = prediction.peak_mass
        details[f"refiner_{edge}_tta_disagreement"] = prediction.tta_disagreement
    height, width = image.shape[:2]
    return clip_box(candidate, width, height), details


def fit_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    config: dict[str, Any] = {"fit_split": "val", "edges": {}}
    confidence_values = np.arange(0.20, 0.91, 0.05)
    max_moves = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 24.0, 32.0)
    blends = (0.25, 0.50, 0.75, 1.0)

    for edge in EDGES:
        base_errors = [abs(float(row[f"base_{edge}"]) - float(row[f"gt_{edge}"])) for row in rows]
        base_mean = statistics.fmean(base_errors)
        base_p95 = float(np.percentile(base_errors, 95))
        base_max = max(base_errors)
        best = {
            "confidence": 1.01,
            "max_move": 0.0,
            "blend": 0.0,
            "mae": base_mean,
            "p95": base_p95,
            "max": base_max,
            "accepted": 0,
            "score": base_mean + 0.14 * base_p95 + 0.02 * base_max,
        }
        for confidence in confidence_values:
            for max_move in max_moves:
                for blend in blends:
                    errors: list[float] = []
                    accepted = 0
                    for row in rows:
                        offset = float(row[f"refiner_{edge}_offset"])
                        use = (
                            float(row[f"refiner_{edge}_confidence"]) >= float(confidence)
                            and abs(offset) <= max_move
                        )
                        value = float(row[f"base_{edge}"])
                        if use:
                            value += blend * offset
                            accepted += 1
                        errors.append(abs(value - float(row[f"gt_{edge}"])))
                    mean = statistics.fmean(errors)
                    p95 = float(np.percentile(errors, 95))
                    maximum = max(errors)
                    score = mean + 0.14 * p95 + 0.02 * maximum
                    safe_tail = p95 <= base_p95 + 0.35 and maximum <= base_max + 0.75
                    if safe_tail and score < float(best["score"]):
                        best = {
                            "confidence": round(float(confidence), 4),
                            "max_move": max_move,
                            "blend": blend,
                            "mae": mean,
                            "p95": p95,
                            "max": maximum,
                            "accepted": accepted,
                            "score": score,
                        }
        config["edges"][edge] = best
    return config


def apply_gate(row: dict[str, Any], gate: dict[str, Any], width: int, height: int) -> dict[str, float]:
    output = {edge: float(row[f"base_{edge}"]) for edge in EDGES}
    for edge in EDGES:
        settings = gate["edges"][edge]
        offset = float(row[f"refiner_{edge}_offset"])
        if (
            float(row[f"refiner_{edge}_confidence"]) >= float(settings["confidence"])
            and abs(offset) <= float(settings["max_move"])
        ):
            output[edge] += float(settings["blend"]) * offset
    return clip_box(output, width, height)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end evaluation for the learned edge refiner")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "training" / "data" / "inner_refiner_manifest.csv",
    )
    parser.add_argument(
        "--yolo",
        type=Path,
        default=ROOT / "models" / "inner_frame_yolo_v3_base_candidate.pt",
    )
    parser.add_argument(
        "--refiner",
        type=Path,
        default=ROOT / "models" / "inner_frame_edge_refiner_v4_candidate.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "training_outputs" / "inner_refiner_evaluation",
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows_by_split: dict[str, list[dict[str, Any]]] = {"val": [], "test": []}
    manifest_rows = read_manifest(args.manifest.resolve(), set(rows_by_split))
    yolo = YOLO(str(args.yolo.resolve()))
    refiner, refiner_config = load_refiner(args.refiner.resolve(), device)

    for index, source_row in enumerate(manifest_rows, 1):
        split = source_row["split"]
        image = read_image(Path(source_row["image"]))
        height, width = image.shape[:2]
        gt = gt_box(source_row)
        result = yolo.predict(
            source=image,
            imgsz=args.imgsz,
            conf=0.05,
            device=0 if device.type == "cuda" else "cpu",
            retina_masks=False,
            verbose=False,
        )[0]
        raw, confidence = prediction_box(result)
        row: dict[str, Any] = {
            "id": source_row["id"],
            "image": source_row["image"],
            "split": split,
            "source": source_row["source"],
            "width": width,
            "height": height,
            "detected": raw is not None,
            "yolo_confidence": confidence,
        }
        for edge in EDGES:
            row[f"gt_{edge}"] = gt[edge]
        if raw is None:
            rows_by_split[split].append(row)
            continue

        stabilized = stabilize_inner_frame_box(image, to_internal(raw))
        base = from_internal(calibrate_inner_frame_box(stabilized.box, width, height))
        candidate, details = refine_candidate(refiner, refiner_config, image, base, device)
        row.update(details)
        add_mode_metrics(row, "base", base, gt, width, height)
        add_mode_metrics(row, "candidate", candidate, gt, width, height)
        rows_by_split[split].append(row)
        if index % 25 == 0:
            print(f"evaluated {index}/{len(manifest_rows)}", flush=True)

    gate = fit_gate([row for row in rows_by_split["val"] if row.get("detected")])
    for split, rows in rows_by_split.items():
        for row in rows:
            if not row.get("detected"):
                continue
            gated = apply_gate(row, gate, int(row["width"]), int(row["height"]))
            gt = {edge: float(row[f"gt_{edge}"]) for edge in EDGES}
            add_mode_metrics(row, "gated", gated, gt, int(row["width"]), int(row["height"]))

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    summaries: dict[str, Any] = {
        "yolo": str(args.yolo.resolve()),
        "refiner": str(args.refiner.resolve()),
        "refiner_config": refiner_config,
        "gate": gate,
        "splits": {},
    }
    for split, rows in rows_by_split.items():
        write_rows(output / split / "per_image.csv", rows)
        split_summary = {mode: summarize(rows, mode) for mode in ("base", "candidate", "gated")}
        summaries["splits"][split] = split_summary
        (output / split / "summary.json").write_text(
            json.dumps(split_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output / "comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
