from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(SCRIPT_DIR / "runtime" / "yolo"))
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / "runtime" / "matplotlib"))

from ultralytics import YOLO

from calibrate_inner_frame_box import calibrate_inner_frame_box
from edge_refiner import EDGE_TO_KEY, EDGES, load_refiner, predict_edge
from stabilize_inner_frame_box import stabilize_inner_frame_box


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def clip_box(box: dict[str, float], width: int, height: int) -> dict[str, float]:
    left = float(np.clip(box["left"], 0, width - 2))
    right = float(np.clip(box["right"], left + 1, width - 1))
    top = float(np.clip(box["top"], 0, height - 2))
    bottom = float(np.clip(box["bottom"], top + 1, height - 1))
    return {
        "left": round(left, 3),
        "right": round(right, 3),
        "top": round(top, 3),
        "bottom": round(bottom, 3),
    }


def internal(box: dict[str, float]) -> dict[str, float]:
    return {
        "x_left": float(box["left"]),
        "x_right": float(box["right"]),
        "y_top": float(box["top"]),
        "y_bottom": float(box["bottom"]),
    }


def external(box: dict[str, float]) -> dict[str, float]:
    return {
        "left": float(box["x_left"]),
        "right": float(box["x_right"]),
        "top": float(box["y_top"]),
        "bottom": float(box["y_bottom"]),
    }


def draw_box(image: np.ndarray, box: dict[str, float], color: tuple[int, int, int], thickness: int) -> None:
    cv2.rectangle(
        image,
        (round(box["left"]), round(box["top"])),
        (round(box["right"]), round(box["bottom"])),
        color,
        thickness,
        cv2.LINE_AA,
    )


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def resolve_torch_device(value: str) -> torch.device:
    normalized = value.strip().lower()
    if normalized == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if normalized.startswith("cuda:"):
        return torch.device(normalized)
    index = normalized.split(",", 1)[0]
    return torch.device(f"cuda:{index}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inner-frame detector with learned high-resolution edge refinement")
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--yolo",
        default="models/inner_frame_yolo_v3_base_candidate.pt",
    )
    parser.add_argument(
        "--refiner",
        default="models/inner_frame_edge_refiner_v4_candidate.pt",
    )
    parser.add_argument("--gate", default="gate.json")
    parser.add_argument("--output", default="prediction_corrected_v4")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    yolo_path = resolve(SCRIPT_DIR, args.yolo).resolve()
    refiner_path = resolve(SCRIPT_DIR, args.refiner).resolve()
    gate_path = resolve(SCRIPT_DIR, args.gate).resolve()

    image = read_image(image_path)
    height, width = image.shape[:2]
    torch_device = resolve_torch_device(args.device)
    yolo_device = args.device if torch.cuda.is_available() or args.device.strip().lower() == "cpu" else "cpu"
    result = YOLO(str(yolo_path)).predict(
        source=image,
        imgsz=args.imgsz,
        conf=args.conf,
        device=yolo_device,
        retina_masks=False,
        verbose=False,
    )[0]
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError("No inner_frame mask detected")
    confidences = result.boxes.conf.detach().cpu().numpy()
    index = int(np.argmax(confidences))
    points = np.asarray(result.masks.xy[index], dtype=np.float64)
    if len(points) < 3:
        raise RuntimeError("Detected mask polygon is invalid")
    raw = {
        "left": float(points[:, 0].min()),
        "right": float(points[:, 0].max()),
        "top": float(points[:, 1].min()),
        "bottom": float(points[:, 1].max()),
    }

    stabilized_result = stabilize_inner_frame_box(image, internal(raw))
    base = external(calibrate_inner_frame_box(stabilized_result.box, width, height))
    refiner, config = load_refiner(refiner_path, torch_device)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    base_internal = internal(base)
    final = dict(base)
    edge_details: dict[str, dict[str, float | bool]] = {}

    for edge in EDGES:
        key = EDGE_TO_KEY[edge]
        prediction = predict_edge(
            refiner,
            image,
            edge,
            base_internal[key],
            base_internal,
            device=torch_device,
            band_half=int(config.get("band_half", 32)),
            patch_width=int(config.get("patch_width", 96)),
            patch_height=int(config.get("patch_height", 192)),
        )
        settings = gate["edges"][edge]
        accepted = (
            prediction.confidence >= float(settings["confidence"])
            and abs(prediction.offset) <= float(settings["max_move"])
        )
        applied_offset = float(settings["blend"]) * prediction.offset if accepted else 0.0
        final[edge] = base[edge] + applied_offset
        edge_details[edge] = {
            "proposed_offset": round(prediction.offset, 4),
            "applied_offset": round(applied_offset, 4),
            "confidence": round(prediction.confidence, 4),
            "entropy": round(prediction.entropy, 4),
            "peak_mass": round(prediction.peak_mass, 4),
            "accepted": accepted,
        }
    final = clip_box(final, width, height)

    payload = {
        "success": True,
        "version": "inner_frame_v4_learned_edge_refinement_candidate",
        "image": str(image_path),
        "image_width": width,
        "image_height": height,
        "yolo_confidence": round(float(confidences[index]), 6),
        "raw_box": {key: round(value, 3) for key, value in raw.items()},
        "base_v3_box": {key: round(value, 3) for key, value in base.items()},
        "final_box": final,
        "stabilizer": {
            "status": stabilized_result.status,
            "reason": stabilized_result.reason,
            "evidence": stabilized_result.evidence,
        },
        "edge_refinement": edge_details,
        "models": {"yolo": str(yolo_path), "refiner": str(refiner_path)},
    }
    (output / "prediction.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    overlay = image.copy()
    draw_box(overlay, base, (0, 0, 255), 2)
    draw_box(overlay, final, (255, 220, 0), 2)
    legend = "v3 red / v4 cyan"
    cv2.putText(overlay, legend, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, legend, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 94])[1].tofile(str(output / "prediction_overlay.jpg"))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
