from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


SPLITS = ("train", "val", "test")
SOURCE_ASPECT = 630.0 / 880.0


def _seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _read_splits(path: Path) -> dict[str, list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {split: [] for split in SPLITS}
    for row in rows:
        if row["split"] in result:
            result[row["split"]].append(row)
    return result


def _select(rows: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, str]]:
    ranked = sorted(rows, key=lambda row: _seed(row["entry_name"], seed))
    return ranked[: min(limit, len(ranked))]


def _background(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    palettes = np.asarray(
        [
            (235, 235, 232),
            (202, 198, 188),
            (116, 105, 94),
            (63, 68, 70),
            (28, 34, 32),
            (171, 154, 132),
            (210, 214, 218),
            (91, 112, 124),
        ],
        dtype=np.float32,
    )
    base = palettes[int(rng.integers(0, len(palettes)))].copy()
    image = np.empty((height, width, 3), dtype=np.float32)
    image[:] = base
    yy, xx = np.mgrid[:height, :width]
    angle = float(rng.uniform(0.0, math.tau))
    gradient = (
        np.cos(angle) * (xx / max(width - 1, 1) - 0.5)
        + np.sin(angle) * (yy / max(height - 1, 1) - 0.5)
    )
    image += gradient[..., None] * float(rng.uniform(-42.0, 42.0))
    coarse = rng.normal(0.0, 1.0, (max(2, height // 48), max(2, width // 48))).astype(
        np.float32
    )
    coarse = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    fine = rng.normal(0.0, float(rng.uniform(1.0, 7.0)), image.shape).astype(np.float32)
    image += coarse[..., None] * float(rng.uniform(4.0, 18.0)) + fine
    if rng.random() < 0.30:
        period = float(rng.uniform(24.0, 100.0))
        wood = np.sin((xx + 0.14 * coarse * width) / period * math.tau)
        image += wood[..., None] * float(rng.uniform(2.0, 11.0))
    return np.clip(image, 0, 255).astype(np.uint8)


def _quad(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    close = bool(rng.random() < 0.28)
    card_height = height * float(rng.uniform(0.82, 0.93) if close else rng.uniform(0.58, 0.84))
    card_width = card_height * SOURCE_ASPECT * float(rng.uniform(0.985, 1.015))
    angle_limit = 4.0 if close else 11.0
    angle = math.radians(float(rng.uniform(-angle_limit, angle_limit)))
    center = np.asarray(
        [
            width * 0.5 + rng.uniform(-0.08, 0.08) * width,
            height * 0.5 + rng.uniform(-0.055, 0.055) * height,
        ],
        dtype=np.float32,
    )
    local = np.asarray(
        [
            (-card_width / 2, -card_height / 2),
            (card_width / 2, -card_height / 2),
            (card_width / 2, card_height / 2),
            (-card_width / 2, card_height / 2),
        ],
        dtype=np.float32,
    )
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float32,
    )
    points = local @ rotation.T + center
    perspective = (0.010 if close else 0.035) * card_height
    points += rng.normal(0.0, perspective, points.shape).astype(np.float32)
    margin = 8.0
    span = points.max(axis=0) - points.min(axis=0)
    scale = min(1.0, (width - 2 * margin) / max(span[0], 1.0), (height - 2 * margin) / max(span[1], 1.0))
    if scale < 1.0:
        points = (points - center) * scale + center
    shift = np.zeros(2, dtype=np.float32)
    low, high = points.min(axis=0), points.max(axis=0)
    shift += np.maximum(margin - low, 0.0)
    shift -= np.maximum(high + shift - np.asarray([width - margin, height - margin]), 0.0)
    return (points + shift).astype(np.float32)


def _source_image(blob: bytes, alpha_candidate: bool) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(io.BytesIO(blob)) as source:
        source.seek(0)
        normalized = ImageOps.exif_transpose(source)
        rgba = np.asarray(normalized.convert("RGBA"), dtype=np.uint8)
    rgb = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)
    if alpha_candidate:
        mask = rgba[..., 3]
    else:
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        radius = max(2, round(rgb.shape[1] * 0.035))
        cv2.rectangle(mask, (radius, 0), (rgb.shape[1] - radius - 1, rgb.shape[0] - 1), 255, -1)
        cv2.rectangle(mask, (0, radius), (rgb.shape[1] - 1, rgb.shape[0] - radius - 1), 255, -1)
        for x, y in (
            (radius, radius),
            (rgb.shape[1] - radius - 1, radius),
            (rgb.shape[1] - radius - 1, rgb.shape[0] - radius - 1),
            (radius, rgb.shape[0] - radius - 1),
        ):
            cv2.circle(mask, (x, y), radius, 255, -1)
    return rgb, mask


def _expanded(points: np.ndarray, factor: float) -> np.ndarray:
    center = points.mean(axis=0, keepdims=True)
    return (center + (points - center) * factor).astype(np.int32)


def _render(
    card: np.ndarray,
    source_mask: np.ndarray,
    points: np.ndarray,
    rng: np.random.Generator,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    image = _background(rng, height, width)
    source_points = np.asarray(
        [[0, 0], [card.shape[1] - 1, 0], [card.shape[1] - 1, card.shape[0] - 1], [0, card.shape[0] - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source_points, points)
    warped_card = cv2.warpPerspective(card, transform, (width, height), flags=cv2.INTER_CUBIC)
    warped_mask = cv2.warpPerspective(source_mask, transform, (width, height), flags=cv2.INTER_LINEAR)

    sleeve = bool(rng.random() < 0.55)
    sleeve_width = 0.0
    if sleeve:
        sleeve_width = float(rng.uniform(1.018, 1.105))
        sleeve_poly = _expanded(points, sleeve_width)
        sleeve_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(sleeve_mask, sleeve_poly, 255)
        rim = sleeve_mask.copy()
        inner = np.zeros_like(rim)
        cv2.fillConvexPoly(inner, points.astype(np.int32), 255)
        rim = cv2.subtract(rim, cv2.erode(inner, np.ones((3, 3), np.uint8)))
        tint = np.full_like(image, int(rng.integers(150, 245)))
        alpha = (rim.astype(np.float32) / 255.0 * float(rng.uniform(0.12, 0.42)))[..., None]
        image = np.clip(image * (1.0 - alpha) + tint * alpha, 0, 255).astype(np.uint8)
        cv2.polylines(
            image,
            [sleeve_poly],
            True,
            tuple(int(value) for value in rng.integers(100, 235, 3)),
            int(rng.integers(1, 4)),
            cv2.LINE_AA,
        )

    shadow_type = str(rng.choice(["soft", "hard_gray", "double", "none"], p=[0.34, 0.34, 0.20, 0.12]))
    if shadow_type != "none":
        dx = int(rng.integers(-24, 25))
        dy = int(rng.integers(5, 29))
        shifted = cv2.warpAffine(warped_mask, np.float32([[1, 0, dx], [0, 1, dy]]), (width, height))
        blur = int(rng.integers(1, 5) if shadow_type == "hard_gray" else rng.integers(9, 31))
        if blur % 2 == 0:
            blur += 1
        shifted = cv2.GaussianBlur(shifted, (blur, blur), 0)
        opacity = float(rng.uniform(0.17, 0.52))
        alpha = shifted.astype(np.float32)[..., None] / 255.0 * opacity
        shadow_color = np.full_like(image, int(rng.integers(30, 125)))
        image = np.clip(image * (1.0 - alpha) + shadow_color * alpha, 0, 255).astype(np.uint8)
        if shadow_type == "double":
            shifted2 = cv2.warpAffine(warped_mask, np.float32([[1, 0, -dx // 2], [0, 1, dy // 2]]), (width, height))
            shifted2 = cv2.GaussianBlur(shifted2, (41, 41), 0)
            alpha2 = shifted2.astype(np.float32)[..., None] / 255.0 * 0.12
            image = np.clip(image * (1.0 - alpha2), 0, 255).astype(np.uint8)

    alpha = (warped_mask.astype(np.float32) / 255.0)[..., None]
    image = np.clip(image * (1.0 - alpha) + warped_card * alpha, 0, 255).astype(np.uint8)

    glare = bool(rng.random() < 0.48)
    if glare:
        overlay = np.zeros_like(image)
        x1 = int(rng.uniform(-0.15, 0.65) * width)
        x2 = x1 + int(rng.uniform(0.12, 0.34) * width)
        cv2.line(overlay, (x1, 0), (x2, height), (255, 255, 255), int(rng.integers(18, 76)), cv2.LINE_AA)
        overlay = cv2.GaussianBlur(overlay, (0, 0), float(rng.uniform(7.0, 22.0)))
        glare_alpha = (overlay.max(axis=2).astype(np.float32) / 255.0) * (warped_mask.astype(np.float32) / 255.0)
        glare_alpha = (glare_alpha * float(rng.uniform(0.12, 0.42)))[..., None]
        image = np.clip(image * (1.0 - glare_alpha) + 255.0 * glare_alpha, 0, 255).astype(np.uint8)

    gamma = float(rng.uniform(0.72, 1.35))
    lut = np.asarray([np.clip((value / 255.0) ** gamma * 255.0, 0, 255) for value in range(256)], dtype=np.uint8)
    image = cv2.LUT(image, lut)
    if rng.random() < 0.30:
        sigma = float(rng.uniform(0.35, 1.45))
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    if rng.random() < 0.36:
        noise = rng.normal(0.0, float(rng.uniform(1.0, 6.0)), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image, warped_mask, {
        "shadow_type": shadow_type,
        "sleeve": sleeve,
        "sleeve_scale": sleeve_width,
        "glare": glare,
        "gamma": gamma,
    }


def _segmentation_label(mask: np.ndarray, width: int, height: int) -> str:
    contours, _ = cv2.findContours((mask >= 96).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("empty warped card mask")
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(0.8, 0.0025 * cv2.arcLength(contour, True))
    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(polygon) > 48:
        indices = np.linspace(0, len(polygon) - 1, 48).round().astype(int)
        polygon = polygon[indices]
    values = ["0"]
    for x, y in polygon:
        values.extend((f"{np.clip(x / width, 0, 1):.6f}", f"{np.clip(y / height, 0, 1):.6f}"))
    return " ".join(values) + "\n"


def _pose_label(points: np.ndarray, width: int, height: int) -> str:
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2.0
    size = high - low
    values: list[str] = [
        "0",
        f"{center[0] / width:.6f}",
        f"{center[1] / height:.6f}",
        f"{size[0] / width:.6f}",
        f"{size[1] / height:.6f}",
    ]
    for x, y in points:
        values.extend((f"{x / width:.6f}", f"{y / height:.6f}", "2"))
    return " ".join(values) + "\n"


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _add_history(output: Path, history: Path, repeats: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        image_dir = history / "images" / split
        label_dir = history / "labels" / split
        split_repeats = repeats if split == "train" else 1
        for image_path in sorted(path for path in image_dir.glob("*") if path.is_file()):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue
            for repeat in range(split_repeats):
                name = f"history_r{repeat}_{image_path.stem}"
                target_image = output / "outer_seg" / "images" / split / f"{name}{image_path.suffix.lower()}"
                target_label = output / "outer_seg" / "labels" / split / f"{name}.txt"
                _link(image_path, target_image)
                _link(label_path, target_label)
                rows.append({"sample_id": name, "split": split, "source": "historical_real", "entry_name": str(image_path)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate exact-geometry outer-card synthetic training data.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=6000)
    parser.add_argument("--val", type=int, default=900)
    parser.add_argument("--test", type=int, default=900)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--history-repeats", type=int, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=896)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output is not empty: {args.output}")

    split_rows = _read_splits(args.splits)
    limits = {"train": args.train, "val": args.val, "test": args.test}
    selected = {
        split: _select(split_rows[split], limits[split], args.seed + index)
        for index, split in enumerate(SPLITS)
    }
    metadata: list[dict[str, object]] = []
    with zipfile.ZipFile(args.archive) as archive:
        total = sum(len(rows) for rows in selected.values())
        done = 0
        for split in SPLITS:
            for index, row in enumerate(selected[split]):
                entry_name = row["entry_name"]
                rng = np.random.default_rng(_seed(entry_name, args.seed))
                card, source_mask = _source_image(
                    archive.read(entry_name), bool(int(row["official_alpha_candidate"] or 0))
                )
                points = _quad(rng, args.height, args.width)
                image, mask, factors = _render(card, source_mask, points, rng, args.height, args.width)
                sample_id = f"official_{split}_{index:05d}_{str(row['blob_sha256'])[:10]}"
                image_path = args.output / "outer_seg" / "images" / split / f"{sample_id}.jpg"
                seg_label = args.output / "outer_seg" / "labels" / split / f"{sample_id}.txt"
                pose_image = args.output / "outer_pose" / "images" / split / f"{sample_id}.jpg"
                pose_label = args.output / "outer_pose" / "labels" / split / f"{sample_id}.txt"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                seg_label.parent.mkdir(parents=True, exist_ok=True)
                pose_label.parent.mkdir(parents=True, exist_ok=True)
                cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])[1].tofile(str(image_path))
                seg_label.write_text(_segmentation_label(mask, args.width, args.height), encoding="utf-8")
                _link(image_path, pose_image)
                pose_label.write_text(_pose_label(points, args.width, args.height), encoding="utf-8")
                metadata.append(
                    {
                        "sample_id": sample_id,
                        "split": split,
                        "source": "official_synthetic",
                        "entry_name": entry_name,
                        "group_id": row["group_id"],
                        "locale": row["locale"],
                        "era": row["era"],
                        "points": json.dumps(points.round(3).tolist()),
                        **factors,
                    }
                )
                done += 1
                if done % 250 == 0:
                    print(f"generated {done}/{total}", flush=True)

    if args.history:
        metadata.extend(_add_history(args.output, args.history, max(1, args.history_repeats)))

    for task, extra in (("outer_seg", {}), ("outer_pose", {"kpt_shape": [4, 3], "flip_idx": [1, 0, 3, 2]})):
        data = {
            "path": str((args.output / task).resolve()).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: "card"},
            **extra,
        }
        lines = [f"path: {data['path']}", "train: images/train", "val: images/val", "test: images/test", "names:", "  0: card"]
        if task == "outer_pose":
            lines.extend(("kpt_shape: [4, 3]", "flip_idx: [1, 0, 3, 2]"))
        (args.output / task / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fields = sorted({key for row in metadata for key in row})
    with (args.output / "metadata.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata)
    summary = {
        "official": {split: len(selected[split]) for split in SPLITS},
        "historical_real": {
            split: sum(row["source"] == "historical_real" and row["split"] == split for row in metadata)
            for split in SPLITS
        },
        "total": {split: sum(row["split"] == split for row in metadata) for split in SPLITS},
        "image_size": [args.width, args.height],
        "history_repeats": args.history_repeats if args.history else 0,
        "seed": args.seed,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
