from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / "runtime" / "matplotlib"))
sys.path.insert(0, str(ROOT))

from card_quality_processor.outer_detection import order_points  # noqa: E402
from card_quality_processor.outer_line_refiner import (  # noqa: E402
    CANONICAL_SIDE_LENGTHS,
    make_outer_side_patch,
    source_offset_to_canonical_position,
)
from inner_frame.edge_refiner import (  # noqa: E402
    EdgeRefinerV5,
    augment_patch_v5,
    localization_loss,
    patch_to_tensor,
)


def _read_rows(path: Path, split: str) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("split") == split]
    normalized: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        image = Path(row["image"])
        label = Path(row["label"])
        if not image.is_absolute():
            image = (path.parent / image).resolve()
        if not label.is_absolute():
            label = (path.parent / label).resolve()
        if not image.is_file() or not label.is_file():
            continue
        normalized.append(
            {
                "id": str(row.get("id") or f"{path.stem}_{index}"),
                "image": str(image),
                "label": str(label),
                "source": str(
                    row.get("source")
                    or row.get("dataset_source")
                    or path.stem
                ),
            }
        )
    return normalized


@lru_cache(maxsize=8)
def _read_image(path_text: str) -> np.ndarray:
    data = np.fromfile(path_text, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path_text}")
    return image


@lru_cache(maxsize=512)
def _read_quad(label_text: str, width: int, height: int) -> np.ndarray:
    line = Path(label_text).read_text(encoding="utf-8-sig").splitlines()[0].strip()
    values = [float(value) for value in line.split()]
    coordinates = values[1:]
    if len(coordinates) < 8 or len(coordinates) % 2:
        raise ValueError(f"Expected polygon label: {label_text}")
    polygon = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
    polygon[:, 0] *= float(width)
    polygon[:, 1] *= float(height)
    if len(polygon) == 4:
        return order_points(polygon)
    hull = cv2.convexHull(polygon.reshape(-1, 1, 2))
    rectangle = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    return order_points(rectangle)


def _outward_normal(
    start: np.ndarray,
    end: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    tangent = end - start
    tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    if float(np.dot(normal, 0.5 * (start + end) - center)) < 0.0:
        normal = -normal
    return normal


def _sample_jitter(rng: np.random.Generator, hard_limit: float) -> float:
    selector = float(rng.random())
    if selector < 0.58:
        value = float(np.clip(rng.normal(0.0, 4.5), -13.0, 13.0))
    elif selector < 0.88:
        value = float(rng.uniform(-18.0, 18.0))
    else:
        value = float(rng.uniform(-hard_limit, hard_limit))
    return float(np.clip(value, -hard_limit, hard_limit))


def _add_outer_distractors(
    patch: np.ndarray,
    target_position: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add broad cast-shadow and sleeve-like decoy edges without moving the label."""

    image = patch.astype(np.float32)
    height, width = image.shape[:2]
    target = float(np.clip(target_position, 4.0, width - 5.0))

    if rng.random() < 0.68 and target > 8.0:
        shadow_edge = float(rng.uniform(1.5, max(2.0, target - 3.0)))
        x = np.arange(width, dtype=np.float32)
        softness = float(rng.uniform(1.5, 8.0))
        mask = 1.0 / (1.0 + np.exp((x - shadow_edge) / softness))
        strength = float(rng.uniform(0.28, 0.70))
        tint = rng.uniform(0.78, 1.08, (1, 1, 3)).astype(np.float32)
        image *= 1.0 - mask[None, :, None] * strength * tint

    if rng.random() < 0.42:
        candidates = [
            value
            for value in (
                target - float(rng.uniform(4.0, 22.0)),
                target + float(rng.uniform(4.0, 20.0)),
            )
            if 2.0 <= value <= width - 3.0
        ]
        if candidates:
            edge = int(round(float(rng.choice(candidates))))
            overlay = np.clip(image, 0, 255).astype(np.uint8)
            color_level = int(rng.uniform(45, 235))
            for _ in range(int(rng.integers(1, 4))):
                x = int(np.clip(edge + int(rng.integers(-2, 3)), 1, width - 2))
                cv2.line(
                    overlay,
                    (x, 0),
                    (x + int(rng.integers(-1, 2)), height - 1),
                    (color_level, color_level, color_level),
                    int(rng.integers(1, 3)),
                    cv2.LINE_AA,
                )
            alpha = float(rng.uniform(0.25, 0.65))
            image = cv2.addWeighted(
                overlay.astype(np.float32),
                alpha,
                image,
                1.0 - alpha,
                0.0,
            )
    return np.clip(image, 0, 255).astype(np.uint8)


class OuterSideDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        *,
        training: bool,
        seed: int,
        repeats: int,
        band_canonical_px: float,
        patch_width: int,
        patch_height: int,
        feedback_repeat: int,
    ) -> None:
        self.rows = rows
        self.training = training
        self.seed = seed
        self.repeats = repeats
        self.band_canonical_px = band_canonical_px
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.feedback_repeat = feedback_repeat
        self.epoch = 0
        self.order: list[tuple[int, int]] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        base: list[int] = []
        for row_index, row in enumerate(self.rows):
            repeat = (
                self.feedback_repeat
                if "feedback" in row["source"].lower()
                else 1
            )
            base.extend([row_index] * max(1, repeat))
        rng = random.Random(self.seed + epoch)
        self.order = []
        for _ in range(self.repeats):
            epoch_rows = list(base)
            if self.training:
                rng.shuffle(epoch_rows)
            for row_index in epoch_rows:
                sides = list(range(4))
                if self.training:
                    rng.shuffle(sides)
                self.order.extend((row_index, side_index) for side_index in sides)

    def __len__(self) -> int:
        return len(self.order)

    def _rng(self, row: dict[str, str], side_index: int, index: int) -> np.random.Generator:
        scope = self.epoch if self.training else "validation"
        token = f"{self.seed}:{scope}:{index}:{row['id']}:{side_index}"
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "little"))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row_index, side_index = self.order[index]
        row = self.rows[row_index]
        image = _read_image(row["image"])
        target_quad = _read_quad(row["label"], image.shape[1], image.shape[0]).copy()
        rng = self._rng(row, side_index, index)

        center_fraction = float(rng.uniform(0.16, 0.84))
        span_fraction = float(rng.uniform(0.24, 0.40))
        hard_limit = max(20.0, self.band_canonical_px - 3.0)
        jitter_start = _sample_jitter(rng, hard_limit)
        jitter_end = _sample_jitter(rng, hard_limit)
        target_start = target_quad[side_index].copy()
        target_end = target_quad[(side_index + 1) % 4].copy()
        target_center = target_quad.mean(axis=0)
        target_normal = _outward_normal(target_start, target_end, target_center)
        side_length = float(np.linalg.norm(target_end - target_start))
        source_per_canonical = side_length / CANONICAL_SIDE_LENGTHS[side_index]

        coarse_quad = target_quad.copy()
        coarse_quad[side_index] += jitter_start * source_per_canonical * target_normal
        coarse_quad[(side_index + 1) % 4] += (
            jitter_end * source_per_canonical * target_normal
        )
        patch, metadata = make_outer_side_patch(
            image,
            coarse_quad,
            side_index,
            center_fraction=center_fraction,
            span_fraction=span_fraction,
            band_canonical_px=self.band_canonical_px,
            patch_width=self.patch_width,
            patch_height=self.patch_height,
        )

        target_point = (
            target_start
            + center_fraction * (target_end - target_start)
        )
        base_point = np.asarray(metadata["base_point"], dtype=np.float32)
        outward = np.asarray(metadata["outward_normal"], dtype=np.float32)
        target_source_offset = float(np.dot(target_point - base_point, outward))
        target_position = source_offset_to_canonical_position(
            target_source_offset,
            band_source_px=float(metadata["band_source_px"]),
            patch_width=self.patch_width,
        )
        if self.training:
            patch = _add_outer_distractors(patch, target_position, rng)
            patch = augment_patch_v5(patch, rng)
        tensor = patch_to_tensor(patch, input_channels=7)
        return tensor, torch.tensor(target_position, dtype=torch.float32)


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    pixel_scale: float,
    inward_penalty_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    errors: list[float] = []
    signed_errors: list[float] = []
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss, details = localization_loss(
                    logits,
                    targets,
                    sigma=1.4,
                    regression_weight=0.16,
                )
                if training and inward_penalty_weight > 0.0:
                    inward_index_error = torch.relu(details["expected"] - targets)
                    inward_penalty = F.smooth_l1_loss(
                        inward_index_error,
                        torch.zeros_like(inward_index_error),
                        beta=1.0,
                    )
                    loss = loss + inward_penalty_weight * inward_penalty
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            losses.append(float(loss.detach().item()))
            expected = details["expected"]
            signed = (expected - targets) * pixel_scale
            signed_values = [float(value) for value in signed.detach().cpu().tolist()]
            signed_errors.extend(signed_values)
            errors.extend(abs(value) for value in signed_values)
    inward_errors = [max(value, 0.0) for value in signed_errors]
    return {
        "loss": statistics.fmean(losses),
        "mae_canonical_px": statistics.fmean(errors),
        "p95_canonical_px": float(np.percentile(errors, 95)),
        "max_canonical_px": max(errors),
        "signed_bias_canonical_px": statistics.fmean(signed_errors),
        "mean_inward_canonical_px": statistics.fmean(inward_errors),
        "p95_inward_canonical_px": float(np.percentile(inward_errors, 95)),
    }


def _save(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train independent high-resolution outer-side refiner.")
    parser.add_argument("--history-manifest", type=Path, required=True)
    parser.add_argument("--feedback-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--weight-decay", type=float, default=0.00015)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--train-repeats", type=int, default=2)
    parser.add_argument("--feedback-repeat", type=int, default=3)
    parser.add_argument("--band-canonical-px", type=float, default=32.0)
    parser.add_argument("--patch-width", type=int, default=96)
    parser.add_argument("--patch-height", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--initial-model", type=Path, default=None)
    parser.add_argument("--inward-penalty-weight", type=float, default=0.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    history_manifest = args.history_manifest.resolve()
    feedback_manifest = args.feedback_manifest.resolve() if args.feedback_manifest else None
    train_rows = _read_rows(history_manifest, "train")
    val_rows = _read_rows(history_manifest, "val")
    if feedback_manifest is not None:
        train_rows.extend(_read_rows(feedback_manifest, "train"))
        val_rows.extend(_read_rows(feedback_manifest, "val"))
    if not train_rows or not val_rows:
        raise RuntimeError("Training and validation rows are required")

    config = {
        "architecture": "outer_line_refiner_v1",
        "input_channels": 7,
        "history_manifest": str(history_manifest),
        "feedback_manifest": str(feedback_manifest) if feedback_manifest else None,
        "train_images": len(train_rows),
        "val_images": len(val_rows),
        "band_canonical_px": args.band_canonical_px,
        "patch_width": args.patch_width,
        "patch_height": args.patch_height,
        "train_repeats": args.train_repeats,
        "feedback_repeat": args.feedback_repeat,
        "seed": args.seed,
        "shadow_distractor_augmentation": True,
        "sleeve_line_distractor_augmentation": True,
        "initial_model": str(args.initial_model.resolve()) if args.initial_model else None,
        "inward_penalty_weight": args.inward_penalty_weight,
    }
    train_dataset = OuterSideDataset(
        train_rows,
        training=True,
        seed=args.seed,
        repeats=args.train_repeats,
        band_canonical_px=args.band_canonical_px,
        patch_width=args.patch_width,
        patch_height=args.patch_height,
        feedback_repeat=args.feedback_repeat,
    )
    val_dataset = OuterSideDataset(
        val_rows,
        training=False,
        seed=args.seed,
        repeats=1,
        band_canonical_px=args.band_canonical_px,
        patch_width=args.patch_width,
        patch_height=args.patch_height,
        feedback_repeat=1,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = EdgeRefinerV5(input_channels=7).to(device)
    if args.initial_model is not None:
        payload = torch.load(
            str(args.initial_model.resolve()),
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(payload["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.04,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    pixel_scale = (2.0 * args.band_canonical_px) / float(args.patch_width - 1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    best_score = math.inf
    best_metrics: dict[str, float] | None = None
    history: list[dict[str, Any]] = []
    no_improvement = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            pixel_scale=pixel_scale,
            inward_penalty_weight=args.inward_penalty_weight,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            device=device,
            optimizer=None,
            scaler=scaler,
            pixel_scale=pixel_scale,
            inward_penalty_weight=0.0,
        )
        selection_score = (
            val_metrics["mae_canonical_px"]
            + 0.18 * val_metrics["p95_canonical_px"]
            + 0.01 * val_metrics["max_canonical_px"]
        )
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "selection_score": selection_score,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        _save(output / "last.pt", model, optimizer, epoch, config, val_metrics)
        if selection_score < best_score:
            best_score = selection_score
            best_metrics = dict(val_metrics)
            no_improvement = 0
            _save(output / "best.pt", model, optimizer, epoch, config, val_metrics)
        else:
            no_improvement += 1
        (output / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)
        scheduler.step()
        if no_improvement >= args.patience:
            break

    summary = {
        "config": config,
        "epochs_completed": len(history),
        "best_selection_score": best_score,
        "best_validation_metrics": best_metrics,
        "last": history[-1],
        "elapsed_seconds": time.time() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
