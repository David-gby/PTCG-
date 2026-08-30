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
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / "runtime" / "matplotlib"))
sys.path.insert(0, str(ROOT))

from inner_frame.edge_refiner import (  # noqa: E402
    EDGES,
    EDGE_TO_KEY,
    EdgeRefiner,
    EdgeRefinerV5,
    augment_bottom_logo_distractor,
    augment_patch,
    augment_patch_v5,
    canonical_target_index,
    localization_loss,
    make_edge_patch,
    patch_to_tensor,
)


def read_manifest(path: Path, split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    for row in rows:
        if row.get("archive") and row.get("entry_name"):
            archive_path = Path(row["archive"])
            if not archive_path.is_absolute():
                archive_path = (path.parent / archive_path).resolve()
            row["archive"] = str(archive_path)
            row["image"] = f"zip::{archive_path}::{row['entry_name']}"
            continue
        image_path = Path(row["image"])
        if not image_path.is_absolute():
            row["image"] = str((path.parent / image_path).resolve())
    return rows


@lru_cache(maxsize=4)
def _open_archive(path_text: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(path_text)


@lru_cache(maxsize=24)
def read_image_cached(path_text: str, expected_width: int, expected_height: int) -> np.ndarray:
    if path_text.startswith("zip::"):
        archive_text, entry_name = path_text[5:].split("::", 1)
        payload = _open_archive(archive_text).read(entry_name)
        data = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot decode ZIP image: {entry_name}")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            image = np.clip(
                image[:, :, :3].astype(np.float32) * alpha + 242.0 * (1.0 - alpha),
                0,
                255,
            ).astype(np.uint8)
        if image.shape[1] != expected_width or image.shape[0] != expected_height:
            image = cv2.resize(
                image,
                (expected_width, expected_height),
                interpolation=cv2.INTER_AREA,
            )
        return image
    path = Path(path_text)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def row_box(row: dict[str, str]) -> dict[str, float]:
    width, height = float(row["width"]), float(row["height"])
    return {
        "x_left": float(row["left"]) * width,
        "x_right": float(row["right"]) * width,
        "y_top": float(row["top"]) * height,
        "y_bottom": float(row["bottom"]) * height,
    }


def sample_jitter(rng: np.random.Generator, band_half: int) -> float:
    selector = float(rng.random())
    if selector < 0.68:
        jitter = float(np.clip(rng.normal(0.0, 4.5), -12.0, 12.0))
    elif selector < 0.92:
        jitter = float(rng.uniform(-18.0, 18.0))
    else:
        hard_limit = max(28.0, float(band_half - 3))
        jitter = float(rng.uniform(-hard_limit, hard_limit))
    return float(np.clip(jitter, -band_half + 2, band_half - 2))


class EdgeStripDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        *,
        training: bool,
        seed: int,
        repeats: int,
        band_half: int,
        patch_width: int,
        patch_height: int,
        architecture: str,
        augmentation: str,
        input_channels: int,
        consistency_views: bool,
        sample_order: str = "row_block",
        source_repeats: dict[str, int] | None = None,
        edge_repeats: dict[str, int] | None = None,
    ) -> None:
        self.rows = rows
        self.training = training
        self.seed = seed
        self.repeats = repeats
        self.band_half = band_half
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.architecture = architecture
        self.augmentation = augmentation
        self.input_channels = input_channels
        self.consistency_views = consistency_views
        self.sample_order = sample_order
        self.source_repeats = source_repeats or {}
        self.edge_repeats = edge_repeats or {}
        self.epoch = 0
        self.order: list[tuple[int, str]] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        base_order = [
            index
            for index, row in enumerate(self.rows)
            for _ in range(max(1, int(self.source_repeats.get(row["source"], 1))))
        ]
        if self.training and self.sample_order == "edge_shuffle":
            self.order = [
                (row_index, edge)
                for _ in range(self.repeats)
                for row_index in base_order
                for edge in EDGES
                for _ in range(max(1, int(self.edge_repeats.get(edge, 1))))
            ]
            random.Random(self.seed + epoch).shuffle(self.order)
            return

        # Keep all edge samples from one image adjacent so the decoded image is
        # reused by read_image_cached.  This is especially important for the
        # official corpus stored in ZIP archives: globally shuffling individual
        # edges otherwise decompresses the same image roughly four times per
        # epoch.  Row blocks and the edge order inside each block are still
        # deterministically shuffled, preserving stochastic training.
        rng = random.Random(self.seed + epoch)
        self.order = []
        for _ in range(self.repeats):
            row_order = list(base_order)
            if self.training:
                rng.shuffle(row_order)
            for row_index in row_order:
                edge_order = list(EDGES)
                if self.training:
                    rng.shuffle(edge_order)
                for edge in edge_order:
                    self.order.extend(
                        (row_index, edge)
                        for _ in range(max(1, int(self.edge_repeats.get(edge, 1))))
                    )

    def __len__(self) -> int:
        return len(self.order)

    def _rng(self, row: dict[str, str], edge: str, index: int) -> np.random.Generator:
        if self.training:
            token = f"{self.seed}:{self.epoch}:{index}:{row['id']}:{edge}"
        else:
            token = f"validation:{self.seed}:{row['id']}:{edge}"
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "little"))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        row_index, edge = self.order[index]
        row = self.rows[row_index]
        image = read_image_cached(row["image"], int(float(row["width"])), int(float(row["height"])))
        box = row_box(row)
        target = box[EDGE_TO_KEY[edge]]
        rng = self._rng(row, edge, index)
        jitter = sample_jitter(rng, self.band_half)
        coarse = target + jitter
        patch = make_edge_patch(
            image,
            edge,
            coarse,
            box,
            band_half=self.band_half,
            patch_width=self.patch_width,
            patch_height=self.patch_height,
        )
        target_position = canonical_target_index(
            edge,
            target,
            coarse,
            band_half=self.band_half,
            patch_width=self.patch_width,
        )
        target_tensor = torch.tensor(target_position, dtype=torch.float32)
        if self.training and self.consistency_views:
            augment = augment_patch_v5 if self.augmentation == "robust" else augment_patch
            clean = patch_to_tensor(patch, input_channels=self.input_channels)
            first_patch = augment(patch, rng)
            second_patch = augment(patch, rng)
            if edge == "bottom":
                first_patch = augment_bottom_logo_distractor(first_patch, target_position, rng)
                second_patch = augment_bottom_logo_distractor(second_patch, target_position, rng)
            view_one = patch_to_tensor(
                first_patch,
                input_channels=self.input_channels,
            )
            view_two = patch_to_tensor(
                second_patch,
                input_channels=self.input_channels,
            )
            return clean, view_one, view_two, target_tensor
        if self.training:
            patch = augment_patch_v5(patch, rng) if self.augmentation == "robust" else augment_patch(patch, rng)
            if edge == "bottom":
                patch = augment_bottom_logo_distractor(patch, target_position, rng)
        tensor = patch_to_tensor(patch, input_channels=self.input_channels)
        return tensor, target_tensor


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp: bool,
    pixel_scale: float,
    consistency_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    distribution_losses: list[float] = []
    regression_losses: list[float] = []
    consistency_losses: list[float] = []
    errors: list[float] = []

    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch in loader:
            if len(batch) == 4:
                clean_images, view_one, view_two, target_positions = batch
                images = torch.cat((clean_images, view_one, view_two), dim=0).to(
                    device, non_blocking=True
                )
                target_positions = target_positions.to(device, non_blocking=True)
                repeated_targets = target_positions.repeat(3)
            else:
                images, target_positions = batch
                images = images.to(device, non_blocking=True)
                target_positions = target_positions.to(device, non_blocking=True)
                repeated_targets = target_positions
            target_positions = target_positions.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss, details = localization_loss(logits, repeated_targets)
                consistency_loss = logits.new_zeros(())
                metric_logits = logits
                if len(batch) == 4:
                    clean_logits, first_logits, second_logits = logits.chunk(3, dim=0)
                    probabilities = torch.stack(
                        (
                            clean_logits.softmax(dim=1),
                            first_logits.softmax(dim=1),
                            second_logits.softmax(dim=1),
                        ),
                        dim=0,
                    )
                    mixture = probabilities.mean(dim=0).clamp_min(1e-8)
                    consistency_loss = (
                        probabilities
                        * (probabilities.clamp_min(1e-8).log() - mixture.log()[None])
                    ).sum(dim=2).mean()
                    loss = loss + consistency_weight * consistency_loss
                    metric_logits = (clean_logits + first_logits + second_logits) / 3.0
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            losses.append(float(loss.detach().item()))
            distribution_losses.append(float(details["distribution"].item()))
            regression_losses.append(float(details["regression"].item()))
            consistency_losses.append(float(consistency_loss.detach().item()))
            metric_probabilities = metric_logits.detach().softmax(dim=1)
            coordinates = torch.arange(
                metric_logits.shape[1],
                device=metric_logits.device,
                dtype=metric_logits.dtype,
            )
            expected = (metric_probabilities * coordinates[None]).sum(dim=1)
            batch_errors = (expected - target_positions).abs() * pixel_scale
            errors.extend(float(value) for value in batch_errors.detach().cpu().tolist())

    return {
        "loss": statistics.fmean(losses),
        "distribution_loss": statistics.fmean(distribution_losses),
        "regression_loss": statistics.fmean(regression_losses),
        "consistency_loss": statistics.fmean(consistency_losses),
        "mae_px": statistics.fmean(errors),
        "p95_px": float(np.percentile(errors, 95)),
        "max_px": max(errors),
    }


def save_checkpoint(
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a high-resolution four-edge strip refiner")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "training" / "data" / "inner_refiner_manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "training_outputs" / "inner_refiner",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--train-repeats", type=int, default=2)
    parser.add_argument("--band-half", type=int, default=32)
    parser.add_argument("--patch-width", type=int, default=96)
    parser.add_argument("--patch-height", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--balance-sources", action="store_true")
    parser.add_argument("--feedback-repeat", type=int, default=1)
    parser.add_argument("--left-repeat", type=int, default=1)
    parser.add_argument("--right-repeat", type=int, default=1)
    parser.add_argument("--top-repeat", type=int, default=1)
    parser.add_argument("--bottom-repeat", type=int, default=1)
    parser.add_argument("--architecture", choices=("v4", "v5"), default="v4")
    parser.add_argument("--augmentation", choices=("auto", "legacy", "robust"), default="auto")
    parser.add_argument("--consistency-weight", type=float, default=0.35)
    parser.add_argument("--consistency-views", action="store_true")
    parser.add_argument("--no-consistency-views", action="store_true")
    parser.add_argument("--selection", choices=("mean", "tail"), default="mean")
    parser.add_argument(
        "--sample-order",
        choices=("row_block", "edge_shuffle"),
        default="row_block",
        help="row_block reuses each decoded image across its four edge samples",
    )
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    train_rows = read_manifest(args.manifest.resolve(), "train")
    val_rows = read_manifest(args.manifest.resolve(), "val")
    if not train_rows or not val_rows:
        raise RuntimeError("Training and validation rows are required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    initial_model = args.model.resolve() if args.model else None

    input_channels = 7 if args.architecture == "v5" else 4
    augmentation = args.augmentation
    if augmentation == "auto":
        augmentation = "robust" if args.architecture == "v5" else "legacy"
    consistency_views = (
        (args.architecture == "v5" or args.consistency_views)
        and not args.no_consistency_views
    )
    config = {
        "architecture": args.architecture,
        "augmentation": augmentation,
        "input_channels": input_channels,
        "band_half": args.band_half,
        "patch_width": args.patch_width,
        "patch_height": args.patch_height,
        "manifest": str(args.manifest.resolve()),
        "initial_model": str(initial_model) if initial_model else None,
        "train_images": len(train_rows),
        "val_images": len(val_rows),
        "seed": args.seed,
        "balance_sources": args.balance_sources,
        "feedback_repeat": max(1, args.feedback_repeat),
        "left_repeat": max(1, args.left_repeat),
        "right_repeat": max(1, args.right_repeat),
        "top_repeat": max(1, args.top_repeat),
        "bottom_repeat": max(1, args.bottom_repeat),
        "weight_decay": args.weight_decay,
        "patience": max(1, args.patience),
        "consistency_views": consistency_views,
        "consistency_weight": args.consistency_weight if consistency_views else 0.0,
        "tta_height_flip": args.architecture == "v5",
        "selection": args.selection,
        "sample_order": args.sample_order,
    }
    source_repeats: dict[str, int] = {}
    if args.balance_sources:
        source_repeats.update({
            "final_v4_hardcase": 1,
            "manual_v2_precision": 1,
            "real_rectified_20260713": 5,
            "xiuzheng_20260713": 6,
            "jiance_20260716": 5,
        })
    if args.feedback_repeat > 1:
        source_repeats["human_feedback"] = max(1, args.feedback_repeat)
        source_repeats["human_feedback_hard"] = max(1, args.feedback_repeat + 1)
    if source_repeats:
        config["source_repeats"] = source_repeats
    edge_repeats = {
        "left": max(1, args.left_repeat),
        "right": max(1, args.right_repeat),
        "top": max(1, args.top_repeat),
        "bottom": max(1, args.bottom_repeat),
    }
    config["edge_repeats"] = edge_repeats
    train_dataset = EdgeStripDataset(
        train_rows,
        training=True,
        seed=args.seed,
        repeats=args.train_repeats,
        band_half=args.band_half,
        patch_width=args.patch_width,
        patch_height=args.patch_height,
        architecture=args.architecture,
        augmentation=augmentation,
        input_channels=input_channels,
        consistency_views=consistency_views,
        sample_order=args.sample_order,
        source_repeats=source_repeats or None,
        edge_repeats=edge_repeats,
    )
    val_dataset = EdgeStripDataset(
        val_rows,
        training=False,
        seed=args.seed,
        repeats=1,
        band_half=args.band_half,
        patch_width=args.patch_width,
        patch_height=args.patch_height,
        architecture=args.architecture,
        augmentation=augmentation,
        input_channels=input_channels,
        consistency_views=False,
        sample_order="row_block",
        edge_repeats=None,
    )
    workers = max(0, int(args.workers))
    loader_options = {
        "num_workers": workers,
        "pin_memory": amp,
    }
    if workers > 0:
        loader_options.update({"prefetch_factor": 2})
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=False,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        **loader_options,
    )

    if args.architecture == "v5":
        model: torch.nn.Module = EdgeRefinerV5(input_channels=input_channels).to(device)
    else:
        model = EdgeRefiner(input_channels=input_channels).to(device)
    if initial_model is not None:
        payload = torch.load(str(initial_model), map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.04)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    pixel_scale = (2.0 * args.band_half) / float(args.patch_width - 1)

    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_validation_metrics: dict[str, float] | None = None
    epochs_without_improvement = 0
    stopped_early = False
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
            pixel_scale=pixel_scale,
            consistency_weight=args.consistency_weight if consistency_views else 0.0,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device=device,
            optimizer=None,
            scaler=scaler,
            amp=amp,
            pixel_scale=pixel_scale,
            consistency_weight=0.0,
        )
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "elapsed_seconds": time.time() - started,
        }
        validation_score = (
            val_metrics["mae_px"]
            if args.selection == "mean"
            else val_metrics["mae_px"]
            + 0.15 * val_metrics["p95_px"]
            + 0.02 * val_metrics["max_px"]
        )
        record["validation_selection_score"] = validation_score
        history.append(record)
        save_checkpoint(output / "last.pt", model, optimizer, epoch, config, val_metrics)
        if validation_score < best_score:
            best_score = validation_score
            best_validation_metrics = dict(val_metrics)
            epochs_without_improvement = 0
            save_checkpoint(output / "best.pt", model, optimizer, epoch, config, val_metrics)
        else:
            epochs_without_improvement += 1
        (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        scheduler.step()
        if epochs_without_improvement >= max(1, args.patience):
            stopped_early = True
            break

    summary = {
        "config": config,
        "epochs": args.epochs,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "best_validation_selection_score": best_score,
        "best_validation_metrics": best_validation_metrics,
        "last": history[-1],
        "elapsed_seconds": time.time() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
