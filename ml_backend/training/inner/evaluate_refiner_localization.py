from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from inner_frame.edge_refiner import EdgeRefiner, EdgeRefinerV5  # noqa: E402
from train_refiner import EdgeStripDataset, read_manifest  # noqa: E402


def metrics(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mae_px": statistics.fmean(values) if values else None,
        "p95_px": float(np.percentile(values, 95)) if values else None,
        "max_px": max(values) if values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an edge refiner without YOLO")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload = torch.load(str(args.model.resolve()), map_location=device, weights_only=False)
    config = dict(payload.get("config", {}))
    architecture = str(config.get("architecture", "v4"))
    input_channels = int(config.get("input_channels", 7 if architecture == "v5" else 4))
    model: torch.nn.Module
    if architecture == "v5":
        model = EdgeRefinerV5(input_channels=input_channels).to(device)
    else:
        model = EdgeRefiner(input_channels=input_channels).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    rows = read_manifest(args.manifest.resolve(), args.split)
    dataset = EdgeStripDataset(
        rows,
        training=False,
        seed=int(config.get("seed", 20260717)),
        repeats=1,
        band_half=int(config.get("band_half", 32)),
        patch_width=int(config.get("patch_width", 96)),
        patch_height=int(config.get("patch_height", 256)),
        architecture=architecture,
        augmentation=str(config.get("augmentation", "robust" if architecture == "v5" else "legacy")),
        input_channels=input_channels,
        consistency_views=False,
        sample_order="row_block",
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    scale = (2.0 * dataset.band_half) / float(dataset.patch_width - 1)
    samples: list[dict[str, Any]] = []
    cursor = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probabilities = logits.softmax(dim=1)
            coordinates = torch.arange(logits.shape[1], device=device, dtype=logits.dtype)
            predicted = (probabilities * coordinates[None]).sum(dim=1).cpu().numpy()
            targets_np = targets.numpy()
            for predicted_position, target_position in zip(predicted, targets_np):
                row_index, edge = dataset.order[cursor]
                row = rows[row_index]
                samples.append(
                    {
                        "id": row.get("id", str(row_index)),
                        "source": row.get("source", ""),
                        "edge": edge,
                        "error_px": abs(float(predicted_position) - float(target_position)) * scale,
                    }
                )
                cursor += 1

    grouped: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        error = float(sample["error_px"])
        grouped["all"].append(error)
        grouped[f"edge:{sample['edge']}"] .append(error)
        grouped[f"source:{sample['source']}"] .append(error)
    summary = {
        "manifest": str(args.manifest.resolve()),
        "model": str(args.model.resolve()),
        "split": args.split,
        "architecture": architecture,
        "groups": {name: metrics(values) for name, values in sorted(grouped.items())},
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "per_sample.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "source", "edge", "error_px"))
        writer.writeheader()
        writer.writerows(samples)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
