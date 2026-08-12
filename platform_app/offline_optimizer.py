from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


PROFILE_LABELS = {
    "balanced": "均衡微调",
    "precision": "高分辨率精修",
    "robust": "实拍鲁棒增强",
}


def _bounded_int(value: Any, minimum: int, maximum: int) -> int:
    return max(minimum, min(int(value), maximum))


def _segmentation_profiles(target: str, *, memory_gb: float, domain_gap: dict[str, Any]) -> list[dict[str, Any]]:
    base_batch = 2 if memory_gb <= 8.5 else 4
    deltas = domain_gap.get("delta_real_minus_training", {}) if isinstance(domain_gap, dict) else {}
    difficult_lighting = (
        float(deltas.get("highlight_fraction", 0.0)) > 0.012
        or float(deltas.get("shadow_fraction", 0.0)) > 0.012
        or float(deltas.get("sharpness_log", 0.0)) < -0.18
    )
    balanced = {
        "id": "balanced",
        "label": PROFILE_LABELS["balanced"],
        "parameters": {
            "imgsz": 640,
            "batch": base_batch,
            "lr0": 0.0005,
            "optimizer": "AdamW",
            "weight_decay": 0.0005,
            "degrees": 2.0,
            "translate": 0.025,
            "scale": 0.12,
            "perspective": 0.0003,
            "hsv_v": 0.16,
            "close_mosaic": 4,
        },
    }
    precision = {
        "id": "precision",
        "label": PROFILE_LABELS["precision"],
        "parameters": {
            "imgsz": 768,
            "batch": max(1, base_batch // 2),
            "lr0": 0.00025,
            "optimizer": "AdamW",
            "weight_decay": 0.00025,
            "degrees": 1.0,
            "translate": 0.015,
            "scale": 0.08,
            "perspective": 0.0002,
            "hsv_v": 0.12,
            "close_mosaic": 3,
        },
    }
    robust = {
        "id": "robust",
        "label": PROFILE_LABELS["robust"],
        "parameters": {
            "imgsz": 640,
            "batch": base_batch,
            "lr0": 0.0007,
            "optimizer": "AdamW",
            "weight_decay": 0.0007,
            "degrees": 5.0,
            "translate": 0.055,
            "scale": 0.22,
            "perspective": 0.0012,
            "hsv_v": 0.28,
            "close_mosaic": 5,
        },
    }
    # Inner-frame labels are already rectified, so excessive perspective augmentation
    # would teach a geometry that production inference does not normally see.
    if target == "inner_seg":
        for profile in (balanced, precision, robust):
            profile["parameters"]["perspective"] *= 0.5
            profile["parameters"]["degrees"] *= 0.6
    return [balanced, robust, precision] if difficult_lighting else [balanced, precision, robust]


def _refiner_profiles(*, hard_example_replay: bool) -> list[dict[str, Any]]:
    feedback_repeat = 2 if hard_example_replay else 1
    return [
        {
            "id": "balanced",
            "label": PROFILE_LABELS["balanced"],
            "parameters": {
                "lr": 0.0002,
                "weight_decay": 0.0001,
                "train_repeats": 2,
                "band_half": 32,
                "patch_width": 96,
                "patch_height": 256,
                "feedback_repeat": feedback_repeat,
            },
        },
        {
            "id": "precision",
            "label": PROFILE_LABELS["precision"],
            "parameters": {
                "lr": 0.0001,
                "weight_decay": 0.00005,
                "train_repeats": 2,
                "band_half": 24,
                "patch_width": 128,
                "patch_height": 256,
                "feedback_repeat": min(3, feedback_repeat + 1),
            },
        },
        {
            "id": "robust",
            "label": PROFILE_LABELS["robust"],
            "parameters": {
                "lr": 0.00035,
                "weight_decay": 0.0002,
                "train_repeats": 3,
                "band_half": 40,
                "patch_width": 112,
                "patch_height": 256,
                "feedback_repeat": min(4, feedback_repeat + 1),
            },
        },
    ]


def build_search_plan(
    *,
    targets: list[str],
    trial_count: int,
    screening_epochs: int,
    full_epochs: int,
    memory_gb: float,
    domain_gap: dict[str, Any] | None = None,
    hard_example_replay: bool = True,
) -> dict[str, Any]:
    """Create a bounded deterministic search plan with no API or network dependency."""

    count = _bounded_int(trial_count, 1, 3)
    screen = _bounded_int(screening_epochs, 1, max(1, int(full_epochs)))
    plan: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        profiles = (
            _refiner_profiles(hard_example_replay=hard_example_replay)
            if target == "inner_refiner"
            else _segmentation_profiles(target, memory_gb=memory_gb, domain_gap=domain_gap or {})
        )
        selected = []
        for profile in profiles[:count]:
            item = json.loads(json.dumps(profile))
            item["screening_epochs"] = screen
            item["full_epochs"] = int(full_epochs)
            selected.append(item)
        plan[target] = selected
    return {
        "schema_version": "1.0",
        "mode": "offline_bounded_search",
        "trial_count_per_target": count,
        "screening_epochs": screen,
        "full_epochs": int(full_epochs),
        "targets": plan,
        "selection_policy": "validation proxy selects one profile per target; fixed real-photo holdout remains the deployment gate",
    }


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _best_yolo_metric(results_path: Path) -> tuple[float | None, str | None]:
    if not results_path.is_file():
        return None, None
    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None, None
    names = list(rows[0])
    preferred = [
        name for name in names
        if "map50-95" in name.lower() and ("(m)" in name.lower() or "mask" in name.lower())
    ]
    if not preferred:
        preferred = [name for name in names if "map50-95" in name.lower()]
    if not preferred:
        preferred = [name for name in names if "fitness" in name.lower()]
    for name in preferred:
        values = [_float(row.get(name)) for row in rows]
        valid = [value for value in values if value is not None]
        if valid:
            return max(valid), name.strip()
    return None, None


def read_validation_proxy(target: str, report_root: Path) -> dict[str, Any]:
    """Read a comparable validation proxy without loading candidate models onto the GPU."""

    if target in {"outer_seg", "inner_seg"}:
        results = report_root / "run" / "results.csv"
        value, metric = _best_yolo_metric(results)
        return {
            "available": value is not None,
            "metric": metric or "YOLO validation mAP50-95",
            "value": value,
            "direction": "maximize",
            "source": str(results),
        }
    summary_path = report_root / "summary.json"
    summary = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
    value = _float(summary.get("best_validation_mae_px")) if isinstance(summary, dict) else None
    return {
        "available": value is not None,
        "metric": "best_validation_mae_px",
        "value": value,
        "direction": "minimize",
        "source": str(summary_path),
    }


def choose_trial(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        raise ValueError("at least one offline optimization trial is required")
    valid = [trial for trial in trials if trial.get("validation", {}).get("available")]
    if not valid:
        chosen = trials[0]
        reason = "训练脚本没有产生可比较的验证指标，安全回退到均衡方案"
    else:
        direction = valid[0]["validation"].get("direction")
        reverse = direction == "maximize"
        chosen = sorted(valid, key=lambda row: float(row["validation"]["value"]), reverse=reverse)[0]
        reason = f"按 {chosen['validation']['metric']} 自动选择最佳筛选方案"
    return {
        "trial_id": chosen["id"],
        "profile": chosen["profile"],
        "label": chosen["label"],
        "parameters": chosen["parameters"],
        "validation": chosen.get("validation"),
        "reason": reason,
    }


__all__ = ["PROFILE_LABELS", "build_search_plan", "choose_trial", "read_validation_proxy"]
