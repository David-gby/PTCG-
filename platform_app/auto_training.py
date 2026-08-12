from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import secrets
import shutil
import statistics
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

from studio.errors import StudioError
from studio.security import atomic_json, read_json, sha256_file, utc_now


APP_ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = APP_ROOT / "ml_backend"
MODEL_ROOT = ML_ROOT / "models"
MODEL_MANIFEST = ML_ROOT / "model_manifest.json"
ACTIVE_JOB_STATUSES = {"queued", "preparing", "training", "evaluating", "promoting"}
TERMINAL_JOB_STATUSES = {"completed", "failed", "blocked"}
VALID_TARGETS = {"outer_seg", "inner_seg", "inner_refiner"}
MODEL_FILES = {
    "outer_seg": "outer_seg.pt",
    "inner_seg": "inner_frame_yolo_v3_base_candidate.pt",
    "inner_refiner": "inner_frame_edge_refiner_v4_candidate.pt",
}
DEFAULT_SETTINGS: dict[str, Any] = {
    "schema_version": "1.0",
    "enabled": False,
    "minimum_approved_samples": 20,
    "minimum_new_samples": 10,
    "epochs": 25,
    "history_limit": 100,
    "offline_optimization": True,
    "optimization_trials": 2,
    "screening_epochs": 6,
    "hard_example_replay": True,
    "targets": ["outer_seg", "inner_seg", "inner_refiner"],
    "auto_promote": True,
    "require_quality_gate": True,
    "updated_at": None,
}


def _json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        value = read_json(path)
    except (OSError, ValueError, StudioError):
        return default
    return value


def training_root(private_root: Path) -> Path:
    root = private_root / "auto_training"
    root.mkdir(parents=True, exist_ok=True)
    (root / "jobs").mkdir(parents=True, exist_ok=True)
    (root / "deployments").mkdir(parents=True, exist_ok=True)
    return root


def settings_path(private_root: Path) -> Path:
    return training_root(private_root) / "settings.json"


def load_settings(private_root: Path) -> dict[str, Any]:
    value = _json(settings_path(private_root), {})
    result = dict(DEFAULT_SETTINGS)
    if isinstance(value, dict):
        result.update({key: value[key] for key in DEFAULT_SETTINGS if key in value})
    return result


def save_settings(private_root: Path, value: dict[str, Any]) -> dict[str, Any]:
    settings = dict(DEFAULT_SETTINGS)
    settings.update(value)
    settings["schema_version"] = "1.0"
    settings["updated_at"] = utc_now()
    atomic_json(settings_path(private_root), settings)
    return settings


def list_jobs(private_root: Path, limit: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in (training_root(private_root) / "jobs").glob("trn_*/job.json"):
        value = _json(path, None)
        if isinstance(value, dict):
            rows.append(value)
    rows.sort(key=lambda row: str(row.get("created_at", "")), reverse=True)
    return rows[: max(1, min(int(limit), 100))]


def active_job(private_root: Path) -> dict[str, Any] | None:
    return next((job for job in list_jobs(private_root, 100) if job.get("status") in ACTIVE_JOB_STATUSES), None)


def active_deployment(private_root: Path) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    for path in (training_root(private_root) / "deployments").glob("mdl_*/deployment.json"):
        value = _json(path, None)
        if isinstance(value, dict) and value.get("status") == "active":
            rows.append(value)
    rows.sort(key=lambda row: str(row.get("created_at", "")), reverse=True)
    return rows[0] if rows else None


def gpu_status() -> dict[str, Any]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        name = torch.cuda.get_device_name(0) if available else None
        memory_gb = None
        if available:
            memory_gb = round(float(torch.cuda.get_device_properties(0).total_memory) / (1024**3), 2)
        return {"available": available, "name": name, "memory_gb": memory_gb}
    except Exception as exc:
        return {"available": False, "name": None, "memory_gb": None, "error": f"{type(exc).__name__}: {exc}"}


def active_model_manifest() -> dict[str, Any]:
    value = _json(MODEL_MANIFEST, {})
    if not isinstance(value, dict):
        value = {}
    return {
        "package_version": value.get("package_version", "unknown"),
        "pipeline_version": value.get("pipeline_version", "unknown"),
        "manifest_sha256": sha256_file(MODEL_MANIFEST) if MODEL_MANIFEST.is_file() else None,
        "models": value.get("models", []),
    }


def launch_worker(job_path: Path) -> int:
    log_path = job_path.parent / "worker.log"
    command = [sys.executable, str(APP_ROOT / "platform_training_worker.py"), "--job", str(job_path.resolve())]
    creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    with log_path.open("ab", buffering=0) as handle:
        process = subprocess.Popen(
            command,
            cwd=str(APP_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            close_fds=True,
        )
    return int(process.pid)


def create_job(private_root: Path, feedback_ids: list[str], settings: dict[str, Any], *, trigger: str) -> dict[str, Any]:
    if active_job(private_root):
        raise RuntimeError("another automatic training job is already active")
    job_id = f"trn_{secrets.token_hex(10)}"
    root = training_root(private_root) / "jobs" / job_id
    root.mkdir(parents=False, exist_ok=False)
    path = root / "job.json"
    now = utc_now()
    job = {
        "schema_version": "1.0",
        "id": job_id,
        "status": "queued",
        "trigger": trigger,
        "created_at": now,
        "updated_at": now,
        "workspace_root": str(private_root.parent.resolve()),
        "app_root": str(APP_ROOT.resolve()),
        "job_path": str(path.resolve()),
        "feedback_ids": list(feedback_ids),
        "approved_snapshot_count": len(feedback_ids),
        "targets": list(settings["targets"]),
        "epochs": int(settings["epochs"]),
        "history_limit": int(settings["history_limit"]),
        "offline_optimization": bool(settings["offline_optimization"]),
        "optimization_trials": int(settings["optimization_trials"]),
        "screening_epochs": int(settings["screening_epochs"]),
        "hard_example_replay": bool(settings["hard_example_replay"]),
        "auto_promote": bool(settings["auto_promote"]),
        "require_quality_gate": bool(settings["require_quality_gate"]),
        "dataset_dir": str((root / "dataset").resolve()),
        "candidate_dir": str((root / "candidates").resolve()),
        "report_dir": str((root / "reports").resolve()),
        "log_path": str((root / "worker.log").resolve()),
        "target_statuses": {target: {"status": "queued"} for target in settings["targets"]},
        "candidate_models": [],
        "offline_optimization_report": None,
        "quality_gate": None,
        "deployment": None,
        "production_models_changed": False,
        "safety_note": "Production weights change only after a paired real-photo holdout quality gate passes.",
    }
    atomic_json(path, job)
    try:
        job["worker_pid"] = launch_worker(path)
        job["updated_at"] = utc_now()
        atomic_json(path, job)
    except Exception:
        job["status"] = "failed"
        job["failed_at"] = utc_now()
        job["error"] = "automatic training worker could not be launched"
        atomic_json(path, job)
        raise
    return job


def _write_job(job_path: Path, job: dict[str, Any], status: str | None = None, **updates: Any) -> None:
    if status is not None:
        job["status"] = status
    job.update(updates)
    job["updated_at"] = utc_now()
    atomic_json(job_path, job)


def _stable_splits(private_root: Path, ids: list[str]) -> dict[str, str]:
    path = training_root(private_root) / "split_registry.json"
    registry = _json(path, {"schema_version": "1.0", "assignments": {}})
    if not isinstance(registry, dict) or not isinstance(registry.get("assignments"), dict):
        registry = {"schema_version": "1.0", "assignments": {}}
    assignments: dict[str, str] = {
        str(key): str(value)
        for key, value in registry["assignments"].items()
        if value in {"train", "val", "test"}
    }
    new_ids = [value for value in ids if value not in assignments]
    if not assignments and new_ids:
        ranked = sorted(new_ids, key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
        test_count = max(5, round(len(ranked) * 0.20))
        val_count = max(3, round(len(ranked) * 0.15))
        if test_count + val_count >= len(ranked):
            test_count = max(1, len(ranked) // 5)
            val_count = max(1, len(ranked) // 6)
        for index, value in enumerate(ranked):
            assignments[value] = "test" if index < test_count else "val" if index < test_count + val_count else "train"
    else:
        for value in new_ids:
            bucket = int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 100
            assignments[value] = "test" if bucket < 20 else "val" if bucket < 35 else "train"
    registry = {"schema_version": "1.0", "updated_at": utc_now(), "assignments": assignments}
    atomic_json(path, registry)
    return {value: assignments[value] for value in ids}


def _copy_approved_feedback(service: Any, selected_ids: set[str], destination: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    (destination / "annotations").mkdir(parents=True, exist_ok=True)
    exported_ids: list[str] = []
    annotations: dict[str, dict[str, Any]] = {}
    for item in service.database.list_feedback(status="approved", limit=1000):
        if item["id"] not in selected_ids or not item.get("exported_feedback_id"):
            continue
        exported_id = str(item["exported_feedback_id"])
        source_root = Path(service.store.root) / "ml_feedback" / item["project_id"]
        annotation_path = source_root / "annotations" / f"{exported_id}.json"
        if not annotation_path.is_file():
            continue
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        source_image = source_root / str(annotation.get("image", {}).get("path") or "")
        source_rectified = source_root / str(annotation.get("rectification", {}).get("image_path") or "")
        if not source_image.is_file() or not source_rectified.is_file():
            continue
        image_target = destination / "original_images" / f"{exported_id}{source_image.suffix.lower()}"
        rectified_target = destination / "rectified_images" / f"{exported_id}{source_rectified.suffix.lower()}"
        image_target.parent.mkdir(parents=True, exist_ok=True)
        rectified_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, image_target)
        shutil.copy2(source_rectified, rectified_target)
        annotation["image"]["path"] = image_target.relative_to(destination).as_posix()
        annotation["rectification"]["image_path"] = rectified_target.relative_to(destination).as_posix()
        (destination / "annotations" / f"{exported_id}.json").write_text(
            json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        exported_ids.append(exported_id)
        annotations[exported_id] = annotation
    return exported_ids, annotations


def _remove_holdout_leakage(dataset_root: Path, holdout_hashes: set[str]) -> dict[str, int]:
    removed = {"outer_seg": 0, "inner_seg": 0, "inner_refiner": 0}
    for target in ("outer_seg", "inner_seg"):
        image_dir = dataset_root / target / "images" / "train"
        label_dir = dataset_root / target / "labels" / "train"
        if not image_dir.is_dir():
            continue
        for image in list(image_dir.glob("history_*")):
            if sha256_file(image) not in holdout_hashes:
                continue
            image.unlink(missing_ok=True)
            (label_dir / f"{image.stem}.txt").unlink(missing_ok=True)
            removed[target] += 1
    manifest = dataset_root / "inner_refiner_manifest.csv"
    if manifest.is_file():
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0]) if rows else ["id", "image", "width", "height", "left", "right", "top", "bottom", "split", "source"]
        retained: list[dict[str, Any]] = []
        for row in rows:
            image = dataset_root / row["image"] if not Path(row["image"]).is_absolute() else Path(row["image"])
            if row.get("source") == "historical_human_label" and image.is_file() and sha256_file(image) in holdout_hashes:
                removed["inner_refiner"] += 1
                continue
            retained.append(row)
        with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(retained)
    return removed


def _annotation_difficulty(annotation: dict[str, Any]) -> dict[str, Any]:
    issue_tags = list(annotation.get("review", {}).get("issue_tags") or [])
    outer_error = 0.0
    inner_error = 0.0
    try:
        predicted = annotation.get("outer_frame", {}).get("prediction", {}).get("points")
        corrected = annotation.get("outer_frame", {}).get("correction", {}).get("points")
        width = float(annotation.get("image", {}).get("width") or 1)
        height = float(annotation.get("image", {}).get("height") or 1)
        if isinstance(predicted, list) and isinstance(corrected, list) and len(predicted) == len(corrected) == 4:
            distances = [
                math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))
                for left, right in zip(predicted, corrected)
            ]
            outer_error = statistics.fmean(distances) / max(1.0, math.hypot(width, height)) * 100.0
    except (KeyError, TypeError, ValueError, statistics.StatisticsError):
        outer_error = 0.0
    try:
        predicted_box = annotation.get("inner_frame", {}).get("prediction", {}).get("box")
        corrected_box = annotation.get("inner_frame", {}).get("correction", {}).get("box")
        if isinstance(predicted_box, dict) and isinstance(corrected_box, dict):
            inner_error = statistics.fmean(
                abs(float(predicted_box[edge]) - float(corrected_box[edge]))
                for edge in ("left", "right", "top", "bottom")
            )
    except (KeyError, TypeError, ValueError, statistics.StatisticsError):
        inner_error = 0.0
    hard = bool(issue_tags) or outer_error >= 0.35 or inner_error >= 1.0
    return {
        "hard": hard,
        "issue_tags": issue_tags,
        "outer_error_percent_diagonal": round(outer_error, 6),
        "inner_error_px": round(inner_error, 6),
    }


def _apply_hard_example_replay(
    dataset_root: Path,
    annotations: dict[str, dict[str, Any]],
    splits: dict[str, str],
) -> dict[str, Any]:
    """Add one extra training-only copy of corrected high-error feedback samples."""

    hard_ids = [
        sample_id for sample_id, annotation in annotations.items()
        if splits.get(sample_id) == "train" and _annotation_difficulty(annotation)["hard"]
    ]
    duplicated = {"outer_seg": 0, "inner_seg": 0, "inner_refiner": 0}
    for target in ("outer_seg", "inner_seg"):
        image_dir = dataset_root / target / "images" / "train"
        label_dir = dataset_root / target / "labels" / "train"
        for sample_id in hard_ids:
            source_image = next((path for path in image_dir.glob(f"{sample_id}.*") if path.is_file()), None)
            source_label = label_dir / f"{sample_id}.txt"
            if source_image is None or not source_label.is_file():
                continue
            replay_stem = f"hard_replay_{sample_id}"
            shutil.copy2(source_image, image_dir / f"{replay_stem}{source_image.suffix.lower()}")
            shutil.copy2(source_label, label_dir / f"{replay_stem}.txt")
            duplicated[target] += 1

    manifest_path = dataset_root / "inner_refiner_manifest.csv"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        fields = list(rows[0]) if rows else [
            "id", "image", "width", "height", "left", "right", "top", "bottom", "split", "source"
        ]
        extra: list[dict[str, Any]] = []
        hard_set = set(hard_ids)
        for row in rows:
            if row.get("split") != "train" or row.get("id") not in hard_set:
                continue
            copy_row = dict(row)
            copy_row["id"] = f"hard_replay_{row['id']}"
            copy_row["source"] = "human_feedback_hard"
            extra.append(copy_row)
        if extra:
            with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows([*rows, *extra])
            duplicated["inner_refiner"] = len(extra)
    return {
        "enabled": True,
        "hard_sample_count": len(hard_ids),
        "hard_sample_ids": hard_ids,
        "duplicates_added": duplicated,
        "policy": "only corrected training-split feedback is replayed; validation and test samples are never duplicated",
    }


def prepare_dataset(service: Any, job: dict[str, Any]) -> dict[str, Any]:
    feedback_root = Path(job["job_path"]).parent / "feedback"
    exported_ids, annotations = _copy_approved_feedback(service, set(job["feedback_ids"]), feedback_root)
    if len(exported_ids) < 5:
        raise RuntimeError("fewer than five approved feedback packages are still available")
    splits = _stable_splits(service.private_root, exported_ids)
    split_counts = dict(Counter(splits.values()))
    if split_counts.get("test", 0) < 1 or split_counts.get("val", 0) < 1 or split_counts.get("train", 0) < 1:
        raise RuntimeError("stable train/validation/test split could not be created")

    if str(ML_ROOT) not in sys.path:
        sys.path.insert(0, str(ML_ROOT))
    from feedback_dataset import convert_feedback_to_training

    dataset_root = Path(job["dataset_dir"])
    conversion = convert_feedback_to_training(feedback_root, dataset_root, sample_splits=splits)
    inner_history = service._append_historical_inner(dataset_root, int(job["history_limit"]))
    outer_history = service._append_historical_outer(dataset_root, int(job["history_limit"]))
    holdout_hashes: set[str] = set()
    for exported_id, split in splits.items():
        if split != "test":
            continue
        annotation = annotations[exported_id]
        for key in ("image", "rectification"):
            relative = annotation["image"]["path"] if key == "image" else annotation["rectification"]["image_path"]
            path = feedback_root / relative
            if path.is_file():
                holdout_hashes.add(sha256_file(path))
    leakage = _remove_holdout_leakage(dataset_root, holdout_hashes)
    hard_replay = (
        _apply_hard_example_replay(dataset_root, annotations, splits)
        if job.get("hard_example_replay")
        else {"enabled": False, "hard_sample_count": 0, "duplicates_added": {}}
    )
    summary = {
        "feedback_count": len(exported_ids),
        "feedback_splits": split_counts,
        "split_registry": str(training_root(service.private_root) / "split_registry.json"),
        "conversion": conversion,
        "historical_inner": inner_history,
        "historical_outer": outer_history,
        "holdout_leakage_removed": leakage,
        "hard_example_replay": hard_replay,
        "feedback_root": str(feedback_root),
    }
    (dataset_root / "auto_training_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load training script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_module(path: Path, name: str, args: list[str]) -> None:
    module = _load_module(path, name)
    result = module.main(args)
    if result not in (None, 0):
        raise RuntimeError(f"training script {path.name} exited with {result}")


def _release_training_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _optimization_input_analysis(job: dict[str, Any]) -> dict[str, Any]:
    dataset_root = Path(job["dataset_dir"])
    feedback_root = Path(job["job_path"]).parent / "feedback"
    registry = _json(training_root(Path(job["workspace_root"]) / "private") / "split_registry.json", {})
    assignments = registry.get("assignments", {}) if isinstance(registry, dict) else {}
    train_images = sorted(path for path in (dataset_root / "outer_seg" / "images" / "train").glob("*") if path.is_file())
    holdout_images: list[Path] = []
    issue_tags: Counter[str] = Counter()
    difficulty: list[dict[str, Any]] = []
    for annotation_path in sorted((feedback_root / "annotations").glob("*.json")):
        annotation = _json(annotation_path, {})
        if not isinstance(annotation, dict):
            continue
        sample_id = annotation_path.stem
        if assignments.get(sample_id) == "test":
            image = feedback_root / str(annotation.get("image", {}).get("path") or "")
            if image.is_file():
                holdout_images.append(image)
        if assignments.get(sample_id) == "train":
            issue_tags.update(annotation.get("review", {}).get("issue_tags") or [])
            item = _annotation_difficulty(annotation)
            item["sample_id"] = sample_id
            difficulty.append(item)
    training_features = _mean_features(train_images)
    real_features = _mean_features(holdout_images)
    domain_gap = {
        "training_mix": training_features,
        "real_photo_holdout": real_features,
        "delta_real_minus_training": {
            key: round(real_features[key] - training_features[key], 6)
            for key in training_features.keys() & real_features.keys()
        },
    }
    return {
        "train_image_count": len(train_images),
        "holdout_image_count": len(holdout_images),
        "issue_tags": dict(issue_tags.most_common()),
        "hard_training_sample_count": sum(1 for row in difficulty if row.get("hard")),
        "domain_gap": domain_gap,
    }


def _run_target_training(
    *,
    job: dict[str, Any],
    target: str,
    model: Path,
    output: Path,
    run_root: Path,
    epochs: int,
    parameters: dict[str, Any],
    module_suffix: str,
    memory_gb: float,
) -> None:
    scripts = ML_ROOT / "training"
    dataset = Path(job["dataset_dir"])
    output.parent.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    if target in {"outer_seg", "inner_seg"}:
        arguments = [
            "--data", str(dataset / target / "data.yaml"), "--model", str(model),
            "--epochs", str(epochs), "--imgsz", str(parameters["imgsz"]),
            "--batch", str(parameters["batch"]), "--device", "0",
            "--lr0", str(parameters["lr0"]), "--seed", "20260721",
            "--optimizer", str(parameters["optimizer"]),
            "--weight-decay", str(parameters["weight_decay"]),
            "--patience", str(max(2, min(epochs, int(parameters.get("patience", 5))))),
            "--degrees", str(parameters["degrees"]), "--translate", str(parameters["translate"]),
            "--scale", str(parameters["scale"]), "--perspective", str(parameters["perspective"]),
            "--hsv-v", str(parameters["hsv_v"]),
            "--close-mosaic", str(min(epochs, int(parameters["close_mosaic"]))),
        ]
        if target == "outer_seg":
            arguments.extend(["--output", str(output), "--reports-dir", str(run_root)])
            script = scripts / "outer" / "train_outer_seg.py"
        else:
            arguments.extend(["--workers", "0", "--project", str(run_root), "--name", "run"])
            script = scripts / "inner" / "train_segmentation.py"
        _run_module(script, f"offline_{target}_{job['id']}_{module_suffix}", arguments)
        if target == "inner_seg":
            source = run_root / "run" / "weights" / "best.pt"
            if not source.is_file():
                raise RuntimeError("inner segmentation did not produce best.pt")
            shutil.copy2(source, output)
    elif target == "inner_refiner":
        refiner_batch = 16 if memory_gb <= 8.5 else 32
        arguments = [
            "--manifest", str(dataset / "inner_refiner_manifest.csv"), "--output", str(run_root),
            "--model", str(model), "--epochs", str(epochs), "--batch", str(refiner_batch),
            "--lr", str(parameters["lr"]), "--weight-decay", str(parameters["weight_decay"]),
            "--patience", str(max(2, min(epochs, int(parameters.get("patience", 5))))),
            "--seed", "20260721", "--train-repeats", str(parameters["train_repeats"]),
            "--band-half", str(parameters["band_half"]), "--patch-width", str(parameters["patch_width"]),
            "--patch-height", str(parameters["patch_height"]),
            "--feedback-repeat", str(parameters["feedback_repeat"]), "--device", "cuda:0",
        ]
        _run_module(
            scripts / "inner" / "train_refiner.py",
            f"offline_{target}_{job['id']}_{module_suffix}",
            arguments,
        )
        source = run_root / "best.pt"
        if not source.is_file():
            raise RuntimeError("inner refiner did not produce best.pt")
        shutil.copy2(source, output)
    else:
        raise RuntimeError(f"unsupported training target: {target}")
    if not output.is_file():
        raise RuntimeError(f"{target} did not produce a candidate model")
    _release_training_memory()


def train_candidates(job_path: Path, job: dict[str, Any]) -> list[dict[str, Any]]:
    import torch

    from .offline_optimizer import build_search_plan, choose_trial, read_validation_proxy

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; safe automatic training requires an NVIDIA GPU")
    memory_gb = float(torch.cuda.get_device_properties(0).total_memory) / (1024**3)
    analysis = _optimization_input_analysis(job)
    enabled = bool(job.get("offline_optimization"))
    search_plan = build_search_plan(
        targets=list(job["targets"]),
        trial_count=int(job.get("optimization_trials", 1)) if enabled else 1,
        screening_epochs=int(job.get("screening_epochs", job["epochs"])) if enabled else int(job["epochs"]),
        full_epochs=int(job["epochs"]),
        memory_gb=memory_gb,
        domain_gap=analysis["domain_gap"],
        hard_example_replay=bool(job.get("hard_example_replay")),
    )
    optimization_report: dict[str, Any] = {
        "enabled": enabled,
        "mode": "offline_bounded_search" if enabled else "single_fixed_candidate",
        "created_at": utc_now(),
        "input_analysis": analysis,
        "search_plan": search_plan,
        "targets": {},
        "api_used": False,
        "network_used": False,
    }
    job["training_runtime"] = {
        "gpu": torch.cuda.get_device_name(0),
        "memory_gb": round(memory_gb, 2),
        "offline_optimization": enabled,
        "trial_count_per_target": search_plan["trial_count_per_target"],
        "screening_epochs": search_plan["screening_epochs"],
    }
    job["offline_optimization_report"] = optimization_report
    _write_job(job_path, job)
    candidates = Path(job["candidate_dir"])
    reports = Path(job["report_dir"])
    candidates.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    for target in job["targets"]:
        status_row = job["target_statuses"][target]
        status_row.update({"status": "screening" if enabled else "training", "started_at": utc_now()})
        _write_job(job_path, job)
        trials: list[dict[str, Any]] = []
        for index, profile in enumerate(search_plan["targets"][target], start=1):
            profile_id = str(profile["id"])
            status_row.update({
                "status": "screening" if enabled else "training",
                "trial": index,
                "trial_count": len(search_plan["targets"][target]),
                "profile": profile_id,
            })
            _write_job(job_path, job)
            trial_root = reports / target / "search" / profile_id
            trial_output = candidates / "search" / target / f"{profile_id}.pt"
            _run_target_training(
                job=job,
                target=target,
                model=MODEL_ROOT / MODEL_FILES[target],
                output=trial_output,
                run_root=trial_root,
                epochs=int(profile["screening_epochs"]),
                parameters=dict(profile["parameters"]),
                module_suffix=f"screen_{profile_id}",
                memory_gb=memory_gb,
            )
            validation = read_validation_proxy(target, trial_root)
            trial = {
                "id": f"{target}_{profile_id}",
                "profile": profile_id,
                "label": profile["label"],
                "parameters": profile["parameters"],
                "epochs": int(profile["screening_epochs"]),
                "candidate_path": str(trial_output),
                "validation": validation,
                "status": "screened",
            }
            trials.append(trial)
            optimization_report["targets"][target] = {"trials": trials, "selected": None}
            job["offline_optimization_report"] = optimization_report
            _write_job(job_path, job)
        selected = choose_trial(trials)
        selected_trial = next(row for row in trials if row["id"] == selected["trial_id"])
        status_row.update({"status": "training_best", "profile": selected["profile"]})
        _write_job(job_path, job)
        final_output = candidates / f"{target}_candidate.pt"
        remaining_epochs = max(0, int(job["epochs"]) - int(search_plan["screening_epochs"]))
        if enabled and remaining_epochs > 0:
            _run_target_training(
                job=job,
                target=target,
                model=Path(selected_trial["candidate_path"]),
                output=final_output,
                run_root=reports / target / "final",
                epochs=remaining_epochs,
                parameters=dict(selected["parameters"]),
                module_suffix=f"final_{selected['profile']}",
                memory_gb=memory_gb,
            )
        else:
            shutil.copy2(selected_trial["candidate_path"], final_output)
        record = {
            "target": target,
            "path": str(final_output),
            "filename": final_output.name,
            "size": final_output.stat().st_size,
            "sha256": sha256_file(final_output),
            "deployment_status": "candidate_only",
            "optimization": selected,
            "screening_epochs": int(search_plan["screening_epochs"]),
            "continuation_epochs": remaining_epochs,
        }
        output_rows.append(record)
        optimization_report["targets"][target] = {"trials": trials, "selected": selected}
        status_row.update({"status": "trained", "completed_at": utc_now(), "candidate": record})
        job["candidate_models"] = output_rows
        job["offline_optimization_report"] = optimization_report
        _write_job(job_path, job)
    return output_rows


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] * (high - index) + ordered[high] * (index - low))


def _summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    outer = [float(row[f"{prefix}_outer_error"]) for row in rows if row.get(f"{prefix}_outer_error") is not None]
    inner = [float(row[f"{prefix}_inner_error"]) for row in rows if row.get(f"{prefix}_inner_error") is not None]
    failures = sum(1 for row in rows if row.get(f"{prefix}_failed"))
    return {
        "sample_count": len(rows),
        "failure_count": failures,
        "failure_rate": round(failures / len(rows), 6) if rows else None,
        "outer_corner_mean_percent_diagonal": round(statistics.fmean(outer), 6) if outer else None,
        "outer_corner_p95_percent_diagonal": round(_percentile(outer, 0.95) or 0.0, 6) if outer else None,
        "inner_edge_mae_px": round(statistics.fmean(inner), 6) if inner else None,
        "inner_edge_p95_px": round(_percentile(inner, 0.95) or 0.0, 6) if inner else None,
    }


def _image_features(path: Path) -> dict[str, float]:
    import cv2
    import numpy as np

    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        return {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return {
        "brightness": float(gray.mean() / 255.0),
        "contrast": float(gray.std() / 255.0),
        "sharpness_log": float(math.log1p(cv2.Laplacian(gray, cv2.CV_64F).var())),
        "highlight_fraction": float((gray >= 245).mean()),
        "shadow_fraction": float((gray <= 20).mean()),
        "saturation": float(hsv[:, :, 1].mean() / 255.0),
    }


def _mean_features(paths: Iterable[Path], limit: int = 100) -> dict[str, float]:
    rows = [_image_features(path) for path in list(paths)[:limit]]
    rows = [row for row in rows if row]
    if not rows:
        return {}
    return {key: round(statistics.fmean(row[key] for row in rows), 6) for key in rows[0]}


def _recommendations(domain_gap: dict[str, Any], rows: list[dict[str, Any]], gate: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    deltas = domain_gap.get("delta_real_minus_training") or domain_gap.get("delta_real_minus_history", {})
    if float(deltas.get("highlight_fraction", 0.0)) > 0.02:
        recommendations.append("实拍高光比例明显更高：增加评级壳反光、局部镜面高光增强，并补采不同灯位样本。")
    if float(deltas.get("shadow_fraction", 0.0)) > 0.02:
        recommendations.append("实拍暗部/阴影更多：增加单侧阴影和曝光变化增强，并按阴影方向分层采样。")
    if float(deltas.get("sharpness_log", 0.0)) < -0.25:
        recommendations.append("实拍清晰度低于历史训练图：加入轻度失焦、运动模糊和压缩噪声增强。")
    tags = Counter(tag for row in rows for tag in row.get("issue_tags", []))
    if tags:
        top = "、".join(f"{name}({count})" for name, count in tags.most_common(3))
        recommendations.append(f"留出集中最常见的问题类型为：{top}；下一轮优先补齐这些类别。")
    signed: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for edge, value in (row.get("candidate_inner_signed") or {}).items():
            signed[edge].append(float(value))
    biased = [(edge, statistics.fmean(values)) for edge, values in signed.items() if values and abs(statistics.fmean(values)) >= 0.5]
    if biased:
        text = "、".join(f"{edge} {value:+.2f}px" for edge, value in biased)
        recommendations.append(f"检测存在稳定方向偏差（{text}）：增加对应边的高分辨率条带样本，并检查标注中心线定义。")
    if not gate.get("passed"):
        recommendations.append("候选模型未通过质量门禁：保留线上旧模型，扩大独立实拍留出集后再训练。")
    if not recommendations:
        recommendations.append("当前实拍域差距未显示单一主因；继续按卡种、语言、反光强度和评级壳类型分层收集。")
    return recommendations


def evaluate_candidates(job: dict[str, Any]) -> dict[str, Any]:
    if str(ML_ROOT) not in sys.path:
        sys.path.insert(0, str(ML_ROOT))
    import cv2
    import numpy as np
    from ptcg_inference import CardFramePipeline, PipelineModels

    candidate_by_target = {row["target"]: Path(row["path"]) for row in job["candidate_models"]}
    baseline_models = PipelineModels()
    candidate_models = PipelineModels(
        outer_seg=candidate_by_target.get("outer_seg", baseline_models.outer_seg),
        outer_pose=baseline_models.outer_pose,
        inner_yolo=candidate_by_target.get("inner_seg", baseline_models.inner_yolo),
        inner_refiner=candidate_by_target.get("inner_refiner", baseline_models.inner_refiner),
        inner_gate=baseline_models.inner_gate,
    )
    feedback_root = Path(job["job_path"]).parent / "feedback"
    split_registry = _json(training_root(Path(job["workspace_root"]) / "private") / "split_registry.json", {})
    assignments = split_registry.get("assignments", {}) if isinstance(split_registry, dict) else {}
    rows: list[dict[str, Any]] = []
    records: list[tuple[dict[str, Any], Path, Any, Any]] = []
    test_images: list[Path] = []
    for annotation_path in sorted((feedback_root / "annotations").glob("*.json")):
        sample_id = annotation_path.stem
        if assignments.get(sample_id) != "test":
            continue
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_path = feedback_root / annotation["image"]["path"]
        test_images.append(image_path)
        gt_outer = annotation.get("outer_frame", {}).get("correction", {}).get("points")
        gt_inner = annotation.get("inner_frame", {}).get("correction", {}).get("box")
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "issue_tags": list(annotation.get("review", {}).get("issue_tags") or []),
            "features": _image_features(image_path),
        }
        rows.append(row)
        records.append((row, image_path, gt_outer, gt_inner))

    # Evaluate the current and candidate pipelines in separate GPU lifetimes. Keeping
    # two complete YOLO/refiner stacks resident at once can exhaust an 8 GB card and
    # turn a valid candidate into a false evaluation failure.
    for prefix, models in (("baseline", baseline_models), ("candidate", candidate_models)):
        pipeline = CardFramePipeline(device="0", models=models)
        for row, image_path, gt_outer, gt_inner in records:
            try:
                encoded = np.fromfile(str(image_path), dtype=np.uint8)
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError("holdout image could not be decoded")
                result = pipeline.infer_image(image)
                pred_outer = result.get("outer_frame", {}).get("points")
                pred_inner = result.get("inner_frame", {}).get("final_box")
                failed = not isinstance(pred_outer, list) or not isinstance(pred_inner, dict)
                row[f"{prefix}_failed"] = failed
                if isinstance(gt_outer, list) and isinstance(pred_outer, list) and len(gt_outer) == len(pred_outer) == 4:
                    gt = np.asarray(gt_outer, dtype=np.float64)
                    pred = np.asarray(pred_outer, dtype=np.float64)
                    diagonal = math.hypot(image.shape[1], image.shape[0])
                    row[f"{prefix}_outer_error"] = float(np.linalg.norm(pred - gt, axis=1).mean() / diagonal * 100.0)
                else:
                    row[f"{prefix}_outer_error"] = None
                if isinstance(gt_inner, dict) and isinstance(pred_inner, dict):
                    signed = {edge: float(pred_inner[edge]) - float(gt_inner[edge]) for edge in ("left", "right", "top", "bottom")}
                    row[f"{prefix}_inner_signed"] = signed
                    row[f"{prefix}_inner_error"] = statistics.fmean(abs(value) for value in signed.values())
                else:
                    row[f"{prefix}_inner_error"] = None
            except Exception as exc:
                row[f"{prefix}_failed"] = True
                row[f"{prefix}_outer_error"] = None
                row[f"{prefix}_inner_error"] = None
                row[f"{prefix}_error"] = f"{type(exc).__name__}: {exc}"
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    baseline = _summary(rows, "baseline")
    candidate = _summary(rows, "candidate")
    outer_requested = "outer_seg" in job["targets"]
    inner_requested = bool({"inner_seg", "inner_refiner"} & set(job["targets"]))
    reasons: list[str] = []
    holdout_ok = len(rows) >= 5
    if not holdout_ok:
        reasons.append("独立实拍测试集少于 5 张")
    failures_ok = (candidate.get("failure_count") or 0) <= (baseline.get("failure_count") or 0)
    if not failures_ok:
        reasons.append("候选模型检测失败数增加")

    def improved(metric: str, p95: str, tolerance: float) -> bool:
        before, after = baseline.get(metric), candidate.get(metric)
        before_p95, after_p95 = baseline.get(p95), candidate.get(p95)
        if before is None or after is None or before_p95 is None or after_p95 is None:
            return False
        mean_improved = after <= before * 0.995 or after <= before - tolerance
        tail_safe = after_p95 <= before_p95 + tolerance * 2.0
        return bool(mean_improved and tail_safe)

    outer_passed = (not outer_requested) or (holdout_ok and failures_ok and improved(
        "outer_corner_mean_percent_diagonal", "outer_corner_p95_percent_diagonal", 0.01
    ))
    inner_passed = (not inner_requested) or (holdout_ok and failures_ok and improved(
        "inner_edge_mae_px", "inner_edge_p95_px", 0.10
    ))
    if outer_requested and not outer_passed:
        reasons.append("外框候选模型未达到平均误差改善且尾部误差不回退的要求")
    if inner_requested and not inner_passed:
        reasons.append("内框候选模型未达到平均误差改善且尾部误差不回退的要求")
    passed = holdout_ok and failures_ok and outer_passed and inner_passed
    outer_train_root = Path(job["dataset_dir"]) / "outer_seg" / "images" / "train"
    training_images = sorted(path for path in outer_train_root.glob("*") if path.is_file())
    history_images = [path for path in training_images if path.name.startswith("history_")]
    feedback_train_images = [path for path in training_images if not path.name.startswith("history_")]
    training_features = _mean_features(training_images)
    history_features = _mean_features(history_images)
    feedback_train_features = _mean_features(feedback_train_images)
    real_features = _mean_features(test_images)
    domain_gap = {
        "training_mix": training_features,
        "approved_feedback_train": feedback_train_features,
        "historical_train": history_features,
        "real_photo_holdout": real_features,
        "delta_real_minus_training": {
            key: round(real_features[key] - training_features[key], 6)
            for key in training_features.keys() & real_features.keys()
        },
        "delta_real_minus_history": {
            key: round(real_features[key] - history_features[key], 6)
            for key in history_features.keys() & real_features.keys()
        },
    }
    gate = {
        "passed": passed,
        "holdout_count": len(rows),
        "failure_gate_passed": failures_ok,
        "outer_gate_passed": outer_passed,
        "inner_gate_passed": inner_passed,
        "reasons": reasons,
    }
    recommendations = _recommendations(domain_gap, rows, gate)
    optimization = job.get("offline_optimization_report") or {}
    if optimization.get("enabled"):
        selected_text = "、".join(
            f"{target}={value.get('selected', {}).get('label', '筛选中')}"
            for target, value in optimization.get("targets", {}).items()
        )
        recommendations.insert(0, f"离线智能优化已完成本地候选筛选（{selected_text}）；全程未调用 API。")
    replay = (job.get("dataset") or {}).get("hard_example_replay") or {}
    if replay.get("hard_sample_count"):
        recommendations.append(
            f"本轮将 {int(replay['hard_sample_count'])} 张人工修正难例仅在训练集中加权回放，固定留出集未参与训练。"
        )
    report = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "baseline": baseline,
        "candidate": candidate,
        "quality_gate": gate,
        "domain_gap": domain_gap,
        "recommendations": recommendations,
        "per_sample": rows,
        "method": "Paired current-vs-candidate inference on a persistent real-photo holdout that is never used for training.",
    }
    return report


def promote_candidates(private_root: Path, job: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    gate = report["quality_gate"]
    if not gate.get("passed"):
        raise RuntimeError("quality gate did not pass")
    deployment_id = "mdl_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + secrets.token_hex(3)
    root = training_root(private_root) / "deployments" / deployment_id
    previous_root = root / "previous"
    previous_root.mkdir(parents=True, exist_ok=False)
    previous_manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    (root / "previous_manifest.json").write_text(
        json.dumps(previous_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    promoted: list[dict[str, Any]] = []
    staged: list[tuple[Path, Path, dict[str, Any]]] = []
    replaced: list[Path] = []
    try:
        # Phase one: validate every source, back up every live weight and stage every
        # candidate. Nothing visible to inference changes during this phase.
        for candidate in job["candidate_models"]:
            target = candidate["target"]
            production = MODEL_ROOT / MODEL_FILES[target]
            source = Path(candidate["path"])
            if not production.is_file():
                raise RuntimeError(f"production model is missing before promotion: {target}")
            if not source.is_file() or sha256_file(source) != candidate["sha256"]:
                raise RuntimeError(f"candidate hash changed before promotion: {target}")
            shutil.copy2(production, previous_root / production.name)
            temporary = production.with_name(production.name + f".{deployment_id}.tmp")
            shutil.copy2(source, temporary)
            if sha256_file(temporary) != candidate["sha256"]:
                raise RuntimeError(f"candidate staging hash mismatch: {target}")
            record = {"target": target, "file": production.name, "sha256": candidate["sha256"]}
            staged.append((temporary, production, record))

        # Phase two: atomically replace each individual weight, then publish the
        # manifest last. The live pipeline reloads only after this manifest changes.
        for temporary, production, record in staged:
            os.replace(temporary, production)
            replaced.append(production)
            promoted.append(record)

        manifest = json.loads(json.dumps(previous_manifest))
        version_stamp = deployment_id.removeprefix("mdl_")
        manifest["package_version"] = f"auto-{version_stamp}"
        manifest["pipeline_version"] = f"ptcg_outer_inner_auto_{version_stamp}"
        for model in manifest.get("models", []):
            filename = Path(str(model.get("file", ""))).name
            match = next((item for item in promoted if item["file"] == filename), None)
            if match:
                path = MODEL_ROOT / filename
                model["size_bytes"] = path.stat().st_size
                model["sha256"] = sha256_file(path).upper()
                model["auto_training_job"] = job["id"]
        atomic_json(MODEL_MANIFEST, manifest)
    except Exception:
        # A multi-model release must never leave a partially updated pipeline.
        for production in replaced:
            backup = previous_root / production.name
            if backup.is_file():
                restore = production.with_name(production.name + ".promotion-rollback.tmp")
                shutil.copy2(backup, restore)
                os.replace(restore, production)
        atomic_json(MODEL_MANIFEST, previous_manifest)
        for temporary, _, _ in staged:
            temporary.unlink(missing_ok=True)
        raise
    deployment = {
        "schema_version": "1.0",
        "id": deployment_id,
        "created_at": utc_now(),
        "job_id": job["id"],
        "previous_manifest": str(root / "previous_manifest.json"),
        "backup_dir": str(previous_root),
        "promoted": promoted,
        "active_manifest_sha256": sha256_file(MODEL_MANIFEST),
        "status": "active",
    }
    atomic_json(root / "deployment.json", deployment)
    return deployment


def rollback_latest(private_root: Path) -> dict[str, Any]:
    deployments: list[tuple[Path, dict[str, Any]]] = []
    for path in (training_root(private_root) / "deployments").glob("mdl_*/deployment.json"):
        value = _json(path, None)
        if isinstance(value, dict) and value.get("status") == "active":
            deployments.append((path, value))
    if not deployments:
        raise RuntimeError("there is no active automatic deployment to roll back")
    path, deployment = sorted(deployments, key=lambda item: str(item[1].get("created_at", "")), reverse=True)[0]
    backup = Path(deployment["backup_dir"])
    previous_manifest_path = Path(deployment["previous_manifest"])
    previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
    for item in deployment.get("promoted", []):
        production = MODEL_ROOT / item["file"]
        source = backup / item["file"]
        temporary = production.with_name(production.name + ".rollback.tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, production)
    atomic_json(MODEL_MANIFEST, previous_manifest)
    deployment["status"] = "rolled_back"
    deployment["rolled_back_at"] = utc_now()
    atomic_json(path, deployment)
    return deployment


def run_worker(service: Any, job_path: Path) -> int:
    job_path = job_path.resolve()
    try:
        job = read_json(job_path)
        if not isinstance(job, dict) or Path(str(job.get("job_path", ""))).resolve() != job_path:
            raise RuntimeError("training job provenance is invalid")
        if Path(str(job.get("app_root", ""))).resolve() != APP_ROOT.resolve():
            raise RuntimeError("training job belongs to a different application root")
        _write_job(job_path, job, "preparing", started_at=utc_now(), worker_pid=os.getpid())
        dataset = prepare_dataset(service, job)
        _write_job(job_path, job, "training", dataset=dataset)
        candidates = train_candidates(job_path, job)
        _write_job(job_path, job, "evaluating", candidate_models=candidates)
        report = evaluate_candidates(job)
        report["offline_optimization"] = job.get("offline_optimization_report")
        report_dir = Path(job["report_dir"])
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "sim_to_real_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        job["quality_gate"] = report["quality_gate"]
        job["analysis"] = {
            "baseline": report["baseline"],
            "candidate": report["candidate"],
            "domain_gap": report["domain_gap"],
            "recommendations": report["recommendations"],
            "offline_optimization": job.get("offline_optimization_report"),
            "report_path": str(report_path),
        }
        deployment = None
        if job["auto_promote"] and report["quality_gate"]["passed"]:
            _write_job(job_path, job, "promoting")
            deployment = promote_candidates(service.private_root, job, report)
            job["production_models_changed"] = True
            for candidate in job["candidate_models"]:
                candidate["deployment_status"] = "promoted"
        job["deployment"] = deployment
        _write_job(job_path, job, "completed", completed_at=utc_now())
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        try:
            job = _json(job_path, {})
            if not isinstance(job, dict):
                job = {}
            error_path = job_path.parent / "error_traceback.txt"
            error_path.write_text(trace, encoding="utf-8")
            _write_job(
                job_path,
                job,
                "failed",
                failed_at=utc_now(),
                error=f"{type(exc).__name__}: {exc}",
                error_traceback_path=str(error_path),
                production_models_changed=False,
            )
        except Exception:
            pass
        print(trace, file=sys.stderr, flush=True)
        return 1


__all__ = [
    "ACTIVE_JOB_STATUSES",
    "DEFAULT_SETTINGS",
    "TERMINAL_JOB_STATUSES",
    "VALID_TARGETS",
    "active_job",
    "active_deployment",
    "active_model_manifest",
    "create_job",
    "gpu_status",
    "list_jobs",
    "load_settings",
    "rollback_latest",
    "run_worker",
    "save_settings",
]
