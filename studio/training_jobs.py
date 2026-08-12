from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

from .config import RESOURCE_ROOT, AppConfig
from .security import atomic_json, read_json, sha256_file, utc_now


TERMINAL_STATUSES = {"completed", "failed"}


def launch_training_worker(job_path: Path) -> int:
    """Start a detached worker and return its process id."""
    job_path = job_path.resolve()
    log_path = job_path.parent / "worker.log"
    if getattr(sys, "frozen", False):
        command = [sys.executable, "--training-worker", str(job_path)]
    else:
        command = [sys.executable, str(RESOURCE_ROOT / "server.py"), "--training-worker", str(job_path)]
    creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    with log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(job_path.parent),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            close_fds=True,
        )
    return int(process.pid)


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load training script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_job(job_path: Path, job: dict[str, Any]) -> None:
    job["updated_at"] = utc_now()
    atomic_json(job_path, job)


def _run_script(module: ModuleType, args: list[str], description: str) -> None:
    result = module.main(args)
    if result not in (None, 0):
        raise RuntimeError(f"{description} exited with status {result}.")


def _candidate_record(path: Path, target: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{target} did not produce a candidate weight file.")
    return {
        "target": target,
        "path": str(path),
        "filename": path.name,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "deployment_status": "candidate_only",
    }


def _set_target(job_path: Path, job: dict[str, Any], target: str, status: str, **extra: Any) -> None:
    row = job["target_statuses"][target]
    row.update({"status": status, **extra})
    _write_job(job_path, job)


def run_training_worker(job_path: str | Path) -> int:
    """Prepare feedback data and optionally train candidate models for one job."""
    job_path = Path(job_path).resolve()
    try:
        job = read_json(job_path)
        if not isinstance(job, dict) or Path(str(job.get("job_path", ""))).resolve() != job_path:
            raise RuntimeError("Training job provenance is invalid.")
        workspace = Path(str(job["workspace_root"])).resolve()
        if job_path.parent.parent != workspace / "training_jobs":
            raise RuntimeError("Training job is outside the declared workspace.")

        job["status"] = "preparing"
        job["worker_pid"] = os.getpid()
        job["started_at"] = utc_now()
        _write_job(job_path, job)

        ml_root = RESOURCE_ROOT / "ml_backend"
        if str(ml_root) not in sys.path:
            sys.path.insert(0, str(ml_root))
        from ml_backend.feedback_dataset import convert_feedback_to_training
        from studio.store import StudioStore

        store = StudioStore(AppConfig(workspace_root=workspace))
        feedback_splits: dict[str, str] = {}
        feedback_root: Path | None = None
        for row in job["samples"]:
            exported = store.export_ml_feedback(
                job["project_id"],
                row["sample_id"],
                row["revision"],
                allow_batch_approval=True,
                training_job_id=job["id"],
            )
            feedback_root = Path(exported["feedback_root"])
            feedback_splits[str(exported["sample_id"])] = row["split"]

        if feedback_root is None:
            raise RuntimeError("No eligible feedback samples were exported.")
        dataset_root = Path(job["dataset_dir"])
        conversion = convert_feedback_to_training(
            feedback_root,
            dataset_root,
            allow_accepted_predictions=job["include_accepted_predictions"],
            sample_splits=feedback_splits,
        )
        job["conversion"] = conversion
        job["feedback_root"] = str(feedback_root)

        job["status"] = "merging_history"
        _write_job(job_path, job)
        from studio.historical_replay import prepare_combined_replay

        replay = prepare_combined_replay(
            dataset_root,
            job["targets"],
            registry_path=job["historical_replay_registry"],
        )
        job["historical_replay"] = replay
        job["prepared_at"] = utc_now()
        _write_job(job_path, job)

        if not job["run_training"]:
            for target in job["targets"]:
                job["target_statuses"][target]["status"] = "prepared"
            job["status"] = "completed"
            job["completed_at"] = utc_now()
            _write_job(job_path, job)
            return 0

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; candidate training requires an NVIDIA GPU.")

        job["status"] = "training"
        _write_job(job_path, job)
        scripts = ml_root / "training"
        models = ml_root / "models"
        candidates = Path(job["candidate_dir"])
        reports = Path(job["report_dir"])
        candidates.mkdir(parents=True, exist_ok=True)
        reports.mkdir(parents=True, exist_ok=True)
        candidate_models: list[dict[str, Any]] = []

        for target in job["targets"]:
            _set_target(job_path, job, target, "training", started_at=utc_now())
            if target == "outer_seg":
                module = _load_module(scripts / "outer" / "train_outer_seg.py", f"outer_train_{job['id']}")
                output = candidates / "outer_seg_candidate.pt"
                _run_script(
                    module,
                    [
                        "--data", str(dataset_root / "outer_seg" / "data.yaml"),
                        "--model", str(models / "outer_seg.pt"),
                        "--epochs", str(job["epochs"]),
                        "--imgsz", "640", "--batch", "4", "--device", "0",
                        "--lr0", "0.0005", "--seed", "20260719",
                        "--output", str(output), "--reports-dir", str(reports / "outer_seg"),
                    ],
                    "outer segmentation training",
                )
            elif target == "inner_seg":
                module = _load_module(scripts / "inner" / "train_segmentation.py", f"inner_train_{job['id']}")
                run_root = reports / "inner_seg"
                _run_script(
                    module,
                    [
                        "--data", str(dataset_root / "inner_seg" / "data.yaml"),
                        "--model", str(models / "inner_frame_yolo_v3_base_candidate.pt"),
                        "--epochs", str(job["epochs"]),
                        "--imgsz", "640", "--batch", "4", "--device", "0",
                        "--workers", "0", "--patience", str(min(job["epochs"], 10)),
                        "--seed", "20260719", "--lr0", "0.0005",
                        "--project", str(run_root), "--name", "run",
                    ],
                    "inner segmentation training",
                )
                output = candidates / "inner_frame_yolo_candidate.pt"
                source = run_root / "run" / "weights" / "best.pt"
                if not source.is_file():
                    raise RuntimeError("Inner segmentation training did not produce best.pt.")
                shutil.copy2(source, output)
            elif target == "inner_refiner":
                module = _load_module(scripts / "inner" / "train_refiner.py", f"refiner_train_{job['id']}")
                run_root = reports / "inner_refiner"
                _run_script(
                    module,
                    [
                        "--manifest", str(dataset_root / "inner_refiner_manifest.csv"),
                        "--output", str(run_root),
                        "--model", str(models / "inner_frame_edge_refiner_v4_candidate.pt"),
                        "--epochs", str(job["epochs"]), "--batch", "32",
                        "--lr", "0.0002", "--seed", "20260719",
                        "--train-repeats", "2", "--device", "cuda:0",
                    ],
                    "inner edge refiner training",
                )
                output = candidates / "inner_frame_edge_refiner_candidate.pt"
                source = run_root / "best.pt"
                if not source.is_file():
                    raise RuntimeError("Inner refiner training did not produce best.pt.")
                shutil.copy2(source, output)
            else:
                raise RuntimeError(f"Unsupported training target: {target}")

            record = _candidate_record(output, target)
            candidate_models.append(record)
            _set_target(job_path, job, target, "completed", completed_at=utc_now(), candidate=record)

        job["candidate_models"] = candidate_models
        job["production_models_changed"] = False
        job["status"] = "completed"
        job["completed_at"] = utc_now()
        _write_job(job_path, job)
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        try:
            job = read_json(job_path) if job_path.is_file() else {}
            if not isinstance(job, dict):
                job = {}
            error_path = job_path.parent / "error_traceback.txt"
            error_path.write_text(trace, encoding="utf-8")
            job.update(
                {
                    "status": "failed",
                    "failed_at": utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "error_traceback_path": str(error_path),
                    "production_models_changed": False,
                }
            )
            for row in job.get("target_statuses", {}).values():
                if isinstance(row, dict) and row.get("status") in {"queued", "training"}:
                    row.update({"status": "failed", "failed_at": utc_now(), "error": str(exc)})
            _write_job(job_path, job)
        except Exception:
            pass
        print(trace, file=sys.stderr, flush=True)
        return 1
