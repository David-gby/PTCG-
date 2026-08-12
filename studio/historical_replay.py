from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .config import APP_ROOT
from .security import sha256_file


REGISTRY_SCHEMA_VERSION = "1.0"
MANIFEST_FIELDS = (
    "id",
    "split",
    "image",
    "label",
    "image_sha256",
    "label_sha256",
    "width",
    "height",
    "left",
    "right",
    "top",
    "bottom",
    "source",
)
VALID_TARGETS = {"outer_seg", "inner_seg", "inner_refiner"}
VALID_SPLITS = {"train", "val", "test"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".jpe", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def default_registry_path() -> Path:
    return (APP_ROOT / "training_history" / "registry.json").resolve()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _contained(root: Path, relative: str, description: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{description} escapes its declared root: {relative}") from exc
    return candidate


def _load_registry(registry_path: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    path = Path(registry_path).resolve() if registry_path is not None else default_registry_path()
    if not path.is_file():
        raise FileNotFoundError(f"Historical replay registry is missing: {path}")
    value = _read_json(path)
    required = {"schema_version", "replay_required", "created_at", "policy", "sources"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Historical replay registry has missing or unknown fields.")
    if value["schema_version"] != REGISTRY_SCHEMA_VERSION or value["replay_required"] is not True:
        raise ValueError("Historical replay registry must require schema version 1.0 replay.")
    policy = value["policy"]
    if not isinstance(policy, dict) or policy.get("training_splits") != ["train", "val"]:
        raise ValueError("Historical replay policy must train on train/val and keep test isolated.")
    if policy.get("protected_splits") != ["val", "test"]:
        raise ValueError("Historical replay policy must protect val/test from feedback leakage.")
    if not isinstance(value["sources"], list) or not value["sources"]:
        raise ValueError("Historical replay registry contains no sources.")
    return path, value


def _source_paths(registry_path: Path, source: dict[str, Any]) -> tuple[Path, Path]:
    required = {
        "id",
        "description",
        "targets",
        "format",
        "root",
        "manifest",
        "manifest_sha256",
        "counts",
    }
    if not isinstance(source, dict) or set(source) != required:
        raise ValueError("Historical replay source has missing or unknown fields.")
    source_id = source.get("id")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("Historical replay source id is invalid.")
    targets = source.get("targets")
    if (
        not isinstance(targets, list)
        or not targets
        or any(target not in VALID_TARGETS for target in targets)
        or len(set(targets)) != len(targets)
    ):
        raise ValueError(f"Historical replay source {source_id} has invalid targets.")
    root_value = source.get("root")
    manifest_value = source.get("manifest")
    if not isinstance(root_value, str) or not isinstance(manifest_value, str):
        raise ValueError(f"Historical replay source {source_id} has invalid paths.")
    root = _contained(registry_path.parent, root_value, f"source {source_id}")
    manifest = _contained(root, manifest_value, f"manifest for {source_id}")
    return root, manifest


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError(f"Historical replay manifest has an invalid header: {path}")
        return [dict(row) for row in reader]


def _count_rows(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row["split"] for row in rows)
    return {split: int(counts.get(split, 0)) for split in ("train", "val", "test")}


def _validate_source(
    registry_path: Path,
    source: dict[str, Any],
    *,
    full: bool,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    source_id = str(source.get("id", ""))
    errors: list[str] = []
    rows: list[dict[str, str]] = []
    try:
        root, manifest = _source_paths(registry_path, source)
        if not root.is_dir():
            raise FileNotFoundError(f"source directory is missing: {root}")
        if not manifest.is_file():
            raise FileNotFoundError(f"manifest is missing: {manifest}")
        actual_manifest_hash = sha256_file(manifest)
        if actual_manifest_hash != source["manifest_sha256"]:
            raise ValueError("manifest SHA-256 does not match the registry")
        rows = _read_manifest(manifest)
        expected_counts = source.get("counts")
        actual_counts = _count_rows(rows)
        if expected_counts != actual_counts:
            raise ValueError(f"split counts do not match registry: {actual_counts}")

        seen_ids: set[str] = set()
        image_splits: dict[str, set[str]] = defaultdict(set)
        for index, row in enumerate(rows, 2):
            row_id = row["id"]
            split = row["split"]
            if not row_id or row_id in seen_ids:
                raise ValueError(f"duplicate or empty id at manifest line {index}")
            seen_ids.add(row_id)
            if split not in VALID_SPLITS:
                raise ValueError(f"invalid split at manifest line {index}: {split}")
            image_splits[row["image_sha256"]].add(split)
            image_path = _contained(root, row["image"], f"image at line {index}")
            label_path = _contained(root, row["label"], f"label at line {index}")
            if not image_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(f"image or label is missing at manifest line {index}")
            if full:
                if sha256_file(image_path) != row["image_sha256"]:
                    raise ValueError(f"image SHA-256 mismatch at manifest line {index}")
                if sha256_file(label_path) != row["label_sha256"]:
                    raise ValueError(f"label SHA-256 mismatch at manifest line {index}")
        leakage = sorted(key for key, splits in image_splits.items() if len(splits) > 1)
        if leakage:
            raise ValueError(f"exact image leakage exists across splits ({len(leakage)} hashes)")
    except (OSError, TypeError, ValueError) as exc:
        errors.append(str(exc))

    return (
        {
            "id": source_id,
            "description": source.get("description", ""),
            "targets": source.get("targets", []),
            "ready": not errors,
            "counts": _count_rows(rows) if rows else {"train": 0, "val": 0, "test": 0},
            "row_count": len(rows),
            "errors": errors,
            "integrity": "sha256_verified" if full and not errors else "manifest_and_files_verified" if not errors else "failed",
        },
        rows,
    )


def historical_replay_status(
    targets: Iterable[str] | None = None,
    *,
    registry_path: str | Path | None = None,
    full: bool = False,
) -> dict[str, Any]:
    requested = sorted(set(targets or VALID_TARGETS))
    invalid = set(requested) - VALID_TARGETS
    if invalid:
        raise ValueError(f"Unsupported historical replay targets: {sorted(invalid)}")
    try:
        path, registry = _load_registry(registry_path)
    except (OSError, TypeError, ValueError) as exc:
        return {
            "ready": False,
            "required": True,
            "registry_path": str(Path(registry_path).resolve()) if registry_path else str(default_registry_path()),
            "registry_sha256": None,
            "targets": requested,
            "target_counts": {},
            "sources": [],
            "errors": [str(exc)],
            "test_policy": "Historical test data is isolated and never mixed into training.",
        }

    selected_sources = [source for source in registry["sources"] if set(source["targets"]) & set(requested)]
    source_reports: list[dict[str, Any]] = []
    errors: list[str] = []
    covered: set[str] = set()
    target_counts = {target: {"train": 0, "val": 0, "test": 0} for target in requested}
    for source in selected_sources:
        report, _ = _validate_source(path, source, full=full)
        source_reports.append(report)
        if not report["ready"]:
            errors.extend(f"{report['id']}: {message}" for message in report["errors"])
            continue
        for target in set(source["targets"]) & set(requested):
            covered.add(target)
            for split in VALID_SPLITS:
                target_counts[target][split] += int(report["counts"][split])
    missing_targets = sorted(set(requested) - covered)
    if missing_targets:
        errors.append(f"No ready historical source covers: {', '.join(missing_targets)}")
    return {
        "ready": not errors,
        "required": True,
        "registry_path": str(path),
        "registry_sha256": sha256_file(path),
        "targets": requested,
        "target_counts": target_counts,
        "sources": source_reports,
        "errors": errors,
        "test_policy": "Historical test data is isolated and never mixed into training.",
    }


def _feedback_segmentation_rows(dataset_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in ("train", "val"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.is_dir():
            continue
        for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise ValueError(f"Feedback label is missing: {label_path}")
            rows.append(
                {
                    "id": image_path.stem,
                    "split": split,
                    "image": str(image_path.resolve()),
                    "label": str(label_path.resolve()),
                    "image_sha256": sha256_file(image_path),
                    "source": "human_feedback",
                }
            )
    return rows


def _feedback_refiner_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    for row in rows:
        image_path = Path(row["image"])
        if not image_path.is_absolute():
            image_path = (path.parent / image_path).resolve()
        if not image_path.is_file():
            raise ValueError(f"Feedback refiner image is missing: {image_path}")
        row["image"] = str(image_path)
        row["image_sha256"] = sha256_file(image_path)
    return rows


def _merge_rows(
    history_rows: list[dict[str, str]],
    feedback_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    history_by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in history_rows:
        history_by_hash[row["image_sha256"]].append(row)

    feedback_by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in feedback_rows:
        feedback_by_hash[row["image_sha256"]].append(row)

    selected_feedback: list[dict[str, str]] = []
    feedback_duplicates_removed = 0
    for rows in feedback_by_hash.values():
        rows.sort(key=lambda row: (0 if row["split"] == "val" else 1, row["id"]))
        selected_feedback.append(rows[0])
        feedback_duplicates_removed += len(rows) - 1

    superseded_hashes: set[str] = set()
    protected_feedback: list[dict[str, str]] = []
    included_feedback: list[dict[str, str]] = []
    for row in selected_feedback:
        matches = history_by_hash.get(row["image_sha256"], [])
        matched_splits = {match["split"] for match in matches}
        if matched_splits & {"val", "test"}:
            protected_feedback.append(
                {"id": row["id"], "history_splits": sorted(matched_splits & {"val", "test"})}
            )
            continue
        if "train" in matched_splits:
            superseded_hashes.add(row["image_sha256"])
        included_feedback.append(row)

    retained_history = [
        row for row in history_rows if not (row["split"] == "train" and row["image_sha256"] in superseded_hashes)
    ]
    merged = retained_history + included_feedback
    summary = {
        "historical_rows": len(history_rows),
        "feedback_rows": len(feedback_rows),
        "feedback_included": len(included_feedback),
        "feedback_duplicates_removed": feedback_duplicates_removed,
        "feedback_excluded_by_protected_split": protected_feedback,
        "historical_train_rows_superseded": len(history_rows) - len(retained_history),
        "combined_counts": _count_rows(merged),
    }
    return merged, summary


def _write_image_list(path: Path, rows: list[dict[str, str]], split: str) -> None:
    values = [row["absolute_image"] for row in rows if row["split"] == split]
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def _write_combined_yaml(dataset_root: Path, class_name: str) -> None:
    lines = [
        f"path: {json.dumps(dataset_root.resolve().as_posix(), ensure_ascii=False)}",
        "train: combined_train.txt",
        "val: combined_val.txt",
        "test: combined_test.txt",
        "names:",
        f"  0: {class_name}",
    ]
    (dataset_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_refiner_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["id", "image", "width", "height", "left", "right", "top", "bottom", "split", "source"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def prepare_combined_replay(
    feedback_dataset_root: str | Path,
    targets: Iterable[str],
    *,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset_root = Path(feedback_dataset_root).resolve()
    requested = sorted(set(targets))
    status = historical_replay_status(requested, registry_path=registry_path, full=True)
    if not status["ready"]:
        raise RuntimeError("Required historical replay validation failed: " + "; ".join(status["errors"]))
    path, registry = _load_registry(status["registry_path"])

    target_history: dict[str, list[dict[str, str]]] = {target: [] for target in requested}
    for source in registry["sources"]:
        source_targets = set(source["targets"]) & set(requested)
        if not source_targets:
            continue
        source_root, manifest_path = _source_paths(path, source)
        rows = _read_manifest(manifest_path)
        expanded: list[dict[str, str]] = []
        for row in rows:
            item = dict(row)
            item["absolute_image"] = str(_contained(source_root, row["image"], "historical image"))
            item["absolute_label"] = str(_contained(source_root, row["label"], "historical label"))
            item["image"] = item["absolute_image"]
            expanded.append(item)
        for target in source_targets:
            target_history[target].extend(dict(row) for row in expanded)

    target_summaries: dict[str, Any] = {}
    for target in requested:
        history_rows = target_history[target]
        if target in {"outer_seg", "inner_seg"}:
            target_root = dataset_root / target
            feedback_rows = _feedback_segmentation_rows(target_root)
            merged, summary = _merge_rows(history_rows, feedback_rows)
            for row in feedback_rows:
                row["absolute_image"] = row["image"]
                row["absolute_label"] = row["label"]
            for row in merged:
                row.setdefault("absolute_image", row["image"])
                row.setdefault("absolute_label", row["label"])
            for split in ("train", "val", "test"):
                _write_image_list(target_root / f"combined_{split}.txt", merged, split)
            _write_combined_yaml(target_root, "card" if target == "outer_seg" else "inner_frame")
            target_summaries[target] = summary
        elif target == "inner_refiner":
            manifest = dataset_root / "inner_refiner_manifest.csv"
            feedback_rows = _feedback_refiner_rows(manifest)
            feedback_copy = dataset_root / "feedback_inner_refiner_manifest.csv"
            if not feedback_copy.exists():
                feedback_copy.write_bytes(manifest.read_bytes())
            merged, summary = _merge_rows(history_rows, feedback_rows)
            normalized: list[dict[str, str]] = []
            for row in merged:
                item = dict(row)
                item["image"] = item.get("absolute_image", item["image"])
                normalized.append(item)
            _write_refiner_manifest(manifest, normalized)
            target_summaries[target] = summary

    summary = {
        "required": True,
        "ready": True,
        "registry_path": status["registry_path"],
        "registry_sha256": status["registry_sha256"],
        "integrity": "all_history_images_and_labels_sha256_verified",
        "targets": target_summaries,
        "sources": status["sources"],
        "test_policy": status["test_policy"],
    }
    summary_path = dataset_root / "historical_replay_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
