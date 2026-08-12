from __future__ import annotations

import copy
import hashlib
import json
import os
import secrets
import shutil
import threading
from pathlib import Path
from typing import Any

from .config import AppConfig
from .contracts import default_label, validate_label, validate_outer_corners, validate_project_create
from .errors import RevisionConflict, SourceIntegrityError, StudioError
from .images import (
    OPTIONAL_CODEC_STATUS,
    build_preview_png,
    decode_image,
    visual_fingerprint,
    visual_fingerprints_match,
)
from .ml_rectification import rectify_ml_png
from .security import (
    append_jsonl,
    atomic_json,
    atomic_write,
    ensure_no_reparse,
    read_json,
    json_bytes,
    safe_path,
    sha256_bytes,
    sha256_file,
    utc_now,
    validate_id,
)


PROJECT_SCHEMA = "1.0"


def _run_local_prelabel(normalized_png: bytes, layout_id: str) -> dict[str, Any]:
    from .prelabel import generate_prelabel

    return generate_prelabel(normalized_png, layout_id)


def _local_prelabel_cache_key() -> str:
    from .prelabel import get_prelabel_cache_key

    return get_prelabel_cache_key()


def _inner_center_box(lines: Any) -> dict[str, float]:
    if not isinstance(lines, dict):
        raise StudioError(422, "ML_FEEDBACK_INNER_REQUIRED", "ML feedback requires four inner centerlines.")
    try:
        return {
            "left": round(sum(float(point[0]) for point in lines["left"]) / 2.0, 4),
            "right": round(sum(float(point[0]) for point in lines["right"]) / 2.0, 4),
            "top": round(sum(float(point[1]) for point in lines["top"]) / 2.0, 4),
            "bottom": round(sum(float(point[1]) for point in lines["bottom"]) / 2.0, 4),
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise StudioError(
            422,
            "ML_FEEDBACK_INNER_INVALID",
            "ML feedback inner geometry is invalid.",
        ) from exc


class StudioStore:
    def __init__(self, config: AppConfig) -> None:
        self.config = config.normalized()
        self.root = self.config.workspace_root
        self.projects_root = self.root / "projects"
        self.prediction_cache_root = self.root / "prediction_consistency"
        self._lock = threading.RLock()
        self._prelabel_lock = threading.Lock()
        self._feedback_lock = threading.Lock()
        self._training_lock = threading.Lock()
        self._preview_lock = threading.Lock()
        self._initialize()

    def _initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        ensure_no_reparse(self.root, self.root)
        self.projects_root.mkdir(parents=True, exist_ok=True)
        ensure_no_reparse(self.projects_root, self.root)
        self.prediction_cache_root.mkdir(parents=True, exist_ok=True)
        ensure_no_reparse(self.prediction_cache_root, self.root)
        marker = self.root / "workspace.json"
        if not marker.exists():
            atomic_json(
                marker,
                {
                    "schema_version": "1.0",
                    "application": "PTCG Annotation Studio ML",
                    "created_at": utc_now(),
                },
            )

    def _project_root(self, project_id: str) -> Path:
        validate_id(project_id, "project")
        path = safe_path(self.projects_root, project_id)
        if not path.is_dir():
            raise StudioError(404, "PROJECT_NOT_FOUND", "Project does not exist.")
        return path

    def _project_file(self, project_id: str) -> Path:
        return safe_path(self._project_root(project_id), "project.json")

    def _read_project(self, project_id: str) -> dict[str, Any]:
        project = read_json(self._project_file(project_id))
        if not isinstance(project, dict) or project.get("id") != project_id:
            raise StudioError(500, "CORRUPT_PROJECT", "Project metadata is invalid.")
        return project

    def _update_project_timestamp(self, project: dict[str, Any]) -> None:
        project["updated_at"] = utc_now()
        project["revision"] = int(project.get("revision", 0)) + 1
        atomic_json(self._project_file(project["id"]), project)

    def list_projects(self) -> list[dict[str, Any]]:
        with self._lock:
            projects: list[dict[str, Any]] = []
            for directory in self.projects_root.iterdir():
                if not directory.is_dir() or not directory.name.startswith("prj_"):
                    continue
                try:
                    project = self._read_project(directory.name)
                    summary = copy.deepcopy(project)
                    samples = self.list_samples(directory.name)
                    summary["sample_count"] = len(samples)
                    summary["progress"] = self._progress(directory.name, samples)
                    projects.append(summary)
                except StudioError:
                    continue
            projects.sort(key=lambda value: value.get("updated_at", ""), reverse=True)
            return projects

    def create_project(self, payload: Any) -> dict[str, Any]:
        validated = validate_project_create(
            payload,
            self.config.rectified_width,
            self.config.rectified_height,
        )
        with self._lock:
            for _ in range(20):
                project_id = f"prj_{secrets.token_hex(12)}"
                root = self.projects_root / project_id
                if not root.exists():
                    break
            else:
                raise StudioError(500, "ID_ALLOCATION_FAILED", "Could not allocate a project ID.")
            root.mkdir()
            for name in ("assets", "originals", "normalized", "previews", "prelabels", "labels", "imports", "audit"):
                (root / name).mkdir()
            now = utc_now()
            project = {
                "schema_version": PROJECT_SCHEMA,
                "id": project_id,
                "name": validated["name"],
                "description": validated["description"],
                "created_at": now,
                "updated_at": now,
                "revision": 1,
                "rectified_size": validated["rectified_size"],
                "default_classification": validated["default_classification"],
                "custom_metadata": validated["custom_metadata"],
            }
            atomic_json(root / "project.json", project)
            append_jsonl(
                root / "audit" / "events.jsonl",
                {"event": "project_created", "at": now, "project_id": project_id},
            )
            result = copy.deepcopy(project)
            result["sample_count"] = 0
            result["progress"] = self._empty_progress()
            return result

    @staticmethod
    def _empty_progress() -> dict[str, int]:
        return {
            "total": 0,
            "unlabeled": 0,
            "in_progress": 0,
            "accepted": 0,
            "review": 0,
            "rejected": 0,
            "saved": 0,
        }

    def _progress(self, project_id: str, samples: list[dict[str, Any]]) -> dict[str, int]:
        progress = self._empty_progress()
        progress["total"] = len(samples)
        labels_root = safe_path(self._project_root(project_id), "labels")
        for sample in samples:
            label_path = safe_path(labels_root, f"{sample['id']}.json")
            if not label_path.exists():
                progress["unlabeled"] += 1
                continue
            label = read_json(label_path)
            status = label.get("annotation_status", "unlabeled")
            if status not in {"unlabeled", "in_progress", "accepted", "review", "rejected"}:
                status = "unlabeled"
            progress[status] += 1
            progress["saved"] += 1
        return progress

    def project_detail(self, project_id: str) -> dict[str, Any]:
        with self._lock:
            project = self._read_project(project_id)
            samples = self.list_samples(project_id)
            imports_root = safe_path(self._project_root(project_id), "imports")
            recent_imports: list[dict[str, Any]] = []
            for path in imports_root.glob("imp_*.json"):
                report = read_json(safe_path(imports_root, path.name))
                report.pop("results", None)
                recent_imports.append(report)
            recent_imports.sort(key=lambda item: item.get("created_at", ""), reverse=True)
            result_project = copy.deepcopy(project)
            result_project["sample_count"] = len(samples)
            progress = self._progress(project_id, samples)
            result_project["progress"] = progress
            return {
                "project": result_project,
                "samples": samples,
                "progress": progress,
                "recent_imports": recent_imports[:20],
            }

    def list_samples(self, project_id: str) -> list[dict[str, Any]]:
        root = safe_path(self._project_root(project_id), "assets")
        samples: list[dict[str, Any]] = []
        for path in root.glob("img_*.json"):
            record = read_json(safe_path(root, path.name))
            if isinstance(record, dict) and record.get("id") == path.stem:
                samples.append(record)
        samples.sort(key=lambda value: (value.get("imported_at", ""), value.get("filename", "")))
        return samples

    def sample(self, project_id: str, sample_id: str) -> dict[str, Any]:
        validate_id(sample_id, "sample")
        path = safe_path(self._project_root(project_id), "assets", f"{sample_id}.json")
        if not path.is_file():
            raise StudioError(404, "SAMPLE_NOT_FOUND", "Sample does not exist.")
        sample = read_json(path)
        if sample.get("id") != sample_id:
            raise StudioError(500, "CORRUPT_SAMPLE", "Sample metadata is invalid.")
        return sample

    @staticmethod
    def _clean_filename(filename: str) -> str:
        if not isinstance(filename, str):
            raise StudioError(400, "FILENAME_REQUIRED", "A filename is required.")
        filename = filename.replace("\\", "/").split("/")[-1]
        filename = "".join("_" if ord(character) < 32 else character for character in filename).strip()
        if not filename or filename in {".", ".."}:
            raise StudioError(400, "FILENAME_REQUIRED", "A filename is required.")
        if len(filename) > 255:
            filename = filename[:255]
        return filename

    def create_import(self, project_id: str, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or set(payload) - {"expected_files", "source"}:
            raise StudioError(422, "INVALID_IMPORT_REQUEST", "Import request contains unknown fields.")
        expected = payload.get("expected_files", 0)
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0 or expected > self.config.max_project_assets:
            raise StudioError(422, "INVALID_IMPORT_REQUEST", "expected_files is invalid.")
        source = payload.get("source", "browser-picker")
        if not isinstance(source, str) or not source.strip() or len(source) > 128:
            raise StudioError(422, "INVALID_IMPORT_REQUEST", "source is invalid.")
        with self._lock:
            project = self._read_project(project_id)
            batch_id = f"imp_{secrets.token_hex(10)}"
            now = utc_now()
            report = {
                "schema_version": "1.0",
                "id": batch_id,
                "project_id": project_id,
                "source": source.strip(),
                "expected_files": expected,
                "created_at": now,
                "updated_at": now,
                "totals": {"processed": 0, "imported": 0, "duplicate": 0, "failed": 0},
                "results": [],
            }
            atomic_json(safe_path(self._project_root(project_id), "imports", f"{batch_id}.json"), report)
            append_jsonl(
                safe_path(self._project_root(project_id), "audit", "events.jsonl"),
                {"event": "import_started", "at": now, "project_id": project_id, "import_id": batch_id},
            )
            self._update_project_timestamp(project)
            return copy.deepcopy(report)

    def import_report(self, project_id: str, batch_id: str) -> dict[str, Any]:
        validate_id(batch_id, "import")
        path = safe_path(self._project_root(project_id), "imports", f"{batch_id}.json")
        if not path.is_file():
            raise StudioError(404, "IMPORT_NOT_FOUND", "Import batch does not exist.")
        report = read_json(path)
        if report.get("project_id") != project_id:
            raise StudioError(500, "CORRUPT_IMPORT_REPORT", "Import report is invalid.")
        return report

    def _append_import_result(
        self,
        project_id: str,
        batch_id: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        report = self.import_report(project_id, batch_id)
        report["results"].append(result)
        status = result["status"]
        report["totals"]["processed"] += 1
        report["totals"][status] += 1
        report["updated_at"] = utc_now()
        atomic_json(safe_path(self._project_root(project_id), "imports", f"{batch_id}.json"), report)
        return report

    def record_import_failure(
        self,
        project_id: str,
        batch_id: str,
        filename: str,
        error: StudioError,
        *,
        byte_size: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            clean_name = self._clean_filename(filename)
            result = {
                "status": "failed",
                "filename": clean_name,
                "byte_size": byte_size,
                "error_code": error.code,
                "message": error.message,
                "at": utc_now(),
            }
            self._append_import_result(project_id, batch_id, result)
            return result

    def import_file(
        self,
        project_id: str,
        batch_id: str,
        filename: str,
        data: bytes,
    ) -> dict[str, Any]:
        clean_name = self._clean_filename(filename)
        digest = sha256_bytes(data)
        sample_id = f"img_{digest}"

        # Validate the destination first, but do not hold the global store
        # lock while Pillow decodes and encodes a potentially large photo.
        with self._lock:
            self._read_project(project_id)
            self.import_report(project_id, batch_id)
            project_root = self._project_root(project_id)
            assets_root = safe_path(project_root, "assets")
            asset_path = safe_path(assets_root, f"{sample_id}.json")
            if (
                not asset_path.exists()
                and sum(1 for path in assets_root.glob("img_*.json") if path.is_file())
                >= self.config.max_project_assets
            ):
                error = StudioError(
                    409,
                    "PROJECT_ASSET_LIMIT",
                    "The project has reached its asset limit.",
                )
                self.record_import_failure(
                    project_id,
                    batch_id,
                    clean_name,
                    error,
                    byte_size=len(data),
                )
                raise error

        try:
            decoded = decode_image(data, clean_name, self.config)
        except StudioError as error:
            self.record_import_failure(project_id, batch_id, clean_name, error, byte_size=len(data))
            raise

        normalized_hash = sha256_bytes(decoded.normalized_png)
        preview_hash = sha256_bytes(decoded.preview_png)
        fingerprint = visual_fingerprint(decoded.normalized_png)

        # Capacity, deduplication, repair and metadata updates remain one
        # serialized commit so concurrent imports cannot overfill a project or
        # create multiple records for the same immutable source bytes.
        with self._lock:
            project = self._read_project(project_id)
            self.import_report(project_id, batch_id)
            project_root = self._project_root(project_id)
            assets_root = safe_path(project_root, "assets")
            asset_path = safe_path(assets_root, f"{sample_id}.json")
            if asset_path.exists():
                existing = read_json(asset_path)
                if (
                    not isinstance(existing, dict)
                    or existing.get("id") != sample_id
                    or existing.get("sha256") != digest
                    or existing.get("width") != decoded.normalized_width
                    or existing.get("height") != decoded.normalized_height
                ):
                    raise SourceIntegrityError("The duplicate sample metadata failed verification.")

                repaired: list[str] = []
                originals_root = safe_path(project_root, "originals")
                original_path = safe_path(originals_root, f"{digest}.bin")
                if not original_path.is_file() or sha256_file(original_path) != digest:
                    atomic_write(original_path, data, overwrite=True)
                    if sha256_file(original_path) != digest:
                        raise SourceIntegrityError("The restored original failed verification.")
                    repaired.append("original")

                metadata_changed = False
                normalized_path = safe_path(project_root, "normalized", f"{sample_id}.png")
                normalized_valid = (
                    normalized_path.is_file()
                    and existing.get("normalized_sha256") == normalized_hash
                    and sha256_file(normalized_path) == normalized_hash
                )
                if not normalized_valid:
                    atomic_write(normalized_path, decoded.normalized_png, overwrite=True)
                    if sha256_file(normalized_path) != normalized_hash:
                        raise SourceIntegrityError("The restored normalized image failed verification.")
                    repaired.append("normalized")
                if existing.get("normalized_sha256") != normalized_hash:
                    existing["normalized_sha256"] = normalized_hash
                    metadata_changed = True
                if existing.get("visual_fingerprint") != fingerprint:
                    existing["visual_fingerprint"] = fingerprint
                    metadata_changed = True

                preview_path = safe_path(project_root, "previews", f"{sample_id}.png")
                if preview_hash == normalized_hash:
                    # A missing preview is a valid logical alias. Preserve a
                    # sound legacy preview file, but drop a corrupt one so the
                    # verified normalized bytes are used instead.
                    if preview_path.exists() and sha256_file(preview_path) != preview_hash:
                        preview_path.unlink()
                        repaired.append("previews")
                    if existing.get("preview_sha256") != preview_hash:
                        existing["preview_sha256"] = preview_hash
                        metadata_changed = True
                        if "previews" not in repaired:
                            repaired.append("previews")
                else:
                    preview_valid = (
                        preview_path.is_file()
                        and existing.get("preview_sha256") == preview_hash
                        and sha256_file(preview_path) == preview_hash
                    )
                    if not preview_valid:
                        atomic_write(preview_path, decoded.preview_png, overwrite=True)
                        if sha256_file(preview_path) != preview_hash:
                            raise SourceIntegrityError("The restored preview image failed verification.")
                        repaired.append("previews")
                    if existing.get("preview_sha256") != preview_hash:
                        existing["preview_sha256"] = preview_hash
                        metadata_changed = True

                if metadata_changed:
                    atomic_json(asset_path, existing)
                if repaired:
                    now = utc_now()
                    append_jsonl(
                        safe_path(self._project_root(project_id), "audit", "events.jsonl"),
                        {
                            "event": "sample_repaired_on_duplicate",
                            "at": now,
                            "project_id": project_id,
                            "sample_id": sample_id,
                            "sha256": digest,
                            "repaired": repaired,
                            "import_id": batch_id,
                        },
                    )
                    self._update_project_timestamp(project)
                result = {
                    "status": "duplicate",
                    "filename": clean_name,
                    "sample_id": sample_id,
                    "sha256": digest,
                    "message": (
                        f"Identical image bytes already exist; repaired: {', '.join(repaired)}."
                        if repaired
                        else "Identical image bytes already exist in this project."
                    ),
                    "at": utc_now(),
                }
                self._append_import_result(project_id, batch_id, result)
                return {"result": result, "sample": existing}

            if (
                sum(1 for path in assets_root.glob("img_*.json") if path.is_file())
                >= self.config.max_project_assets
            ):
                error = StudioError(409, "PROJECT_ASSET_LIMIT", "The project has reached its asset limit.")
                self.record_import_failure(project_id, batch_id, clean_name, error, byte_size=len(data))
                raise error

            originals = safe_path(project_root, "originals")
            original_path = safe_path(originals, f"{digest}.bin")
            if original_path.exists():
                if sha256_file(original_path) != digest:
                    raise SourceIntegrityError()
            else:
                atomic_write(original_path, data, overwrite=False)

            normalized_path = safe_path(project_root, "normalized", f"{sample_id}.png")
            if normalized_path.exists():
                if sha256_file(normalized_path) != normalized_hash:
                    raise SourceIntegrityError("An existing normalized image failed verification.")
            else:
                atomic_write(normalized_path, decoded.normalized_png, overwrite=False)

            preview_path = safe_path(project_root, "previews", f"{sample_id}.png")
            if preview_hash == normalized_hash:
                # New small images use normalized bytes as their preview and
                # never consume storage for a duplicate physical PNG.
                preview_path.unlink(missing_ok=True)
            elif preview_path.exists():
                if sha256_file(preview_path) != preview_hash:
                    raise SourceIntegrityError("An existing preview image failed verification.")
            else:
                atomic_write(preview_path, decoded.preview_png, overwrite=False)
            now = utc_now()
            sample = {
                "schema_version": "1.0",
                "id": sample_id,
                "filename": clean_name,
                "sha256": digest,
                "byte_size": len(data),
                **decoded.metadata(),
                "normalized_sha256": normalized_hash,
                "preview_sha256": preview_hash,
                "visual_fingerprint": fingerprint,
                "imported_at": now,
                "import_id": batch_id,
                "annotation_status": "unlabeled",
            }
            atomic_json(asset_path, sample)
            result = {
                "status": "imported",
                "filename": clean_name,
                "sample_id": sample_id,
                "sha256": digest,
                "message": "Imported.",
                "at": now,
            }
            self._append_import_result(project_id, batch_id, result)
            append_jsonl(
                safe_path(self._project_root(project_id), "audit", "events.jsonl"),
                {
                    "event": "sample_imported",
                    "at": now,
                    "project_id": project_id,
                    "sample_id": sample_id,
                    "sha256": digest,
                    "import_id": batch_id,
                },
            )
            self._update_project_timestamp(project)
            return {"result": result, "sample": sample}

    def _verify_original(self, project_id: str, sample: dict[str, Any]) -> Path:
        path = safe_path(self._project_root(project_id), "originals", f"{sample['sha256']}.bin")
        if not path.is_file() or sha256_file(path) != sample["sha256"]:
            raise SourceIntegrityError()
        return path

    def _preview_bytes(self, project_id: str, sample_id: str) -> tuple[bytes, str, str]:
        # Legacy workspaces may logically alias preview to a multi-megapixel
        # normalized PNG. Migrate those records lazily so existing projects do
        # not need to be re-imported after the preview limit is lowered.
        with self._preview_lock:
            with self._lock:
                sample = self.sample(project_id, sample_id)
                project_root = self._project_root(project_id)
                preview_path = safe_path(project_root, "previews", f"{sample_id}.png")
                normalized_path = safe_path(project_root, "normalized", f"{sample_id}.png")
                normalized_hash = sample.get("normalized_sha256")
                preview_hash = sample.get("preview_sha256")

                if preview_hash != normalized_hash:
                    if not preview_path.is_file() or sha256_file(preview_path) != preview_hash:
                        raise SourceIntegrityError("The stored preview image failed verification.")
                    return (
                        preview_path.read_bytes(),
                        "image/png",
                        f"{Path(sample['filename']).stem}_preview.png",
                    )

                if not normalized_path.is_file() or sha256_file(normalized_path) != normalized_hash:
                    raise SourceIntegrityError("The stored preview alias failed verification.")
                normalized_png = normalized_path.read_bytes()
                if max(int(sample["width"]), int(sample["height"])) <= self.config.preview_max_dimension:
                    preview_path.unlink(missing_ok=True)
                    return (
                        normalized_png,
                        "image/png",
                        f"{Path(sample['filename']).stem}_preview.png",
                    )
                source_hash = str(normalized_hash)
                filename = str(sample["filename"])

            # Pillow decoding/resizing intentionally runs outside the global
            # store lock so health checks, labels and other imports stay live.
            preview_png = build_preview_png(normalized_png, self.config.preview_max_dimension)
            migrated_hash = sha256_bytes(preview_png)

            with self._lock:
                current = self.sample(project_id, sample_id)
                if current.get("normalized_sha256") != source_hash:
                    raise SourceIntegrityError("The normalized image changed during preview migration.")
                if current.get("preview_sha256") != current.get("normalized_sha256"):
                    if not preview_path.is_file() or sha256_file(preview_path) != current.get("preview_sha256"):
                        raise SourceIntegrityError("The stored preview image failed verification.")
                    return (
                        preview_path.read_bytes(),
                        "image/png",
                        f"{Path(current['filename']).stem}_preview.png",
                    )
                atomic_write(preview_path, preview_png, overwrite=True)
                if sha256_file(preview_path) != migrated_hash:
                    raise SourceIntegrityError("The migrated preview image failed verification.")
                current["preview_sha256"] = migrated_hash
                atomic_json(
                    safe_path(self._project_root(project_id), "assets", f"{sample_id}.json"),
                    current,
                )
                append_jsonl(
                    safe_path(self._project_root(project_id), "audit", "events.jsonl"),
                    {
                        "event": "preview_migrated",
                        "at": utc_now(),
                        "project_id": project_id,
                        "sample_id": sample_id,
                        "normalized_sha256": source_hash,
                        "preview_sha256": migrated_hash,
                        "preview_max_dimension": self.config.preview_max_dimension,
                    },
                )
                return preview_png, "image/png", f"{Path(filename).stem}_preview.png"

    def image_bytes(self, project_id: str, sample_id: str, variant: str) -> tuple[bytes, str, str]:
        if variant == "preview":
            return self._preview_bytes(project_id, sample_id)
        with self._lock:
            sample = self.sample(project_id, sample_id)
            if variant == "original":
                path = self._verify_original(project_id, sample)
                return path.read_bytes(), sample["media_type"], sample["filename"]
            if variant != "normalized":
                raise StudioError(400, "INVALID_IMAGE_VARIANT", "Unknown image variant.")
            path = safe_path(self._project_root(project_id), "normalized", f"{sample_id}.png")
            if not path.is_file() or sha256_file(path) != sample["normalized_sha256"]:
                raise SourceIntegrityError("The stored normalized image failed verification.")
            return path.read_bytes(), "image/png", f"{Path(sample['filename']).stem}_{variant}.png"

    @staticmethod
    def _prelabel_digest(value: dict[str, Any]) -> str:
        unsigned = copy.deepcopy(value)
        unsigned.pop("prelabel_sha256", None)
        return sha256_bytes(json_bytes(unsigned))

    @staticmethod
    def _consistency_cache_digest(value: dict[str, Any]) -> str:
        unsigned = copy.deepcopy(value)
        unsigned.pop("cache_sha256", None)
        return sha256_bytes(json_bytes(unsigned))

    def _consistency_cache_directory(self, engine_cache_key: str, layout_id: str, bucket: str) -> Path:
        engine_id = sha256_bytes(engine_cache_key.encode("utf-8"))[:24]
        layout_key = sha256_bytes(layout_id.encode("utf-8"))[:24]
        if len(bucket) != 24 or any(character not in "0123456789abcdef" for character in bucket):
            raise StudioError(500, "INVALID_VISUAL_FINGERPRINT", "Visual fingerprint bucket is invalid.")
        path = self.prediction_cache_root / engine_id / layout_key / bucket
        path.mkdir(parents=True, exist_ok=True)
        ensure_no_reparse(path, self.root)
        return path

    @staticmethod
    def _inner_consistency_payload(candidate: dict[str, Any]) -> dict[str, Any] | None:
        keys = (
            "inner_border_rectified",
            "inner_lines_rectified",
            "inner_line_centers_px",
            "inner_line_midpoints_px",
            "centering_measurements",
        )
        if any(candidate.get(key) is None for key in keys):
            return None
        model_result = candidate.get("model_result")
        model_inner = model_result.get("inner_frame") if isinstance(model_result, dict) else None
        stages = candidate.get("stages")
        stage_inner = stages.get("inner") if isinstance(stages, dict) else None
        if not isinstance(model_inner, dict) or not isinstance(stage_inner, dict):
            return None
        return {
            **{key: copy.deepcopy(candidate[key]) for key in keys},
            "model_inner_frame": copy.deepcopy(model_inner),
            "stage_inner": copy.deepcopy(stage_inner),
        }

    @staticmethod
    def _apply_inner_consistency_payload(
        candidate: dict[str, Any], payload: dict[str, Any]
    ) -> dict[str, Any]:
        output = copy.deepcopy(candidate)
        for key in (
            "inner_border_rectified",
            "inner_lines_rectified",
            "inner_line_centers_px",
            "inner_line_midpoints_px",
            "centering_measurements",
        ):
            output[key] = copy.deepcopy(payload[key])
        if isinstance(output.get("model_result"), dict):
            output["model_result"]["inner_frame"] = copy.deepcopy(payload["model_inner_frame"])
        if isinstance(output.get("stages"), dict):
            output["stages"]["inner"] = copy.deepcopy(payload["stage_inner"])
        return output

    def _read_consistency_entries(
        self, engine_cache_key: str, layout_id: str, fingerprint: dict[str, Any]
    ) -> tuple[Path, list[dict[str, Any]]]:
        directory = self._consistency_cache_directory(
            engine_cache_key, layout_id, str(fingerprint["bucket"])
        )
        entries: list[dict[str, Any]] = []
        for path in directory.glob("*.json"):
            try:
                value = read_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if (
                isinstance(value, dict)
                and value.get("engine_cache_key") == engine_cache_key
                and value.get("layout_id") == layout_id
                and value.get("cache_sha256") == self._consistency_cache_digest(value)
            ):
                entries.append(value)
        return directory, entries

    def _cached_candidate(
        self,
        normalized_sha256: str,
        engine_cache_key: str,
        layout_id: str,
        fingerprint: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
        _, entries = self._read_consistency_entries(engine_cache_key, layout_id, fingerprint)
        for entry in entries:
            if entry.get("normalized_sha256") == normalized_sha256 and isinstance(entry.get("candidate"), dict):
                return copy.deepcopy(entry["candidate"]), entry, "exact_normalized_image"
        for entry in entries:
            if visual_fingerprints_match(fingerprint, entry.get("visual_fingerprint")):
                payload = entry.get("inner_payload")
                if isinstance(payload, dict):
                    return None, entry, "verified_visual_duplicate"
        return None, None, "new_visual_observation"

    def _write_consistency_entry(
        self,
        *,
        normalized_sha256: str,
        engine_cache_key: str,
        layout_id: str,
        fingerprint: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        directory, _ = self._read_consistency_entries(engine_cache_key, layout_id, fingerprint)
        entry = {
            "schema_version": "1.0",
            "normalized_sha256": normalized_sha256,
            "engine_cache_key": engine_cache_key,
            "layout_id": layout_id,
            "visual_fingerprint": copy.deepcopy(fingerprint),
            "inner_payload": self._inner_consistency_payload(candidate),
            "candidate": copy.deepcopy(candidate),
            "created_at": utc_now(),
        }
        entry["cache_sha256"] = self._consistency_cache_digest(entry)
        atomic_json(directory / f"{normalized_sha256}.json", entry)
        return entry

    def get_prelabel(self, project_id: str, sample_id: str) -> dict[str, Any] | None:
        with self._lock:
            sample = self.sample(project_id, sample_id)
            path = safe_path(self._project_root(project_id), "prelabels", f"{sample_id}.json")
            if not path.is_file():
                return None
            value = read_json(path)
            if (
                not isinstance(value, dict)
                or value.get("sample_id") != sample_id
                or value.get("source_sha256") != sample["sha256"]
                or value.get("prelabel_sha256") != self._prelabel_digest(value)
            ):
                raise StudioError(500, "CORRUPT_PRELABEL", "Stored pre-label provenance is invalid.")
            return value

    def generate_prelabel(self, project_id: str, sample_id: str, layout_id: Any) -> dict[str, Any]:
        if not isinstance(layout_id, str) or not layout_id.strip() or len(layout_id) > 128:
            raise StudioError(422, "INVALID_LAYOUT_ID", "layout_id must be non-empty text up to 128 characters.")
        layout_id = layout_id.strip()
        cached = self.get_prelabel(project_id, sample_id)
        if (
            cached is not None
            and cached.get("layout_id") == layout_id
            and cached.get("engine_cache_key") == _local_prelabel_cache_key()
        ):
            return cached

        with self._lock:
            project = self._read_project(project_id)
            sample = self.sample(project_id, sample_id)
            normalized, _, _ = self.image_bytes(project_id, sample_id, "normalized")
            source_sha256 = sample["sha256"]
            normalized_sha256 = sample["normalized_sha256"]
            fingerprint = sample.get("visual_fingerprint")
            if not isinstance(fingerprint, dict):
                fingerprint = visual_fingerprint(normalized)
                sample["visual_fingerprint"] = fingerprint
                atomic_json(
                    safe_path(self._project_root(project_id), "assets", f"{sample_id}.json"),
                    sample,
                )

        engine_cache_key = _local_prelabel_cache_key()

        # The OpenCV pipeline is deliberately outside the storage lock. Only
        # one local pre-label job runs at a time to bound CPU and memory use.
        with self._prelabel_lock:
            exact_candidate, matched_entry, consistency_mode = self._cached_candidate(
                normalized_sha256, engine_cache_key, layout_id, fingerprint
            )
            candidate = exact_candidate or _run_local_prelabel(normalized, layout_id)
            canonical_normalized_sha256 = (
                matched_entry.get("normalized_sha256") if matched_entry is not None else normalized_sha256
            )
            if matched_entry is not None and consistency_mode == "verified_visual_duplicate":
                payload = matched_entry.get("inner_payload")
                if isinstance(payload, dict):
                    candidate = self._apply_inner_consistency_payload(candidate, payload)
                # Store the current source's outer geometry together with the
                # canonical inner geometry. Future uploads of these exact
                # normalized pixels can then skip model execution entirely.
                self._write_consistency_entry(
                    normalized_sha256=normalized_sha256,
                    engine_cache_key=engine_cache_key,
                    layout_id=layout_id,
                    fingerprint=fingerprint,
                    candidate=candidate,
                )
            elif matched_entry is None:
                matched_entry = self._write_consistency_entry(
                    normalized_sha256=normalized_sha256,
                    engine_cache_key=engine_cache_key,
                    layout_id=layout_id,
                    fingerprint=fingerprint,
                    candidate=candidate,
                )
        if not isinstance(candidate, dict):
            raise StudioError(500, "PRELABEL_ENGINE_ERROR", "Local pre-label engine returned invalid data.")
        record = copy.deepcopy(candidate)
        record.update(
            {
                "sample_id": sample_id,
                "source_sha256": source_sha256,
                "layout_id": layout_id,
                "engine_cache_key": engine_cache_key,
                "consistency_guard": {
                    "mode": consistency_mode,
                    "visual_group": fingerprint["bucket"],
                    "input_normalized_sha256": normalized_sha256,
                    "canonical_normalized_sha256": canonical_normalized_sha256,
                },
                "generated_at": utc_now(),
            }
        )
        record["prelabel_sha256"] = self._prelabel_digest(record)

        with self._lock:
            current = self.sample(project_id, sample_id)
            if current["sha256"] != source_sha256:
                raise SourceIntegrityError("The source changed while generating its pre-label.")
            project_root = self._project_root(project_id)
            path = safe_path(project_root, "prelabels", f"{sample_id}.json")
            atomic_json(path, record)
            append_jsonl(
                safe_path(project_root, "audit", "events.jsonl"),
                {
                    "event": "prelabel_generated",
                    "at": record["generated_at"],
                    "project_id": project_id,
                    "sample_id": sample_id,
                    "source_sha256": source_sha256,
                    "prelabel_sha256": record["prelabel_sha256"],
                    "layout_id": layout_id,
                    "status": record.get("status"),
                },
            )
            self._update_project_timestamp(project)
        return record

    def export_ml_feedback(
        self,
        project_id: str,
        sample_id: str,
        expected_revision: Any,
        *,
        allow_batch_approval: bool = False,
        training_job_id: str | None = None,
    ) -> dict[str, Any]:
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 1:
            raise StudioError(
                422,
                "INVALID_EXPECTED_REVISION",
                "ML feedback requires a saved positive revision.",
            )
        with self._lock:
            project = self._read_project(project_id)
            sample = self.sample(project_id, sample_id)
            label = self.get_label(project_id, sample_id)
            revision = int(label["annotation"]["revision"])
            if revision != expected_revision:
                raise RevisionConflict(revision)
            if label["annotation_status"] != "accepted":
                raise StudioError(
                    422,
                    "ML_FEEDBACK_ACCEPTED_REQUIRED",
                    "Only accepted human-reviewed labels may enter the ML feedback pool.",
                )
            feedback_options = label.get("custom_metadata", {}).get("ml_feedback", {})
            individually_approved = (
                isinstance(feedback_options, dict)
                and feedback_options.get("approved_for_training") is True
            )
            if not individually_approved and not allow_batch_approval:
                raise StudioError(
                    422,
                    "ML_FEEDBACK_APPROVAL_REQUIRED",
                    "The label must explicitly approve ML training feedback.",
                )
            if not isinstance(feedback_options, dict):
                feedback_options = {}
            if training_job_id is not None:
                validate_id(training_job_id, "training job")
            outer = copy.deepcopy(label["geometry"]["outer_corners"])
            inner = _inner_center_box(label["geometry"]["inner_lines_rectified"])
            normalized_path = safe_path(self._project_root(project_id), "normalized", f"{sample_id}.png")
            if not normalized_path.is_file() or sha256_file(normalized_path) != sample["normalized_sha256"]:
                raise SourceIntegrityError("The normalized image required for ML feedback failed verification.")
            prelabel = self.get_prelabel(project_id, sample_id)
            prediction = copy.deepcopy(prelabel.get("model_result")) if isinstance(prelabel, dict) else None
            if not isinstance(prediction, dict):
                prediction = {
                    "version": "no_model_prediction",
                    "outer_frame": {},
                    "inner_frame": {},
                }
            feedback_root = safe_path(self.root, "ml_feedback", project_id)
            feedback_id = f"{project_id}_{sample_id}_r{revision}"
            existing = feedback_root / "annotations" / f"{feedback_id}.json"
            if existing.is_file():
                return {
                    "success": True,
                    "already_exported": True,
                    "sample_id": feedback_id,
                    "annotation_path": str(existing),
                    "feedback_root": str(feedback_root),
                    "training_eligible": True,
                }
            issue_tags_raw = feedback_options.get("issue_tags", [])
            issue_tags = (
                [str(value).strip() for value in issue_tags_raw if str(value).strip()]
                if isinstance(issue_tags_raw, list)
                else []
            )
            unchanged_prediction = (
                label["annotation"].get("preannotation_disposition") == "accepted_unchanged"
                and label["annotation"].get("outer_source") == "prelabel_accepted"
                and label["annotation"].get("inner_source") == "prelabel_accepted"
            )
            snapshot = {
                "project": project,
                "sample": sample,
                "label": label,
                "outer": outer,
                "inner": inner,
                "normalized_path": normalized_path,
                "prediction": prediction,
                "feedback_root": feedback_root,
                "feedback_id": feedback_id,
                "issue_tags": issue_tags,
                "review_status": "accepted_prediction" if unchanged_prediction else "corrected",
            }

        # PyTorch/OpenCV work and image copying stay outside the main storage lock.
        with self._feedback_lock:
            try:
                from .ml_feedback import export_reviewed_feedback

                exported = export_reviewed_feedback(**snapshot)
            except StudioError:
                raise
            except Exception as exc:
                raise StudioError(
                    500,
                    "ML_FEEDBACK_EXPORT_FAILED",
                    f"ML feedback export failed: {type(exc).__name__}: {exc}",
                ) from exc

        with self._lock:
            append_jsonl(
                safe_path(self._project_root(project_id), "audit", "events.jsonl"),
                {
                    "event": "ml_feedback_exported",
                    "at": utc_now(),
                    "project_id": project_id,
                    "sample_id": sample_id,
                    "revision": expected_revision,
                    "feedback_sample_id": exported.get("sample_id"),
                    "training_eligible": exported.get("training_eligible"),
                    "approval_source": "individual_label" if individually_approved else "bulk_training_selection",
                    "training_job_id": training_job_id,
                },
            )
        return {**exported, "feedback_root": str(snapshot["feedback_root"])}

    def revoke_ml_feedback(
        self,
        project_id: str,
        sample_id: str,
        feedback_id: str | None,
        *,
        reviewer: str,
        reason: str,
    ) -> dict[str, Any]:
        """Make a previously exported revision ineligible while retaining an audit copy."""
        validate_id(project_id, "project")
        validate_id(sample_id, "sample")
        if feedback_id is not None and (
            not isinstance(feedback_id, str)
            or not feedback_id
            or len(feedback_id) > 240
            or feedback_id[0] not in "abcdefghijklmnopqrstuvwxyz"
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in feedback_id)
        ):
            raise StudioError(404, "NOT_FOUND", "Unknown ML feedback.")
        revoked_at = utc_now()
        feedback_root = safe_path(self.root, "ml_feedback", project_id)
        annotation_path = (
            safe_path(feedback_root, "annotations", f"{feedback_id}.json")
            if feedback_id is not None
            else None
        )
        annotation_revoked = False

        with self._feedback_lock:
            if annotation_path is not None and annotation_path.is_file():
                annotation = read_json(annotation_path)
                if not isinstance(annotation, dict) or str(annotation.get("sample_id")) != feedback_id:
                    raise StudioError(500, "ML_FEEDBACK_CORRUPT", "The exported feedback identity is invalid.")
                review = annotation.get("review")
                if not isinstance(review, dict):
                    raise StudioError(500, "ML_FEEDBACK_CORRUPT", "The exported feedback review block is invalid.")
                review.update(
                    {
                        "status": "reopened",
                        "approved_for_training": False,
                        "training_eligible": False,
                        "revoked_at_utc": revoked_at,
                        "revoked_by": reviewer,
                        "revocation_reason": reason,
                    }
                )
                atomic_json(annotation_path, annotation)
                annotation_revoked = True

                manifest_path = feedback_root / "manifest.jsonl"
                if manifest_path.is_file():
                    updated_lines: list[str] = []
                    for raw_line in manifest_path.read_text(encoding="utf-8-sig").splitlines():
                        try:
                            row = json.loads(raw_line)
                        except (TypeError, ValueError):
                            updated_lines.append(raw_line)
                            continue
                        if feedback_id is not None and isinstance(row, dict) and str(row.get("sample_id")) == feedback_id:
                            row.update(
                                {
                                    "approved_for_training": False,
                                    "training_eligible": False,
                                    "review_status": "reopened",
                                    "revoked_at_utc": revoked_at,
                                }
                            )
                        updated_lines.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                    atomic_write(
                        manifest_path,
                        (("\n".join(updated_lines) + "\n") if updated_lines else "").encode("utf-8"),
                    )

        with self._lock:
            current = self.get_label(project_id, sample_id)
            label = copy.deepcopy(current)
            metadata = copy.deepcopy(label.get("custom_metadata") or {})
            options = copy.deepcopy(metadata.get("ml_feedback") or {})
            options.update(
                {
                    "approved_for_training": False,
                    "revoked_at": revoked_at,
                    "revoked_by": reviewer,
                    "revocation_reason": reason,
                }
            )
            metadata["ml_feedback"] = options
            label["custom_metadata"] = metadata
            label["annotation_status"] = "review"
            label["annotation"]["reviewer"] = reviewer
            saved, _ = self.save_label(project_id, sample_id, label, int(current["annotation"]["revision"]))
            append_jsonl(
                safe_path(self._project_root(project_id), "audit", "events.jsonl"),
                {
                    "event": "ml_feedback_revoked",
                    "at": revoked_at,
                    "project_id": project_id,
                    "sample_id": sample_id,
                    "feedback_sample_id": feedback_id,
                    "reviewer": reviewer,
                    "reason": reason,
                    "annotation_revoked": annotation_revoked,
                    "new_label_revision": int(saved["annotation"]["revision"]),
                },
            )
        return {
            "success": True,
            "sample_id": feedback_id,
            "annotation_revoked": annotation_revoked,
            "training_eligible": False,
            "label_revision": int(saved["annotation"]["revision"]),
        }

    def delete_ml_feedback(self, project_id: str, feedback_id: str | None) -> dict[str, Any]:
        """Permanently remove one exported feedback artifact and its manifest row."""
        validate_id(project_id, "project")
        if not feedback_id:
            return {"deleted": False, "sample_id": None}
        if (
            not isinstance(feedback_id, str)
            or len(feedback_id) > 240
            or feedback_id[0] not in "abcdefghijklmnopqrstuvwxyz"
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in feedback_id)
        ):
            raise StudioError(404, "NOT_FOUND", "Unknown ML feedback.")
        feedback_root = safe_path(self.root, "ml_feedback", project_id)
        if not feedback_root.is_dir():
            return {"deleted": False, "sample_id": feedback_id}
        annotation_path = safe_path(feedback_root, "annotations", f"{feedback_id}.json")
        removed_files: list[str] = []
        with self._feedback_lock:
            referenced: list[str] = []
            if annotation_path.is_file():
                annotation = read_json(annotation_path)
                if isinstance(annotation, dict):
                    image = annotation.get("image") if isinstance(annotation.get("image"), dict) else {}
                    rectification = (
                        annotation.get("rectification")
                        if isinstance(annotation.get("rectification"), dict)
                        else {}
                    )
                    for value in (image.get("path"), rectification.get("image_path")):
                        if isinstance(value, str) and value:
                            referenced.append(value)
            for relative in referenced:
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise StudioError(500, "ML_FEEDBACK_CORRUPT", "Feedback file path is invalid.")
                target = safe_path(feedback_root, *relative_path.parts)
                if target.is_file():
                    target.unlink()
                    removed_files.append(str(target.relative_to(feedback_root)))
            for folder in ("original_images", "rectified_images"):
                directory = safe_path(feedback_root, folder)
                if directory.is_dir():
                    for target in directory.glob(f"{feedback_id}.*"):
                        safe_target = safe_path(directory, target.name)
                        if safe_target.is_file():
                            safe_target.unlink()
                            removed_files.append(str(safe_target.relative_to(feedback_root)))
            if annotation_path.is_file():
                annotation_path.unlink()
                removed_files.append(str(annotation_path.relative_to(feedback_root)))
            manifest_path = safe_path(feedback_root, "manifest.jsonl")
            if manifest_path.is_file():
                kept: list[str] = []
                for raw_line in manifest_path.read_text(encoding="utf-8-sig").splitlines():
                    try:
                        row = json.loads(raw_line)
                    except (TypeError, ValueError):
                        kept.append(raw_line)
                        continue
                    if isinstance(row, dict) and str(row.get("sample_id")) == feedback_id:
                        continue
                    kept.append(raw_line)
                atomic_write(
                    manifest_path,
                    (("\n".join(kept) + "\n") if kept else "").encode("utf-8"),
                )
        return {"deleted": bool(removed_files), "sample_id": feedback_id, "files": removed_files}

    def delete_ml_feedback_for_sample(
        self,
        project_id: str,
        sample_id: str,
        latest_feedback_id: str | None = None,
    ) -> dict[str, Any]:
        """Remove every exported revision associated with one Studio sample."""
        validate_id(project_id, "project")
        validate_id(sample_id, "sample")
        feedback_root = safe_path(self.root, "ml_feedback", project_id)
        identifiers: set[str] = set()
        if latest_feedback_id:
            identifiers.add(latest_feedback_id)
        annotations = safe_path(feedback_root, "annotations")
        if annotations.is_dir():
            prefix = f"{project_id}_{sample_id}_r"
            for path in annotations.glob(f"{prefix}*.json"):
                identifiers.add(path.stem)
        results = [self.delete_ml_feedback(project_id, identifier) for identifier in sorted(identifiers)]
        return {
            "deleted": any(bool(result.get("deleted")) for result in results),
            "sample_id": sample_id,
            "exported_revisions": results,
        }

    def delete_project(self, project_id: str) -> dict[str, Any]:
        """Permanently remove a platform-managed project and its ML feedback artifacts."""
        validate_id(project_id, "project")
        project_root = self._project_root(project_id)
        ensure_no_reparse(project_root, self.root)
        feedback_root = safe_path(self.root, "ml_feedback", project_id)
        with self._feedback_lock:
            if feedback_root.is_dir():
                ensure_no_reparse(feedback_root, self.root)
                shutil.rmtree(feedback_root)
        with self._lock:
            if project_root.is_dir():
                ensure_no_reparse(project_root, self.root)
                shutil.rmtree(project_root)
        return {"deleted": True, "project_id": project_id}

    def _training_jobs_root(self) -> Path:
        root = safe_path(self.root, "training_jobs")
        root.mkdir(parents=True, exist_ok=True)
        ensure_no_reparse(root, self.root)
        return root

    def training_job(self, project_id: str, job_id: str) -> dict[str, Any]:
        self._project_root(project_id)
        validate_id(job_id, "training job")
        path = safe_path(self._training_jobs_root(), job_id, "job.json")
        if not path.is_file():
            raise StudioError(404, "TRAINING_JOB_NOT_FOUND", "Training job does not exist.")
        job = read_json(path)
        if not isinstance(job, dict) or job.get("id") != job_id or job.get("project_id") != project_id:
            raise StudioError(404, "TRAINING_JOB_NOT_FOUND", "Training job does not exist.")
        return copy.deepcopy(job)

    def list_training_jobs(self, project_id: str) -> list[dict[str, Any]]:
        self._project_root(project_id)
        jobs: list[dict[str, Any]] = []
        for path in self._training_jobs_root().glob("trn_*/job.json"):
            value = read_json(path)
            if isinstance(value, dict) and value.get("project_id") == project_id:
                jobs.append(value)
        jobs.sort(key=lambda value: str(value.get("created_at", "")), reverse=True)
        return copy.deepcopy(jobs[:50])

    @staticmethod
    def _training_split(project_id: str, sample_ids: list[str], ratio: float) -> dict[str, str]:
        ranked = sorted(
            sample_ids,
            key=lambda sample_id: hashlib.sha256(f"{project_id}:{sample_id}".encode("utf-8")).hexdigest(),
        )
        validation_count = max(1, min(len(ranked) - 1, round(len(ranked) * ratio)))
        validation_ids = set(ranked[:validation_count])
        return {sample_id: ("val" if sample_id in validation_ids else "train") for sample_id in sample_ids}

    def create_training_job(self, project_id: str, payload: Any) -> dict[str, Any]:
        allowed_fields = {
            "sample_ids",
            "targets",
            "run_training",
            "validation_ratio",
            "epochs",
            "include_accepted_predictions",
            "confirm_training_approval",
        }
        if not isinstance(payload, dict) or set(payload) != allowed_fields:
            raise StudioError(
                422,
                "INVALID_TRAINING_REQUEST",
                "Batch training request has missing or unknown fields.",
            )
        if payload["confirm_training_approval"] is not True:
            raise StudioError(
                422,
                "TRAINING_APPROVAL_REQUIRED",
                "Batch selection must be explicitly approved for model training.",
            )
        sample_ids = payload["sample_ids"]
        if not isinstance(sample_ids, list) or not 1 <= len(sample_ids) <= self.config.max_project_assets:
            raise StudioError(422, "INVALID_SAMPLE_IDS", "sample_ids must be a non-empty array.")
        validated_ids: list[str] = []
        seen: set[str] = set()
        for value in sample_ids:
            validate_id(value, "sample")
            if value in seen:
                raise StudioError(422, "DUPLICATE_SAMPLE_ID", "sample_ids must not contain duplicates.")
            seen.add(value)
            validated_ids.append(value)

        valid_targets = {"outer_seg", "inner_seg", "inner_refiner"}
        targets = payload["targets"]
        if (
            not isinstance(targets, list)
            or not targets
            or any(not isinstance(value, str) for value in targets)
            or len(set(targets)) != len(targets)
            or set(targets) - valid_targets
        ):
            raise StudioError(422, "INVALID_TRAINING_TARGETS", "targets contains an unsupported or duplicate target.")
        from .historical_replay import historical_replay_status

        replay_preflight = historical_replay_status(targets, full=False)
        if not replay_preflight["ready"]:
            raise StudioError(
                422,
                "HISTORICAL_REPLAY_UNAVAILABLE",
                "Required historical training data is missing or invalid; training was not started.",
                details=replay_preflight,
            )
        run_training = payload["run_training"]
        include_predictions = payload["include_accepted_predictions"]
        if not isinstance(run_training, bool) or not isinstance(include_predictions, bool):
            raise StudioError(422, "INVALID_TRAINING_REQUEST", "Training mode flags must be booleans.")
        epochs = payload["epochs"]
        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 100:
            raise StudioError(422, "INVALID_TRAINING_EPOCHS", "epochs must be an integer from 1 to 100.")
        ratio_value = payload["validation_ratio"]
        if isinstance(ratio_value, bool) or not isinstance(ratio_value, (int, float)):
            raise StudioError(422, "INVALID_VALIDATION_RATIO", "validation_ratio must be a number.")
        validation_ratio = float(ratio_value)
        if not 0.1 <= validation_ratio <= 0.4:
            raise StudioError(422, "INVALID_VALIDATION_RATIO", "validation_ratio must be from 0.1 to 0.4.")

        self._project_root(project_id)
        eligible: list[dict[str, Any]] = []
        excluded: list[dict[str, str]] = []
        for sample_id in validated_ids:
            sample = self.sample(project_id, sample_id)
            label = self.get_label(project_id, sample_id)
            reason = None
            if label.get("annotation_status") != "accepted":
                reason = "accepted_label_required"
            elif label.get("classification", {}).get("layout_id") != "gx_current":
                reason = "gx_current_layout_required"
            elif label.get("assessment", {}).get("planarity") != "planar":
                reason = "planar_card_required"
            else:
                try:
                    validate_outer_corners(label.get("geometry", {}).get("outer_corners"), sample["width"], sample["height"])
                    _inner_center_box(label.get("geometry", {}).get("inner_lines_rectified"))
                except StudioError:
                    reason = "complete_outer_and_inner_geometry_required"
            unchanged_prediction = (
                label.get("annotation", {}).get("preannotation_disposition") == "accepted_unchanged"
                and label.get("annotation", {}).get("outer_source") == "prelabel_accepted"
                and label.get("annotation", {}).get("inner_source") == "prelabel_accepted"
            )
            if reason is None and unchanged_prediction and not include_predictions:
                reason = "unchanged_prediction_excluded_by_default"
            if reason is not None:
                excluded.append({"sample_id": sample_id, "reason": reason})
                continue
            revision = label.get("annotation", {}).get("revision")
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
                excluded.append({"sample_id": sample_id, "reason": "saved_revision_required"})
                continue
            eligible.append({"sample_id": sample_id, "revision": revision})

        minimum = 5 if run_training else 2
        if len(eligible) < minimum:
            raise StudioError(
                422,
                "INSUFFICIENT_TRAINING_SAMPLES",
                f"At least {minimum} eligible samples are required for this mode.",
                details={"eligible_count": len(eligible), "minimum": minimum, "excluded": excluded},
            )
        split_map = self._training_split(project_id, [row["sample_id"] for row in eligible], validation_ratio)
        for row in eligible:
            row["split"] = split_map[row["sample_id"]]

        with self._training_lock:
            active_statuses = {"queued", "preparing", "merging_history", "training"}
            for path in self._training_jobs_root().glob("trn_*/job.json"):
                existing = read_json(path)
                if isinstance(existing, dict) and existing.get("status") in active_statuses:
                    raise StudioError(
                        409,
                        "TRAINING_JOB_ACTIVE",
                        "Another training job is already active. Wait for it to finish.",
                        details={"job_id": existing.get("id"), "status": existing.get("status")},
                    )

            job_id = f"trn_{secrets.token_hex(10)}"
            job_root = safe_path(self._training_jobs_root(), job_id)
            job_root.mkdir(parents=False, exist_ok=False)
            job_path = safe_path(job_root, "job.json")
            now = utc_now()
            job: dict[str, Any] = {
                "schema_version": "1.1",
                "id": job_id,
                "project_id": project_id,
                "status": "queued",
                "created_at": now,
                "updated_at": now,
                "workspace_root": str(self.root.resolve()),
                "job_path": str(job_path.resolve()),
                "run_training": run_training,
                "targets": list(targets),
                "epochs": epochs,
                "validation_ratio": validation_ratio,
                "include_accepted_predictions": include_predictions,
                "historical_replay_required": True,
                "historical_replay_registry": replay_preflight["registry_path"],
                "historical_replay_preflight": replay_preflight,
                "requested_count": len(validated_ids),
                "eligible_count": len(eligible),
                "excluded": excluded,
                "samples": eligible,
                "dataset_dir": str((job_root / "dataset").resolve()),
                "candidate_dir": str((job_root / "candidates").resolve()),
                "report_dir": str((job_root / "reports").resolve()),
                "log_path": str((job_root / "worker.log").resolve()),
                "target_statuses": {target: {"status": "queued"} for target in targets},
                "candidate_models": [],
                "production_models_changed": False,
                "safety_note": "Candidates are never copied over production models automatically.",
            }
            atomic_json(job_path, job)
            append_jsonl(
                safe_path(self._project_root(project_id), "audit", "events.jsonl"),
                {
                    "event": "bulk_training_selection",
                    "at": now,
                    "project_id": project_id,
                    "training_job_id": job_id,
                    "sample_ids": [row["sample_id"] for row in eligible],
                    "excluded": excluded,
                    "targets": targets,
                    "run_training": run_training,
                    "include_accepted_predictions": include_predictions,
                    "historical_replay_required": True,
                    "historical_replay_registry_sha256": replay_preflight["registry_sha256"],
                },
            )
            try:
                from .training_jobs import launch_training_worker

                launch_training_worker(job_path)
            except Exception as exc:
                job["status"] = "failed"
                job["failed_at"] = utc_now()
                job["error"] = f"Cannot start training worker: {type(exc).__name__}: {exc}"
                atomic_json(job_path, job)
                raise StudioError(500, "TRAINING_WORKER_START_FAILED", job["error"]) from exc
        return copy.deepcopy(job)

    def delete_samples(self, project_id: str, payload: Any) -> dict[str, list[str]]:
        if not isinstance(payload, dict) or set(payload) != {"sample_ids"}:
            raise StudioError(
                422,
                "INVALID_DELETE_REQUEST",
                "Sample deletion requires exactly one sample_ids field.",
            )
        sample_ids = payload.get("sample_ids")
        if not isinstance(sample_ids, list) or len(sample_ids) > self.config.max_project_assets:
            raise StudioError(
                422,
                "INVALID_SAMPLE_IDS",
                "sample_ids must be an array within the project asset limit.",
            )
        validated: list[str] = []
        seen: set[str] = set()
        for index, sample_id in enumerate(sample_ids):
            try:
                validate_id(sample_id, f"sample_ids[{index}]")
            except StudioError as exc:
                raise StudioError(
                    422,
                    "INVALID_SAMPLE_IDS",
                    f"sample_ids[{index}] is not a valid sample ID.",
                ) from exc
            if not sample_id.startswith("img_"):
                raise StudioError(
                    422,
                    "INVALID_SAMPLE_IDS",
                    f"sample_ids[{index}] is not a valid sample ID.",
                )
            if sample_id in seen:
                raise StudioError(
                    422,
                    "DUPLICATE_SAMPLE_ID",
                    "sample_ids must not contain duplicates.",
                )
            seen.add(sample_id)
            validated.append(sample_id)

        with self._lock:
            project = self._read_project(project_id)
            project_root = self._project_root(project_id)
            assets_root = safe_path(project_root, "assets")

            # Read every asset record before mutating anything. Apart from
            # detecting corrupt metadata early, this ensures a content-addressed
            # original is never removed while another sample still references it.
            records: dict[str, dict[str, Any]] = {}
            for path in assets_root.glob("img_*.json"):
                record = read_json(safe_path(assets_root, path.name))
                digest = record.get("sha256") if isinstance(record, dict) else None
                if (
                    not isinstance(record, dict)
                    or record.get("id") != path.stem
                    or not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise StudioError(500, "CORRUPT_SAMPLE", "Sample metadata is invalid.")
                records[path.stem] = record

            deleted = [sample_id for sample_id in validated if sample_id in records]
            not_found = [sample_id for sample_id in validated if sample_id not in records]
            if not deleted:
                return {"deleted": [], "not_found": not_found}

            deleting = set(deleted)
            remaining_digests = {
                record["sha256"]
                for sample_id, record in records.items()
                if sample_id not in deleting
            }

            # Remove the discoverable asset record first. If a later cleanup is
            # interrupted, the result is harmless orphaned data rather than a
            # visible sample whose required image has disappeared.
            for sample_id in deleted:
                safe_path(assets_root, f"{sample_id}.json").unlink(missing_ok=True)
            for sample_id in deleted:
                for folder in ("labels", "prelabels", "normalized", "previews"):
                    suffix = ".json" if folder in {"labels", "prelabels"} else ".png"
                    safe_path(project_root, folder, f"{sample_id}{suffix}").unlink(missing_ok=True)

            originals_deleted: list[str] = []
            for digest in sorted({records[sample_id]["sha256"] for sample_id in deleted}):
                if digest not in remaining_digests:
                    safe_path(project_root, "originals", f"{digest}.bin").unlink(missing_ok=True)
                    originals_deleted.append(digest)

            now = utc_now()
            append_jsonl(
                safe_path(project_root, "audit", "events.jsonl"),
                {
                    "event": "samples_deleted",
                    "at": now,
                    "project_id": project_id,
                    "sample_ids": deleted,
                    "not_found": not_found,
                    "original_sha256_deleted": originals_deleted,
                },
            )
            self._update_project_timestamp(project)
            return {"deleted": deleted, "not_found": not_found}

    def rectified_bytes(
        self,
        project_id: str,
        sample_id: str,
        outer_corners: Any,
        width: int | None,
        height: int | None,
    ) -> bytes:
        with self._lock:
            project = self._read_project(project_id)
            sample = self.sample(project_id, sample_id)
            normalized, _, _ = self.image_bytes(project_id, sample_id, "normalized")
            corners = validate_outer_corners(outer_corners, sample["width"], sample["height"])
            if corners is None:
                raise StudioError(422, "INVALID_GEOMETRY", "outer_corners are required.")
            target_width = project["rectified_size"]["width"] if width is None else width
            target_height = project["rectified_size"]["height"] if height is None else height
            for value, field in ((target_width, "width"), (target_height, "height")):
                if isinstance(value, bool) or not isinstance(value, int) or value < 128 or value > 4096:
                    raise StudioError(422, "INVALID_RECTIFIED_SIZE", f"{field} must be an integer from 128 to 4096.")
        return rectify_ml_png(normalized, corners, target_width, target_height)

    def get_label(self, project_id: str, sample_id: str) -> dict[str, Any]:
        with self._lock:
            project = self._read_project(project_id)
            sample = self.sample(project_id, sample_id)
            path = safe_path(self._project_root(project_id), "labels", f"{sample_id}.json")
            if path.exists():
                return read_json(path)
            return default_label(project, sample)

    def save_label(
        self,
        project_id: str,
        sample_id: str,
        incoming: Any,
        expected_revision: int,
    ) -> tuple[dict[str, Any], bool]:
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
            raise StudioError(422, "INVALID_EXPECTED_REVISION", "expected_revision must be a non-negative integer.")
        with self._lock:
            project = self._read_project(project_id)
            sample = self.sample(project_id, sample_id)
            self._verify_original(project_id, sample)
            current = self.get_label(project_id, sample_id)
            current_revision = int(current["annotation"]["revision"])
            if expected_revision != current_revision:
                raise RevisionConflict(current_revision)
            candidate = validate_label(
                incoming,
                project,
                sample,
                current_revision=current_revision,
            )
            candidate["annotation"]["created_at"] = current["annotation"].get("created_at")
            candidate["annotation"]["updated_at"] = current["annotation"].get("updated_at")
            if candidate == current:
                return current, True

            now = utc_now()
            candidate["annotation"]["revision"] = current_revision + 1
            candidate["annotation"]["created_at"] = current["annotation"].get("created_at") or now
            candidate["annotation"]["updated_at"] = now
            label_path = safe_path(self._project_root(project_id), "labels", f"{sample_id}.json")
            atomic_json(label_path, candidate)

            sample["annotation_status"] = candidate["annotation_status"]
            atomic_json(safe_path(self._project_root(project_id), "assets", f"{sample_id}.json"), sample)
            append_jsonl(
                safe_path(self._project_root(project_id), "audit", "events.jsonl"),
                {
                    "event": "label_saved",
                    "at": now,
                    "project_id": project_id,
                    "sample_id": sample_id,
                    "revision": candidate["annotation"]["revision"],
                    "status": candidate["annotation_status"],
                    "labeler": candidate["annotation"]["labeler"],
                },
            )
            self._update_project_timestamp(project)
            return candidate, False

    def codec_status(self) -> dict[str, Any]:
        return copy.deepcopy(OPTIONAL_CODEC_STATUS)
