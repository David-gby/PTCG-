from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .contracts import validate_label
from .errors import StudioError
from .images import draw_annotated_overlay
from .measurements import center_geometry_from_centers, centers_from_lines, centering_measurements
from .security import json_bytes, safe_path, sha256_file, utc_now, validate_id
from .store import StudioStore


EXPORT_MODES = {"labels-only", "full", "annotated"}
SPREADSHEET_FORMULA_PREFIXES = frozenset("=+-@")


@dataclass(slots=True)
class ExportArtifact:
    path: Path
    filename: str
    size: int

    def cleanup(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


def _safe_zip_filename(value: str) -> str:
    value = value.replace("\\", "/").split("/")[-1]
    value = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    return value[:180] or "image.bin"


def spreadsheet_safe(value: Any) -> Any:
    """Neutralize spreadsheet formulas without changing canonical JSON labels."""

    if value is None or not isinstance(value, str):
        return value
    if not value:
        return value
    probe = value.lstrip(" \t\r\n\v\f")
    leading_control = ord(value[0]) < 32 or ord(value[0]) == 127
    if leading_control or (probe and probe[0] in SPREADSHEET_FORMULA_PREFIXES):
        return "'" + value
    return value


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


class _ArchiveBuilder:
    def __init__(self, archive: zipfile.ZipFile) -> None:
        self.archive = archive
        self.members: list[dict[str, Any]] = []

    def add_bytes(self, name: str, data: bytes) -> None:
        self.archive.writestr(_zip_info(name), data)
        self.members.append(
            {"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )

    def add_file(self, name: str, source: Path, *, expected_sha256: str | None = None) -> None:
        digest = hashlib.sha256()
        size = 0
        with source.open("rb") as source_handle, self.archive.open(_zip_info(name), "w") as target:
            for block in iter(lambda: source_handle.read(1024 * 1024), b""):
                target.write(block)
                digest.update(block)
                size += len(block)
        actual_digest = digest.hexdigest()
        if expected_sha256 is not None and actual_digest != expected_sha256:
            raise StudioError(409, "SOURCE_INTEGRITY_ERROR", f"{name} changed while exporting.")
        self.members.append({"path": name, "size": size, "sha256": actual_digest})


def _csv_bytes(samples: list[dict[str, Any]], labels: dict[str, dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    fields = [
        "sample_id",
        "filename",
        "sha256",
        "width",
        "height",
        "annotation_status",
        "planarity",
        "reason_codes",
        "shadow_sides",
        "face",
        "layout_id",
        "orientation_degrees_cw",
        "card_type",
        "labeler",
        "reviewer",
        "revision",
        "outer_source",
        "inner_source",
        "preannotation_disposition",
        "prelabel_sha256",
        "outer_corners_json",
        "inner_lines_rectified_json",
        "inner_line_centers_px_json",
        "inner_line_midpoints_px_json",
        "centering_measurements_json",
        "custom_metadata_json",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for sample in samples:
        label = labels[sample["id"]]
        centers, midpoints = _inner_center_geometry(label["geometry"]["inner_lines_rectified"])
        measurements = centering_measurements(centers, 630, 880) if centers is not None else None
        row = {
                "sample_id": sample["id"],
                "filename": sample["filename"],
                "sha256": sample["sha256"],
                "width": sample["width"],
                "height": sample["height"],
                "annotation_status": label["annotation_status"],
                "planarity": label["assessment"]["planarity"],
                "reason_codes": "|".join(label["assessment"]["reason_codes"]),
                "shadow_sides": "|".join(label["assessment"]["shadow_sides"]),
                "face": label["classification"]["face"],
                "layout_id": label["classification"]["layout_id"],
                "orientation_degrees_cw": label["classification"]["orientation_degrees_cw"],
                "card_type": label["classification"]["card_type"],
                "labeler": label["annotation"]["labeler"],
                "reviewer": label["annotation"]["reviewer"],
                "revision": label["annotation"]["revision"],
                "outer_source": label["annotation"]["outer_source"],
                "inner_source": label["annotation"]["inner_source"],
                "preannotation_disposition": label["annotation"]["preannotation_disposition"],
                "prelabel_sha256": label["annotation"]["prelabel_sha256"],
                "outer_corners_json": json.dumps(label["geometry"]["outer_corners"], ensure_ascii=False, separators=(",", ":")),
                "inner_lines_rectified_json": json.dumps(label["geometry"]["inner_lines_rectified"], ensure_ascii=False, separators=(",", ":")),
                "inner_line_centers_px_json": json.dumps(centers, ensure_ascii=False, separators=(",", ":")),
                "inner_line_midpoints_px_json": json.dumps(midpoints, ensure_ascii=False, separators=(",", ":")),
                "centering_measurements_json": json.dumps(measurements, ensure_ascii=False, separators=(",", ":")),
                "custom_metadata_json": json.dumps(label["custom_metadata"], ensure_ascii=False, separators=(",", ":")),
            }
        writer.writerow({key: spreadsheet_safe(value) for key, value in row.items()})
    return ("\ufeff" + stream.getvalue()).encode("utf-8")


def _normalize_points(points: Any, width: float, height: float) -> Any:
    if points is None:
        return None
    return [
        [round(float(point[0]) / width, 8), round(float(point[1]) / height, 8)]
        for point in points
    ]


def _inner_center_geometry(lines: Any) -> tuple[dict[str, float] | None, dict[str, list[float]] | None]:
    centers = centers_from_lines(lines)
    if centers is None:
        return None, None
    try:
        geometry = center_geometry_from_centers(centers, 630, 880)
    except ValueError:
        return None, None
    return geometry["line_centers_px"], geometry["line_midpoints_px"]


def _annotations_jsonl_bytes(
    samples: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
) -> bytes:
    lines: list[str] = []
    for sample in samples:
        label = labels[sample["id"]]
        geometry = label["geometry"]
        rectified = geometry["rectified_size"]
        inner_pixels = geometry["inner_lines_rectified"]
        inner_centers, inner_midpoints = _inner_center_geometry(inner_pixels)
        inner_measurements = (
            centering_measurements(inner_centers, rectified["width"], rectified["height"])
            if inner_centers is not None
            else None
        )
        inner_normalized = None
        if inner_pixels is not None:
            inner_normalized = {
                side: _normalize_points(points, rectified["width"], rectified["height"])
                for side, points in inner_pixels.items()
            }
        record = {
            "schema_version": "1.0",
            "sample_id": sample["id"],
            "file_name": sample["filename"],
            "sha256": sample["sha256"],
            "image": {"width": sample["width"], "height": sample["height"]},
            "status": label["annotation_status"],
            "outer": {
                "state": geometry["outer_state"],
                "coordinate_space": "exif_normalized_pixels",
                "corners_pixels": geometry["outer_corners"],
                "corners_normalized_0_1": _normalize_points(
                    geometry["outer_corners"], sample["width"], sample["height"]
                ),
            },
            "inner": {
                "state": geometry["inner_state"],
                "coordinate_space": "rectified_card_pixels",
                "rectified_size": rectified,
                "lines_pixels": inner_pixels,
                "lines_normalized_0_1": inner_normalized,
                "line_centers_pixels": inner_centers,
                "line_midpoints_pixels": inner_midpoints,
                "coordinate_semantics": "zero_width_red_line_center",
                "centering_measurements": inner_measurements,
            },
            "assessment": label["assessment"],
            "classification": label["classification"],
            "custom_metadata": label["custom_metadata"],
            "annotation": label["annotation"],
        }
        lines.append(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _coco_bytes(samples: list[dict[str, Any]], labels: dict[str, dict[str, Any]]) -> bytes:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for image_id, sample in enumerate(samples, start=1):
        label = labels[sample["id"]]
        images.append(
            {
                "id": image_id,
                "file_name": f"images/normalized/{sample['id']}.png",
                "width": sample["width"],
                "height": sample["height"],
                "sample_id": sample["id"],
                "sha256": sample["sha256"],
                "annotation_status": label["annotation_status"],
            }
        )
        corners = label["geometry"]["outer_corners"]
        if label["annotation_status"] != "accepted" or corners is None:
            continue
        xs = [float(point[0]) for point in corners]
        ys = [float(point[1]) for point in corners]
        polygon = [coordinate for point in corners for coordinate in (float(point[0]), float(point[1]))]
        area = abs(
            sum(
                float(corners[index][0]) * float(corners[(index + 1) % 4][1])
                - float(corners[index][1]) * float(corners[(index + 1) % 4][0])
                for index in range(4)
            )
        ) / 2.0
        inner_centers, inner_midpoints = _inner_center_geometry(
            label["geometry"]["inner_lines_rectified"]
        )
        inner_size = label["geometry"]["rectified_size"]
        inner_measurements = (
            centering_measurements(inner_centers, inner_size["width"], inner_size["height"])
            if inner_centers is not None
            else None
        )
        annotations.append(
            {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [polygon],
                "area": round(area, 4),
                "bbox": [
                    round(min(xs), 4),
                    round(min(ys), 4),
                    round(max(xs) - min(xs), 4),
                    round(max(ys) - min(ys), 4),
                ],
                "iscrowd": 0,
                "keypoints": [coordinate for point in corners for coordinate in (float(point[0]), float(point[1]), 2)],
                "num_keypoints": 4,
                "sample_id": sample["id"],
                "annotation_status": label["annotation_status"],
                "planarity": label["assessment"]["planarity"],
                "reason_codes": label["assessment"]["reason_codes"],
                "classification": label["classification"],
                "preannotation_provenance": {
                    "outer_source": label["annotation"]["outer_source"],
                    "inner_source": label["annotation"]["inner_source"],
                    "disposition": label["annotation"]["preannotation_disposition"],
                    "prelabel_sha256": label["annotation"]["prelabel_sha256"],
                },
                "inner_lines_rectified": label["geometry"]["inner_lines_rectified"],
                "inner_line_centers_px": inner_centers,
                "inner_line_midpoints_px": inner_midpoints,
                "inner_coordinate_semantics": "zero_width_red_line_center",
                "centering_measurements": inner_measurements,
                "inner_rectified_size": label["geometry"]["rectified_size"],
                "custom_metadata": label["custom_metadata"],
            }
        )
    coco = {
        "info": {
            "description": "PTCG Annotation Studio ML export. Annotations contain accepted labels only; images may include all statuses.",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "card",
                "supercategory": "object",
                "keypoints": ["top_left", "top_right", "bottom_right", "bottom_left"],
                "skeleton": [[1, 2], [2, 3], [3, 4], [4, 1]],
            }
        ],
    }
    return json_bytes(coco)


def build_export(
    store: StudioStore,
    project_id: str,
    mode: str,
    sample_ids: list[str] | None = None,
) -> ExportArtifact:
    if mode not in EXPORT_MODES:
        raise StudioError(400, "INVALID_EXPORT_MODE", "Unknown export mode.")
    if sample_ids is not None:
        if not isinstance(sample_ids, list) or len(sample_ids) > store.config.max_project_assets:
            raise StudioError(
                422,
                "INVALID_SAMPLE_IDS",
                "sample_ids must be an array within the project asset limit.",
            )
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

    with store._lock:
        detail = store.project_detail(project_id)
        project = detail["project"]
        samples = detail["samples"]
        if sample_ids is not None:
            available = {sample["id"] for sample in samples}
            missing = [sample_id for sample_id in sample_ids if sample_id not in available]
            if missing:
                raise StudioError(
                    404,
                    "SAMPLE_NOT_FOUND",
                    "One or more selected samples do not exist.",
                    details={"sample_ids": missing},
                )
            selected = set(sample_ids)
            samples = [sample for sample in samples if sample["id"] in selected]
        labels: dict[str, dict[str, Any]] = {}
        for sample in samples:
            label = store.get_label(project_id, sample["id"])
            revision = label.get("annotation", {}).get("revision")
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
                raise StudioError(
                    409,
                    "INVALID_STORED_LABEL",
                    f"Stored label {sample['id']} has an invalid revision and cannot be exported.",
                )
            # Export is a trust boundary: re-run the complete semantic gate so
            # legacy or externally edited labels cannot bypass newer rules.
            labels[sample["id"]] = validate_label(
                label,
                project,
                sample,
                current_revision=revision,
            )
        labels_root = safe_path(store._project_root(project_id), "labels")
        selected_counts = {
            "samples": len(samples),
            "saved_labels": sum(
                safe_path(labels_root, f"{sample['id']}.json").is_file()
                for sample in samples
            ),
            "accepted": sum(
                labels[sample["id"]]["annotation_status"] == "accepted" for sample in samples
            ),
            "review": sum(
                labels[sample["id"]]["annotation_status"] == "review" for sample in samples
            ),
            "rejected": sum(
                labels[sample["id"]]["annotation_status"] == "rejected" for sample in samples
            ),
        }
        # Every export profile claims sample identities derived from immutable
        # source bytes, so verify all originals even when the ZIP profile does
        # not embed them.
        verified_originals = {
            sample["id"]: store._verify_original(project_id, sample)
            for sample in samples
        }
        temporary_root = safe_path(store.root, "export_tmp")
        temporary_root.mkdir(exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f"{project_id}_{mode}_",
            suffix=".zip",
            dir=temporary_root,
        )
        os.close(descriptor)
        path = Path(temporary_name)
        try:
            with zipfile.ZipFile(
                path,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
                allowZip64=True,
            ) as archive:
                builder = _ArchiveBuilder(archive)
                export_project = {
                    key: value
                    for key, value in project.items()
                    if key not in {"sample_count", "progress"}
                }
                builder.add_bytes("project.json", json_bytes(export_project))
                for sample in samples:
                    builder.add_bytes(f"labels/{sample['id']}.json", json_bytes(labels[sample["id"]]))
                builder.add_bytes("labels.csv", _csv_bytes(samples, labels))
                builder.add_bytes("annotations.jsonl", _annotations_jsonl_bytes(samples, labels))
                builder.add_bytes("coco.json", _coco_bytes(samples, labels))

                # Import reports are project-wide and can mention samples that
                # were not selected. Include them only for whole-project
                # exports so a filtered archive contains no unrelated records.
                if sample_ids is None:
                    imports_root = safe_path(store._project_root(project_id), "imports")
                    for report_path in sorted(
                        imports_root.glob("imp_*.json"), key=lambda item: item.name
                    ):
                        builder.add_file(
                            f"import_reports/{report_path.name}",
                            safe_path(imports_root, report_path.name),
                        )

                if mode == "full":
                    for sample in samples:
                        original = verified_originals[sample["id"]]
                        normalized = safe_path(
                            store._project_root(project_id),
                            "normalized",
                            f"{sample['id']}.png",
                        )
                        if sha256_file(normalized) != sample["normalized_sha256"]:
                            raise StudioError(409, "SOURCE_INTEGRITY_ERROR", "A normalized image failed verification.")
                        original_name = _safe_zip_filename(sample["filename"])
                        builder.add_file(
                            f"images/original/{sample['id']}/{original_name}",
                            original,
                            expected_sha256=sample["sha256"],
                        )
                        builder.add_file(
                            f"images/normalized/{sample['id']}.png",
                            normalized,
                            expected_sha256=sample["normalized_sha256"],
                        )

                if mode == "annotated":
                    for sample in samples:
                        normalized, _, _ = store.image_bytes(project_id, sample["id"], "normalized")
                        overlay = draw_annotated_overlay(normalized, labels[sample["id"]])
                        builder.add_bytes(f"images/annotated/{sample['id']}.png", overlay)

                manifest = {
                    "schema_version": "1.0",
                    "application": "PTCG Annotation Studio ML",
                    "profile": mode,
                    "project_id": project_id,
                    "project_name": project["name"],
                    "created_at": utc_now(),
                    "counts": selected_counts,
                    "coordinate_spaces": {
                        "outer": "exif_normalized_pixels",
                        "inner": "rectified_card_pixels",
                        "outer_corner_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
                    },
                    "overlay_style": {
                        "outer_color_rgb": [0, 220, 110],
                        "inner_color_rgb": [245, 52, 80],
                        "line_width": "1-4 pixels, proportional to source resolution",
                    },
                    "members": sorted(builder.members, key=lambda item: item["path"]),
                }
                archive.writestr(_zip_info("manifest.json"), json_bytes(manifest))
            safe_project_name = _safe_zip_filename(project["name"]).strip("._") or "project"
            filename = f"{safe_project_name}_{mode}.zip"
            return ExportArtifact(path=path, filename=filename, size=path.stat().st_size)
        except Exception:
            path.unlink(missing_ok=True)
            raise
