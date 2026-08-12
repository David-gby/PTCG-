from __future__ import annotations

import copy
from typing import Any

from .errors import StudioError
from .security import SAFE_REASON, finite_number, reject_unknown, require_text, validate_json_value


LABEL_FIELDS = {
    "schema_version",
    "project_id",
    "sample_id",
    "source",
    "annotation_status",
    "geometry",
    "assessment",
    "classification",
    "custom_metadata",
    "annotation",
}
SOURCE_FIELDS = {"sha256", "width", "height", "coordinate_space"}
GEOMETRY_FIELDS = {
    "coordinate_space",
    "outer_state",
    "outer_corners",
    "inner_state",
    "inner_lines_rectified",
    "rectified_size",
}
RECTIFIED_SIZE_FIELDS = {"width", "height"}
INNER_SIDES = {"left", "right", "top", "bottom"}
ASSESSMENT_FIELDS = {"planarity", "reason_codes", "shadow_sides", "notes"}
CLASSIFICATION_FIELDS = {"face", "layout_id", "orientation_degrees_cw", "card_type"}
ANNOTATION_FIELDS = {
    "labeler",
    "reviewer",
    "revision",
    "created_at",
    "updated_at",
    "outer_source",
    "inner_source",
    "preannotation_disposition",
    "prelabel_sha256",
}

STATUSES = {"unlabeled", "in_progress", "accepted", "review", "rejected"}
GEOMETRY_STATES = {"unset", "applicable", "not_applicable", "unresolved"}
PLANARITY_STATES = {"unknown", "planar", "non_planar", "unresolved"}
FACES = {"front", "back", "unknown"}
SHADOW_SIDES = {"left", "right", "top", "bottom"}
ORIENTATIONS = {0, 90, 180, 270, None}
GEOMETRY_SOURCES = {"manual_unset", "manual", "prelabel_accepted", "prelabel_modified"}
PREANNOTATION_DISPOSITIONS = {
    "not_available",
    "available_unconfirmed",
    "accepted_unchanged",
    "accepted_modified",
    "rejected",
}
REASON_CODES = {
    "IMAGE_TOO_SMALL",
    "IMAGE_BLURRED",
    "GLARE_OCCLUDES_BOUNDARY",
    "CARD_CROPPED",
    "CARD_TOUCHES_FRAME",
    "CARD_NONPLANAR",
    "PERSPECTIVE_TOO_EXTREME",
    "OUTER_EDGE_INCOMPLETE",
    "SHADOW_EDGE_AMBIGUOUS",
    "MATERIAL_BOUNDARY_AMBIGUOUS",
    "INNER_BOUNDARY_AMBIGUOUS",
    "LENS_DISTORTION_UNMODELED",
    "WRONG_FACE_OR_LAYOUT",
    "NOT_IN_SCOPE",
    "OTHER_REVIEW_REASON",
}
def default_label(project: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    classification = copy.deepcopy(project["default_classification"])
    return {
        "schema_version": "1.0",
        "project_id": project["id"],
        "sample_id": sample["id"],
        "source": {
            "sha256": sample["sha256"],
            "width": sample["width"],
            "height": sample["height"],
            "coordinate_space": "exif_normalized_pixels",
        },
        "annotation_status": "unlabeled",
        "geometry": {
            "coordinate_space": "exif_normalized_pixels",
            "outer_state": "unset",
            "outer_corners": None,
            "inner_state": "unset",
            "inner_lines_rectified": None,
            "rectified_size": copy.deepcopy(project["rectified_size"]),
        },
        "assessment": {
            "planarity": "unknown",
            "reason_codes": [],
            "shadow_sides": [],
            "notes": "",
        },
        "classification": classification,
        "custom_metadata": {},
        "annotation": {
            "labeler": "",
            "reviewer": "",
            "revision": 0,
            "created_at": None,
            "updated_at": None,
            "outer_source": "manual_unset",
            "inner_source": "manual_unset",
            "preannotation_disposition": "not_available",
            "prelabel_sha256": None,
        },
    }


def _point(value: Any, field: str, max_x: float, max_y: float) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise StudioError(422, "INVALID_GEOMETRY", f"{field} must be [x,y].")
    x = finite_number(value[0], f"{field}.x")
    y = finite_number(value[1], f"{field}.y")
    if x < 0 or y < 0 or x > max_x or y > max_y:
        raise StudioError(422, "GEOMETRY_OUT_OF_BOUNDS", f"{field} is outside its image.")
    return [round(x, 4), round(y, 4)]


def validate_outer_corners(value: Any, width: int, height: int) -> list[list[float]] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise StudioError(422, "INVALID_GEOMETRY", "outer_corners must contain four points.")
    points = [_point(point, f"outer_corners[{index}]", width - 1, height - 1) for index, point in enumerate(value)]
    signed_twice_area = sum(
        points[index][0] * points[(index + 1) % 4][1]
        - points[index][1] * points[(index + 1) % 4][0]
        for index in range(4)
    )
    if signed_twice_area <= max(20.0, width * height * 0.0001):
        raise StudioError(
            422,
            "INVALID_CORNER_ORDER",
            "Corners must be non-degenerate and ordered top-left, top-right, bottom-right, bottom-left.",
        )
    cross_products: list[float] = []
    for index in range(4):
        first = points[index]
        second = points[(index + 1) % 4]
        third = points[(index + 2) % 4]
        cross_products.append(
            (second[0] - first[0]) * (third[1] - second[1])
            - (second[1] - first[1]) * (third[0] - second[0])
        )
    if any(value <= 1e-6 for value in cross_products):
        raise StudioError(
            422,
            "INVALID_CORNER_ORDER",
            "Outer corners must form a clockwise-on-screen convex quadrilateral.",
        )
    return points


def _validate_classification(value: Any) -> dict[str, Any]:
    item = reject_unknown(value, CLASSIFICATION_FIELDS, "classification")
    face = item.get("face")
    if face not in FACES:
        raise StudioError(422, "INVALID_FIELD", "classification.face is invalid.")
    orientation = item.get("orientation_degrees_cw")
    if orientation not in ORIENTATIONS:
        raise StudioError(422, "INVALID_FIELD", "classification.orientation_degrees_cw is invalid.")
    return {
        "face": face,
        "layout_id": require_text(item.get("layout_id"), "classification.layout_id", maximum=128),
        "orientation_degrees_cw": orientation,
        "card_type": require_text(item.get("card_type"), "classification.card_type", maximum=128),
    }


def validate_label(
    incoming: Any,
    project: dict[str, Any],
    sample: dict[str, Any],
    *,
    current_revision: int,
) -> dict[str, Any]:
    label = reject_unknown(incoming, LABEL_FIELDS, "label")
    if label.get("schema_version") != "1.0":
        raise StudioError(422, "INVALID_SCHEMA_VERSION", "label.schema_version must be 1.0.")

    source = reject_unknown(label.get("source"), SOURCE_FIELDS, "source")
    expected_source = {
        "sha256": sample["sha256"],
        "width": sample["width"],
        "height": sample["height"],
        "coordinate_space": "exif_normalized_pixels",
    }
    if source != expected_source:
        raise StudioError(422, "SOURCE_FIELDS_IMMUTABLE", "Source metadata is server-owned.")
    if label.get("project_id") != project["id"] or label.get("sample_id") != sample["id"]:
        raise StudioError(422, "IDENTIFIERS_IMMUTABLE", "Project and sample IDs are server-owned.")

    status = label.get("annotation_status")
    if status not in STATUSES:
        raise StudioError(422, "INVALID_FIELD", "annotation_status is invalid.")

    geometry = reject_unknown(label.get("geometry"), GEOMETRY_FIELDS, "geometry")
    if geometry.get("coordinate_space") != "exif_normalized_pixels":
        raise StudioError(422, "INVALID_FIELD", "geometry.coordinate_space is invalid.")
    outer_state = geometry.get("outer_state")
    inner_state = geometry.get("inner_state")
    if outer_state not in GEOMETRY_STATES or inner_state not in GEOMETRY_STATES:
        raise StudioError(422, "INVALID_FIELD", "A geometry state is invalid.")
    outer = validate_outer_corners(geometry.get("outer_corners"), sample["width"], sample["height"])

    rectified_size = reject_unknown(geometry.get("rectified_size"), RECTIFIED_SIZE_FIELDS, "rectified_size")
    canonical = project["rectified_size"]
    if rectified_size != canonical:
        raise StudioError(422, "RECTIFIED_SIZE_IMMUTABLE", "rectified_size is fixed by the project.")

    inner_raw = geometry.get("inner_lines_rectified")
    inner: dict[str, list[list[float]]] | None = None
    if inner_raw is not None:
        inner_object = reject_unknown(inner_raw, INNER_SIDES, "inner_lines_rectified")
        if set(inner_object) != INNER_SIDES:
            raise StudioError(422, "INVALID_GEOMETRY", "All four inner lines are required.")
        inner = {}
        for side in ("left", "right", "top", "bottom"):
            endpoints = inner_object[side]
            if not isinstance(endpoints, list) or len(endpoints) != 2:
                raise StudioError(422, "INVALID_GEOMETRY", f"inner line {side} needs two endpoints.")
            inner[side] = [
                _point(point, f"inner.{side}[{index}]", canonical["width"] - 1, canonical["height"] - 1)
                for index, point in enumerate(endpoints)
            ]

    if (outer_state == "applicable") != (outer is not None):
        raise StudioError(422, "GEOMETRY_STATE_MISMATCH", "outer_state and outer_corners disagree.")
    if (inner_state == "applicable") != (inner is not None):
        raise StudioError(422, "GEOMETRY_STATE_MISMATCH", "inner_state and inner lines disagree.")
    if inner is not None:
        if outer is None:
            raise StudioError(422, "GEOMETRY_STATE_MISMATCH", "Inner lines require complete outer corners.")
        average_left = sum(point[0] for point in inner["left"]) / 2.0
        average_right = sum(point[0] for point in inner["right"]) / 2.0
        average_top = sum(point[1] for point in inner["top"]) / 2.0
        average_bottom = sum(point[1] for point in inner["bottom"]) / 2.0
        if average_right - average_left < 2.0 or average_bottom - average_top < 2.0:
            raise StudioError(
                422,
                "INVALID_INNER_LINE_ORDER",
                "Inner lines must be ordered left/right and top/bottom with a non-zero interior.",
            )

    assessment = reject_unknown(label.get("assessment"), ASSESSMENT_FIELDS, "assessment")
    planarity = assessment.get("planarity")
    if planarity not in PLANARITY_STATES:
        raise StudioError(422, "INVALID_FIELD", "assessment.planarity is invalid.")
    reason_codes = assessment.get("reason_codes")
    if not isinstance(reason_codes, list) or len(reason_codes) > 32:
        raise StudioError(422, "INVALID_FIELD", "assessment.reason_codes must be a short list.")
    normalized_reasons: list[str] = []
    for code in reason_codes:
        if not isinstance(code, str) or not SAFE_REASON.fullmatch(code):
            raise StudioError(422, "INVALID_FIELD", "A reason code is invalid.")
        if code not in normalized_reasons:
            normalized_reasons.append(code)
    shadows = assessment.get("shadow_sides")
    if not isinstance(shadows, list) or any(side not in SHADOW_SIDES for side in shadows):
        raise StudioError(422, "INVALID_FIELD", "assessment.shadow_sides is invalid.")
    normalized_shadows = [side for side in ("left", "right", "top", "bottom") if side in shadows]
    notes = require_text(assessment.get("notes"), "assessment.notes", maximum=4000)

    classification = _validate_classification(label.get("classification"))
    custom_metadata = copy.deepcopy(label.get("custom_metadata"))
    if not isinstance(custom_metadata, dict):
        raise StudioError(422, "INVALID_FIELD", "custom_metadata must be an object.")
    validate_json_value(custom_metadata)

    annotation = reject_unknown(label.get("annotation"), ANNOTATION_FIELDS, "annotation")
    if annotation.get("revision") != current_revision:
        raise StudioError(422, "REVISION_FIELD_MISMATCH", "label.annotation.revision is stale.")
    labeler = require_text(annotation.get("labeler"), "annotation.labeler", maximum=128)
    reviewer = require_text(annotation.get("reviewer"), "annotation.reviewer", maximum=128)
    outer_source = annotation.get("outer_source", "manual_unset")
    inner_source = annotation.get("inner_source", "manual_unset")
    disposition = annotation.get("preannotation_disposition", "not_available")
    prelabel_sha256 = annotation.get("prelabel_sha256")
    if outer_source not in GEOMETRY_SOURCES or inner_source not in GEOMETRY_SOURCES:
        raise StudioError(422, "INVALID_FIELD", "annotation geometry source is invalid.")
    if disposition not in PREANNOTATION_DISPOSITIONS:
        raise StudioError(422, "INVALID_FIELD", "annotation.preannotation_disposition is invalid.")
    if prelabel_sha256 is not None and (
        not isinstance(prelabel_sha256, str)
        or len(prelabel_sha256) != 64
        or any(character not in "0123456789abcdef" for character in prelabel_sha256)
    ):
        raise StudioError(422, "INVALID_FIELD", "annotation.prelabel_sha256 is invalid.")
    uses_prelabel = outer_source.startswith("prelabel_") or inner_source.startswith("prelabel_")
    if (uses_prelabel or disposition.startswith("accepted_")) and prelabel_sha256 is None:
        raise StudioError(422, "PRELABEL_PROVENANCE_REQUIRED", "Adopted pre-label geometry requires its SHA-256 provenance.")

    if status == "accepted":
        if planarity != "planar" or outer is None or inner is None or not labeler:
            raise StudioError(422, "INCOMPLETE_ACCEPTED_LABEL", "Accepted labels require a labeler and complete planar geometry.")
        if normalized_reasons:
            raise StudioError(
                422,
                "ACCEPTED_LABEL_HAS_REASON_CODES",
                "Accepted labels cannot contain reason codes; resolve or clear them before acceptance.",
            )
    if status == "rejected" and not normalized_reasons:
        raise StudioError(422, "REJECT_REASON_REQUIRED", "Rejected labels require at least one reason code.")
    if planarity == "non_planar":
        if status != "rejected" or "CARD_NONPLANAR" not in normalized_reasons or inner is not None:
            raise StudioError(
                422,
                "NON_PLANAR_RULE_VIOLATION",
                "Non-planar cards must be rejected with CARD_NONPLANAR and no inner lines.",
            )
        inner_state = "not_applicable"

    result = default_label(project, sample)
    result.update(
        {
            "annotation_status": status,
            "geometry": {
                "coordinate_space": "exif_normalized_pixels",
                "outer_state": outer_state,
                "outer_corners": outer,
                "inner_state": inner_state,
                "inner_lines_rectified": inner,
                "rectified_size": copy.deepcopy(canonical),
            },
            "assessment": {
                "planarity": planarity,
                "reason_codes": normalized_reasons,
                "shadow_sides": normalized_shadows,
                "notes": notes,
            },
            "classification": classification,
            "custom_metadata": custom_metadata,
            "annotation": {
                "labeler": labeler,
                "reviewer": reviewer,
                "revision": current_revision,
                "created_at": annotation.get("created_at"),
                "updated_at": annotation.get("updated_at"),
                "outer_source": outer_source,
                "inner_source": inner_source,
                "preannotation_disposition": disposition,
                "prelabel_sha256": prelabel_sha256,
            },
        }
    )
    return result


def validate_project_create(payload: Any, rectified_width: int, rectified_height: int) -> dict[str, Any]:
    fields = {"name", "description", "default_classification", "custom_metadata"}
    value = reject_unknown(payload, fields, "project")
    classification = value.get(
        "default_classification",
        {"face": "unknown", "layout_id": "", "orientation_degrees_cw": None, "card_type": ""},
    )
    metadata = copy.deepcopy(value.get("custom_metadata", {}))
    if not isinstance(metadata, dict):
        raise StudioError(422, "INVALID_FIELD", "custom_metadata must be an object.")
    validate_json_value(metadata)
    return {
        "name": require_text(value.get("name"), "name", maximum=160, allow_empty=False),
        "description": require_text(value.get("description", ""), "description", maximum=4000),
        "default_classification": _validate_classification(classification),
        "custom_metadata": metadata,
        "rectified_size": {"width": rectified_width, "height": rectified_height},
    }
