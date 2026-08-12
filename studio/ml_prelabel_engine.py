"""Candidate-only adapter for the machine-learning PTCG frame pipeline.

The heavy PyTorch/Ultralytics package is imported lazily so the Studio can
start, manage projects, and inspect labels before CUDA is available. Public
results always remain human-review candidates.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any, Callable

from .measurements import centering_measurements


ENGINE_NAME = "ptcg_ml_prelabel"
ENGINE_VERSION = "3"
RECTIFIED_WIDTH = 630
RECTIFIED_HEIGHT = 880
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_PNG_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 60_000_000

_ROOT = Path(__file__).resolve().parents[1]
_ML_ROOT = _ROOT / "ml_backend"
_MANIFEST_PATH = _ML_ROOT / "model_manifest.json"


def _manifest() -> dict[str, Any]:
    try:
        value = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _manifest_metadata() -> dict[str, str]:
    manifest = _manifest()
    try:
        digest = hashlib.sha256(_MANIFEST_PATH.read_bytes()).hexdigest()
    except OSError:
        digest = "0" * 64
    package_version = str(manifest.get("package_version") or "unknown")
    pipeline_version = str(
        manifest.get("pipeline_version") or "ptcg_outer_inner_pipeline_20260718"
    )
    return {
        "package_version": package_version,
        "pipeline_version": pipeline_version,
        "manifest_sha256": digest,
        "cache_key": f"{ENGINE_NAME}:{ENGINE_VERSION}:{pipeline_version}:{package_version}:{digest}",
    }


_INITIAL_METADATA = _manifest_metadata()
PACKAGE_VERSION = _INITIAL_METADATA["package_version"]
PIPELINE_VERSION = _INITIAL_METADATA["pipeline_version"]
_MANIFEST_SHA256 = _INITIAL_METADATA["manifest_sha256"]

# Included in the Studio pre-label cache key. A changed model manifest
# invalidates old predictions instead of silently displaying stale geometry.
PRELABEL_CACHE_KEY = (
    f"{ENGINE_NAME}:{ENGINE_VERSION}:{PIPELINE_VERSION}:{PACKAGE_VERSION}:{_MANIFEST_SHA256}"
)


def get_prelabel_cache_key() -> str:
    """Return a live cache key so an atomically promoted model invalidates old predictions."""
    return _manifest_metadata()["cache_key"]

_pipeline: Any | None = None
_pipeline_lock = threading.Lock()
_pipeline_manifest_sha256: str | None = None


def _stage(
    status: str,
    generator: str,
    reason_codes: list[str] | tuple[str, ...] = (),
    **details: Any,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason_codes": sorted({str(code) for code in reason_codes if code}),
        "generator": generator,
        **details,
    }


def _base_result(layout_id: Any) -> dict[str, Any]:
    metadata = _manifest_metadata()
    return {
        "schema_version": "1.0",
        "generator": {
            "name": ENGINE_NAME,
            "version": ENGINE_VERSION,
            "algorithm_version": metadata["pipeline_version"],
            "package_version": metadata["package_version"],
            "manifest_sha256": metadata["manifest_sha256"],
            "cache_key": metadata["cache_key"],
            "execution": "local",
            "network_used": False,
            "device_requested": os.environ.get("PTCG_ML_DEVICE", "auto"),
        },
        "candidate_only": True,
        "requires_human_confirmation": True,
        "annotation_status": "unlabeled",
        "status": "unavailable",
        "reason_codes": [],
        "layout_id": layout_id if isinstance(layout_id, str) else None,
        "outer_corners": None,
        "inner_border_rectified": None,
        "inner_lines_rectified": None,
        "inner_line_centers_px": None,
        "inner_line_midpoints_px": None,
        "centering_measurements": None,
        "rectified_size": {"width": RECTIFIED_WIDTH, "height": RECTIFIED_HEIGHT},
        "stages": {
            "decode": _stage("not_run", "opencv_imdecode"),
            "outer": _stage("not_run", "ptcg_ml_outer_pipeline"),
            "rectification": _stage("not_run", "ptcg_ml_rectification"),
            "layout": _stage("provided", "caller_layout_id"),
            "inner": _stage("not_run", "ptcg_ml_inner_pipeline"),
        },
        # JSON-safe model result retained for reviewed-feedback export.
        "model_result": None,
    }


def _finish(result: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    if result.get("outer_corners") is not None:
        reasons.append("PRELABEL_REQUIRES_HUMAN_CONFIRMATION")
        result["status"] = "review"
    else:
        result["status"] = "unavailable"
    result["reason_codes"] = sorted({str(code) for code in reasons if code})
    return result


def _png_dimensions(payload: memoryview) -> tuple[int, int] | None:
    if payload.nbytes < 24 or bytes(payload[:8]) != PNG_SIGNATURE:
        return None
    width = int.from_bytes(payload[16:20], "big")
    height = int.from_bytes(payload[20:24], "big")
    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        return None
    return width, height


def _decode_png(normalized_png: Any) -> tuple[Any | None, str | None, tuple[int, int] | None]:
    if not isinstance(normalized_png, (bytes, bytearray, memoryview)):
        return None, "PNG_BYTES_REQUIRED", None
    try:
        payload = memoryview(normalized_png).cast("B")
    except (TypeError, ValueError):
        return None, "PNG_BYTES_REQUIRED", None
    if payload.nbytes == 0:
        return None, "IMAGE_UNREADABLE", None
    if payload.nbytes > MAX_PNG_BYTES:
        return None, "IMAGE_PAYLOAD_TOO_LARGE", None
    dimensions = _png_dimensions(payload)
    if dimensions is None:
        return None, "NORMALIZED_PNG_REQUIRED", None
    try:
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except (ImportError, BufferError, ValueError, MemoryError):
        return None, "ML_RUNTIME_DEPENDENCY_MISSING", dimensions
    except Exception:
        return None, "IMAGE_DECODE_FAILED", dimensions
    if (
        image is None
        or getattr(image, "ndim", None) != 3
        or image.shape[2] not in (3, 4)
    ):
        return None, "IMAGE_DECODE_FAILED", dimensions
    return image, None, dimensions


def _load_pipeline() -> Any:
    global _pipeline, _pipeline_manifest_sha256
    current_manifest_sha256 = _manifest_metadata()["manifest_sha256"]
    if _pipeline is not None and _pipeline_manifest_sha256 == current_manifest_sha256:
        return _pipeline
    with _pipeline_lock:
        current_manifest_sha256 = _manifest_metadata()["manifest_sha256"]
        if _pipeline is not None and _pipeline_manifest_sha256 == current_manifest_sha256:
            return _pipeline
        _pipeline = None
        if not _ML_ROOT.is_dir():
            raise FileNotFoundError(f"Machine-learning backend is missing: {_ML_ROOT}")
        if str(_ML_ROOT) not in sys.path:
            sys.path.insert(0, str(_ML_ROOT))
        from ptcg_inference import CardFramePipeline

        device = os.environ.get("PTCG_ML_DEVICE") or None
        _pipeline = CardFramePipeline(device=device)
        _pipeline_manifest_sha256 = current_manifest_sha256
        return _pipeline


def _public_model_result(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    public = {key: item for key, item in value.items() if not str(key).startswith("_")}
    try:
        return json.loads(json.dumps(public, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError):
        return None


def _outer_candidate(value: Any, width: int, height: int) -> list[list[float]] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    result: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        try:
            x_value = float(point[0])
            y_value = float(point[1])
        except (TypeError, ValueError):
            return None
        if not (0.0 <= x_value <= width - 1 and 0.0 <= y_value <= height - 1):
            return None
        result.append([round(x_value, 4), round(y_value, 4)])
    return result


def inner_geometry_from_box(box: Any) -> dict[str, Any] | None:
    """Convert an ML box into mathematical zero-width red centerlines."""

    if not isinstance(box, dict):
        return None
    try:
        left = float(box["left"])
        right = float(box["right"])
        top = float(box["top"])
        bottom = float(box["bottom"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (
        0.0 <= left < right <= RECTIFIED_WIDTH - 1
        and 0.0 <= top < bottom <= RECTIFIED_HEIGHT - 1
    ):
        return None
    centers = {
        "left": round(left, 4),
        "right": round(right, 4),
        "top": round(top, 4),
        "bottom": round(bottom, 4),
    }
    middle_x = round(RECTIFIED_WIDTH / 2.0, 4)
    middle_y = round(RECTIFIED_HEIGHT / 2.0, 4)
    lines = {
        "left": [[centers["left"], 0.0], [centers["left"], float(RECTIFIED_HEIGHT - 1)]],
        "right": [[centers["right"], 0.0], [centers["right"], float(RECTIFIED_HEIGHT - 1)]],
        "top": [[0.0, centers["top"]], [float(RECTIFIED_WIDTH - 1), centers["top"]]],
        "bottom": [[0.0, centers["bottom"]], [float(RECTIFIED_WIDTH - 1), centers["bottom"]]],
    }
    midpoints = {
        "left": [centers["left"], middle_y],
        "right": [centers["right"], middle_y],
        "top": [middle_x, centers["top"]],
        "bottom": [middle_x, centers["bottom"]],
    }
    return {"centers": centers, "lines": lines, "midpoints": midpoints}


def generate_prelabel(
    normalized_png: bytes,
    layout_id: str,
    *,
    pipeline_loader: Callable[[], Any] = _load_pipeline,
) -> dict[str, Any]:
    result = _base_result(layout_id)
    reasons: list[str] = []
    image, decode_error, dimensions = _decode_png(normalized_png)
    if decode_error is not None or image is None or dimensions is None:
        code = decode_error or "IMAGE_DECODE_FAILED"
        result["stages"]["decode"] = _stage("unavailable", "opencv_imdecode", [code])
        return _finish(result, [code])
    width, height = dimensions
    result["stages"]["decode"] = _stage(
        "available",
        "opencv_imdecode",
        width=width,
        height=height,
        channels=int(image.shape[2]),
        alpha_preserved=bool(image.shape[2] == 4),
    )

    try:
        pipeline = pipeline_loader()
        raw = pipeline.infer_image(image)
    except FileNotFoundError:
        result["stages"]["outer"] = _stage(
            "unavailable", "ptcg_ml_outer_pipeline", ["ML_MODEL_FILE_MISSING"]
        )
        return _finish(result, ["ML_MODEL_FILE_MISSING"])
    except ImportError:
        result["stages"]["outer"] = _stage(
            "unavailable", "ptcg_ml_outer_pipeline", ["ML_RUNTIME_DEPENDENCY_MISSING"]
        )
        return _finish(result, ["ML_RUNTIME_DEPENDENCY_MISSING"])
    except Exception as exc:
        result["stages"]["outer"] = _stage(
            "unavailable",
            "ptcg_ml_outer_pipeline",
            ["ML_INFERENCE_ERROR"],
            error_type=type(exc).__name__,
        )
        return _finish(result, ["ML_INFERENCE_ERROR"])

    public = _public_model_result(raw)
    result["model_result"] = public
    if not isinstance(raw, dict):
        result["stages"]["outer"] = _stage(
            "unavailable", "ptcg_ml_outer_pipeline", ["ML_RESULT_INVALID"]
        )
        return _finish(result, ["ML_RESULT_INVALID"])

    outer = raw.get("outer_frame") if isinstance(raw.get("outer_frame"), dict) else {}
    corners = _outer_candidate(outer.get("points"), width, height)
    model_stage = str(raw.get("stage") or "unknown")
    error_code = str(raw.get("error_code") or outer.get("error_code") or "").strip()
    if corners is None:
        code = error_code or "OUTER_FRAME_NOT_DETECTED"
        result["stages"]["outer"] = _stage(
            "unavailable",
            "ptcg_ml_outer_pipeline",
            [code],
            confidence=outer.get("confidence"),
            model_stage=model_stage,
        )
        return _finish(result, [code])

    result["outer_corners"] = corners
    result["stages"]["outer"] = _stage(
        "available",
        "ptcg_ml_outer_pipeline",
        confidence=outer.get("confidence"),
        keypoint_confidence=outer.get("keypoint_confidence"),
        metrics=outer.get("metrics"),
        model_stage=model_stage,
    )

    rectification = raw.get("rectification") if isinstance(raw.get("rectification"), dict) else {}
    output_size = rectification.get("output_size")
    if output_size == [RECTIFIED_WIDTH, RECTIFIED_HEIGHT]:
        result["stages"]["rectification"] = _stage(
            "available",
            "ptcg_ml_rectification",
            confidence=rectification.get("confidence"),
        )
    else:
        code = error_code or "RECTIFICATION_FAILED"
        reasons.append(code)
        result["stages"]["rectification"] = _stage(
            "unavailable", "ptcg_ml_rectification", [code], output_size=output_size
        )
        result["stages"]["inner"] = _stage(
            "not_run", "ptcg_ml_inner_pipeline", ["OUTER_OR_RECTIFICATION_NOT_ACCEPTED"]
        )
        return _finish(result, reasons)

    inner = raw.get("inner_frame") if isinstance(raw.get("inner_frame"), dict) else {}
    geometry = inner_geometry_from_box(inner.get("final_box"))
    if geometry is None:
        code = error_code or "INNER_FRAME_NOT_DETECTED"
        reasons.append(code)
        result["stages"]["inner"] = _stage(
            "unavailable",
            "ptcg_ml_inner_pipeline",
            [code],
            yolo_confidence=inner.get("yolo_confidence"),
            model_stage=model_stage,
        )
        return _finish(result, reasons)

    result["inner_border_rectified"] = geometry["centers"]
    result["inner_lines_rectified"] = geometry["lines"]
    result["inner_line_centers_px"] = geometry["centers"]
    result["inner_line_midpoints_px"] = geometry["midpoints"]
    result["centering_measurements"] = centering_measurements(
        geometry["centers"], RECTIFIED_WIDTH, RECTIFIED_HEIGHT
    )
    result["stages"]["inner"] = _stage(
        "available",
        "ptcg_ml_inner_pipeline",
        yolo_confidence=inner.get("yolo_confidence"),
        edge_refinement=inner.get("edge_refinement"),
        centerline_semantics="zero_width_red_line_center",
    )
    return _finish(result, reasons)


__all__ = [
    "ENGINE_NAME",
    "ENGINE_VERSION",
    "PACKAGE_VERSION",
    "PIPELINE_VERSION",
    "PRELABEL_CACHE_KEY",
    "generate_prelabel",
    "inner_geometry_from_box",
]
