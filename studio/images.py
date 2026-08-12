from __future__ import annotations

import io
import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from .config import AppConfig, OPTIONAL_FORMATS, SUPPORTED_BASE_FORMATS
from .errors import StudioError


EXIF_ORIENTATION_TAG = 274


def _register_optional_codecs() -> dict[str, Any]:
    result: dict[str, Any] = {
        "pillow_heif_installed": False,
        "pillow_heif_usable": False,
        "heif": False,
        "heic": False,
        "avif": False,
        "details": "",
    }
    errors: list[str] = []
    try:
        import pillow_heif  # type: ignore[import-not-found]

        result["pillow_heif_installed"] = True
        register_heif = getattr(pillow_heif, "register_heif_opener", None)
        if callable(register_heif):
            try:
                register_heif()
            except Exception as exc:  # Optional plugin must never block core startup.
                errors.append(f"HEIF registration failed: {type(exc).__name__}: {exc}")
        else:
            errors.append("pillow_heif has no callable register_heif_opener")
        register_avif = getattr(pillow_heif, "register_avif_opener", None)
        if callable(register_avif):
            try:
                register_avif()
            except Exception as exc:  # Pillow may still provide native AVIF support.
                errors.append(f"AVIF registration failed: {type(exc).__name__}: {exc}")
    except Exception as exc:  # ImportError, DLL load failure, incompatible plugin, etc.
        errors.append(f"pillow_heif unavailable: {type(exc).__name__}: {exc}")

    try:
        extensions = Image.registered_extensions()
        formats = {str(value).upper() for value in extensions.values()}
    except Exception as exc:  # Defensive: optional capability reporting is fail-soft.
        formats = set()
        errors.append(f"codec capability scan failed: {type(exc).__name__}: {exc}")
    result["heif"] = "HEIF" in formats
    result["heic"] = result["heif"] or "HEIC" in formats
    result["avif"] = "AVIF" in formats
    result["pillow_heif_usable"] = bool(
        result["pillow_heif_installed"] and (result["heif"] or result["heic"])
    )
    result["details"] = "; ".join(errors)
    return result


OPTIONAL_CODEC_STATUS = _register_optional_codecs()


@dataclass(slots=True)
class DecodedImage:
    detected_format: str
    media_type: str
    source_width: int
    source_height: int
    normalized_width: int
    normalized_height: int
    exif_orientation_original: int
    frame_count: int
    animated: bool
    normalized_png: bytes
    preview_png: bytes

    def metadata(self) -> dict[str, Any]:
        return {
            "detected_format": self.detected_format,
            "media_type": self.media_type,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "width": self.normalized_width,
            "height": self.normalized_height,
            "exif_orientation_original": self.exif_orientation_original,
            "frame_count": self.frame_count,
            "first_frame_only": self.frame_count > 1,
            "animated": self.animated,
        }


def _optional_codec_for(filename: str) -> str | None:
    suffix = Path(filename).suffix.lower()
    for name, details in OPTIONAL_FORMATS.items():
        if suffix in details["extensions"]:
            return name
    return None


def _png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", compress_level=6, optimize=False)
    return output.getvalue()


def _bit_hex(values: list[bool]) -> str:
    output = bytearray((len(values) + 7) // 8)
    for index, value in enumerate(values):
        if value:
            output[index // 8] |= 1 << (7 - index % 8)
    return output.hex()


def _pixels(image: Image.Image) -> list[int]:
    flattened = getattr(image, "get_flattened_data", None)
    return list(flattened() if callable(flattened) else image.getdata())


def visual_fingerprint(normalized_png: bytes) -> dict[str, Any]:
    """Return a privacy-safe fingerprint for resized/re-encoded duplicates.

    Exact SHA-256 remains the primary identity.  This fingerprint is only used
    to recognize the same photograph after messenger/web compression.  A
    coarse hash selects a small bucket; callers must still verify the thumbnail
    distance before treating two images as equivalent.
    """

    with Image.open(io.BytesIO(normalized_png)) as image:
        image.load()
        gray = image.convert("L")
        width, height = gray.size
        thumbnail = _pixels(gray.resize((16, 16), Image.Resampling.LANCZOS))
        average = sum(thumbnail) / len(thumbnail)
        ahash = _bit_hex([value >= average for value in thumbnail])
        differences = _pixels(gray.resize((17, 16), Image.Resampling.LANCZOS))
        dhash_bits: list[bool] = []
        for row in range(16):
            offset = row * 17
            dhash_bits.extend(
                differences[offset + column + 1] >= differences[offset + column]
                for column in range(16)
            )
        dhash = _bit_hex(dhash_bits)
    return {
        "version": "gray16-v1",
        "bucket": hashlib.sha256(ahash.encode("ascii")).hexdigest()[:24],
        "ahash": ahash,
        "dhash": dhash,
        "thumbnail": thumbnail,
        "width": width,
        "height": height,
        "aspect_ratio": round(width / max(1, height), 8),
    }


def visual_fingerprints_match(left: Any, right: Any) -> bool:
    """Strictly verify that two coarse-hash candidates are the same view."""

    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    if left.get("version") != "gray16-v1" or right.get("version") != "gray16-v1":
        return False
    if left.get("bucket") != right.get("bucket") or left.get("ahash") != right.get("ahash"):
        return False
    try:
        left_aspect = float(left["aspect_ratio"])
        right_aspect = float(right["aspect_ratio"])
        left_thumb = [int(value) for value in left["thumbnail"]]
        right_thumb = [int(value) for value in right["thumbnail"]]
    except (KeyError, TypeError, ValueError):
        return False
    if len(left_thumb) != 256 or len(right_thumb) != 256:
        return False
    if abs(left_aspect - right_aspect) > 0.001:
        return False
    differences = [abs(a - b) for a, b in zip(left_thumb, right_thumb)]
    if max(differences, default=255) > 6 or sum(differences) / 256.0 > 1.25:
        return False
    try:
        left_dhash = bytes.fromhex(str(left["dhash"]))
        right_dhash = bytes.fromhex(str(right["dhash"]))
    except (KeyError, TypeError, ValueError):
        return False
    if len(left_dhash) != len(right_dhash) or not left_dhash:
        return False
    hamming = sum((a ^ b).bit_count() for a, b in zip(left_dhash, right_dhash))
    return hamming <= 8


def build_preview_png(normalized_png: bytes, max_dimension: int) -> bytes:
    """Build a bounded PNG preview while preserving normalized coordinates.

    The browser stretches this preview over the immutable normalized image
    dimensions, so labels remain expressed in full-resolution source pixels.
    """
    if max_dimension < 1:
        raise ValueError("max_dimension must be positive")
    with Image.open(io.BytesIO(normalized_png)) as image:
        image.load()
        if max(image.size) <= max_dimension:
            return normalized_png
        preview = image.copy()
        preview.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        return _png_bytes(preview)


def decode_image(data: bytes, filename: str, config: AppConfig) -> DecodedImage:
    if not data:
        raise StudioError(422, "EMPTY_FILE", "The uploaded file is empty.")
    if len(data) > config.max_upload_bytes:
        raise StudioError(
            413,
            "UPLOAD_TOO_LARGE",
            f"The file exceeds the {config.max_upload_bytes} byte upload limit.",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(data))
            detected_format = str(image.format or "").upper()
            # Pillow reports some ordinary phone-camera .JPG files as MPO
            # because they contain an MPF auxiliary-image segment.  MPO is a
            # JPEG container; v1 intentionally uses its first frame and stores
            # the immutable original bytes exactly as uploaded.
            if detected_format == "MPO":
                detected_format = "JPEG"
            source_width, source_height = image.size
            if source_width < 1 or source_height < 1:
                raise StudioError(422, "INVALID_IMAGE_DIMENSIONS", "Image dimensions are invalid.")
            if source_width > config.max_dimension or source_height > config.max_dimension:
                raise StudioError(
                    413,
                    "IMAGE_DIMENSION_LIMIT",
                    f"Image dimensions exceed {config.max_dimension} pixels per side.",
                )
            if source_width * source_height > config.max_pixels:
                raise StudioError(
                    413,
                    "IMAGE_PIXEL_LIMIT",
                    f"Image exceeds the {config.max_pixels} decoded-pixel limit.",
                )
            if detected_format not in SUPPORTED_BASE_FORMATS and detected_format not in {
                "HEIC",
                "HEIF",
                "AVIF",
            }:
                raise StudioError(
                    415,
                    "UNSUPPORTED_IMAGE_FORMAT",
                    f"Decoded image format {detected_format or 'UNKNOWN'} is not supported.",
                )
            exif_orientation = int(image.getexif().get(EXIF_ORIENTATION_TAG, 1) or 1)
            if exif_orientation not in range(1, 9):
                exif_orientation = 1
            frame_count = int(getattr(image, "n_frames", 1) or 1)
            animated = bool(getattr(image, "is_animated", False))
            if frame_count > 1:
                image.seek(0)
            image.load()
            normalized = ImageOps.exif_transpose(image)
            has_alpha = normalized.mode in {"RGBA", "LA"} or "transparency" in normalized.info
            normalized = normalized.convert("RGBA" if has_alpha else "RGB")
            normalized_width, normalized_height = normalized.size
            normalized_png = _png_bytes(normalized)
            if max(normalized.size) <= config.preview_max_dimension:
                # A second full-resolution PNG encode is pure overhead when
                # the normalized image already satisfies the preview limit.
                # Sharing the immutable bytes also lets storage use the
                # normalized image as the preview without a duplicate file.
                preview_png = normalized_png
            else:
                preview = normalized.copy()
                preview.thumbnail(
                    (config.preview_max_dimension, config.preview_max_dimension),
                    Image.Resampling.LANCZOS,
                )
                preview_png = _png_bytes(preview)
    except StudioError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise StudioError(413, "DECOMPRESSION_BOMB_BLOCKED", "Image decompression limit exceeded.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        optional_format = _optional_codec_for(filename)
        codec_key = optional_format.lower() if optional_format else ""
        if optional_format and not bool(OPTIONAL_CODEC_STATUS.get(codec_key, False)):
            plugin = OPTIONAL_FORMATS[optional_format]["plugin"]
            raise StudioError(
                415,
                "OPTIONAL_CODEC_REQUIRED",
                f"{optional_format} could not be decoded. Install/enable {plugin}, or convert to JPEG/PNG.",
                details={"format": optional_format, "plugin": plugin},
            ) from exc
        raise StudioError(415, "IMAGE_DECODE_FAILED", "The file is corrupt or not a supported image.") from exc

    if detected_format in SUPPORTED_BASE_FORMATS:
        media_type = str(SUPPORTED_BASE_FORMATS[detected_format]["mime"])
    elif detected_format in {"HEIC", "HEIF"}:
        media_type = "image/heic" if detected_format == "HEIC" else "image/heif"
    else:
        media_type = "image/avif"
    return DecodedImage(
        detected_format=detected_format,
        media_type=media_type,
        source_width=source_width,
        source_height=source_height,
        normalized_width=normalized_width,
        normalized_height=normalized_height,
        exif_orientation_original=exif_orientation,
        frame_count=frame_count,
        animated=animated,
        normalized_png=normalized_png,
        preview_png=preview_png,
    )


def _solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise StudioError(422, "DEGENERATE_GEOMETRY", "Outer corners do not define a valid card.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(augmented[row], augmented[column])
                ]
    return [augmented[row][-1] for row in range(size)]


def perspective_coefficients(
    output_points: list[tuple[float, float]],
    source_points: list[tuple[float, float]],
) -> tuple[float, ...]:
    matrix: list[list[float]] = []
    vector: list[float] = []
    for (u, v), (x, y) in zip(output_points, source_points):
        matrix.append([u, v, 1.0, 0.0, 0.0, 0.0, -u * x, -v * x])
        vector.append(x)
        matrix.append([0.0, 0.0, 0.0, u, v, 1.0, -u * y, -v * y])
        vector.append(y)
    return tuple(_solve_linear(matrix, vector))


def rectify_png(normalized_png: bytes, outer_corners: list[list[float]], width: int, height: int) -> bytes:
    image = Image.open(io.BytesIO(normalized_png)).convert("RGB")
    output_points = [
        (0.0, 0.0),
        (float(width - 1), 0.0),
        (float(width - 1), float(height - 1)),
        (0.0, float(height - 1)),
    ]
    source_points = [(float(point[0]), float(point[1])) for point in outer_corners]
    coefficients = perspective_coefficients(output_points, source_points)
    corrected = image.transform(
        (width, height),
        Image.Transform.PERSPECTIVE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
    )
    return _png_bytes(corrected)


def draw_annotated_overlay(normalized_png: bytes, label: dict[str, Any]) -> bytes:
    image = Image.open(io.BytesIO(normalized_png)).convert("RGB")
    draw = ImageDraw.Draw(image)
    scale = max(image.size) / 1800.0
    line_width = max(1, min(4, round(scale)))
    corners = label.get("geometry", {}).get("outer_corners")
    if corners:
        outer = [(float(point[0]), float(point[1])) for point in corners]
        draw.line(outer + [outer[0]], fill=(0, 220, 110), width=line_width, joint="curve")

        inner = label.get("geometry", {}).get("inner_lines_rectified")
        size = label.get("geometry", {}).get("rectified_size", {})
        if inner and size:
            rectified_width = float(size.get("width", 630))
            rectified_height = float(size.get("height", 880))
            output_points = [
                (0.0, 0.0),
                (rectified_width - 1.0, 0.0),
                (rectified_width - 1.0, rectified_height - 1.0),
                (0.0, rectified_height - 1.0),
            ]
            # This homography maps rectified points back to normalized source pixels.
            coefficients = perspective_coefficients(output_points, outer)

            def project(point: list[float]) -> tuple[float, float]:
                u, v = float(point[0]), float(point[1])
                a, b, c, d, e, f, g, h = coefficients
                divisor = g * u + h * v + 1.0
                return ((a * u + b * v + c) / divisor, (d * u + e * v + f) / divisor)

            for side in ("left", "right", "top", "bottom"):
                endpoints = inner.get(side)
                if endpoints:
                    draw.line([project(endpoints[0]), project(endpoints[1])], fill=(245, 52, 80), width=line_width)
    return _png_bytes(image)
