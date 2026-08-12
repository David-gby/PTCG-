from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if getattr(sys, "frozen", False):
    APP_ROOT = Path(sys.executable).resolve().parent
    RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", APP_ROOT)).resolve()
else:
    APP_ROOT = _SOURCE_ROOT
    RESOURCE_ROOT = _SOURCE_ROOT
DEFAULT_WORKSPACE_ROOT = APP_ROOT / "workspace"


@dataclass(frozen=True, slots=True)
class AppConfig:
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT
    host: str = "127.0.0.1"
    port: int = 0
    max_upload_bytes: int = 100 * 1024 * 1024
    max_json_bytes: int = 512 * 1024
    max_pixels: int = 60_000_000
    max_dimension: int = 24_000
    preview_max_dimension: int = 2048
    max_project_assets: int = 100_000
    rectified_width: int = 630
    rectified_height: int = 880

    def normalized(self) -> "AppConfig":
        return AppConfig(
            workspace_root=self.workspace_root.expanduser().absolute(),
            host=self.host,
            port=self.port,
            max_upload_bytes=self.max_upload_bytes,
            max_json_bytes=self.max_json_bytes,
            max_pixels=self.max_pixels,
            max_dimension=self.max_dimension,
            preview_max_dimension=self.preview_max_dimension,
            max_project_assets=self.max_project_assets,
            rectified_width=self.rectified_width,
            rectified_height=self.rectified_height,
        )


SUPPORTED_BASE_FORMATS = {
    "JPEG": {"extensions": [".jpg", ".jpeg", ".jpe"], "mime": "image/jpeg"},
    "PNG": {"extensions": [".png"], "mime": "image/png"},
    "WEBP": {"extensions": [".webp"], "mime": "image/webp"},
    "BMP": {"extensions": [".bmp", ".dib"], "mime": "image/bmp"},
    "TIFF": {"extensions": [".tif", ".tiff"], "mime": "image/tiff"},
    "GIF": {"extensions": [".gif"], "mime": "image/gif"},
}

OPTIONAL_FORMATS = {
    "HEIC": {"extensions": [".heic"], "plugin": "pillow-heif"},
    "HEIF": {"extensions": [".heif", ".hif"], "plugin": "pillow-heif"},
    "AVIF": {"extensions": [".avif"], "plugin": "Pillow AVIF support or pillow-heif"},
}


def load_app_config(
    path: Path,
    *,
    workspace_override: Path | None = None,
    port_override: int | None = None,
) -> AppConfig:
    from .security import strict_json_loads

    try:
        value = strict_json_loads(path.read_bytes().decode("utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Cannot read strict studio config {path}: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"schema_version", "workspace_directory", "server", "limits", "rectified_size"}:
        raise ValueError("studio_config.json has missing or unknown top-level fields")
    if value["schema_version"] != "1.0":
        raise ValueError("studio_config.json schema_version must be 1.0")
    server = value["server"]
    limits = value["limits"]
    rectified = value["rectified_size"]
    if not isinstance(server, dict) or set(server) != {"host", "port"}:
        raise ValueError("server config must contain only host and port")
    if server["host"] != "127.0.0.1":
        raise ValueError("server.host must be 127.0.0.1")
    if not isinstance(limits, dict) or set(limits) != {
        "max_upload_bytes",
        "max_json_bytes",
        "max_pixels",
        "max_dimension",
        "preview_max_dimension",
        "max_project_assets",
    }:
        raise ValueError("limits config has missing or unknown fields")
    if not isinstance(rectified, dict) or set(rectified) != {"width", "height"}:
        raise ValueError("rectified_size must contain only width and height")
    numeric: dict[str, tuple[int, int]] = {
        "max_upload_bytes": (1024 * 1024, 2 * 1024 * 1024 * 1024),
        "max_json_bytes": (16 * 1024, 10 * 1024 * 1024),
        "max_pixels": (1_000_000, 500_000_000),
        "max_dimension": (1000, 100_000),
        "preview_max_dimension": (512, 8192),
        "max_project_assets": (1, 1_000_000),
    }
    checked: dict[str, int] = {}
    for key, (minimum, maximum) in numeric.items():
        item = limits[key]
        if isinstance(item, bool) or not isinstance(item, int) or not minimum <= item <= maximum:
            raise ValueError(f"limits.{key} must be an integer from {minimum} to {maximum}")
        checked[key] = item
    for key in ("width", "height"):
        item = rectified[key]
        if isinstance(item, bool) or not isinstance(item, int) or not 128 <= item <= 4096:
            raise ValueError(f"rectified_size.{key} must be an integer from 128 to 4096")
    configured_port = server["port"] if port_override is None else port_override
    if isinstance(configured_port, bool) or not isinstance(configured_port, int) or not 0 <= configured_port <= 65535:
        raise ValueError("server.port must be an integer from 0 to 65535")
    workspace_value = value["workspace_directory"]
    if not isinstance(workspace_value, str) or not workspace_value.strip():
        raise ValueError("workspace_directory must be non-empty text")
    workspace = workspace_override
    if workspace is None:
        configured_path = Path(workspace_value)
        workspace = configured_path if configured_path.is_absolute() else path.parent / configured_path
    return AppConfig(
        workspace_root=workspace,
        host="127.0.0.1",
        port=configured_port,
        rectified_width=rectified["width"],
        rectified_height=rectified["height"],
        **checked,
    ).normalized()
