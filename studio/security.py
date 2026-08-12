from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import StudioError


SAFE_ID = re.compile(r"^[a-z][a-z0-9_-]{2,95}$")
SAFE_REASON = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def strict_json_loads(raw: str | bytes) -> Any:
    return json.loads(
        raw,
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )


def read_json(path: Path) -> Any:
    try:
        return strict_json_loads(path.read_bytes().decode("utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise StudioError(500, "CORRUPT_DATA", f"Cannot read {path.name}: {exc}") from exc


def reject_unknown(value: Any, allowed: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StudioError(422, "INVALID_FIELD", f"{field} must be an object.")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise StudioError(
            422,
            "UNKNOWN_FIELD",
            f"Unknown {field} field(s): {', '.join(unknown)}",
        )
    return value


def finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StudioError(422, "INVALID_FIELD", f"{field} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise StudioError(422, "INVALID_FIELD", f"{field} must be finite.")
    return result


def require_text(value: Any, field: str, *, maximum: int, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise StudioError(422, "INVALID_FIELD", f"{field} must be text.")
    normalized = value.strip()
    if not allow_empty and not normalized:
        raise StudioError(422, "INVALID_FIELD", f"{field} cannot be empty.")
    if len(normalized) > maximum:
        raise StudioError(422, "INVALID_FIELD", f"{field} is longer than {maximum} characters.")
    return normalized


def validate_id(value: str, field: str = "id") -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise StudioError(404, "NOT_FOUND", f"Unknown {field}.")
    return value


def ensure_no_reparse(path: Path, root: Path) -> None:
    root_abs = Path(os.path.abspath(root))
    path_abs = Path(os.path.abspath(path))
    try:
        relative = path_abs.relative_to(root_abs)
    except ValueError as exc:
        raise StudioError(403, "PATH_OUTSIDE_WORKSPACE", "Path is outside the workspace.") from exc
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    current = root_abs
    if os.path.lexists(current):
        root_info = os.lstat(current)
        if stat.S_ISLNK(root_info.st_mode) or int(getattr(root_info, "st_file_attributes", 0)) & reparse_flag:
            raise StudioError(403, "REPARSE_POINT_FORBIDDEN", "Links and junctions are forbidden.")
    for part in relative.parts:
        current = current / part
        if not os.path.lexists(current):
            break
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode) or int(getattr(info, "st_file_attributes", 0)) & reparse_flag:
            raise StudioError(403, "REPARSE_POINT_FORBIDDEN", "Links and junctions are forbidden.")


def safe_path(root: Path, *parts: str) -> Path:
    for part in parts:
        if not isinstance(part, str) or not part or part in {".", ".."} or "/" in part or "\\" in part:
            raise StudioError(400, "INVALID_PATH_COMPONENT", "Invalid storage identifier.")
    candidate = root.joinpath(*parts)
    ensure_no_reparse(candidate, root)
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise StudioError(403, "PATH_OUTSIDE_WORKSPACE", "Path is outside the workspace.") from exc
    return candidate


def atomic_write(path: Path, data: bytes, *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_no_reparse(path.parent, path.parent.parent if path.parent != path.parent.parent else path.parent)
    if not overwrite and path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if not overwrite and path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, json_bytes(value))


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def validate_json_value(value: Any, *, field: str = "custom_metadata", depth: int = 0) -> None:
    if depth > 8:
        raise StudioError(422, "INVALID_FIELD", f"{field} is nested too deeply.")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StudioError(422, "INVALID_FIELD", f"{field} contains a non-finite number.")
        return
    if isinstance(value, list):
        if len(value) > 1000:
            raise StudioError(422, "INVALID_FIELD", f"{field} contains too many items.")
        for item in value:
            validate_json_value(item, field=field, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 200:
            raise StudioError(422, "INVALID_FIELD", f"{field} contains too many keys.")
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 128:
                raise StudioError(422, "INVALID_FIELD", f"{field} contains an invalid key.")
            validate_json_value(item, field=field, depth=depth + 1)
        return
    raise StudioError(422, "INVALID_FIELD", f"{field} contains an unsupported value.")
