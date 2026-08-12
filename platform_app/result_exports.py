from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_INSPECTION_ID = re.compile(r"^ins_[a-z0-9_]{3,80}$")


def inspection_result_path(workspace: Path, inspection_id: str) -> Path:
    if not _INSPECTION_ID.fullmatch(str(inspection_id)):
        raise ValueError("invalid inspection id")
    return workspace / "exports" / "inspections" / inspection_id / "result.json"


def write_inspection_result(workspace: Path, inspection: dict[str, Any]) -> Path:
    path = inspection_result_path(workspace, str(inspection["id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(inspection, ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
