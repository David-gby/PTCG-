from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def source_path(row: dict[str, str], manifest: Path) -> Path | None:
    raw = row.get("archive") or row.get("image")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (manifest.parent / path).resolve()
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter a manifest to locally readable sources")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    available: list[dict[str, str]] = []
    for row in rows:
        path = source_path(row, input_path)
        if path is not None and path.is_file():
            available.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(available)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "total_rows": len(rows),
        "available_rows": len(available),
        "missing_rows": len(rows) - len(available),
        "by_split": dict(Counter(row.get("split", "") for row in available)),
        "by_source": dict(Counter(row.get("source", "") for row in available)),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
