from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        image = row.get("image", "")
        if image and not Path(image).is_absolute():
            row["image"] = str((path.parent / image).resolve())
        archive = row.get("archive", "")
        if archive and not Path(archive).is_absolute():
            row["archive"] = str((path.parent / archive).resolve())
    return rows


def _rank(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def _balanced_train_limit(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0:
        return rows
    train = [row for row in rows if row.get("split") == "train"]
    retained = [row for row in rows if row.get("split") != "train"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in train:
        grouped[row.get("group_id") or row.get("set_name") or row["id"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: _rank(row["id"]))
    keys = sorted(grouped, key=_rank)
    selected: list[dict[str, str]] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for key in keys:
            if depth < len(grouped[key]):
                selected.append(grouped[key][depth])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        depth += 1
    return selected + retained


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine manual inner labels with official consensus labels.")
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-official-train", type=int, default=3200)
    args = parser.parse_args()
    manual = _read(args.manual.resolve())
    official = _balanced_train_limit(
        _read(args.official.resolve()),
        max(0, args.max_official_train),
    )
    rows = manual + official
    fields = sorted({key for row in rows for key in row})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(
        {
            "manual": len(manual),
            "official": len(official),
            "combined": len(rows),
            "splits": {
                split: sum(row.get("split") == split for row in rows)
                for split in ("train", "val", "test")
            },
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
