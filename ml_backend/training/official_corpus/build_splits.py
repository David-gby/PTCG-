from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def stable_number(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build set-isolated, duplicate-safe official-card splits."
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    conn = sqlite3.connect(args.catalog)
    conn.row_factory = sqlite3.Row
    records = list(
        conn.execute(
            """SELECT entry_name, locale, era, category, set_name, extension,
                      blob_sha256, thumbnail_sha256, dhash128, width, height,
                      image_format, has_alpha, official_alpha_candidate,
                      aspect_error, min_dimension
                 FROM images
                WHERE status='ok' AND card_like=1
                ORDER BY entry_name"""
        )
    )
    if not records:
        raise SystemExit("catalog contains no card-like records")

    def set_key(row: sqlite3.Row) -> str:
        return "\x1f".join(
            str(row[key] or "") for key in ("locale", "era", "category", "set_name")
        )

    keys = sorted({set_key(row) for row in records})
    union = UnionFind(keys)
    duplicate_sets: dict[str, set[str]] = defaultdict(set)
    for row in records:
        duplicate_sets[str(row["thumbnail_sha256"])].add(set_key(row))
    for members in duplicate_sets.values():
        ordered = sorted(members)
        for other in ordered[1:]:
            union.union(ordered[0], other)

    components: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in records:
        components[union.find(set_key(row))].append(row)

    strata: dict[tuple[str, str], list[tuple[str, list[sqlite3.Row]]]] = defaultdict(list)
    for component_id, members in components.items():
        majority = Counter(
            (str(row["locale"] or ""), str(row["era"] or "")) for row in members
        ).most_common(1)[0][0]
        strata[majority].append((component_id, members))

    component_split: dict[str, str] = {}
    for stratum, groups in sorted(strata.items()):
        groups.sort(key=lambda item: stable_number(item[0], args.seed))
        total = sum(len(members) for _, members in groups)
        test_target = max(1, round(total * 0.10)) if len(groups) >= 3 else 0
        val_target = max(1, round(total * 0.10)) if len(groups) >= 2 else 0
        assigned = Counter()
        for component_id, members in groups:
            if assigned["test"] < test_target:
                split = "test"
            elif assigned["val"] < val_target:
                split = "val"
            else:
                split = "train"
            component_split[component_id] = split
            assigned[split] += len(members)

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "entry_name",
        "split",
        "group_id",
        "locale",
        "era",
        "category",
        "set_name",
        "extension",
        "blob_sha256",
        "thumbnail_sha256",
        "dhash128",
        "width",
        "height",
        "image_format",
        "has_alpha",
        "official_alpha_candidate",
        "aspect_error",
        "min_dimension",
    ]
    split_counts = Counter()
    group_counts = Counter()
    seen_hash_splits: dict[str, set[str]] = defaultdict(set)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            group_id = union.find(set_key(row))
            split = component_split[group_id]
            item = {key: row[key] for key in fields if key in row.keys()}
            item.update({"split": split, "group_id": group_id})
            writer.writerow(item)
            split_counts[split] += 1
            group_counts[split] += 0
            seen_hash_splits[str(row["thumbnail_sha256"])].add(split)
    for group_id, split in component_split.items():
        group_counts[split] += 1

    leakage = {
        value: sorted(splits)
        for value, splits in seen_hash_splits.items()
        if len(splits) > 1
    }
    summary = {
        "catalog": str(args.catalog.resolve()),
        "seed": args.seed,
        "eligible_images": len(records),
        "set_keys": len(keys),
        "connected_groups": len(components),
        "images_by_split": dict(sorted(split_counts.items())),
        "groups_by_split": dict(sorted(group_counts.items())),
        "thumbnail_hash_cross_split_leakage": len(leakage),
        "split_policy": "whole set; sets connected by exact normalized thumbnails stay together",
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if leakage:
        raise SystemExit("duplicate leakage detected")


if __name__ == "__main__":
    main()
