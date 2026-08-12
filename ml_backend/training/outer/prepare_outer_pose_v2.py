from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine the original pose dataset with MyProject labels")
    parser.add_argument("--existing", type=Path, default=Path("datasets/card_outer_pose"))
    parser.add_argument("--new", type=Path, required=True, help="Extracted MyProject directory")
    parser.add_argument("--pair-matches", type=Path, required=True, help="CSV of rectified-card SIFT matches")
    parser.add_argument("--output", type=Path, default=Path("datasets/card_outer_pose_v2"))
    parser.add_argument("--match-threshold", type=int, default=130)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def repair_pose_label(source: Path) -> tuple[str, str]:
    lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{source}: expected one object, got {len(lines)}")
    values = [float(value) for value in lines[0].split()]
    if len(values) < 17 or (len(values) - 5) % 3:
        raise ValueError(f"{source}: invalid pose column count {len(values)}")
    point_count = (len(values) - 5) // 3
    repair = ""
    if point_count == 5:
        # One export contains two nearly identical BL clicks. The first four
        # vertices are the intended TL/TR/BR/BL polygon.
        values = values[:17]
        repair = "removed_duplicate_fifth_bl_vertex"
    elif point_count != 4:
        raise ValueError(f"{source}: expected four pose points, got {point_count}")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or np.any(array[1:5] < 0) or np.any(array[1:5] > 1):
        raise ValueError(f"{source}: bbox is non-finite or out of bounds")
    points = array[5:].reshape(4, 3)
    if np.any(points[:, :2] < 0) or np.any(points[:, :2] > 1) or np.any(points[:, 2] <= 0):
        raise ValueError(f"{source}: invalid keypoints")
    formatted = [str(int(values[0]))] + [f"{value:.6f}" for value in values[1:]]
    return " ".join(formatted) + "\n", repair


def assign_groups(groups: dict[str, list[str]], seed: int) -> dict[str, str]:
    total = sum(len(images) for images in groups.values())
    targets = {"train": round(total * 0.70), "val": round(total * 0.15)}
    targets["test"] = total - targets["train"] - targets["val"]
    rng = random.Random(seed)
    ranked = [(rng.random(), root, images) for root, images in groups.items()]
    ranked.sort(key=lambda item: (-len(item[2]), item[0], item[1]))
    counts = {split: 0 for split in SPLITS}
    assignments: dict[str, str] = {}
    for _, root, images in ranked:
        size = len(images)
        split = max(
            SPLITS,
            key=lambda name: ((targets[name] - counts[name]) / max(targets[name], 1), -counts[name]),
        )
        assignments[root] = split
        counts[split] += size
    return assignments


def copy_pair(image_source: Path, label_text: str, output: Path, split: str, image_name: str) -> None:
    image_target = output / "images" / split / image_name
    label_target = output / "labels" / split / f"{Path(image_name).stem}.txt"
    image_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_source, image_target)
    label_target.write_text(label_text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    existing = args.existing.resolve()
    new_root = args.new.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    new_images = {
        path.stem: path
        for path in sorted((new_root / "images").iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    new_labels = {path.stem: path for path in sorted((new_root / "labels").glob("*.txt"))}
    labeled_names = sorted(set(new_images) & set(new_labels))
    union_find = UnionFind(labeled_names)
    for row in read_csv(args.pair_matches):
        left = row.get("left", "")
        right = row.get("right", "")
        matches = int(row.get("good_matches", 0))
        if matches >= args.match_threshold and left in union_find.parent and right in union_find.parent:
            union_find.union(left, right)
    groups: dict[str, list[str]] = {}
    for name in labeled_names:
        groups.setdefault(union_find.find(name), []).append(name)
    assignments = assign_groups(groups, args.seed)

    manifest_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []

    old_manifest = read_csv(existing / "split_manifest.csv")
    old_metadata = read_csv(existing / "metadata.csv")
    old_metadata_by_name = {Path(row["image_path"]).name: row for row in old_metadata}
    for row in old_manifest:
        split = row["split"]
        image_name = row["image_name"]
        image_source = existing / "images" / split / image_name
        label_source = existing / "labels" / split / f"{Path(image_name).stem}.txt"
        label_text, repair = repair_pose_label(label_source)
        copy_pair(image_source, label_text, output, split, image_name)
        source_group = f"frame:{row['source_group']}"
        manifest_rows.append(
            {
                "dataset_source": "Frame.v1i.yolov8.zip",
                "source_group": source_group,
                "image_name": image_name,
                "split": split,
                "label_repair": repair,
            }
        )
        old = old_metadata_by_name.get(image_name, {})
        metadata_rows.append(
            {
                "image_id": Path(image_name).stem,
                "image_path": f"images/{split}/{image_name}",
                "split": split,
                "dataset_source": "Frame.v1i.yolov8.zip",
                "background_type": old.get("background_type", ""),
                "card_type": old.get("card_type", ""),
                "glare_level": old.get("glare_level", ""),
                "perspective_level": old.get("perspective_level", ""),
                "corner_visible": old.get("corner_visible", "true"),
                "remark": old.get("remark", ""),
            }
        )

    for name in labeled_names:
        root = union_find.find(name)
        split = assignments[root]
        image_name = f"myproject_{name}.jpg"
        label_text, repair = repair_pose_label(new_labels[name])
        copy_pair(new_images[name], label_text, output, split, image_name)
        source_group = f"myproject:{root}"
        manifest_rows.append(
            {
                "dataset_source": "MyProject.zip",
                "source_group": source_group,
                "image_name": image_name,
                "split": split,
                "label_repair": repair,
            }
        )
        metadata_rows.append(
            {
                "image_id": Path(image_name).stem,
                "image_path": f"images/{split}/{image_name}",
                "split": split,
                "dataset_source": "MyProject.zip",
                "background_type": "green",
                "card_type": "holo",
                "glare_level": "medium",
                "perspective_level": "mild",
                "corner_visible": "true",
                "remark": f"imported from MyProject.zip; source_group={source_group}",
            }
        )
        if repair:
            repair_rows.append({"image_id": name, "repair": repair})

    manifest_fields = ["dataset_source", "source_group", "image_name", "split", "label_repair"]
    metadata_fields = [
        "image_id",
        "image_path",
        "split",
        "dataset_source",
        "background_type",
        "card_type",
        "glare_level",
        "perspective_level",
        "corner_visible",
        "remark",
    ]
    write_csv(output / "split_manifest.csv", manifest_rows, manifest_fields)
    write_csv(output / "metadata.csv", metadata_rows, metadata_fields)

    data = {
        "path": str(args.output).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "card"},
        "kpt_shape": [4, 3],
        "flip_idx": [1, 0, 3, 2],
        "keypoints": {0: "TL", 1: "TR", 2: "BR", 3: "BL"},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    split_counts = {
        split: sum(1 for row in manifest_rows if row["split"] == split) for split in SPLITS
    }
    new_split_counts = {
        split: sum(
            1
            for row in manifest_rows
            if row["split"] == split and row["dataset_source"] == "MyProject.zip"
        )
        for split in SPLITS
    }
    group_splits: dict[str, set[str]] = {}
    for row in manifest_rows:
        group_splits.setdefault(str(row["source_group"]), set()).add(str(row["split"]))
    leaked = {group: sorted(splits) for group, splits in group_splits.items() if len(splits) > 1}
    report = {
        "existing_image_count": len(old_manifest),
        "new_archive_image_count": len(new_images),
        "new_labeled_image_count": len(labeled_names),
        "new_excluded_unlabeled": sorted(set(new_images) - set(new_labels)),
        "new_identity_group_count": len(groups),
        "sift_match_threshold": args.match_threshold,
        "seed": args.seed,
        "split_counts": split_counts,
        "new_split_counts": new_split_counts,
        "label_repairs": repair_rows,
        "cross_split_group_leakage": leaked,
    }
    (output / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
