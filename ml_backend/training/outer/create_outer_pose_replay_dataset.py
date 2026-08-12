from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a hard-linked pose dataset with old-domain replay")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-source", default="Frame.v1i.yolov8.zip")
    parser.add_argument("--repeats", type=int, default=2, help="Total copies of matching train rows")
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def hardlink_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    args = build_parser().parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(source / "split_manifest.csv")
    metadata = read_csv(source / "metadata.csv")
    metadata_by_name = {Path(row["image_path"]).name: row for row in metadata}
    output_manifest: list[dict[str, str]] = []
    output_metadata: list[dict[str, str]] = []
    for row in manifest:
        copies = args.repeats if row["split"] == "train" and row["dataset_source"] == args.replay_source else 1
        for copy_index in range(copies):
            prefix = "" if copy_index == 0 else f"replay{copy_index}_"
            image_name = f"{prefix}{row['image_name']}"
            split = row["split"]
            source_image = source / "images" / split / row["image_name"]
            source_label = source / "labels" / split / f"{Path(row['image_name']).stem}.txt"
            target_image = output / "images" / split / image_name
            target_label = output / "labels" / split / f"{Path(image_name).stem}.txt"
            hardlink_or_copy(source_image, target_image)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_label, target_label)
            manifest_row = dict(row)
            manifest_row["image_name"] = image_name
            output_manifest.append(manifest_row)
            old_metadata = dict(metadata_by_name[row["image_name"]])
            old_metadata["image_id"] = Path(image_name).stem
            old_metadata["image_path"] = f"images/{split}/{image_name}"
            if prefix:
                old_metadata["remark"] = f"{old_metadata.get('remark', '')}; replay copy {copy_index}"
            output_metadata.append(old_metadata)
    shutil.copy2(source / "data.yaml", output / "data.yaml")
    text = (output / "data.yaml").read_text(encoding="utf-8")
    text = text.replace(source.name, output.name, 1)
    (output / "data.yaml").write_text(text, encoding="utf-8")
    write_csv(output / "split_manifest.csv", output_manifest, list(output_manifest[0]))
    write_csv(output / "metadata.csv", output_metadata, list(output_metadata[0]))
    counts = {
        split: sum(1 for row in output_manifest if row["split"] == split)
        for split in ("train", "val", "test")
    }
    report = {
        "source": str(source),
        "replay_source": args.replay_source,
        "repeats": args.repeats,
        "split_counts": counts,
        "hard_linked_images": True,
    }
    (output / "replay_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
