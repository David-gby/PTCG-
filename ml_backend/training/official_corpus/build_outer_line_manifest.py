from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def _rank(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a balanced line-refiner manifest from the synthetic outer corpus.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-train", type=int, default=3000)
    parser.add_argument("--official-val", type=int, default=600)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    with args.metadata.resolve().open("r", encoding="utf-8-sig", newline="") as stream:
        metadata = list(csv.DictReader(stream))
    selected: list[dict[str, str]] = []
    seen_history: set[tuple[str, str]] = set()
    for split in ("train", "val", "test"):
        rows = [row for row in metadata if row.get("split") == split]
        official = sorted(
            (row for row in rows if row.get("source") == "official_synthetic"),
            key=lambda row: _rank(row["sample_id"]),
        )
        limit = args.official_train if split == "train" else args.official_val
        if split == "test":
            limit = 0
        for row in official[: max(0, limit)]:
            selected.append(row)
        for row in rows:
            if row.get("source") != "historical_real":
                continue
            identity = (split, row.get("entry_name", row["sample_id"]))
            if identity in seen_history:
                continue
            seen_history.add(identity)
            selected.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("id", "split", "source", "image", "label"))
        writer.writeheader()
        for row in selected:
            sample_id = row["sample_id"]
            split = row["split"]
            writer.writerow(
                {
                    "id": sample_id,
                    "split": split,
                    "source": row["source"],
                    "image": str(dataset / "images" / split / f"{sample_id}.jpg"),
                    "label": str(dataset / "labels" / split / f"{sample_id}.txt"),
                }
            )
    print({split: sum(row["split"] == split for row in selected) for split in ("train", "val", "test")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
