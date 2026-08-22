from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite official synthetic masks as exact physical edge-intersection quads."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--width", type=float, default=640.0)
    parser.add_argument("--height", type=float, default=896.0)
    args = parser.parse_args()

    with args.metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rewritten = 0
    for row in rows:
        if row.get("source") != "official_synthetic" or not row.get("points"):
            continue
        points = np.asarray(json.loads(row["points"]), dtype=np.float64).reshape(4, 2)
        values = ["0"]
        for x, y in points:
            values.extend(
                (
                    f"{np.clip(x / args.width, 0.0, 1.0):.6f}",
                    f"{np.clip(y / args.height, 0.0, 1.0):.6f}",
                )
            )
        label = args.dataset / "labels" / row["split"] / f"{row['sample_id']}.txt"
        label.write_text(" ".join(values) + "\n", encoding="utf-8")
        rewritten += 1

    for cache in (args.dataset / "labels").glob("*.cache"):
        cache.unlink()
    print(json.dumps({"rewritten": rewritten, "geometry": "physical_quad"}, indent=2))


if __name__ == "__main__":
    main()
