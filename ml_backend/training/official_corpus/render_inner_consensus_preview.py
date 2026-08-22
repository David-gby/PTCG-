from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np


def _decode(archive: zipfile.ZipFile, entry: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(archive.read(entry), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(entry)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        image = np.clip(image[:, :, :3] * alpha + 242.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
    return cv2.resize(image, (315, 440), interpolation=cv2.INTER_AREA)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", choices=("accepted", "rejected"), default="accepted")
    parser.add_argument("--count", type=int, default=24)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines()]
    desired = args.state == "accepted"
    rows = [row for row in rows if bool(row.get("accepted")) == desired]
    rows.sort(key=lambda row: row["entry_name"])
    if len(rows) > args.count:
        indexes = np.linspace(0, len(rows) - 1, args.count).astype(int)
        rows = [rows[index] for index in indexes]
    tiles: list[np.ndarray] = []
    with zipfile.ZipFile(args.archive) as archive:
        for row in rows:
            tile = _decode(archive, row["entry_name"])
            scale_x, scale_y = 0.5, 0.5
            box = row.get("label_box") or row.get("box")
            if box:
                left, right = int(round(box["left"] * scale_x)), int(round(box["right"] * scale_x))
                top, bottom = int(round(box["top"] * scale_y)), int(round(box["bottom"] * scale_y))
                cv2.rectangle(tile, (left, top), (right, bottom), (40, 40, 240), 1, cv2.LINE_AA)
            label = f"{row.get('split')} n={row.get('cluster_size', 0)} {row.get('reason', '')}"
            cv2.rectangle(tile, (0, 0), (315, 24), (12, 20, 18), -1)
            cv2.putText(tile, label[:43], (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (240, 245, 242), 1, cv2.LINE_AA)
            tiles.append(tile)
    columns = 6
    blank = np.full((440, 315, 3), 24, dtype=np.uint8)
    while len(tiles) % columns:
        tiles.append(blank.copy())
    montage = np.vstack([np.hstack(tiles[index : index + columns]) for index in range(0, len(tiles), columns)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), montage):
        raise OSError(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
