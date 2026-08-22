from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from collections import deque
import hashlib
import io
import json
import sqlite3
import time
import warnings
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


TARGET_ASPECT = 880.0 / 630.0
SCHEMA_VERSION = "1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream an official-card ZIP into a resumable metadata catalog."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def _database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS corpus_meta(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS images(
            entry_name TEXT PRIMARY KEY,
            locale TEXT,
            era TEXT,
            category TEXT,
            set_name TEXT,
            extension TEXT,
            file_size INTEGER NOT NULL,
            compressed_size INTEGER NOT NULL,
            crc32 INTEGER NOT NULL,
            blob_sha256 TEXT,
            thumbnail_sha256 TEXT,
            dhash128 TEXT,
            width INTEGER,
            height INTEGER,
            image_format TEXT,
            mode TEXT,
            frame_count INTEGER,
            orientation TEXT,
            aspect_ratio REAL,
            aspect_error REAL,
            min_dimension INTEGER,
            has_alpha INTEGER,
            transparent_ratio REAL,
            partial_alpha_ratio REAL,
            corner_alpha_json TEXT,
            edge_alpha_json TEXT,
            official_alpha_candidate INTEGER,
            card_like INTEGER,
            status TEXT NOT NULL,
            error TEXT,
            processed_at_utc TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_images_sha256 ON images(blob_sha256)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_images_thumbnail ON images(thumbnail_sha256)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_images_layout ON images(locale, era, category, set_name)"
    )
    return connection


def _path_fields(name: str) -> tuple[str | None, str | None, str | None, str | None]:
    parts = PurePosixPath(name).parts
    # Expected: archive-root / locale / era / category / set / filename.
    values = list(parts[1:5])
    values += [None] * (4 - len(values))
    return tuple(values[:4])  # type: ignore[return-value]


def _dhash128(rgb: Image.Image) -> str:
    gray = np.asarray(rgb.convert("L").resize((17, 8), Image.Resampling.BILINEAR))
    bits = gray[:, 1:] > gray[:, :-1]
    packed = np.packbits(bits.reshape(-1).astype(np.uint8))
    return packed.tobytes().hex()


def _thumbnail_sha256(rgb: Image.Image) -> str:
    thumbnail = rgb.resize((64, 90), Image.Resampling.BILINEAR)
    return hashlib.sha256(np.asarray(thumbnail, dtype=np.uint8).tobytes()).hexdigest()


def _alpha_profile(image: Image.Image) -> dict[str, Any]:
    has_alpha = bool(image.mode in {"RGBA", "LA"} or "transparency" in image.info)
    if not has_alpha:
        return {
            "has_alpha": False,
            "transparent_ratio": 0.0,
            "partial_alpha_ratio": 0.0,
            "corner_alpha": [255, 255, 255, 255],
            "edge_alpha": [255, 255, 255, 255, 255],
            "official_alpha_candidate": False,
        }
    alpha = image.convert("RGBA").getchannel("A").resize(
        (128, 128), Image.Resampling.NEAREST
    )
    values = np.asarray(alpha, dtype=np.uint8)
    corners = [
        int(values[0, 0]),
        int(values[0, -1]),
        int(values[-1, -1]),
        int(values[-1, 0]),
    ]
    mid = len(values) // 2
    edge = [
        int(values[0, mid]),
        int(values[mid, -1]),
        int(values[-1, mid]),
        int(values[mid, 0]),
        int(values[mid, mid]),
    ]
    transparent_ratio = float(np.mean(values <= 16))
    partial_ratio = float(np.mean((values > 16) & (values < 250)))
    candidate = bool(
        all(value <= 16 for value in corners)
        and all(value >= 250 for value in edge)
        and 0.00001 <= transparent_ratio <= 0.05
    )
    return {
        "has_alpha": True,
        "transparent_ratio": transparent_ratio,
        "partial_alpha_ratio": partial_ratio,
        "corner_alpha": corners,
        "edge_alpha": edge,
        "official_alpha_candidate": candidate,
    }


def _record(info: zipfile.ZipInfo, blob: bytes) -> dict[str, Any]:
    locale, era, category, set_name = _path_fields(info.filename)
    base: dict[str, Any] = {
        "entry_name": info.filename,
        "locale": locale,
        "era": era,
        "category": category,
        "set_name": set_name,
        "extension": Path(info.filename).suffix.lower(),
        "file_size": int(info.file_size),
        "compressed_size": int(info.compress_size),
        "crc32": int(info.CRC),
        "blob_sha256": hashlib.sha256(blob).hexdigest(),
        "processed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(blob)) as source:
                frame_count = int(getattr(source, "n_frames", 1))
                source.seek(0)
                normalized = ImageOps.exif_transpose(source)
                width, height = normalized.size
                alpha = _alpha_profile(normalized)
                rgb = normalized.convert("RGB")
                ratio = float(height) / max(float(width), 1.0)
                aspect_error = abs(ratio - TARGET_ASPECT) / TARGET_ASPECT
                orientation = (
                    "portrait" if height > width else "landscape" if width > height else "square"
                )
                card_like = bool(
                    orientation == "portrait"
                    and min(width, height) >= 200
                    and aspect_error <= 0.12
                )
                base.update(
                    {
                        "thumbnail_sha256": _thumbnail_sha256(rgb),
                        "dhash128": _dhash128(rgb),
                        "width": int(width),
                        "height": int(height),
                        "image_format": str(source.format or ""),
                        "mode": str(source.mode or ""),
                        "frame_count": frame_count,
                        "orientation": orientation,
                        "aspect_ratio": ratio,
                        "aspect_error": aspect_error,
                        "min_dimension": min(width, height),
                        "has_alpha": int(alpha["has_alpha"]),
                        "transparent_ratio": alpha["transparent_ratio"],
                        "partial_alpha_ratio": alpha["partial_alpha_ratio"],
                        "corner_alpha_json": json.dumps(alpha["corner_alpha"]),
                        "edge_alpha_json": json.dumps(alpha["edge_alpha"]),
                        "official_alpha_candidate": int(alpha["official_alpha_candidate"]),
                        "card_like": int(card_like),
                        "status": "ok",
                        "error": None,
                    }
                )
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        base.update(
            {
                "thumbnail_sha256": None,
                "dhash128": None,
                "width": None,
                "height": None,
                "image_format": None,
                "mode": None,
                "frame_count": None,
                "orientation": None,
                "aspect_ratio": None,
                "aspect_error": None,
                "min_dimension": None,
                "has_alpha": 0,
                "transparent_ratio": None,
                "partial_alpha_ratio": None,
                "corner_alpha_json": None,
                "edge_alpha_json": None,
                "official_alpha_candidate": 0,
                "card_like": 0,
                "status": "decode_error",
                "error": f"{type(exc).__name__}: {exc}"[:1000],
            }
        )
    return base


def _insert(connection: sqlite3.Connection, record: dict[str, Any]) -> None:
    columns = tuple(record)
    connection.execute(
        f"INSERT OR REPLACE INTO images({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
        tuple(record[column] for column in columns),
    )


def _summary(connection: sqlite3.Connection) -> dict[str, Any]:
    scalar = lambda sql: connection.execute(sql).fetchone()[0]
    by_locale = dict(
        connection.execute(
            "SELECT COALESCE(locale,'unknown'), COUNT(*) FROM images GROUP BY locale ORDER BY COUNT(*) DESC"
        ).fetchall()
    )
    by_extension = dict(
        connection.execute(
            "SELECT extension, COUNT(*) FROM images GROUP BY extension ORDER BY COUNT(*) DESC"
        ).fetchall()
    )
    dimensions = [
        {"width": row[0], "height": row[1], "count": row[2]}
        for row in connection.execute(
            """
            SELECT width, height, COUNT(*) AS count
              FROM images WHERE status='ok'
             GROUP BY width, height ORDER BY count DESC LIMIT 30
            """
        )
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "total": scalar("SELECT COUNT(*) FROM images"),
        "decoded": scalar("SELECT COUNT(*) FROM images WHERE status='ok'"),
        "decode_errors": scalar("SELECT COUNT(*) FROM images WHERE status!='ok'"),
        "card_like": scalar("SELECT COUNT(*) FROM images WHERE card_like=1"),
        "has_alpha": scalar("SELECT COUNT(*) FROM images WHERE has_alpha=1"),
        "official_alpha_candidate": scalar(
            "SELECT COUNT(*) FROM images WHERE official_alpha_candidate=1"
        ),
        "exact_duplicate_groups": scalar(
            "SELECT COUNT(*) FROM (SELECT blob_sha256 FROM images WHERE status='ok' GROUP BY blob_sha256 HAVING COUNT(*)>1)"
        ),
        "thumbnail_duplicate_groups": scalar(
            "SELECT COUNT(*) FROM (SELECT thumbnail_sha256 FROM images WHERE status='ok' GROUP BY thumbnail_sha256 HAVING COUNT(*)>1)"
        ),
        "by_locale": by_locale,
        "by_extension": by_extension,
        "top_dimensions": dimensions,
    }


def main() -> None:
    args = parse_args()
    archive = args.archive.resolve()
    output = args.output.resolve()
    if not archive.is_file():
        raise SystemExit(f"archive not found: {archive}")
    connection = _database(output)
    fingerprint = {
        "path": str(archive),
        "size": archive.stat().st_size,
        "mtime_ns": archive.stat().st_mtime_ns,
    }
    existing_fingerprint = connection.execute(
        "SELECT value FROM corpus_meta WHERE key='archive_fingerprint'"
    ).fetchone()
    if existing_fingerprint and json.loads(existing_fingerprint[0]) != fingerprint:
        raise SystemExit("output catalog belongs to a different archive")
    connection.execute(
        "INSERT OR REPLACE INTO corpus_meta(key,value) VALUES('schema_version',?)",
        (SCHEMA_VERSION,),
    )
    connection.execute(
        "INSERT OR REPLACE INTO corpus_meta(key,value) VALUES('archive_fingerprint',?)",
        (json.dumps(fingerprint, ensure_ascii=False, sort_keys=True),),
    )
    connection.commit()

    processed = {
        row[0] for row in connection.execute("SELECT entry_name FROM images")
    }
    started = time.perf_counter()
    counters: Counter[str] = Counter()
    new_count = 0

    def consume(
        pending: deque[tuple[zipfile.ZipInfo, Future[dict[str, Any]]]],
    ) -> None:
        nonlocal new_count
        _info, future = pending.popleft()
        record = future.result()
        _insert(connection, record)
        counters[record["status"]] += 1
        new_count += 1
        if new_count % 100 == 0:
            connection.commit()
        if new_count % max(args.progress_every, 1) == 0:
            elapsed = max(time.perf_counter() - started, 1e-6)
            print(
                f"[new {new_count}] ok={counters['ok']} "
                f"errors={sum(v for k,v in counters.items() if k not in {'ok','resumed'})} "
                f"rate={new_count/elapsed:.2f}/s",
                flush=True,
            )

    workers = max(1, int(args.workers))
    pending: deque[tuple[zipfile.ZipInfo, Future[dict[str, Any]]]] = deque()
    with zipfile.ZipFile(archive) as source, ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="official-card-decode",
    ) as executor:
        entries = [entry for entry in source.infolist() if not entry.is_dir()]
        if args.limit > 0:
            entries = entries[: args.limit]
        for index, info in enumerate(entries, start=1):
            if info.filename in processed:
                counters["resumed"] += 1
                continue
            try:
                blob = source.read(info)
                pending.append((info, executor.submit(_record, info, blob)))
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                locale, era, category, set_name = _path_fields(info.filename)
                record = {
                    "entry_name": info.filename,
                    "locale": locale,
                    "era": era,
                    "category": category,
                    "set_name": set_name,
                    "extension": Path(info.filename).suffix.lower(),
                    "file_size": int(info.file_size),
                    "compressed_size": int(info.compress_size),
                    "crc32": int(info.CRC),
                    "blob_sha256": None,
                    "status": "archive_error",
                    "error": f"{type(exc).__name__}: {exc}"[:1000],
                    "processed_at_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                }
                immediate: Future[dict[str, Any]] = Future()
                immediate.set_result(record)
                pending.append((info, immediate))
            while len(pending) >= workers * 3:
                consume(pending)
        while pending:
            consume(pending)
        connection.commit()

    summary = _summary(connection)
    summary["archive"] = fingerprint
    summary["elapsed_seconds"] = time.perf_counter() - started
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
