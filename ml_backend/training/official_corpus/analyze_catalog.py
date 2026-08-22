from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def rows(conn: sqlite3.Connection, sql: str):
    return [list(row) for row in conn.execute(sql).fetchall()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an official-card SQLite catalog.")
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    conn = sqlite3.connect(args.catalog)
    report = {
        "locale": rows(conn, "SELECT locale, COUNT(*) FROM images GROUP BY locale ORDER BY COUNT(*) DESC"),
        "card_like": rows(conn, "SELECT card_like, COUNT(*) FROM images GROUP BY card_like"),
        "alpha": rows(
            conn,
            "SELECT official_alpha_candidate, COUNT(*) FROM images GROUP BY official_alpha_candidate",
        ),
        "formats": rows(
            conn,
            "SELECT image_format, COUNT(*) FROM images GROUP BY image_format ORDER BY COUNT(*) DESC",
        ),
        "orientation": rows(conn, "SELECT orientation, COUNT(*) FROM images GROUP BY orientation"),
        "distinct_sets": rows(
            conn,
            "SELECT locale, COUNT(DISTINCT set_name) FROM images GROUP BY locale ORDER BY locale",
        ),
        "top_eras": rows(
            conn,
            """SELECT locale, era, COUNT(*)
                 FROM images
                GROUP BY locale, era
                ORDER BY COUNT(*) DESC
                LIMIT 40""",
        ),
        "exact_duplicate_groups": rows(
            conn,
            """SELECT COUNT(*), COALESCE(SUM(n), 0)
                 FROM (SELECT COUNT(*) AS n
                         FROM images
                        GROUP BY blob_sha256
                       HAVING COUNT(*) > 1)""",
        )[0],
        "thumbnail_duplicate_groups": rows(
            conn,
            """SELECT COUNT(*), COALESCE(SUM(n), 0)
                 FROM (SELECT COUNT(*) AS n
                         FROM images
                        GROUP BY thumbnail_sha256
                       HAVING COUNT(*) > 1)""",
        )[0],
        "non_card_like_examples": rows(
            conn,
            """SELECT entry_name, width, height, ROUND(aspect_ratio, 4), image_format
                 FROM images
                WHERE card_like = 0
                ORDER BY entry_name
                LIMIT 100""",
        ),
    }
    conn.close()

    output = args.output or args.catalog.with_suffix(".analysis.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
