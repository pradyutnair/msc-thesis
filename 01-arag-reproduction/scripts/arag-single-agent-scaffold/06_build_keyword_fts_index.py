#!/usr/bin/env python3
"""
Build two artifacts (one-time):
1) id->byte-offset map for wiki18_100w.jsonl
2) SQLite FTS5 index for keyword search

This is streaming and does not load full corpus in memory.
"""
import argparse
import json
import sqlite3
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_jsonl", required=True)
    p.add_argument("--id_offset_json", required=True)
    p.add_argument("--sqlite_db", required=True)
    p.add_argument("--limit", type=int, default=None, help="Optional debug limit")
    args = p.parse_args()

    corpus = Path(args.corpus_jsonl)
    id_offset_json = Path(args.id_offset_json)
    sqlite_db = Path(args.sqlite_db)
    id_offset_json.parent.mkdir(parents=True, exist_ok=True)
    sqlite_db.parent.mkdir(parents=True, exist_ok=True)

    # Create FTS schema
    conn = sqlite3.connect(sqlite_db)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("DROP TABLE IF EXISTS passages_fts;")
    cur.execute("CREATE VIRTUAL TABLE passages_fts USING fts5(id UNINDEXED, contents);")

    id2offset = {}
    count = 0

    with corpus.open("rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            row = json.loads(line.decode("utf-8"))
            cid = str(row["id"])
            text = row["contents"]

            id2offset[cid] = offset
            cur.execute("INSERT INTO passages_fts(id, contents) VALUES (?, ?)", (cid, text))

            count += 1
            if count % 100000 == 0:
                conn.commit()
                print(f"Indexed {count} passages...")
            if args.limit and count >= args.limit:
                break

    conn.commit()
    conn.close()
    id_offset_json.write_text(json.dumps(id2offset), encoding="utf-8")
    print(f"Done. passages={count}")
    print(f"id_offset_json={id_offset_json}")
    print(f"sqlite_db={sqlite_db}")


if __name__ == "__main__":
    main()