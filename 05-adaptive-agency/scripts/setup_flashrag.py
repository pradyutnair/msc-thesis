#!/usr/bin/env python3
"""Setup FlashRAG corpus for ARAG experiments.

Steps:
  1. Unzip wiki18_100w.zip -> wiki18_100w.jsonl
  2. Build SQLite database with FTS5 index for keyword search
  3. Verify FAISS index compatibility
  4. Download FlashRAG dataset splits

Usage:
    python scripts/setup_flashrag.py \
        --flashrag-dir /scratch-shared/pnair/flashrag \
        --output-dir /scratch-shared/pnair/flashrag
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def unzip_corpus(flashrag_dir: Path) -> Path:
    """Unzip wiki18_100w.zip if not already extracted."""
    zip_path = flashrag_dir / "wiki18_100w.zip"
    jsonl_path = flashrag_dir / "wiki18_100w.jsonl"

    if jsonl_path.exists():
        logger.info("Corpus already extracted: %s", jsonl_path)
        return jsonl_path

    if not zip_path.exists():
        logger.error("Zip file not found: %s", zip_path)
        logger.error("Download from: https://huggingface.co/datasets/ignore/FlashRAG_datasets")
        sys.exit(1)

    logger.info("Extracting %s ...", zip_path)
    t0 = time.monotonic()
    with zipfile.ZipFile(zip_path, "r") as zf:
        # The zip may contain the file directly or in a subdirectory
        members = zf.namelist()
        logger.info("Zip contents: %s", members[:10])
        zf.extractall(flashrag_dir)
    elapsed = time.monotonic() - t0
    logger.info("Extraction complete in %.1f seconds", elapsed)

    # Find the extracted jsonl file (might be in a subdirectory)
    if not jsonl_path.exists():
        # Search for it
        for member in members:
            if member.endswith(".jsonl"):
                extracted = flashrag_dir / member
                if extracted.exists():
                    extracted.rename(jsonl_path)
                    logger.info("Moved %s -> %s", extracted, jsonl_path)
                    break

    if not jsonl_path.exists():
        logger.error("Could not find extracted JSONL file")
        sys.exit(1)

    return jsonl_path


def build_sqlite_db(jsonl_path: Path, output_dir: Path) -> Path:
    """Build SQLite database with FTS5 index for keyword search.

    Schema:
        passages(id INTEGER PRIMARY KEY, passage_id TEXT, title TEXT, contents TEXT)
        passages_fts(passage_id, title, contents) USING fts5
    """
    db_path = output_dir / "wiki18_100w.db"

    if db_path.exists():
        # Check if it's complete
        conn = sqlite3.connect(str(db_path))
        try:
            count = conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
            if count > 20_000_000:
                logger.info("SQLite DB already exists with %d passages: %s", count, db_path)
                conn.close()
                return db_path
            else:
                logger.info("DB exists but only has %d passages, rebuilding...", count)
        except sqlite3.OperationalError:
            logger.info("DB exists but is corrupt/incomplete, rebuilding...")
        conn.close()
        db_path.unlink()

    logger.info("Building SQLite database from %s ...", jsonl_path)
    t0 = time.monotonic()

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-1000000")  # 1GB cache

    # Main table
    conn.execute("""
        CREATE TABLE passages (
            id INTEGER PRIMARY KEY,
            passage_id TEXT NOT NULL,
            title TEXT,
            contents TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX idx_passage_id ON passages(passage_id)")

    # FTS5 virtual table for full-text search
    conn.execute("""
        CREATE VIRTUAL TABLE passages_fts USING fts5(
            passage_id UNINDEXED,
            title,
            contents,
            content=passages,
            content_rowid=id
        )
    """)

    batch_size = 50_000
    batch = []
    total = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            passage_id = str(obj.get("id", total))
            title = obj.get("title", "")
            contents = obj.get("contents", "")

            batch.append((passage_id, title, contents))
            total += 1

            if len(batch) >= batch_size:
                conn.executemany(
                    "INSERT INTO passages (passage_id, title, contents) VALUES (?, ?, ?)",
                    batch,
                )
                # Populate FTS index
                conn.executemany(
                    "INSERT INTO passages_fts (rowid, passage_id, title, contents) "
                    "SELECT id, passage_id, title, contents FROM passages "
                    "WHERE id > ? AND id <= ?",
                    [(total - len(batch), total)],
                )
                conn.commit()
                elapsed = time.monotonic() - t0
                rate = total / elapsed
                logger.info(
                    "  Indexed %d passages (%.0f/s, %.1f min elapsed)",
                    total, rate, elapsed / 60,
                )
                batch = []

    # Final batch
    if batch:
        conn.executemany(
            "INSERT INTO passages (passage_id, title, contents) VALUES (?, ?, ?)",
            batch,
        )
        conn.commit()

    # Rebuild FTS index from scratch (more reliable than incremental)
    logger.info("Rebuilding FTS5 index...")
    conn.execute("INSERT INTO passages_fts(passages_fts) VALUES('rebuild')")
    conn.commit()

    elapsed = time.monotonic() - t0
    logger.info(
        "SQLite DB complete: %d passages in %.1f min -> %s",
        total, elapsed / 60, db_path,
    )

    # Verify
    count = conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
    fts_count = conn.execute("SELECT COUNT(*) FROM passages_fts").fetchone()[0]
    logger.info("Verification: passages=%d, fts=%d", count, fts_count)

    conn.close()
    return db_path


def verify_faiss_index(flashrag_dir: Path) -> bool:
    """Verify the FAISS index exists and is loadable."""
    index_dir = flashrag_dir / "wiki18_100w_e5_index"

    if not index_dir.exists():
        # Check for zip
        zip_path = flashrag_dir / "wiki18_100w_e5_index.zip"
        if zip_path.exists():
            logger.info("Extracting FAISS index zip...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(flashrag_dir)
        else:
            logger.warning(
                "FAISS index not found at %s. "
                "Download from ModelScope or build with FlashRAG.",
                index_dir,
            )
            return False

    # Look for the index file
    index_files = list(index_dir.glob("*.index")) + list(index_dir.glob("*.faiss"))
    if not index_files:
        # FlashRAG might store it differently
        all_files = list(index_dir.iterdir())
        logger.info("Files in index dir: %s", [f.name for f in all_files])
        if all_files:
            logger.info("FAISS index directory exists with %d files", len(all_files))
            return True
        logger.warning("No index files found in %s", index_dir)
        return False

    logger.info("FAISS index found: %s", index_files[0])

    try:
        import faiss
        index = faiss.read_index(str(index_files[0]))
        logger.info("FAISS index loaded: %d vectors, dimension=%d", index.ntotal, index.d)
        del index
        return True
    except ImportError:
        logger.warning("faiss not installed, skipping verification")
        return True
    except Exception as e:
        logger.error("Failed to load FAISS index: %s", e)
        return False


def download_flashrag_datasets(output_dir: Path) -> None:
    """Download FlashRAG dataset splits for HotPotQA, 2Wiki, MuSiQue."""
    datasets_dir = output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.warning("huggingface_hub not installed, skipping dataset download")
        logger.info("Install with: pip install huggingface_hub")
        logger.info("Or download manually from: https://huggingface.co/datasets/ignore/FlashRAG_datasets")
        return

    # FlashRAG datasets repo
    repo_id = "ignore/FlashRAG_datasets"

    dataset_files = {
        "hotpotqa": "hotpotqa/dev.jsonl",
        "2wikimultihopqa": "2wikimultihopqa/dev.jsonl",
        "musique": "musique/dev.jsonl",
    }

    for name, path in dataset_files.items():
        out_path = datasets_dir / name / "dev.jsonl"
        if out_path.exists():
            logger.info("Dataset already exists: %s", out_path)
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=path,
                repo_type="dataset",
                local_dir=str(datasets_dir),
            )
            logger.info("Downloaded %s -> %s", path, downloaded)
        except Exception as e:
            logger.warning("Failed to download %s: %s", path, e)


def main():
    parser = argparse.ArgumentParser(description="Setup FlashRAG corpus")
    parser.add_argument(
        "--flashrag-dir",
        default="/scratch-shared/pnair/flashrag",
        help="Directory with wiki18_100w.zip",
    )
    parser.add_argument(
        "--output-dir",
        default="/scratch-shared/pnair/flashrag",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--skip-sqlite", action="store_true",
        help="Skip SQLite DB build",
    )
    parser.add_argument(
        "--skip-datasets", action="store_true",
        help="Skip dataset downloads",
    )
    args = parser.parse_args()

    flashrag_dir = Path(args.flashrag_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Unzip corpus
    jsonl_path = unzip_corpus(flashrag_dir)

    # Step 2: Build SQLite DB
    if not args.skip_sqlite:
        build_sqlite_db(jsonl_path, output_dir)

    # Step 3: Verify FAISS index
    verify_faiss_index(flashrag_dir)

    # Step 4: Download dataset splits
    if not args.skip_datasets:
        download_flashrag_datasets(output_dir)

    logger.info("Setup complete!")
    logger.info("Next: run scripts/convert_flashrag_questions.py to convert datasets")


if __name__ == "__main__":
    main()
