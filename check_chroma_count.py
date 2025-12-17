from __future__ import annotations

import sqlite3
from pathlib import Path
import sys


def find_sqlite_file(persist_dir: Path) -> Path:
    # Common location: <persist_dir>/chroma.sqlite3
    direct = persist_dir / "chroma.sqlite3"
    if direct.exists():
        return direct

    # Fallback: search recursively (sometimes nested)
    candidates = list(persist_dir.rglob("chroma.sqlite3"))
    if not candidates:
        raise FileNotFoundError(f"Could not find chroma.sqlite3 under: {persist_dir}")

    # Pick the most recently modified
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def table_columns(cur: sqlite3.Cursor, table: str) -> set[str]:
    cur.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}  # row[1] = column name


def main():
    persist_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./chroma_db_test")
    collection_name = sys.argv[2] if len(sys.argv) > 2 else "sfs_paragraphs"

    sqlite_path = find_sqlite_file(persist_dir)
    print(f"Using sqlite db: {sqlite_path}")

    con = sqlite3.connect(str(sqlite_path))
    cur = con.cursor()

    # List tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    print("Tables:", ", ".join(sorted(tables)))

    if "collections" not in tables:
        raise RuntimeError("No 'collections' table found. This doesn't look like a Chroma sqlite DB.")

    # Get collection id
    col_cols = table_columns(cur, "collections")
    # Column name varies a bit; usually 'id' and 'name'
    if "name" not in col_cols:
        raise RuntimeError(f"'collections' table has no 'name' column. Columns: {sorted(col_cols)}")

    cur.execute("SELECT id FROM collections WHERE name = ?", (collection_name,))
    row = cur.fetchone()
    if not row:
        raise RuntimeError(f"Collection '{collection_name}' not found in sqlite DB.")
    collection_id = row[0]
    print(f"Collection id: {collection_id}")

    # Best-effort count:
    # Newer Chroma: embeddings table rows belong to segments; segments belong to collections.
    if "segments" in tables and "embeddings" in tables:
        seg_cols = table_columns(cur, "segments")
        emb_cols = table_columns(cur, "embeddings")

        # segments: collection column might be named 'collection' or 'collection_id'
        seg_collection_col = "collection" if "collection" in seg_cols else ("collection_id" if "collection_id" in seg_cols else None)
        if seg_collection_col is None:
            print(f"Warning: couldn't find collection foreign key in segments. Columns: {sorted(seg_cols)}")
        else:
            cur.execute(f"SELECT id FROM segments WHERE {seg_collection_col} = ?", (collection_id,))
            segment_ids = [r[0] for r in cur.fetchall()]
            print(f"Segments for collection: {len(segment_ids)}")

            if segment_ids:
                # embeddings: segment fk might be named 'segment_id'
                if "segment_id" in emb_cols:
                    q_marks = ",".join(["?"] * len(segment_ids))
                    cur.execute(f"SELECT COUNT(*) FROM embeddings WHERE segment_id IN ({q_marks})", segment_ids)
                    count = cur.fetchone()[0]
                    print(f"Chunks/embeddings in collection '{collection_name}': {count}")
                    return

    # Fallbacks
    if "embeddings" in tables:
        cur.execute("SELECT COUNT(*) FROM embeddings")
        count_all = cur.fetchone()[0]
        print(f"Fallback count (ALL embeddings in DB, not filtered by collection): {count_all}")
        return

    raise RuntimeError("Could not count embeddings: no known tables matched expected schema.")


if __name__ == "__main__":
    main()
