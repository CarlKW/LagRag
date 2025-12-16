from __future__ import annotations

import sqlite3
from pathlib import Path

PERSIST_DIR = Path("./chroma_db_test")
COLLECTION_NAME = "sfs_paragraphs"
N = 10


def get_columns(cur: sqlite3.Cursor, table: str) -> list[str]:
    cur.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]  # row[1] = column name


def pick_text_column(cols: list[str]) -> str:
    # Common candidates across Chroma/FTS variants
    preferred = ["content", "text", "document", "documents", "body", "chunk"]
    for name in preferred:
        if name in cols:
            return name

    # FTS tables often have columns named c0, c1, ...
    for name in cols:
        if name.lower().startswith("c") and name[1:].isdigit():
            return name

    # Last resort: first non-rowid column
    for name in cols:
        if name.lower() != "rowid":
            return name

    raise RuntimeError(f"Could not pick a text column from: {cols}")


def main():
    sqlite_path = PERSIST_DIR / "chroma.sqlite3"
    if not sqlite_path.exists():
        raise FileNotFoundError(f"Could not find {sqlite_path}")

    con = sqlite3.connect(str(sqlite_path))
    cur = con.cursor()

    # 1) Find collection id
    cur.execute("SELECT id FROM collections WHERE name = ?", (COLLECTION_NAME,))
    row = cur.fetchone()
    if not row:
        raise RuntimeError(f"Collection '{COLLECTION_NAME}' not found")
    collection_id = row[0]

    # 2) Find segments
    cur.execute("SELECT id FROM segments WHERE collection = ?", (collection_id,))
    segment_ids = [r[0] for r in cur.fetchall()]
    if not segment_ids:
        raise RuntimeError("No segments found for this collection")

    q_marks = ",".join("?" * len(segment_ids))

    # 3) Pick random embedding ids
    cur.execute(
        f"""
        SELECT e.id
        FROM embeddings e
        WHERE e.segment_id IN ({q_marks})
        ORDER BY RANDOM()
        LIMIT {N}
        """,
        segment_ids,
    )
    embedding_ids = [r[0] for r in cur.fetchall()]

    # 4) Detect text column in FTS content table
    content_table = "embedding_fulltext_search_content"
    data_table = "embedding_fulltext_search_data"

    content_cols = get_columns(cur, content_table)
    if not content_cols:
        raise RuntimeError(f"No columns found in {content_table}")

    text_col = pick_text_column(content_cols)
    print(f"Using sqlite db: {sqlite_path}")
    print(f"FTS content table columns: {content_cols}")
    print(f"Picked text column: {text_col}")

    print("=" * 80)
    print(f"Random {len(embedding_ids)} chunks from '{COLLECTION_NAME}'")
    print("=" * 80)

    for i, emb_id in enumerate(embedding_ids, 1):
        # Fetch text via FTS mapping (rowid join)
        cur.execute(
            f"""
            SELECT c.{text_col}
            FROM {content_table} c
            JOIN {data_table} d
              ON d.rowid = c.rowid
            WHERE d.embedding_id = ?
            """,
            (emb_id,),
        )
        text_row = cur.fetchone()
        text = text_row[0] if text_row and text_row[0] is not None else "<NO TEXT FOUND>"

        # Fetch metadata
        cur.execute(
            "SELECT key, string_value FROM embedding_metadata WHERE embedding_id = ?",
            (emb_id,),
        )
        metadata = {k: v for k, v in cur.fetchall()}

        print("\n" + "-" * 80)
        print(f"CHUNK {i}")
        print("-" * 80)

        print("Metadata:")
        for k in sorted(metadata):
            print(f"  {k}: {metadata[k]}")

        print("\nContent:")
        print(text[:2000] + ("…" if len(text) > 2000 else ""))

    con.close()


if __name__ == "__main__":
    main()
