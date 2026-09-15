"""Export legacy text records before an optional destructive index migration.

The runtime never imports this helper.  The default command is recoverable and
only writes a JSONL backup.  ``--drop-legacy`` first verifies that the backup
was written, then removes the old text tables explicitly requested by an
operator; it is never run during normal application startup.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lancedb


def export_legacy_text(db_path: Path, output_path: Path) -> int:
    db = lancedb.connect(str(db_path))
    names = set(db.table_names())
    source_names = [name for name in ("transcripts", "audio_transcripts") if name in names]
    if output_path.exists():
        raise FileExistsError(f"backup already exists; choose a new path: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as stream:
        for table_name in source_names:
            for row in db.open_table(table_name).to_arrow().to_pylist():
                record = {"source_table": table_name, "exported_at": datetime.now(timezone.utc).isoformat(), **row}
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                count += 1
    return count


def drop_legacy_text_tables(db_path: Path) -> list[str]:
    """Drop only known legacy text tables after a successful JSONL export."""
    db = lancedb.connect(str(db_path))
    names = set(db.table_names())
    dropped: list[str] = []
    for table_name in ("transcripts", "audio_transcripts"):
        if table_name in names:
            db.drop_table(table_name)
            dropped.append(table_name)
    return dropped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export legacy text records as JSONL before index migration")
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--drop-legacy",
        action="store_true",
        help="Drop legacy text tables only after the JSONL backup completes",
    )
    args = parser.parse_args()
    exported = export_legacy_text(args.db, args.output)
    print(f"exported {exported} records to {args.output}")
    if args.drop_legacy:
        if not args.output.exists():
            raise RuntimeError("backup file was not created; refusing to drop legacy tables")
        dropped = drop_legacy_text_tables(args.db)
        print(f"dropped legacy tables: {', '.join(dropped) if dropped else 'none'}")
