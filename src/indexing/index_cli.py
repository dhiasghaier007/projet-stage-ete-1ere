import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.indexing.index_vectors import index_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local vector index from chunk JSONL data")
    parser.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--output", required=True, help="Directory for the index artifacts")
    parser.add_argument("--pgvector-dsn", default=None, help="Optional PostgreSQL DSN for pgvector storage")
    parser.add_argument("--table-name", default="rag_chunks", help="Table name to use when writing to pgvector")
    args = parser.parse_args()

    index_path = index_chunks(args.chunks, args.output, pgvector_dsn=args.pgvector_dsn, table_name=args.table_name)
    print(f"Indexed chunks into {index_path}")


if __name__ == "__main__":
    main()
