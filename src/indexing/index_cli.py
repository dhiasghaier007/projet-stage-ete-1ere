import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.indexing.index_vectors import index_chunks, index_chunks_by_department


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local vector index from chunk JSONL data")
    parser.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--output", required=True, help="Directory for the index artifacts")
    parser.add_argument("--pgvector-dsn", default=None, help="Optional PostgreSQL DSN for pgvector storage")
    parser.add_argument("--table-name", default="rag_chunks", help="Table name to use when writing to pgvector (ignored with --by-department)")
    parser.add_argument("--by-department", action="store_true",
                         help="Build one local index file + one pgvector table PER DEPARTMENT "
                              "(real separate collections — the 'Two Department Knowledge Bases' "
                              "deliverable) instead of one combined index/table.")
    parser.add_argument("--no-cache", action="store_true",
                         help="Force a full re-embed of every chunk, ignoring the embedding cache. "
                              "Use this after switching embedding models — cached vectors from a "
                              "different model aren't comparable to new ones and reusing them would "
                              "silently corrupt the index.")
    parser.add_argument("--cache-path", default=None,
                         help="Override the embedding cache location (default: data/manifests/embedding_cache.json)")
    args = parser.parse_args()

    cache_kwargs = {"use_cache": not args.no_cache}
    if args.cache_path:
        cache_kwargs["cache_path"] = args.cache_path

    if args.by_department:
        index_files = index_chunks_by_department(args.chunks, args.output, pgvector_dsn=args.pgvector_dsn, **cache_kwargs)
        print(f"\nIndexed {len(index_files)} department collection(s):")
        for department, path in sorted(index_files.items()):
            print(f"  {department}: {path}")
    else:
        index_path = index_chunks(args.chunks, args.output, pgvector_dsn=args.pgvector_dsn, table_name=args.table_name, **cache_kwargs)
        print(f"Indexed chunks into {index_path}")


if __name__ == "__main__":
    main()
