"""
Ingestion module — Stage 1 of the Atlas-to-RAG pipeline.

Responsibilities (per the project spec):
- Walk a source directory (stand-in for the "File Share Connector")
- Parse PDF / DOCX / XLSX with Docling; CSV with pandas
- Track a manifest (hash + mtime) so re-runs only process new/changed files
  -> this is your "Incremental Synchronization" requirement
- Emit clean Markdown + a metadata JSON per source file into an output folder,
  ready to be picked up by the AI Classification stage next

Run:
    pip install docling pandas --break-system-packages
    python ingestion.py --source ./sample_docs --output ./processed --manifest ./manifest.json
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone

SUPPORTED_DOCLING_EXTS = {".pdf", ".docx", ".pptx", ".xlsx", ".html"}
SUPPORTED_CSV_EXTS = {".csv"}
SUPPORTED_TEXT_EXTS = {".txt"}


@dataclass
class FileRecord:
    source_path: str
    file_hash: str
    mtime: float
    size_bytes: int
    status: str  # "new" | "updated" | "unchanged" | "error"
    output_path: str | None = None
    error: str | None = None
    processed_at: str | None = None


def file_hash(path: Path, block_size: int = 65536) -> str:
    """Content hash so we detect real changes, not just touched mtimes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {}

    try:
        parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        backup_path = manifest_path.with_name(f"{manifest_path.name}.corrupt")
        if backup_path.exists():
            backup_path = manifest_path.with_name(f"{manifest_path.name}.corrupt.{datetime.now().strftime('%Y%m%d%H%M%S')}")
        manifest_path.rename(backup_path)
        manifest_path.write_text("{}", encoding="utf-8")
        print(
            f"Warning: {manifest_path} is corrupted, starting with empty manifest — treating all files as new. "
            f"Backed up original file to {backup_path}",
            file=sys.stderr,
        )
        return {}

    if isinstance(parsed, dict):
        return parsed
    return {}


def save_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.write_text(json.dumps(manifest, indent=2))


def resolve_metadata_value(value):
    if callable(value):
        try:
            return value()
        except TypeError:
            return value
    return value


def prepare_metadata_for_json(metadata: dict) -> dict:
    prepared = {}
    for field_name, value in metadata.items():
        value = resolve_metadata_value(value)
        try:
            json.dumps(value)
        except TypeError as exc:
            raise TypeError(f"Non-serializable value for field '{field_name}': {value!r}") from exc
        prepared[field_name] = value
    return prepared


def discover_files(source_dir: Path) -> list[Path]:
    exts = SUPPORTED_DOCLING_EXTS | SUPPORTED_CSV_EXTS | SUPPORTED_TEXT_EXTS
    return sorted(p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def convert_with_docling(path: Path):
    """Lazy import so the script still runs (for CSV-only sources) without docling installed."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(path))
    doc = result.document
    markdown = doc.export_to_markdown()

    title_value = None
    for attr_name in ("title", "name", "get_title"):
        candidate = getattr(doc, attr_name, None)
        if candidate is None:
            continue
        title_value = resolve_metadata_value(candidate)
        if title_value is not None:
            break
    if title_value is None:
        title_value = path.stem

    num_pages_value = None
    for attr_name in ("num_pages", "get_num_pages"):
        candidate = getattr(doc, attr_name, None)
        if candidate is None:
            continue
        num_pages_value = resolve_metadata_value(candidate)
        if num_pages_value is not None:
            break

    metadata = {
        "num_pages": num_pages_value,
        "title": title_value,
        "origin_filename": path.name,
    }
    return markdown, metadata


def convert_csv(path: Path):
    import pandas as pd

    df = pd.read_csv(path)
    markdown = df.to_markdown(index=False)
    metadata = {
        "rows": len(df),
        "columns": list(df.columns),
        "origin_filename": path.name,
    }
    return markdown, metadata


def convert_text(path: Path):
    markdown = path.read_text(encoding="utf-8")
    metadata = {
        "origin_filename": path.name,
        "title": path.stem,
    }
    return markdown, metadata


def process_file(path: Path, output_dir: Path) -> FileRecord:
    h = file_hash(path)
    stat = path.stat()

    record = FileRecord(
        source_path=str(path),
        file_hash=h,
        mtime=stat.st_mtime,
        size_bytes=stat.st_size,
        status="new",
    )

    try:
        if path.suffix.lower() in SUPPORTED_CSV_EXTS:
            markdown, extra_meta = convert_csv(path)
        elif path.suffix.lower() in SUPPORTED_TEXT_EXTS:
            markdown, extra_meta = convert_text(path)
        else:
            markdown, extra_meta = convert_with_docling(path)

        out_stem = path.stem
        md_path = output_dir / f"{out_stem}.md"
        meta_path = output_dir / f"{out_stem}.meta.json"

        md_path.write_text(markdown)

        full_meta = {
            "source_path": str(path),
            "file_hash": h,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            **prepare_metadata_for_json(extra_meta),
        }
        meta_path.write_text(json.dumps(full_meta, indent=2))

        record.output_path = str(md_path)
        record.processed_at = full_meta["ingested_at"]

    except Exception as e:  # noqa: BLE001 — surface any parser failure into the manifest
        record.status = "error"
        record.error = str(e)

    return record


def run(source_dir: Path, output_dir: Path, manifest_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)

    files = discover_files(source_dir)
    if not files:
        print(f"No supported files found under {source_dir}")
        return

    new_count = updated_count = unchanged_count = error_count = 0

    for path in files:
        h = file_hash(path)
        key = h
        previous = manifest.get(key)

        if previous and previous["file_hash"] == h:
            unchanged_count += 1
            continue  # incremental sync: skip unchanged files entirely

        record = process_file(path, output_dir)
        manifest[key] = asdict(record)

        if record.status == "error":
            error_count += 1
            print(f"  [error] {path.name}: {record.error}")
        elif previous:
            updated_count += 1
            print(f"  [updated] {path.name} -> {record.output_path}")
        else:
            new_count += 1
            print(f"  [new] {path.name} -> {record.output_path}")

    save_manifest(manifest_path, manifest)

    print(
        f"\nDone. new={new_count} updated={updated_count} "
        f"unchanged={unchanged_count} errors={error_count}"
    )


def main():
    parser = argparse.ArgumentParser(description="Ingestion stage: parse source docs into clean Markdown + metadata.")
    parser.add_argument("--source", required=True, help="Folder to scan for PDF/DOCX/XLSX/CSV files")
    parser.add_argument("--output", required=True, help="Folder to write parsed .md + .meta.json files")
    parser.add_argument("--manifest", default="manifest.json", help="Path to the incremental-sync manifest file")
    args = parser.parse_args()

    source_dir = Path(args.source)
    if not source_dir.is_dir():
        print(f"Source directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    run(source_dir, Path(args.output), Path(args.manifest))


if __name__ == "__main__":
    main()
