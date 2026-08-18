"""
RAG Preparation module — Stage 3 of the Atlas-to-RAG pipeline.

Responsibilities:
- Read classified documents from Stage 2
- Clean content (remove boilerplate, headers, footers)
- Perform structure-aware chunking (split by headings, sections)
- Detect and skip exact near-duplicate chunks via SHA-256
- Generate chunk-level metadata (section, source page, lineage)
- Track document state and lineage via AssetRegistry
- Emit chunks + metadata ready for embedding

Run:
    python chunking.py --classified ./classified --output ./chunks --chunk_size 512 --overlap 100
"""

import sys
import argparse
import hashlib
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field

sys.path.insert(0, str(Path(__file__).parent.parent))
from asset_registry import AssetRegistry


@dataclass
class Chunk:
    """Extended chunk with full lineage tracking."""
    chunk_id: str
    document_id: str  # NEW: links to source document
    chunk_number: int  # NEW: position within document (1-indexed)
    total_chunks_in_document: int  # NEW: context about document size
    source_file: str
    department: str
    doc_type: str
    sensitivity: str
    content: str
    section: str = None
    chunk_size: int = 0
    
    # Lineage fields (NEW)
    file_hash: str = None  # SHA-256 from ingestion
    ingested_at: str = None
    classified_at: str = None
    chunked_at: str = None
    
    # Stable content fingerprint (NEW) — same text always hashes the same,
    # regardless of which run produced it. This is what incremental
    # re-indexing checks to decide "did this chunk's content actually
    # change since last time?" without having to re-embed to find out.
    content_hash: str = None
    
    # Traversal info for compliance
    char_offset_start: int = None
    char_offset_end: int = None
    
    created_at: str = None


def split_by_structure(content: str) -> list:
    """
    Split content along natural boundaries (headings, sections).
    Returns list of (section_title, section_content) tuples.
    """
    # Regex to find markdown headings
    heading_pattern = r'^(#{1,6})\s+(.+)$'
    
    lines = content.split('\n')
    sections = []
    current_section = None
    current_content = []
    
    for line in lines:
        match = re.match(heading_pattern, line, re.MULTILINE)
        if match:
            # Save previous section
            if current_section is not None:
                section_content = '\n'.join(current_content).strip()
                if section_content:
                    sections.append((current_section, section_content))
            # Start new section
            current_section = match.group(2)
            current_content = []
        else:
            current_content.append(line)
    
    # Save last section
    if current_section is not None:
        section_content = '\n'.join(current_content).strip()
        if section_content:
            sections.append((current_section, section_content))
    
    return sections if sections else [("Document", content)]


def clean_content(content: str) -> str:
    """
    Remove common boilerplate, repeated page numbers, and footer/header lines.
    This is a shallow heuristic aimed at reducing noise before chunking.
    """
    patterns = [
        r'^\s*Page\s+\d+(?:\s+of\s+\d+)?\s*$',
        r'^\s*Confidential\s*$',
        r'^\s*Draft\s*$',
        r'^\s*[-=]{3,}\s*$',
        r'^\s*Document\s+Title:\s*.*$',
        r'^\s*Company\s+Name:\s*.*$',
    ]
    lines = content.splitlines()
    cleaned_lines = []
    for line in lines:
        if any(re.match(pattern, line, re.IGNORECASE) for pattern in patterns):
            continue
        cleaned_lines.append(line)

    # Collapse repeated blank lines and trim leading/trailing whitespace.
    normalized = []
    previous_blank = False
    for line in cleaned_lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        normalized.append(line.rstrip())
        previous_blank = is_blank

    return '\n'.join(normalized).strip()


def chunk_hash(chunk_text: str) -> str:
    """Return a stable SHA-256 hash for a chunk of text."""
    h = hashlib.sha256()
    normalized = ' '.join(chunk_text.split()).encode('utf-8')
    h.update(normalized)
    return h.hexdigest()


def create_chunks(content: str, source_file: str, section: str = None, chunk_size: int = 512, overlap: int = 100) -> list:
    """
    Create chunks from content with sliding window overlap.
    """
    chunks = []
    words = content.split()
    
    # Simple sliding window by word count
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk_text = ' '.join(words[i:i + chunk_size])
        if chunk_text.strip():
            chunks.append(chunk_text)
    
    return chunks or [content]  # At least one chunk


def run(classified_dir: Path, output_dir: Path, chunk_size: int = 512, overlap: int = 100) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize registry and load any existing data
    registry = AssetRegistry()
    
    all_chunks = []
    # NOTE: chunk_id is no longer a run-global counter (chunk_00001, ...).
    # A counter regenerates different IDs every run even when nothing
    # changed, which made it impossible to tell "same chunk as last time"
    # for incremental re-indexing. chunk_id is now deterministic — built
    # from document_id + chunk_number below — so re-chunking an unchanged
    # document reproduces the exact same IDs every time.
    seen_hashes = set()
    chunking_timestamp = datetime.now(timezone.utc).isoformat()
    
    # PASS 1: Count chunks per document (so we know total_chunks_in_document before writing)
    document_chunk_counts = {}
    document_metadata = {}
    
    print("[Pass 1] Counting chunks per document...")
    for meta_file in sorted(classified_dir.glob("*.classified.json")):
        meta = json.loads(meta_file.read_text())
        file_hash = meta.get("file_hash", "unknown")
        
        # Find corresponding markdown
        md_name = meta_file.name.replace(".classified.json", ".md")
        md_file = classified_dir.parent / "processed" / md_name
        
        if not md_file.exists():
            print(f"  [skip] Markdown not found for {md_file.name}")
            continue
        
        # Store metadata for this document
        content = md_file.read_text()
        doc_type = meta.get("classification", {}).get("doc_type", "Document")
        department = meta.get("classification", {}).get("department", "General")
        sensitivity = meta.get("classification", {}).get("sensitivity", "Public")
        language = meta.get("classification", {}).get("language", "Unknown")
        ingested_at = meta.get("ingested_at", chunking_timestamp)
        classified_at = meta.get("classified_at", chunking_timestamp)
        
        document_metadata[file_hash] = {
            "md_name": md_name,
            "doc_type": doc_type,
            "department": department,
            "sensitivity": sensitivity,
            "language": language,
            "ingested_at": ingested_at,
            "classified_at": classified_at,
        }
        
        # Count chunks
        content = clean_content(content)
        sections = split_by_structure(content)
        
        chunk_count = 0
        for section_title, section_content in sections:
            section_chunks = create_chunks(
                section_content,
                md_name,
                section=section_title,
                chunk_size=chunk_size,
                overlap=overlap
            )
            for chunk_text in section_chunks:
                chunk_signature = chunk_hash(chunk_text)
                if chunk_signature not in seen_hashes:
                    chunk_count += 1
                    seen_hashes.add(chunk_signature)
        
        document_chunk_counts[file_hash] = chunk_count
        
        # Register in AssetRegistry if not already there
        if not registry.get_by_file_hash(file_hash):
            origin_filename = meta.get("origin_filename", md_name)
            source_path = meta.get("source_path", "unknown")
            document_id = registry.register_document(
                origin_filename=origin_filename,
                source_path=source_path,
                file_hash=file_hash,
                ingested_at=ingested_at
            )
            
            # Update classification immediately
            registry.update_classification(
                document_id=document_id,
                department=department,
                doc_type=doc_type,
                language=language,
                sensitivity=sensitivity,
                classified_at=classified_at
            )
        else:
            # Get existing document_id
            doc_record = registry.get_by_file_hash(file_hash)
            document_id = doc_record.document_id
    
    # PASS 2: Create actual chunks with full lineage
    print("[Pass 2] Creating chunks with lineage...")
    seen_hashes.clear()  # Reset for actual chunk creation
    
    for meta_file in sorted(classified_dir.glob("*.classified.json")):
        meta = json.loads(meta_file.read_text())
        file_hash = meta.get("file_hash", "unknown")
        
        if file_hash not in document_metadata:
            continue
        
        md_data = document_metadata[file_hash]
        md_name = md_data["md_name"]
        total_chunks = document_chunk_counts.get(file_hash, 0)
        
        # Get document_id from registry
        doc_record = registry.get_by_file_hash(file_hash)
        document_id = doc_record.document_id if doc_record else f"doc_{file_hash[:16]}"
        
        # Find corresponding markdown
        classified_dir_parent = classified_dir.parent if "processed" not in str(classified_dir) else classified_dir.parent.parent
        md_file = classified_dir_parent / "processed" / md_name
        
        if not md_file.exists():
            continue
        
        content = md_file.read_text()
        content = clean_content(content)
        sections = split_by_structure(content)
        
        file_chunks = 0
        chunk_number = 1
        
        for section_title, section_content in sections:
            section_chunks = create_chunks(
                section_content,
                md_name,
                section=section_title,
                chunk_size=chunk_size,
                overlap=overlap
            )
            
            for chunk_text in section_chunks:
                chunk_signature = chunk_hash(chunk_text)
                if chunk_signature in seen_hashes:
                    continue
                seen_hashes.add(chunk_signature)

                # Deterministic, stable across runs: same document + same
                # position in it => same chunk_id every time. This is what
                # lets pgvector's ON CONFLICT (chunk_id) upsert and the
                # embedding cache actually recognize "unchanged" chunks.
                chunk_id = f"{document_id}_c{chunk_number:04d}"
                file_chunks += 1
                
                chunk = Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    chunk_number=chunk_number,
                    total_chunks_in_document=total_chunks,
                    source_file=md_name,
                    department=md_data["department"],
                    doc_type=md_data["doc_type"],
                    sensitivity=md_data["sensitivity"],
                    content=chunk_text,
                    section=section_title,
                    chunk_size=len(chunk_text.split()),
                    file_hash=file_hash,
                    ingested_at=md_data["ingested_at"],
                    classified_at=md_data["classified_at"],
                    chunked_at=chunking_timestamp,
                    content_hash=chunk_signature,
                    created_at=chunking_timestamp,
                )
                all_chunks.append(chunk)
                chunk_number += 1
        
        # Update registry with chunking results
        if doc_record:
            registry.update_chunking(
                document_id=document_id,
                total_chunks=total_chunks,
                chunked_at=chunking_timestamp
            )
        
        print(f"  [chunked] {md_name} → {len(sections)} sections → {file_chunks} unique chunks")
    
    # Write all chunks
    chunks_file = output_dir / "chunks.jsonl"
    with open(chunks_file, 'w') as f:
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk), default=str) + '\n')
    
    # Save registry
    registry.save()
    
    # Write summary
    summary = {
        "total_chunks": len(all_chunks),
        "total_documents": len(document_chunk_counts),
        "documents_processed": len(list(classified_dir.glob("*.classified.json"))),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "output_file": str(chunks_file),
        "registry_file": str(registry.registry_path),
        "processed_at": chunking_timestamp,
        "registry_summary": registry.summary(),
    }
    
    summary_file = output_dir / "chunking_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, default=str))
    
    print(f"\nChunking complete. {len(all_chunks)} chunks created across {len(document_chunk_counts)} documents.")
    print(f"Registry saved to: {registry.registry_path}")


def main():
    parser = argparse.ArgumentParser(description="RAG Preparation: structure-aware chunking with lineage tracking.")
    parser.add_argument("--classified", required=True, help="Folder with Stage 2 outputs (.classified.json)")
    parser.add_argument("--output", required=True, help="Folder to write chunks.jsonl")
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size in words")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap in words")
    parser.add_argument("--registry", default="data/manifests/asset_registry.json", help="Path to asset registry file")
    args = parser.parse_args()
    
    classified_dir = Path(args.classified)
    if not classified_dir.is_dir():
        print(f"Classified directory not found: {classified_dir}")
        exit(1)
    
    run(classified_dir, Path(args.output), args.chunk_size, args.overlap)


if __name__ == "__main__":
    main()
