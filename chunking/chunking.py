"""
RAG Preparation module — Stage 3 of the Atlas-to-RAG pipeline.

Responsibilities:
- Read classified documents from Stage 2
- Clean content (remove boilerplate, headers, footers)
- Perform structure-aware chunking (split by headings, sections)
- Detect and skip exact near-duplicate chunks via SHA-256
- Generate chunk-level metadata (section, source page, lineage)
- Emit chunks + metadata ready for embedding

Run:
    python chunking.py --classified ./classified --output ./chunks --chunk_size 512 --overlap 100
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    department: str
    doc_type: str
    sensitivity: str
    content: str
    section: str = None
    chunk_size: int = 0
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
    
    all_chunks = []
    chunk_counter = 0
    seen_hashes = set()
    
    # Process each classified document
    for meta_file in sorted(classified_dir.glob("*.classified.json")):
        meta = json.loads(meta_file.read_text())
        
        # Find corresponding markdown
        md_name = meta_file.name.replace(".classified.json", ".md")
        md_file = classified_dir.parent / "processed" / md_name
        
        if not md_file.exists():
            print(f"  [skip] Markdown not found for {md_file.name}")
            continue
        
        content = md_file.read_text()
        doc_type = meta.get("classification", {}).get("doc_type", "Document")
        department = meta.get("classification", {}).get("department", "General")
        sensitivity = meta.get("classification", {}).get("sensitivity", "Public")
        
        # Clean and structure-aware chunking
        content = clean_content(content)
        sections = split_by_structure(content)
        
        file_chunks = 0
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

                chunk_id = f"chunk_{chunk_counter:05d}"
                chunk_counter += 1
                file_chunks += 1
                
                chunk = Chunk(
                    chunk_id=chunk_id,
                    source_file=md_name,
                    department=department,
                    doc_type=doc_type,
                    sensitivity=sensitivity,
                    content=chunk_text,
                    section=section_title,
                    chunk_size=len(chunk_text.split()),
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
                all_chunks.append(chunk)
        
        print(f"  [chunked] {md_name} → {len(sections)} sections → {file_chunks} unique chunks")
    
    # Write all chunks
    chunks_file = output_dir / "chunks.jsonl"
    with open(chunks_file, 'w') as f:
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk)) + '\n')
    
    # Write summary
    summary = {
        "total_chunks": len(all_chunks),
        "documents_processed": len(list(classified_dir.glob("*.classified.json"))),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "output_file": str(chunks_file),
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    
    summary_file = output_dir / "chunking_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    
    print(f"\nChunking complete. {len(all_chunks)} chunks created.")


def main():
    parser = argparse.ArgumentParser(description="RAG Preparation: structure-aware chunking and deduplication.")
    parser.add_argument("--classified", required=True, help="Folder with Stage 2 outputs (.classified.json)")
    parser.add_argument("--output", required=True, help="Folder to write chunks.jsonl")
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size in words")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap in words")
    args = parser.parse_args()
    
    classified_dir = Path(args.classified)
    if not classified_dir.is_dir():
        print(f"Classified directory not found: {classified_dir}")
        exit(1)
    
    run(classified_dir, Path(args.output), args.chunk_size, args.overlap)


if __name__ == "__main__":
    main()
