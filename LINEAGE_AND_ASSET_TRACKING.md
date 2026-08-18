# Source Lineage and Asset Tracking

## Overview

This document describes the **source lineage** and **asset tracking** systems that provide end-to-end visibility into document flow through the RAG pipeline. These features answer critical questions:

- **"Where did this chunk come from, and what happened to it along the way?"** (lineage)
- **"What documents are in the system right now, and what state is each in?"** (asset tracking)

Both capabilities support compliance audits, debugging, and incremental re-indexing.

---

## Architecture

### Two Core Concepts

#### 1. **Source Lineage**
A complete audit trail showing the journey of a single piece of data:

```
Original PDF (SHA-256: 7fa10cc...)
  ↓ [ingested at 2026-08-17T10:00:00Z by ingestion]
  → markdown file (doc_001_hr_policy_english.md)
  ↓ [classified at 2026-08-17T10:05:00Z by ollama/qwen3.6]
  → classified.json (HR, Policy, Internal, EN)
  ↓ [chunked at 2026-08-17T10:10:00Z by chunking.py]
  → chunk_00001, chunk_00002, chunk_00003...
  ↓ [embedded with nomic-embed-text]
  → indexed into Postgres with vector
```

Every chunk carries backward references to:
- Original source file
- Ingestion timestamp & method
- Classification timestamp & classifier
- Chunking timestamp

#### 2. **Asset Tracking**
An inventory registry answering: "What documents exist and in what state?"

```
┌─ Document: doc_7fa10cc...
│  ├─ Origin: /atlas/hr_policies/2026/policy.pdf
│  ├─ Status: indexed
│  ├─ Department: HR
│  ├─ Doc Type: Policy
│  ├─ Sensitivity: Internal
│  ├─ Total Chunks: 42
│  ├─ Chunks Indexed: 42/42
│  └─ Lineage History:
│     ├─ [2026-08-17T10:00:00Z] ingested ✓
│     ├─ [2026-08-17T10:05:00Z] classified ✓
│     ├─ [2026-08-17T10:10:00Z] chunked ✓
│     └─ [2026-08-17T10:15:00Z] indexed ✓
```

---

## Implementation

### 1. AssetRegistry Module (`src/asset_registry.py`)

Core class for tracking document state across all pipeline stages.

#### Key Classes

**`AssetRecord`**: Immutable snapshot of a document at a point in time
```python
@dataclass
class AssetRecord:
    document_id: str           # Stable identifier (hash-based)
    file_hash: str            # SHA-256 from ingestion
    origin_filename: str       # Original file
    department: str
    doc_type: str
    language: str
    sensitivity: str
    
    ingested_at: str          # When it arrived
    classified_at: str        # When it was labeled
    chunked_at: str           # When it was split
    
    total_chunks: int         # Chunk count
    chunks_indexed: int       # How many are searchable
    status: str               # ingested → classified → chunked → indexed
    
    lineage_history: List[LineageEntry]  # Full audit trail
```

**`AssetRegistry`**: Central registry for querying and managing documents
```python
registry = AssetRegistry()

# Register at ingestion stage
document_id = registry.register_document(
    origin_filename="policy.pdf",
    source_path="/data/policy.pdf",
    file_hash="7fa10cc...",
    ingested_at="2026-08-17T10:00:00Z"
)

# Update after classification
registry.update_classification(
    document_id=document_id,
    department="HR",
    doc_type="Policy",
    language="EN",
    sensitivity="Internal",
    classified_at="2026-08-17T10:05:00Z"
)

# Update after chunking
registry.update_chunking(
    document_id=document_id,
    total_chunks=42,
    chunked_at="2026-08-17T10:10:00Z"
)

# Query operations
doc = registry.get_document(document_id)
doc = registry.get_by_file_hash("7fa10cc...")
hr_docs = registry.list_by_department("HR")
chunked_docs = registry.list_by_status("chunked")

# Persistence
registry.save()  # Write to data/manifests/asset_registry.json
registry.load()  # Restore from disk
```

### 2. Extended Chunk Dataclass (`src/chunking/chunking.py`)

Chunks now carry full lineage metadata:

```python
@dataclass
class Chunk:
    chunk_id: str
    document_id: str                  # Links to source document
    chunk_number: int                 # Position in document
    total_chunks_in_document: int     # Context
    source_file: str
    department: str
    doc_type: str
    sensitivity: str
    content: str
    section: str
    
    # Lineage fields (NEW)
    file_hash: str                    # From ingestion
    ingested_at: str
    classified_at: str
    chunked_at: str
    
    char_offset_start: int            # For reconstruction
    char_offset_end: int
    
    created_at: str
```

### 3. Two-Pass Chunking Strategy

The chunking stage now uses a two-pass approach to know document sizes before writing:

**Pass 1: Count chunks per document**
- Read all classified documents
- Count how many chunks would be created from each
- Build registry entries with classification metadata

**Pass 2: Create chunks with full lineage**
- Now knowing `total_chunks_in_document`, create actual chunks
- Each chunk includes `chunk_number` (1-indexed position)
- Carry forward lineage: `file_hash`, `ingested_at`, `classified_at`
- Timestamp the chunking stage

This two-pass approach adds minimal overhead (two linear scans) but enables:
- Each chunk to know its position in the source document
- Tracking "which chunks from a document are indexed"
- Reconstruction of original document from chunks

---

## Usage Example

### Running Chunking with Lineage Tracking

```bash
python src/chunking/chunking.py \
  --classified data/classified \
  --output data/chunks \
  --chunk_size 512 \
  --overlap 100
```

Output:
- `data/chunks/chunks.jsonl` - chunks with full lineage
- `data/chunks/chunking_summary.json` - aggregate stats
- `data/manifests/asset_registry.json` - persistent registry

### Inspecting Lineage

```python
from src.asset_registry import AssetRegistry

registry = AssetRegistry("data/manifests/asset_registry.json")

# Find a document
doc = registry.get_by_file_hash("7fa10cc3ee67d22d1ba315bb9716eb13...")

print(f"Document: {doc.origin_filename}")
print(f"Status: {doc.status}")
print(f"Chunks: {doc.chunks_indexed}/{doc.total_chunks}")
print(f"\nLineage History:")
for entry in doc.lineage_history:
    print(f"  {entry.timestamp} → {entry.stage} ({entry.status})")
    if entry.details:
        print(f"     Details: {entry.details}")
```

### Compliance Queries

```python
# "Show me all confidential documents and their current state"
registry = AssetRegistry()
confidential = [
    d for d in registry.assets.values() 
    if d.sensitivity == "Confidential"
]

for doc in confidential:
    print(f"{doc.origin_filename}")
    print(f"  Indexed: {doc.chunks_indexed}/{doc.total_chunks} chunks")
    print(f"  Department: {doc.department}")
    print(f"  Ingested: {doc.ingested_at}")
    print()

# "Which Finance documents haven't been re-indexed since classification?"
registry = AssetRegistry()
finance_docs = registry.list_by_department("Finance")

for doc in finance_docs:
    # Check if chunked_at is more recent than classified_at
    if doc.chunked_at and doc.classified_at:
        if doc.chunked_at > doc.classified_at:
            print(f"{doc.origin_filename} - re-indexed after classification ✓")
        else:
            print(f"{doc.origin_filename} - needs re-indexing ⚠️")
```

---

## Chunk File Format (JSONL)

Each line is a complete chunk record:

```json
{
  "chunk_id": "chunk_00001",
  "document_id": "doc_7fa10cc3ee67",
  "chunk_number": 1,
  "total_chunks_in_document": 42,
  "source_file": "doc_001_hr_policy_english.md",
  "department": "HR",
  "doc_type": "Policy",
  "sensitivity": "Internal",
  "content": "# HR Policy Document...",
  "section": "Introduction",
  "chunk_size": 487,
  "file_hash": "7fa10cc3ee67d22d1ba315bb9716eb13a94b0650c5e711020eb31d66cd659410",
  "ingested_at": "2026-08-06T11:07:04.224759+00:00",
  "classified_at": "2026-08-06T11:07:06.912604+00:00",
  "chunked_at": "2026-08-17T10:10:00+00:00",
  "char_offset_start": null,
  "char_offset_end": null,
  "created_at": "2026-08-17T10:10:00+00:00"
}
```

---

## Asset Registry File Format (JSON)

Complete snapshot of all documents and their states:

```json
{
  "doc_7fa10cc3ee67": {
    "document_id": "doc_7fa10cc3ee67",
    "file_hash": "7fa10cc3ee67d22d1ba315bb9716eb13a94b0650c5e711020eb31d66cd659410",
    "origin_filename": "doc_001_hr_policy_english.txt",
    "source_path": "data/samples/multilingual/test_multilingual_samples/doc_001_hr_policy_english.txt",
    "department": "HR",
    "doc_type": "Policy",
    "language": "EN",
    "sensitivity": "Internal",
    "ingested_at": "2026-08-06T11:07:04.224759+00:00",
    "classified_at": "2026-08-06T11:07:06.912604+00:00",
    "chunked_at": "2026-08-17T10:10:00+00:00",
    "total_chunks": 42,
    "chunks_indexed": 0,
    "status": "chunked",
    "lineage_history": [
      {
        "stage": "classification",
        "timestamp": "2026-08-06T11:07:06.912604+00:00",
        "processor": "classify.py",
        "status": "success",
        "details": {}
      },
      {
        "stage": "chunking",
        "timestamp": "2026-08-17T10:10:00+00:00",
        "processor": "chunking.py",
        "status": "success",
        "details": {"chunks_created": 42}
      }
    ]
  }
}
```

---

## Test Coverage

Comprehensive tests cover:
- Document registration with stable IDs
- Classification/chunking metadata updates
- Lineage history accumulation
- Queries (by status, department, file hash)
- Chunk indexing tracking
- Registry persistence (save/load)
- Aggregate statistics

Run tests:
```bash
python -m pytest tests/test_asset_registry.py -v
```

---

## Use Cases

### 1. **Compliance Audit**
"Prove this chunk came from an approved, unmodified source document"

→ Use `document_id`, `file_hash`, `ingested_at` in the chunk; verify against registry

### 2. **Incremental Re-Indexing**
"Which chunks from a document need to be re-embedded after an update?"

→ Track `document_id` per chunk; if source document changes, mark all chunks with that `document_id` for re-embedding

### 3. **Data Lineage Reconstruction**
"Show me the full journey of this piece of data"

→ Access chunk's lineage fields (`file_hash`, `ingested_at`, `classified_at`, `chunked_at`) and registry's `lineage_history`

### 4. **Department Isolation Verification**
"Confirm no data leaked across department boundaries"

→ Query registry by department; verify all chunks from each document have matching `department` field

### 5. **Performance Analysis**
"How long does each stage take for different document types?"

→ Calculate differences between consecutive timestamps in `lineage_history` per doc type

---

## Future Enhancements

1. **Character-level offsets** (`char_offset_start`, `char_offset_end`)
   - Already structured for this; populate during chunking
   - Enables precise location of content in original

2. **Change-based re-indexing**
   - Track which chunks changed when document is updated
   - Only re-embed affected chunks, not entire document

3. **Lineage export for visualization**
   - Generate timeline of document through pipeline
   - Support compliance dashboards

4. **Rollback capability**
   - Track which indexing run ingested which chunks
   - Support reverting to previous state

---

## Integration Checklist

- [x] AssetRegistry module created and tested
- [x] Chunk dataclass extended with lineage fields
- [x] Chunking pipeline updated (two-pass approach)
- [x] Registry persistence implemented
- [x] Test suite complete
- [ ] Integration with indexing stage (mark chunks as indexed)
- [ ] Dashboard/query interface for compliance teams
- [ ] Cleanup of old indexed chunks on re-indexing
