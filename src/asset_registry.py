"""
Asset Registry — Central tracking of documents through the RAG pipeline.

Responsibilities:
- Maintain a single source of truth for document metadata
- Track document state at each pipeline stage (ingested, classified, chunked, indexed)
- Enable asset queries and status tracking
- Support lineage reconstruction for compliance/debugging

Structure:
- AssetRecord: immutable document snapshot at a point in time
- AssetRegistry: queryable registry with persistence
"""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, List
import hashlib


@dataclass
class LineageEntry:
    """A single step in a document's journey through the pipeline."""
    stage: str  # "ingestion", "classification", "chunking", "indexing"
    timestamp: str
    processor: str  # e.g., "docling", "ollama/qwen3.6", "nomic-embed-text"
    status: str  # "success", "failed", "skipped"
    details: Dict = field(default_factory=dict)


@dataclass
class AssetRecord:
    """
    Complete record for a source document.
    
    This snapshot captures:
    - Identity (document_id, file_hash, origin path)
    - Metadata (title, department, doc_type, sensitivity)
    - Lineage (where it's been, what happened to it)
    - Current state (which chunks exist, are they indexed?)
    """
    document_id: str  # stable UUID or hash-based identifier
    file_hash: str  # SHA-256 of content
    origin_filename: str
    source_path: str
    
    # Classification
    department: str
    doc_type: str
    language: str
    sensitivity: str
    
    # Lineage chain
    ingested_at: str  # ISO timestamp
    ingested_by: str = "ingestion"
    classified_at: Optional[str] = None
    classified_by: str = "classify.py"
    chunked_at: Optional[str] = None
    chunked_by: str = "chunking.py"
    
    # State tracking
    total_chunks: int = 0
    chunks_indexed: int = 0
    status: str = "ingested"  # ingested → classified → chunked → indexed
    
    # Additional lineage details
    lineage_history: List[LineageEntry] = field(default_factory=list)


class AssetRegistry:
    """
    Queryable registry of all documents in the system.
    Can be persisted to/from JSON for cross-stage communication.
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        self.assets: Dict[str, AssetRecord] = {}  # document_id -> AssetRecord
        self.registry_path = registry_path or Path("data/manifests/asset_registry.json")
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing registry
        if self.registry_path.exists():
            self.load()
    
    def register_document(self, 
                         origin_filename: str,
                         source_path: str,
                         file_hash: str,
                         ingested_at: str) -> str:
        """
        Register a new document from ingestion stage.
        Returns the stable document_id.
        """
        # Use file_hash as basis for document_id (stable across updates)
        document_id = f"doc_{file_hash[:16]}"
        
        record = AssetRecord(
            document_id=document_id,
            file_hash=file_hash,
            origin_filename=origin_filename,
            source_path=source_path,
            department="Unknown",
            doc_type="Unknown",
            language="Unknown",
            sensitivity="Unknown",
            ingested_at=ingested_at,
            status="ingested",
        )
        
        self.assets[document_id] = record
        return document_id
    
    def update_classification(self,
                            document_id: str,
                            department: str,
                            doc_type: str,
                            language: str,
                            sensitivity: str,
                            classified_at: str) -> None:
        """
        Update document with classification results.
        """
        if document_id not in self.assets:
            raise KeyError(f"Document {document_id} not found in registry")
        
        record = self.assets[document_id]
        record.department = department
        record.doc_type = doc_type
        record.language = language
        record.sensitivity = sensitivity
        record.classified_at = classified_at
        record.status = "classified"
        
        # Add to lineage
        record.lineage_history.append(LineageEntry(
            stage="classification",
            timestamp=classified_at,
            processor="classify.py",
            status="success"
        ))
    
    def update_chunking(self,
                       document_id: str,
                       total_chunks: int,
                       chunked_at: str) -> None:
        """
        Update document with chunking results.
        """
        if document_id not in self.assets:
            raise KeyError(f"Document {document_id} not found in registry")
        
        record = self.assets[document_id]
        record.total_chunks = total_chunks
        record.chunked_at = chunked_at
        record.status = "chunked"
        
        # Add to lineage
        record.lineage_history.append(LineageEntry(
            stage="chunking",
            timestamp=chunked_at,
            processor="chunking.py",
            status="success",
            details={"chunks_created": total_chunks}
        ))
    
    def mark_chunk_indexed(self, document_id: str) -> None:
        """Increment the indexed chunk count for a document."""
        if document_id in self.assets:
            self.assets[document_id].chunks_indexed += 1
            # Update status if all chunks are indexed
            if self.assets[document_id].chunks_indexed >= self.assets[document_id].total_chunks:
                self.assets[document_id].status = "indexed"
    
    def get_document(self, document_id: str) -> Optional[AssetRecord]:
        """Retrieve a document record."""
        return self.assets.get(document_id)
    
    def get_by_file_hash(self, file_hash: str) -> Optional[AssetRecord]:
        """Find a document by its file hash."""
        for record in self.assets.values():
            if record.file_hash == file_hash:
                return record
        return None
    
    def list_by_status(self, status: str) -> List[AssetRecord]:
        """Get all documents in a particular status."""
        return [r for r in self.assets.values() if r.status == status]
    
    def list_by_department(self, department: str) -> List[AssetRecord]:
        """Get all documents in a department."""
        return [r for r in self.assets.values() if r.department == department]
    
    def save(self) -> None:
        """Persist registry to disk."""
        data = {
            document_id: asdict(record)
            for document_id, record in self.assets.items()
        }
        self.registry_path.write_text(json.dumps(data, indent=2, default=str))
    
    def load(self) -> None:
        """Load registry from disk."""
        if not self.registry_path.exists():
            return
        
        data = json.loads(self.registry_path.read_text())
        self.assets = {}
        
        for document_id, record_data in data.items():
            # Reconstruct lineage history
            lineage = []
            if 'lineage_history' in record_data and record_data['lineage_history']:
                for entry in record_data['lineage_history']:
                    lineage.append(LineageEntry(**entry))
            
            record_data['lineage_history'] = lineage
            record = AssetRecord(**record_data)
            self.assets[document_id] = record
    
    def summary(self) -> Dict:
        """Get a summary of registry state."""
        return {
            "total_documents": len(self.assets),
            "by_status": {
                status: len(self.list_by_status(status))
                for status in ["ingested", "classified", "chunked", "indexed"]
            },
            "total_chunks": sum(r.total_chunks for r in self.assets.values()),
            "chunks_indexed": sum(r.chunks_indexed for r in self.assets.values()),
        }
