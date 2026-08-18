"""
Tests for the AssetRegistry module.

Pure JSON logic, fully offline. Tests:
- Document registration from ingestion
- Classification metadata updates
- Chunking state tracking
- Lineage history accumulation
- Registry persistence (save/load)
- Queries by status, department, hash
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asset_registry import AssetRegistry, AssetRecord, LineageEntry


@pytest.fixture
def registry(tmp_path):
    """Fresh, isolated AssetRegistry for every test — points at a pytest
    tmp_path, never at the real production registry
    (data/manifests/asset_registry.json). Without this, `AssetRegistry()`
    with no explicit path defaults to that real file — every test in this
    file was silently loading and mutating live pipeline data in memory
    (test_summary / test_list_by_department failed with counts like 52 vs
    7 specifically because they were adding fake test documents on top of
    the real 45 already in production). This is the same class of bug as
    the earlier test_retrieval.py issue that wrote fixture data into the
    real Postgres database — a test must never be able to touch real
    storage just because it forgot to pass an explicit path."""
    return AssetRegistry(registry_path=tmp_path / "test_registry.json")



class TestAssetRegistry:
    """Test the AssetRegistry core functionality."""
    
    def test_register_document_creates_stable_id(self, registry):
        """Document registration should create stable document_id based on file_hash."""
        
        file_hash = "abc123def456"
        doc_id = registry.register_document(
            origin_filename="test.txt",
            source_path="/data/test.txt",
            file_hash=file_hash,
            ingested_at="2026-08-17T10:00:00Z"
        )
        
        assert doc_id.startswith("doc_")
        assert doc_id == f"doc_{file_hash[:16]}"
        
        # Registering again with same hash should return same ID
        doc_id2 = registry.register_document(
            origin_filename="test.txt",
            source_path="/data/test.txt",
            file_hash=file_hash,
            ingested_at="2026-08-17T10:00:00Z"
        )
        
        # Both should reference the same file_hash
        assert registry.get_by_file_hash(file_hash) is not None
    
    def test_update_classification(self, registry):
        """Classification update should set department, doc_type, etc."""
        
        doc_id = registry.register_document(
            origin_filename="hr_policy.pdf",
            source_path="/data/hr_policy.pdf",
            file_hash="hash123",
            ingested_at="2026-08-17T10:00:00Z"
        )
        
        registry.update_classification(
            document_id=doc_id,
            department="HR",
            doc_type="Policy",
            language="EN",
            sensitivity="Internal",
            classified_at="2026-08-17T10:05:00Z"
        )
        
        record = registry.get_document(doc_id)
        assert record.department == "HR"
        assert record.doc_type == "Policy"
        assert record.language == "EN"
        assert record.sensitivity == "Internal"
        assert record.status == "classified"
        assert len(record.lineage_history) == 1
        assert record.lineage_history[0].stage == "classification"
    
    def test_update_chunking(self, registry):
        """Chunking update should set chunk counts and add to lineage."""
        
        doc_id = registry.register_document(
            origin_filename="doc.txt",
            source_path="/data/doc.txt",
            file_hash="hash456",
            ingested_at="2026-08-17T10:00:00Z"
        )
        
        registry.update_classification(
            document_id=doc_id,
            department="IT",
            doc_type="Report",
            language="EN",
            sensitivity="Public",
            classified_at="2026-08-17T10:05:00Z"
        )
        
        registry.update_chunking(
            document_id=doc_id,
            total_chunks=42,
            chunked_at="2026-08-17T10:10:00Z"
        )
        
        record = registry.get_document(doc_id)
        assert record.total_chunks == 42
        assert record.status == "chunked"
        assert len(record.lineage_history) == 2
        
        # Check chunking entry
        chunk_entry = record.lineage_history[1]
        assert chunk_entry.stage == "chunking"
        assert chunk_entry.details["chunks_created"] == 42
    
    def test_lineage_history_accumulates(self, registry):
        """Lineage should accumulate events as document flows through pipeline."""
        
        doc_id = registry.register_document(
            origin_filename="test.pdf",
            source_path="/data/test.pdf",
            file_hash="hash789",
            ingested_at="2026-08-17T10:00:00Z"
        )
        
        # No lineage yet (just ingested)
        assert len(registry.get_document(doc_id).lineage_history) == 0
        
        registry.update_classification(
            document_id=doc_id,
            department="Finance",
            doc_type="Invoice",
            language="FR",
            sensitivity="Confidential",
            classified_at="2026-08-17T10:05:00Z"
        )
        
        assert len(registry.get_document(doc_id).lineage_history) == 1
        
        registry.update_chunking(
            document_id=doc_id,
            total_chunks=10,
            chunked_at="2026-08-17T10:10:00Z"
        )
        
        record = registry.get_document(doc_id)
        assert len(record.lineage_history) == 2
        
        # Timeline should be chronological
        assert record.lineage_history[0].timestamp < record.lineage_history[1].timestamp
    
    def test_get_by_file_hash(self, registry):
        """Should be able to retrieve document by file hash."""
        
        file_hash = "unique_hash"
        doc_id = registry.register_document(
            origin_filename="doc.txt",
            source_path="/data/doc.txt",
            file_hash=file_hash,
            ingested_at="2026-08-17T10:00:00Z"
        )
        
        retrieved = registry.get_by_file_hash(file_hash)
        assert retrieved is not None
        assert retrieved.document_id == doc_id
        
        # Should return None for non-existent hash
        assert registry.get_by_file_hash("nonexistent") is None
    
    def test_list_by_status(self, registry):
        """Should be able to list documents by status."""
        
        # Create 3 documents at different stages
        doc1 = registry.register_document(
            origin_filename="doc1.txt",
            source_path="/data/doc1.txt",
            file_hash="hash1",
            ingested_at="2026-08-17T10:00:00Z"
        )
        
        doc2 = registry.register_document(
            origin_filename="doc2.txt",
            source_path="/data/doc2.txt",
            file_hash="hash2",
            ingested_at="2026-08-17T10:00:00Z"
        )
        
        doc3 = registry.register_document(
            origin_filename="doc3.txt",
            source_path="/data/doc3.txt",
            file_hash="hash3",
            ingested_at="2026-08-17T10:00:00Z"
        )
        
        # Classify two
        registry.update_classification(
            document_id=doc1,
            department="HR", doc_type="Policy", language="EN", sensitivity="Internal",
            classified_at="2026-08-17T10:05:00Z"
        )
        
        registry.update_classification(
            document_id=doc2,
            department="IT", doc_type="Report", language="EN", sensitivity="Public",
            classified_at="2026-08-17T10:05:00Z"
        )
        
        # Check status counts
        ingested = registry.list_by_status("ingested")
        classified = registry.list_by_status("classified")
        
        assert len(ingested) == 1
        assert len(classified) == 2
        assert any(d.document_id == doc3 for d in ingested)
    
    def test_list_by_department(self, registry):
        """Should be able to list documents by department."""
        
        for i, dept in enumerate(["HR", "Finance", "IT", "HR"]):
            doc_id = registry.register_document(
                origin_filename=f"doc{i}.txt",
                source_path=f"/data/doc{i}.txt",
                file_hash=f"hash{i}",
                ingested_at="2026-08-17T10:00:00Z"
            )
            
            registry.update_classification(
                document_id=doc_id,
                department=dept, doc_type="Report", language="EN", sensitivity="Public",
                classified_at="2026-08-17T10:05:00Z"
            )
        
        hr_docs = registry.list_by_department("HR")
        finance_docs = registry.list_by_department("Finance")
        
        assert len(hr_docs) == 2
        assert len(finance_docs) == 1
    
    def test_mark_chunk_indexed(self, registry):
        """Marking chunks as indexed should update status."""
        
        doc_id = registry.register_document(
            origin_filename="doc.txt",
            source_path="/data/doc.txt",
            file_hash="hash",
            ingested_at="2026-08-17T10:00:00Z"
        )
        
        registry.update_chunking(
            document_id=doc_id,
            total_chunks=3,
            chunked_at="2026-08-17T10:10:00Z"
        )
        
        # Index chunks one by one
        registry.mark_chunk_indexed(doc_id)
        assert registry.get_document(doc_id).chunks_indexed == 1
        assert registry.get_document(doc_id).status == "chunked"
        
        registry.mark_chunk_indexed(doc_id)
        registry.mark_chunk_indexed(doc_id)
        
        # All chunks indexed, status should change
        assert registry.get_document(doc_id).chunks_indexed == 3
        assert registry.get_document(doc_id).status == "indexed"
    
    def test_save_and_load(self):
        """Registry should persist to disk and reload correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.json"
            
            # Create and save registry
            registry = AssetRegistry(registry_path=registry_path)
            
            doc_id = registry.register_document(
                origin_filename="doc.txt",
                source_path="/data/doc.txt",
                file_hash="persistent_hash",
                ingested_at="2026-08-17T10:00:00Z"
            )
            
            registry.update_classification(
                document_id=doc_id,
                department="Legal",
                doc_type="Contract",
                language="EN",
                sensitivity="Confidential",
                classified_at="2026-08-17T10:05:00Z"
            )
            
            registry.save()
            assert registry_path.exists()
            
            # Load in new registry instance
            registry2 = AssetRegistry(registry_path=registry_path)
            
            # Should have loaded all documents
            assert len(registry2.assets) == 1
            
            loaded_record = registry2.get_document(doc_id)
            assert loaded_record.department == "Legal"
            assert loaded_record.doc_type == "Contract"
            assert loaded_record.sensitivity == "Confidential"
            assert len(loaded_record.lineage_history) == 1
    
    def test_summary(self, registry):
        """Summary should provide correct aggregate statistics."""
        
        # Create documents at various stages
        # Use hashes with unique first 16 characters
        for i in range(2):
            doc_id = registry.register_document(
                origin_filename=f"doc{i}.txt",
                source_path=f"/data/doc{i}.txt",
                file_hash=f"0000000{i:x}abcdefghijklmnop",
                ingested_at="2026-08-17T10:00:00Z"
            )
        
        for i in range(3):
            doc_id = registry.register_document(
                origin_filename=f"doc{i+2}.txt",
                source_path=f"/data/doc{i+2}.txt",
                file_hash=f"1111111{i:x}abcdefghijklmnop",
                ingested_at="2026-08-17T10:00:00Z"
            )
            registry.update_classification(
                document_id=doc_id,
                department="HR", doc_type="Policy", language="EN", sensitivity="Internal",
                classified_at="2026-08-17T10:05:00Z"
            )
        
        for i in range(2):
            doc_id = registry.register_document(
                origin_filename=f"doc{i+5}.txt",
                source_path=f"/data/doc{i+5}.txt",
                file_hash=f"2222222{i:x}abcdefghijklmnop",
                ingested_at="2026-08-17T10:00:00Z"
            )
            registry.update_classification(
                document_id=doc_id,
                department="IT", doc_type="Report", language="EN", sensitivity="Public",
                classified_at="2026-08-17T10:05:00Z"
            )
            registry.update_chunking(
                document_id=doc_id,
                total_chunks=15,
                chunked_at="2026-08-17T10:10:00Z"
            )
        
        summary = registry.summary()
        
        assert summary["total_documents"] == 7
        assert summary["by_status"]["ingested"] == 2
        assert summary["by_status"]["classified"] == 3
        assert summary["by_status"]["chunked"] == 2
        assert summary["total_chunks"] == 30  # 2 docs * 15 chunks each


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
