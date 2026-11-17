import numpy as np
import pytest
from ragflow_mcp.engine import KnowledgeBase, chunk_text

class FakeEmbedder:
    def encode(self, texts):
        return np.array([[float("fruit" in t.lower()), float("sql" in t.lower()),
                          float("machine" in t.lower())] for t in texts], dtype=np.float32)

def test_chunking_and_validation():
    assert len(chunk_text("one two three four five six", 13, 4)) >= 2
    with pytest.raises(ValueError): chunk_text("hello", 5, 5)

def test_add_search_duplicate_delete(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.db", FakeEmbedder(), 80, 10)
    text = "Apple is a fruit. Fruit is healthy."
    first = kb.add_text("Fruit guide", text)
    assert first["chunks"] >= 1 and not first["duplicate"]
    assert kb.add_text("Other title", text)["duplicate"]
    assert kb.search("apple fruit")[0].title == "Fruit guide"
    assert kb.delete_document(first["document_id"])
    assert kb.list_documents() == []

def test_file_validation(tmp_path):
    path = tmp_path / "bad.pdf"; path.write_text("x")
    kb = KnowledgeBase(tmp_path / "kb.db", FakeEmbedder())
    with pytest.raises(ValueError): kb.ingest_file(path)
