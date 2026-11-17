# RAGFlow MCP — RAG + Vector Search + MCP

A local retrieval-augmented generation knowledge base exposed through Model Context Protocol
tools. Ingest Markdown/text, create overlapping chunks, embed them, search semantically, and
optionally generate answers grounded in retrieved evidence.

## Features
- MCP tools: ingest, add raw text, list, semantic search, delete, and question answering.
- Sentence-transformer embeddings (`all-MiniLM-L6-v2`), lazy-loaded on first use.
- SQLite persistence for documents, chunks, and float32 vectors.
- NumPy cosine-similarity ranking.
- Optional OpenAI answer generation; retrieval works without an API key.
- Tests use a fake embedder, so no model download is needed for testing.

## Architecture
```text
MCP Client -> FastMCP Server
                 |-> Ingest -> chunk -> embed -> SQLite
                 |-> Search -> query embedding -> cosine ranking
                 `-> Q&A -> retrieval -> evidence -> optional LLM
```

## Setup
Python 3.10+:
```bash
git clone https://github.com/YOUR_USERNAME/ragflow-mcp.git
cd ragflow-mcp
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -e .
ragflow-mcp
```
The transformer model downloads the first time a real embedding is requested.
Database defaults to `data/ragflow.sqlite3`; override with `RAGFLOW_DB_PATH`.

## MCP client configuration
Adapt this example to your MCP client:
```json
{
  "mcpServers": {
    "ragflow": {
      "command": "ragflow-mcp",
      "args": []
    }
  }
}
```

## Tools
| Tool | Description |
|---|---|
| `ingest_document(path)` | Index a local `.txt` or `.md` file |
| `add_text_document(title, text)` | Index supplied text |
| `list_documents()` | List indexed documents |
| `search_knowledge_base(query, top_k=5)` | Return ranked chunks |
| `delete_document(document_id)` | Remove a document and chunks |
| `answer_question(question, top_k=5)` | Retrieve evidence and optionally generate an answer |

Resource: `ragflow://about`.

## Optional LLM generation
Install the optional SDK and set your key:
```bash
pip install openai
# macOS/Linux
export OPENAI_API_KEY="your-key"
# PowerShell
$env:OPENAI_API_KEY="your-key"
```
`OPENAI_MODEL` is optional (default `gpt-4o-mini`). Without the key, the Q&A tool returns
retrieved evidence instead of generating a prose answer. When enabled, the question and
retrieved excerpts are sent to the configured provider.

## Test
```bash
pip install -e ".[dev]"
pytest -q
```

## Demo workflow
1. Start the server.
2. Ingest `examples/knowledge_base/ai_engineering_notes.md`.
3. Search for `How does vector search support RAG?`.
4. Ask the same question using `answer_question`.
5. Inspect the document list and delete the demo document if desired.

## Limitations and security
- Only UTF-8 `.txt` and `.md` are supported; PDF/DOCX parsing is not included.
- Retrieval scans vectors from SQLite in-process; suitable for a small demo, not huge corpora.
- Restrict file access and add authentication, authorization, and tenant isolation before
  exposing this server to untrusted clients.
- Retrieved documents are untrusted data. Prompt wording is not a complete prompt-injection
  defence.
- Do not send sensitive excerpts to an external model unless your data policy permits it.

## Extensions
Add PDF/DOCX parsing, FAISS/Qdrant/Chroma, metadata filters, hybrid BM25 search, reranking,
source page references, and a RAG evaluation harness.

## License
MIT
