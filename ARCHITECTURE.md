# Architecture

Ingestion reads a UTF-8 text or Markdown file, normalizes whitespace, creates overlapping
chunks, embeds each chunk, and stores metadata plus float32 vectors in SQLite.

Search embeds the query and computes cosine similarity against stored vectors. The current
implementation scans the small local collection in-process. A dedicated vector database is
the logical next step as the corpus grows.

Question answering retrieves evidence first. If an OpenAI API key is configured, the server
sends the question and retrieved excerpts to the selected model with instructions to answer
only from evidence. Otherwise it returns the excerpts. Retrieved content must always be
treated as untrusted input.
