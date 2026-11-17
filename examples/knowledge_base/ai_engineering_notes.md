# AI Engineering Notes

## Retrieval-Augmented Generation
RAG combines information retrieval with language generation. A typical pipeline ingests
documents, splits them into chunks, embeds each chunk, retrieves relevant chunks for a query,
and supplies those chunks to a language model as context. RAG can reduce unsupported answers,
but does not guarantee correctness: retrieval can miss passages and models can misinterpret context.

## Vector Search
An embedding is a numeric representation of text. Similarity search compares a query embedding
with stored document embeddings. Cosine similarity is common. Choose and evaluate embedding
models for the language, domain, and content type being searched.

## Model Context Protocol
The Model Context Protocol (MCP) is a standard for exposing tools and resources to AI clients.
An MCP server can expose ingestion, search, and question-answering without coupling clients
to the underlying storage implementation.

## Evaluation
Track retrieval recall, ranking quality, answer faithfulness, latency, and cost. Include
questions whose answers are deliberately absent from the knowledge base.
