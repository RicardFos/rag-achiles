# RAG System - Technical Specifications Overview

## Purpose

This directory contains detailed technical specifications for each component of the RAG (Retrieval-Augmented Generation) system, developed using a spec-driven approach.

## Assignment Requirements Coverage

| Requirement | Implementation | Specification |
|-------------|----------------|---------------|
| **PDF Ingestion** | PyPDF2 text extraction with page tracking | [ingestion_spec.md](ingestion_spec.md) |
| **Chunking** | Recursive by tokens (800), 200 overlap | [ingestion_spec.md](ingestion_spec.md) |
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) | [embedding_spec.md](embedding_spec.md) |
| **Vector DB** | FAISS IndexFlatIP with L2 normalization | [vectordb_spec.md](vectordb_spec.md) |
| **Retrieval** | Top-k semantic search (cosine similarity) | [vectordb_spec.md](vectordb_spec.md) |
| **Re-ranking** | Cross-encoder two-stage retrieval | [reranker_spec.md](reranker_spec.md) |
| **LLM Generation** | Gemini with citation extraction | [llm_spec.md](llm_spec.md) |
| **Evaluation** | eval.jsonl with citation metrics | [evaluation_spec.md](evaluation_spec.md) |

## System Architecture

### Component Flow

```
┌──────────────────┐
│  DocumentParser  │  Parse PDFs → chunks with page metadata
└────────┬─────────┘
         │ produces: List[DocumentChunk]
         ↓
┌──────────────────┐
│    Embedder      │  Generate 384-dim vectors
└────────┬─────────┘
         │ produces: np.ndarray (N, 384)
         ↓
┌──────────────────┐
│ FAISSVectorStore │  Index & search with cosine similarity
└────────┬─────────┘
         │ produces: List[SearchResult] (top-20)
         ↓
┌──────────────────┐
│    Reranker      │  Cross-encoder re-ranking (optional)
└────────┬─────────┘
         │ produces: List[RerankResult] (top-5)
         ↓
┌──────────────────┐
│  RAGGenerator    │  LLM generation + citation extraction
└────────┬─────────┘
         │ produces: LLMResponse
         ↓
    Answer + Citations
```

### Data Models (Pydantic)

All components use validated Pydantic models:

- **DocumentChunk** - Text chunk with document name, page number, and index
- **SearchResult** - Retrieved chunk with similarity score
- **RerankResult** - Re-ranked chunk with both bi-encoder and cross-encoder scores
- **Citation** - Source reference with document, page, and text snippet
- **LLMResponse** - Complete response with answer, citations, and metadata

## Design Principles

1. **Type Safety** - Pydantic models throughout for validation and serialization
2. **Immutability** - Frozen configuration models prevent runtime modification
3. **Modularity** - Each component has clear input/output contracts
4. **Testability** - Clean interfaces enable unit and integration testing
5. **Extensibility** - Abstract base classes (e.g., `VectorStore`) allow swapping implementations

## Pipeline Execution

### Indexing (Offline)
```bash
python index_documents.py pdfs/
```
1. Parse PDFs → DocumentChunk objects
2. Generate embeddings → numpy arrays
3. Build FAISS index → persist to disk

### Querying (Runtime)
```bash
python query_rag.py "¿Qué presupuesto se aprobó?"
```
1. Embed query → 384-dim vector
2. Search vector store → top-20 candidates
3. Re-rank with cross-encoder → top-5 results
4. Format context + call LLM → generate answer
5. Extract citations → return LLMResponse

## Specification Index

1. **[Ingestion & Chunking](ingestion_spec.md)** - PDF parsing, recursive text splitting, page tracking
2. **[Embeddings](embedding_spec.md)** - Multilingual sentence transformers, batch processing
3. **[Vector Database](vectordb_spec.md)** - FAISS setup, cosine similarity search, persistence
4. **[Re-ranker](reranker_spec.md)** - Cross-encoder scoring, two-stage retrieval strategy
5. **[LLM Generation](llm_spec.md)** - Gemini integration, prompt engineering, citation extraction
6. **[Evaluation](evaluation_spec.md)** - Test set format, metrics (citation accuracy, precision, recall)

## Key Technical Decisions

- **Local-first approach** - Embeddings and vector DB run locally (no external services except LLM)
- **Two-stage retrieval** - Bi-encoder (fast, 619 docs) → Cross-encoder (accurate, 20 candidates)
- **Spanish language support** - Multilingual models for embeddings and LLM
- **Citation enforcement** - Prompt engineering ensures grounded responses with source attribution
- **Free tier friendly** - Gemini free API, no costs for demo/evaluation

## Evaluation Strategy

Test set (`eval.jsonl`): 10 question-answer pairs with expected source passages

**Metrics:**
- Citation Accuracy: % responses with valid citations (target: ≥90%)
- Source Precision: % cited documents that are correct (target: ≥70%)
- Source Recall: % expected sources that were cited (target: ≥60%)

See [evaluation_spec.md](evaluation_spec.md) for detailed methodology.
