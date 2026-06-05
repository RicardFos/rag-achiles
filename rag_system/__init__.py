"""
RAG System for Document Q&A
Achiles Interview Assignment

A complete RAG pipeline for ingesting PDFs, generating embeddings,
and answering questions with citations.
"""

__version__ = "1.0.0"

# Data models
from rag_system.models import DocumentChunk, ParsedDocument, Citation

# Core components
from rag_system.parser import DocumentParser, ParserConfig
from rag_system.embeddings import Embedder, EmbeddingConfig
from rag_system.vector_store import FAISSVectorStore, VectorStoreConfig, SearchResult
from rag_system.reranker import Reranker, RerankerConfig, RerankResult
from rag_system.llm import RAGGenerator, LLMConfig, LLMResponse

# Evaluation
from rag_system.evaluation import (
    load_eval_data,
    compute_citation_accuracy,
    compute_source_metrics,
    compute_answer_similarity,
    evaluate_rag_system,
    print_evaluation_results,
    analyze_result
)

__all__ = [
    # Models
    "DocumentChunk",
    "ParsedDocument",
    "Citation",
    "LLMResponse",
    "SearchResult",
    "RerankResult",
    # Parser
    "DocumentParser",
    "ParserConfig",
    # Embeddings
    "Embedder",
    "EmbeddingConfig",
    # Vector Store
    "FAISSVectorStore",
    "VectorStoreConfig",
    # Reranker
    "Reranker",
    "RerankerConfig",
    # LLM
    "RAGGenerator",
    "LLMConfig",
    # Evaluation
    "load_eval_data",
    "compute_citation_accuracy",
    "compute_source_metrics",
    "compute_answer_similarity",
    "evaluate_rag_system",
    "print_evaluation_results",
    "analyze_result",
]
