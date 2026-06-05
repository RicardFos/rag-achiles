from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

from sentence_transformers import CrossEncoder
import numpy as np

from rag_system.models import DocumentChunk, SearchResult


class RerankerConfig(BaseModel):
    """Configuration for cross-encoder re-ranker."""

    model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        description="Cross-encoder model name",
    )
    top_n: int = Field(
        default=7,
        ge=1,
        le=20,
        description="Number of results to return after re-ranking",
    )
    batch_size: int = Field(
        default=32, ge=1, le=64, description="Batch size for scoring"
    )
    device: str = Field(default="cpu", description="Device to run on: 'cpu' or 'cuda'")

    model_config = ConfigDict(frozen=True)


class RerankResult(BaseModel):
    """Result after re-ranking with cross-encoder score."""

    chunk: DocumentChunk = Field(..., description="Document chunk")
    retrieval_score: float = Field(
        ..., description="Original bi-encoder similarity score"
    )
    rerank_score: float = Field(..., description="Cross-encoder relevance score")
    rank: int = Field(..., ge=1, description="Final rank after re-ranking")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk": {
                    "text": "El presupuesto aprobado fue...",
                    "document": "ResumenReunionGA_20231124.pdf",
                    "page": 3,
                    "chunk_index": 5,
                },
                "retrieval_score": 0.75,
                "rerank_score": 8.42,
                "rank": 1,
            }
        }
    )


class Reranker:
    """
    Cross-encoder re-ranker for improving retrieval relevance.

    Takes candidates from bi-encoder search and re-ranks them using
    a cross-encoder that processes query+document pairs jointly.
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        """
        Initialize re-ranker with configuration.

        Args:
            config: Re-ranker configuration (uses defaults if None)
        """
        self.config = config or RerankerConfig()
        self._load_model()

    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        print(f"Loading cross-encoder: {self.config.model_name}...")

        self.model = CrossEncoder(
            self.config.model_name, device=self.config.device, max_length=512
        )

        print(f"✓ Cross-encoder loaded")

    def rerank(
        self, query: str, candidates: List[SearchResult], top_n: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Re-rank candidates using cross-encoder.

        Args:
            query: User query string
            candidates: List of SearchResult from vector store
            top_n: Number of top results to return (uses config default if None)

        Returns:
            List of RerankResult sorted by rerank_score (descending)
        """
        if not candidates:
            return []

        top_n = top_n or self.config.top_n

        # Prepare query-document pairs
        pairs = [[query, candidate.chunk.text] for candidate in candidates]

        # Score with cross-encoder (batch processing)
        rerank_scores = self.model.predict(
            pairs, batch_size=self.config.batch_size, show_progress_bar=False
        )

        # Create tuples of (candidate, rerank_score) and sort by score
        scored_candidates = list(zip(candidates, rerank_scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Build results with ranks assigned correctly
        results = []
        for rank, (candidate, rerank_score) in enumerate(scored_candidates[:top_n], 1):
            results.append(
                RerankResult(
                    chunk=candidate.chunk,
                    retrieval_score=candidate.score,
                    rerank_score=float(rerank_score),
                    rank=rank,
                )
            )

        return results

    def compare_rankings(self, query: str, candidates: List[SearchResult]) -> dict:
        """
        Compare bi-encoder vs cross-encoder rankings.

        Args:
            query: User query
            candidates: Search results from vector store

        Returns:
            Dictionary with comparison metrics
        """
        if not candidates:
            return {"error": "No candidates provided"}

        # Get re-ranked results
        reranked = self.rerank(query, candidates, top_n=len(candidates))

        # Compare top-5 from each method
        top_5_biencoder = [c.chunk.chunk_index for c in candidates[:5]]
        top_5_crossencoder = [r.chunk.chunk_index for r in reranked[:5]]

        overlap = len(set(top_5_biencoder) & set(top_5_crossencoder))

        return {
            "query": query,
            "total_candidates": len(candidates),
            "top_5_overlap": overlap,
            "overlap_percentage": (overlap / 5) * 100,
            "biencoder_top_5": top_5_biencoder,
            "crossencoder_top_5": top_5_crossencoder,
            "rank_changes": self._compute_rank_changes(candidates, reranked),
        }

    def _compute_rank_changes(
        self, original: List[SearchResult], reranked: List[RerankResult]
    ) -> List[dict]:
        """
        Compute how ranks changed after re-ranking.

        Returns:
            List of dicts with chunk info and rank change
        """
        # Map chunk_index to original rank
        original_ranks = {c.chunk.chunk_index: i + 1 for i, c in enumerate(original)}

        changes = []
        for new_rank, result in enumerate(reranked, 1):
            chunk_idx = result.chunk.chunk_index
            old_rank = original_ranks.get(chunk_idx, -1)

            if old_rank > 0:
                change = old_rank - new_rank  # Positive = moved up
                if change != 0:
                    changes.append(
                        {
                            "chunk_index": chunk_idx,
                            "document": result.chunk.document,
                            "page": result.chunk.page,
                            "old_rank": old_rank,
                            "new_rank": new_rank,
                            "change": change,
                            "retrieval_score": result.retrieval_score,
                            "rerank_score": result.rerank_score,
                        }
                    )

        # Sort by magnitude of change
        changes.sort(key=lambda x: abs(x["change"]), reverse=True)
        return changes
