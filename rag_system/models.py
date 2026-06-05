from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


class DocumentChunk(BaseModel):
    """Represents a single chunk of text from a document."""

    text: str = Field(..., description="Chunk text content")
    document: str = Field(..., description="Source document filename")
    page: int = Field(..., ge=1, description="Page number in original PDF")
    chunk_index: int = Field(..., ge=0, description="Sequential chunk number")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "El presupuesto aprobado fue...",
                "document": "ResumenReunionGA_20231124.pdf",
                "page": 3,
                "chunk_index": 5,
            }
        }
    )


class ParsedDocument(BaseModel):
    """Represents a fully parsed PDF document with chunks."""

    filename: str = Field(..., description="PDF filename")
    total_pages: int = Field(..., ge=1, description="Total number of pages")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks created")
    chunks: List[DocumentChunk] = Field(default_factory=list)

    # Optional metadata
    file_size_bytes: Optional[int] = None
    processing_time_seconds: Optional[float] = None

    @property
    def chunks_per_page(self) -> float:
        """Average chunks per page."""
        return self.total_chunks / self.total_pages if self.total_pages > 0 else 0

    def get_chunks_by_page(self, page: int) -> List[DocumentChunk]:
        """Filter chunks by page number."""
        return [c for c in self.chunks if c.page == page]

    def get_chunk_texts(self) -> List[str]:
        """Extract all chunk texts for embedding."""
        return [c.text for c in self.chunks]

    model_config = ConfigDict(validate_assignment=True)


class SearchResult(BaseModel):
    """Result from vector similarity search."""

    chunk: DocumentChunk = Field(..., description="Retrieved document chunk")
    score: float = Field(..., ge=0.0, le=1.0, description="Bi-encoder similarity score")
    rerank_score: Optional[float] = Field(
        default=None, description="Cross-encoder rerank score (if re-ranking was used)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk": {
                    "text": "El presupuesto aprobado fue...",
                    "document": "ResumenReunionGA_20231124.pdf",
                    "page": 3,
                    "chunk_index": 5,
                },
                "score": 0.89,
                "rerank_score": 8.42,
            }
        }
    )


class Citation(BaseModel):
    """
    Represents a citation to a (document, page) with associated chunks.

    A citation is a reference to a specific page in a document.
    Multiple chunks from the same page are grouped together.
    """

    document: str = Field(..., description="Source document filename")
    page: int = Field(..., ge=1, description="Page number in document")
    chunks: List["SearchResult"] = Field(
        default_factory=list,
        description="All retrieved chunks from this (document, page)",
    )

    @property
    def num_chunks(self) -> int:
        """Number of chunks associated with this citation."""
        return len(self.chunks)

    @property
    def best_score(self) -> float:
        """Best retrieval score among all chunks."""
        if not self.chunks:
            return 0.0
        return max(chunk.score for chunk in self.chunks)

    @property
    def best_rerank_score(self) -> Optional[float]:
        """Best rerank score among all chunks (if reranking used)."""
        if not self.chunks:
            return None
        rerank_scores = [
            chunk.rerank_score
            for chunk in self.chunks
            if hasattr(chunk, "rerank_score") and chunk.rerank_score is not None
        ]
        return max(rerank_scores) if rerank_scores else None

    @property
    def all_text(self) -> str:
        """Concatenate all chunk texts from this citation."""
        return "\n\n".join(chunk.chunk.text for chunk in self.chunks)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document": "ResumenReunionGA_20231124.pdf",
                "page": 3,
                "chunks": [
                    {
                        "chunk": {
                            "text": "El presupuesto aprobado fue...",
                            "document": "ResumenReunionGA_20231124.pdf",
                            "page": 3,
                            "chunk_index": 5,
                        },
                        "score": 0.85,
                        "rerank_score": 0.92,
                    }
                ],
            }
        }
    )
