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


class Citation(BaseModel):
    """Represents a citation to a source document."""

    document: str = Field(..., description="Source document filename")
    page: int = Field(..., ge=1, description="Page number in document")
    text_snippet: str = Field(
        ..., max_length=200, description="Relevant excerpt from source"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document": "ResumenReunionGA_20231124.pdf",
                "page": 3,
                "text_snippet": "El presupuesto aprobado fue de 5 millones de euros...",
            }
        }
    )
