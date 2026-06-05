import os
import pickle
from pathlib import Path
from typing import List, Optional
from abc import ABC, abstractmethod
from pathlib import Path
import faiss
import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from rag_system.models import DocumentChunk, SearchResult


class VectorStoreConfig(BaseModel):
    """Configuration for FAISS vector store."""

    index_path: Path = Field(
        default=Path("faiss_index/vector_index.faiss"),
        description="Path to save/load FAISS index",
    )
    metadata_path: Path = Field(
        default=Path("faiss_index/metadata.pkl"),
        description="Path to save/load metadata",
    )
    similarity_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for retrieval",
    )
    normalize_vectors: bool = Field(
        default=True, description="L2 normalize vectors before adding to index"
    )

    model_config = ConfigDict(frozen=True)


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def create_index(self, dimension: int) -> None:
        """Create/initialize the vector index."""
        pass

    @abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Add chunks with their embeddings to the store."""
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[SearchResult]:
        """Search for similar vectors and return results with metadata."""
        pass

    @abstractmethod
    def save(self) -> None:
        """Persist index and metadata to disk."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load index and metadata from disk."""
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """Get index statistics."""
        pass


class FAISSVectorStore(VectorStore):
    """
    FAISS-based vector store with metadata.

    Stores embeddings in FAISS IndexFlatIP for cosine similarity search,
    and metadata separately in pickle format.
    """

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        Initialize FAISS vector store.

        Args:
            config: Vector store configuration (uses defaults if None)
        """
        self.config = config or VectorStoreConfig()
        self.index: Optional[faiss.Index] = None
        self.chunks: List[DocumentChunk] = []

        # Ensure directories exist
        self.config.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing index if available
        if self.config.index_path.exists():
            self.load()

    def create_index(self, dimension: int) -> None:
        """
        Create FAISS index for cosine similarity search.

        Args:
            dimension: Embedding dimension (e.g., 384)
        """
        # IndexFlatIP uses inner product (with normalized vectors = cosine)
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []
        print(f"Created FAISS index with dimension {dimension}")

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """
        Add chunks with their embeddings to the store.

        Args:
            chunks: List of DocumentChunk objects
            embeddings: NumPy array of embeddings, shape (len(chunks), dimension)
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        # Convert to float32 and ensure 2D
        vectors = np.array(embeddings).astype("float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # Normalize vectors if configured
        if self.config.normalize_vectors:
            faiss.normalize_L2(vectors)

        # Add to FAISS index
        self.index.add(vectors)

        # Store chunks (metadata)
        self.chunks.extend(chunks)

        print(f"Added {len(chunks)} chunks. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """
        Search for similar vectors and return results with metadata.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return

        Returns:
            List of SearchResult objects with chunks and scores
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # Prepare query vector
        query = np.array([query_embedding]).astype("float32")

        # Normalize query if configured
        if self.config.normalize_vectors:
            faiss.normalize_L2(query)

        # Search
        distances, indices = self.index.search(query, top_k)

        # Build results with metadata
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx != -1 and score >= self.config.similarity_threshold:
                results.append(SearchResult(chunk=self.chunks[idx], score=float(score)))

        return results

    def save(self) -> None:
        """Persist index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save. Create and populate index first.")

        # Save FAISS index
        faiss.write_index(self.index, str(self.config.index_path))

        # Save chunks metadata
        with open(self.config.metadata_path, "wb") as f:
            pickle.dump(self.chunks, f)

        print(
            f"Saved index with {self.index.ntotal} vectors to {self.config.index_path}"
        )

    def load(self) -> None:
        """Load index and metadata from disk."""
        if not self.config.index_path.exists():
            raise FileNotFoundError(f"Index not found at {self.config.index_path}")

        # Load FAISS index
        self.index = faiss.read_index(str(self.config.index_path))

        # Load chunks metadata
        with open(self.config.metadata_path, "rb") as f:
            self.chunks = pickle.load(f)

        print(
            f"Loaded index with {self.index.ntotal} vectors from {self.config.index_path}"
        )

    def get_stats(self) -> dict:
        """
        Get index statistics.

        Returns:
            Dictionary with index info
        """
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.index.d if self.index else 0,
            "index_type": "FAISS-IndexFlatIP",
            "total_chunks": len(self.chunks),
            "similarity_threshold": self.config.similarity_threshold,
            "index_path": str(self.config.index_path),
            "metadata_path": str(self.config.metadata_path),
        }
