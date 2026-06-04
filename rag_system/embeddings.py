import numpy as np
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""

    model_name: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2",
        description="HuggingFace model name",
    )
    device: str = Field(default="cpu", description="Device to run on: 'cpu' or 'cuda'")
    batch_size: int = Field(
        default=64, ge=1, le=256, description="Number of texts to embed per batch"
    )
    normalize_embeddings: bool = Field(
        default=True, description="L2 normalize embeddings for cosine similarity"
    )
    show_progress: bool = Field(
        default=False, description="Show progress bar during batch embedding"
    )

    model_config = ConfigDict(frozen=True)  # Immutable config


class Embedder:
    """
    Generates embeddings using sentence-transformers via LangChain.

    Encapsulates model lifecycle, batch processing, and provides
    consistent interface for document and query embedding.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize embedder with configuration.

        Args:
            config: Embedding configuration (uses defaults if None)
        """
        self.config = config or EmbeddingConfig()
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the embedding model with configured parameters.
        Sets self.model.
        """
        print(f"Loading embedding model: {self.config.model_name}...")

        self.model = HuggingFaceEmbeddings(
            model_name=self.config.model_name,
            model_kwargs={"device": self.config.device},
            encode_kwargs={"normalize_embeddings": self.config.normalize_embeddings},
        )

        # Now self.model is set, so self.dimension works
        print(f"✓ Model loaded (dimension: {self.dimension})")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple document texts in batches.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of embeddings, shape (len(texts), dimension)
        """
        all_embeddings = []

        iterator = range(0, len(texts), self.config.batch_size)
        if self.config.show_progress:
            iterator = tqdm(iterator, desc="Embedding documents", unit="batch")

        for i in iterator:
            batch = texts[i : i + self.config.batch_size]
            batch_embeddings = self.model.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query text.

        Note: Some models differentiate between document and query embedding.
        Always use this method for queries, not embed_documents.

        Args:
            text: Query string to embed

        Returns:
            NumPy array embedding, shape (dimension,)
        """
        embedding = self.model.embed_query(text)
        return np.array(embedding, dtype=np.float32)

    @property
    def dimension(self) -> int:
        """Get the dimensionality of embeddings."""
        dummy_embedding = self.model.embed_query("test")
        return len(dummy_embedding)

    def get_stats(self) -> dict:
        """
        Get embedder statistics and configuration.

        Returns:
            Dictionary with model info
        """
        return {
            "model_name": self.config.model_name,
            "dimension": self.dimension,
            "device": self.config.device,
            "batch_size": self.config.batch_size,
            "normalized": self.config.normalize_embeddings,
        }
