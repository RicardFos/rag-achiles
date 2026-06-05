# Vector Database Specification (FAISS)

## Goal
Store and retrieve embeddings efficiently using FAISS (Facebook AI Similarity Search) for local, fast semantic search.

## Why FAISS?
- **Local**: No external services or API keys
- **Fast**: Optimized C++ implementation
- **Simple**: Easy to use for small-medium datasets
- **Free**: Open source, no costs
- **Persistent**: Can save/load indexes to disk

## Index Type: IndexFlatIP with L2 Normalization

### Why IndexFlatIP?
- **Flat**: Exhaustive search, guaranteed best results
- **IP**: Inner Product similarity
- **With normalization**: Equivalent to cosine similarity
- **Perfect for <1M vectors**: Fast enough for our use case (~619 chunks)

### Formula
```
cosine_similarity(a, b) = dot(a, b) / (||a|| * ||b||)

If vectors are L2-normalized (||a|| = ||b|| = 1):
cosine_similarity(a, b) = dot(a, b) = Inner Product
```

## Architecture

Uses **Pydantic configuration** and **class-based design** for consistency with parser and embedder modules:
- `VectorStoreConfig`: Pydantic model for configuration
- `VectorStore`: Abstract base class defining interface
- `FAISSVectorStore`: Concrete implementation using FAISS

## Data Models

### VectorStoreConfig (Pydantic)
```python
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path
from typing import Optional

class VectorStoreConfig(BaseModel):
    """Configuration for FAISS vector store."""
    index_path: Path = Field(
        default=Path("faiss_index/vector_index.faiss"),
        description="Path to save/load FAISS index"
    )
    metadata_path: Path = Field(
        default=Path("faiss_index/metadata.pkl"),
        description="Path to save/load metadata"
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for retrieval"
    )
    normalize_vectors: bool = Field(
        default=True,
        description="L2 normalize vectors before adding to index"
    )
    
    model_config = ConfigDict(frozen=True)
```

### SearchResult (Pydantic)
```python
from rag_system.models import DocumentChunk
from typing import Optional

class SearchResult(BaseModel):
    """Result from vector similarity search."""
    chunk: DocumentChunk = Field(..., description="Retrieved document chunk")
    score: float = Field(..., ge=0.0, le=1.0, description="Bi-encoder similarity score")
    rerank_score: Optional[float] = Field(
        default=None, 
        description="Cross-encoder rerank score (if re-ranking was used)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk": {
                    "text": "El presupuesto aprobado fue...",
                    "document": "ResumenReunionGA_20231124.pdf",
                    "page": 3,
                    "chunk_index": 5
                },
                "score": 0.89
            }
        }
    )
```

## Implementation

### Imports (Top of File)
```python
import os
import pickle
from pathlib import Path
from typing import List, Optional
from abc import ABC, abstractmethod

import faiss
import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from rag_system.models import DocumentChunk
```

### Abstract Base Class
```python
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
```

### FAISS Implementation
```python
class FAISSVectorStore(VectorStore):
    """
    FAISS-based vector store with metadata.
    
    Stores embeddings in FAISS IndexFlatIP for cosine similarity search,
    and metadata separately in pickle format. Automatically loads existing
    index from disk if available.
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        Initialize FAISS vector store.
        
        Creates necessary directories and automatically loads existing index
        if found at the configured path.
        
        Args:
            config: Vector store configuration (uses defaults if None)
            
        Example:
            >>> vector_store = FAISSVectorStore()  # Auto-loads if exists
            >>> custom_config = VectorStoreConfig(similarity_threshold=0.8)
            >>> vector_store = FAISSVectorStore(custom_config)
        """
    
    def create_index(self, dimension: int) -> None:
        """
        Create FAISS index for cosine similarity search.
        
        Initializes IndexFlatIP (inner product) which is equivalent to cosine
        similarity when vectors are L2-normalized.
        
        Args:
            dimension: Embedding dimension (e.g., 384)
            
        Example:
            >>> vector_store.create_index(dimension=384)
        """
    
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """
        Add chunks with their embeddings to the store.
        
        Normalizes embeddings if configured, adds to FAISS index, and stores
        chunk metadata for retrieval. Validates chunk-embedding count match.
        
        Args:
            chunks: List of DocumentChunk objects
            embeddings: NumPy array of embeddings, shape (len(chunks), dimension)
            
        Raises:
            ValueError: If index not created or chunk/embedding count mismatch
            
        Example:
            >>> vector_store.add_chunks(all_chunks, embeddings)
            Added 619 chunks. Total: 619
        """
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """
        Search for similar vectors and return results with metadata.
        
        Normalizes query embedding if configured, performs FAISS search, and
        filters results by similarity threshold. Returns chunks with scores.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects with chunks and scores, sorted by
            similarity (descending). Empty list if no results meet threshold.
            
        Example:
            >>> results = vector_store.search(query_emb, top_k=5)
            >>> for r in results:
            ...     print(f"{r.chunk.document}, p.{r.chunk.page}: {r.score:.3f}")
        """
    
    def save(self) -> None:
        """
        Persist index and metadata to disk.
        
        Saves FAISS index (binary format) and chunk metadata (pickle) to
        configured paths. Creates parent directories if needed.
        
        Raises:
            ValueError: If no index exists to save
            
        Example:
            >>> vector_store.save()
            Saved index with 619 vectors to faiss_index/vector_index.faiss
        """
    
    def load(self) -> None:
        """
        Load index and metadata from disk.
        
        Restores both FAISS index and chunk metadata from configured paths.
        
        Raises:
            FileNotFoundError: If index file not found
            
        Example:
            >>> vector_store.load()
            Loaded index with 619 vectors from faiss_index/vector_index.faiss
        """
    
    def get_stats(self) -> dict:
        """
        Get index statistics and configuration.
        
        Returns metadata useful for logging, debugging, or displaying system info.
        
        Returns:
            Dictionary with keys:
            - total_vectors: Number of vectors in index
            - dimension: Embedding dimensionality
            - index_type: FAISS index type identifier
            - total_chunks: Number of stored chunks
            - similarity_threshold: Configured minimum similarity
            - index_path: Path to index file
            - metadata_path: Path to metadata file
            
        Example:
            >>> stats = vector_store.get_stats()
            >>> print(f"Indexed {stats['total_vectors']} vectors")
        """
```

## Usage Examples

### Basic Usage
```python
from rag_system.vector_store import FAISSVectorStore, VectorStoreConfig
from rag_system.embeddings import Embedder
from rag_system.parser import DocumentParser

# Initialize with default configuration
vector_store = FAISSVectorStore()

# Or with custom config
config = VectorStoreConfig(
    similarity_threshold=0.75,
    index_path=Path("my_index/vectors.faiss"),
    metadata_path=Path("my_index/metadata.pkl")
)
vector_store = FAISSVectorStore(config)
```

### Complete Pipeline: Parse → Embed → Store
```python
from pathlib import Path

# 1. Parse documents
parser = DocumentParser()
parsed_docs = parser.parse_multiple(Path("pdfs").glob("*.pdf"))

# Collect all chunks
all_chunks = []
for doc in parsed_docs:
    all_chunks.extend(doc.chunks)

# 2. Generate embeddings
embedder = Embedder()
texts = [chunk.text for chunk in all_chunks]
embeddings = embedder.embed_documents(texts)

# 3. Create and populate vector store
vector_store = FAISSVectorStore()
vector_store.create_index(dimension=embedder.dimension)
vector_store.add_chunks(all_chunks, embeddings)

# 4. Save to disk
vector_store.save()

print(f"Indexed {len(all_chunks)} chunks")
print(vector_store.get_stats())
```

### Search and Retrieve
```python
# Query the vector store
question = "¿Qué presupuesto se aprobó?"
query_embedding = embedder.embed_query(question)

results = vector_store.search(query_embedding, top_k=5)

print(f"Found {len(results)} results:\n")
for i, result in enumerate(results, 1):
    print(f"{i}. Score: {result.score:.3f}")
    print(f"   Document: {result.chunk.document}, Page: {result.chunk.page}")
    print(f"   Text: {result.chunk.text[:100]}...")
    print()
```

### Load Existing Index
```python
# Load previously saved index
vector_store = FAISSVectorStore()  # Auto-loads if index exists

print(f"Loaded {vector_store.index.ntotal} vectors")
print(vector_store.get_stats())

# Ready to search
results = vector_store.search(query_embedding, top_k=5)
```

### Filter by Similarity Threshold
```python
# Only return results above threshold
config = VectorStoreConfig(similarity_threshold=0.8)
vector_store = FAISSVectorStore(config)

results = vector_store.search(query_embedding, top_k=10)
# Returns only results with score >= 0.8
```

### Integration with Full Pipeline
```python
from rag_system.parser import DocumentParser
from rag_system.embeddings import Embedder
from rag_system.vector_store import FAISSVectorStore

# End-to-end workflow
def build_index(pdf_dir: Path) -> FAISSVectorStore:
    """Build vector store from PDF directory."""
    
    # Parse all PDFs
    parser = DocumentParser()
    parsed_docs = parser.parse_multiple(pdf_dir.glob("*.pdf"))
    all_chunks = [chunk for doc in parsed_docs for chunk in doc.chunks]
    
    # Generate embeddings
    embedder = Embedder()
    texts = [chunk.text for chunk in all_chunks]
    embeddings = embedder.embed_documents(texts)
    
    # Create and populate index
    vector_store = FAISSVectorStore()
    vector_store.create_index(dimension=embedder.dimension)
    vector_store.add_chunks(all_chunks, embeddings)
    vector_store.save()
    
    return vector_store

# Usage
vector_store = build_index(Path("pdfs"))
```

## Performance

### For ~619 chunks (our corpus):
- **Index creation**: <1ms
- **Adding 619 vectors**: ~15ms
- **Single query (top-5)**: <1ms
- **Save to disk**: ~12ms
- **Load from disk**: ~8ms
- **Disk space**: ~1MB (index) + ~500KB (metadata)

### Scalability
| Vectors | Index Type | Query Time |
|---------|------------|------------|
| <10K | IndexFlatIP | <1ms |
| 10K-100K | IndexFlatIP | ~5ms |
| 100K-1M | IndexIVFFlat | ~10ms |
| >1M | IndexIVFPQ | ~20ms |

For this assignment (619 vectors), IndexFlatIP is perfect.

## Alternative Index Types (Future)

### IndexIVFFlat (for larger datasets)
```python
# Faster approximate search for 10K+ vectors
quantizer = faiss.IndexFlatIP(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)
index.train(training_vectors)  # Requires training step
```

### IndexHNSWFlat (for very fast search)
```python
# Graph-based index, very fast queries
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = connectivity
```

For our use case, stick with **IndexFlatIP** for simplicity and guaranteed accuracy.

## Dependencies
```txt
faiss-cpu==1.8.0
numpy>=1.24.0
```
