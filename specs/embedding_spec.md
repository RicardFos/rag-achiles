# Embedding Specification

## Goal
Generate vector representations of text using a free, local, multilingual embedding model suitable for Spanish documents.

## Model Choice: `paraphrase-multilingual-MiniLM-L12-v2`

### Why This Model?
- **Multilingual**: Trained on 50+ languages including Spanish
- **Free & Local**: No API keys, works offline
- **Lightweight**: 384 dimensions, ~120MB download
- **Fast**: Runs efficiently on CPU, no GPU required
- **Quality**: Good semantic search performance for RAG tasks

### Alternative Considered
- `paraphrase-multilingual-mpnet-base-v2`: Higher quality (768 dim) but slower

## Architecture

Uses **Pydantic configuration** and **class-based design** for consistency with the parser module:
- `EmbeddingConfig`: Pydantic model for configuration
- `Embedder`: Main class that handles model lifecycle and embedding generation

## Data Models

### EmbeddingConfig
Configuration for embedding generation behavior.

**Fields:**
- `model_name` — HuggingFace model (default: `paraphrase-multilingual-MiniLM-L12-v2`)
- `device` — Compute device: `"cpu"` or `"cuda"` (default: `"cpu"`)
- `batch_size` — Texts per batch (default: 64, range: 1-256)
- `normalize_embeddings` — L2 normalize for cosine similarity (default: `True`)
- `show_progress` — Display progress bar (default: `False`)

Config is **immutable** (frozen) after creation.

## API Reference

### Embedder Class

```python
class Embedder:
    """
    Generates embeddings using sentence-transformers.
    
    Encapsulates model lifecycle, batch processing, and provides
    consistent interface for document and query embedding.
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize embedder with configuration.
        
        Loads the configured sentence-transformer model and prepares it for
        embedding generation. Model is loaded on the specified device (CPU/CUDA).
        
        Args:
            config: Embedding configuration (uses defaults if None)
            
        Example:
            >>> embedder = Embedder()  # Uses default config
            >>> embedder = Embedder(EmbeddingConfig(device="cuda"))  # GPU
        """
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple document texts in batches.
        
        Processes texts in batches for efficiency. Supports progress display
        and automatic normalization based on configuration.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings, shape (len(texts), dimension)
            dtype: float32, normalized if configured
            
        Example:
            >>> texts = ["El presupuesto fue...", "La reunión tuvo lugar..."]
            >>> embeddings = embedder.embed_documents(texts)
            >>> embeddings.shape
            (2, 384)
        """
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query text.
        
        Use this method for search queries. Some models differentiate between
        document and query embedding to optimize for asymmetric search.
        
        Args:
            text: Query string to embed
            
        Returns:
            NumPy array embedding, shape (dimension,)
            dtype: float32, normalized if configured
            
        Note:
            Always use embed_query for queries, not embed_documents with a
            single-item list, to benefit from query-specific optimizations.
            
        Example:
            >>> query = "¿Qué presupuesto se aprobó?"
            >>> embedding = embedder.embed_query(query)
            >>> embedding.shape
            (384,)
        """
    
    @property
    def dimension(self) -> int:
        """
        Get the dimensionality of embeddings.
        
        Returns the model's embedding dimension. This is determined by the
        model architecture and is constant for all embeddings.
        
        Returns:
            Embedding dimension (384 for multilingual-MiniLM-L12-v2)
            
        Example:
            >>> embedder.dimension
            384
        """
    
    def get_stats(self) -> dict:
        """
        Get embedder statistics and configuration.
        
        Returns a dictionary with model metadata, useful for logging,
        debugging, or displaying system information.
        
        Returns:
            Dictionary with keys:
            - model_name: HuggingFace model identifier
            - dimension: Embedding dimensionality
            - device: Computation device (cpu/cuda)
            - batch_size: Configured batch size
            - normalized: Whether embeddings are L2-normalized
            
        Example:
            >>> embedder.get_stats()
            {
                'model_name': 'paraphrase-multilingual-MiniLM-L12-v2',
                'dimension': 384,
                'device': 'cpu',
                'batch_size': 64,
                'normalized': True
            }
        """
```

## Performance Characteristics

### Speed (CPU, typical laptop)
- **Model loading**: 2-3 seconds (first time only)
- **Embedding rate**: 100-500 chunks/second
- **Single query**: <0.1 second
- **500 chunks**: 5-10 seconds total

### Memory Usage
- **Model in memory**: ~500MB
- **500 embeddings (384 dim)**: ~1.5MB
- **Total RAM**: ~1GB for complete system

### Comparison to API Embeddings
| Metric | Local (sentence-transformers) | API (Gemini/OpenAI) |
|--------|-------------------------------|---------------------|
| Cost | Free | Free tier limited |
| Speed | ~10ms per embedding | ~200-500ms (network) |
| Rate limits | None | 15-60 requests/min |
| Offline | ✅ Yes | ❌ No |
| Setup | Model download (~120MB) | API key management |

## Usage Examples

### Basic Usage
```python
from rag_system.embeddings import Embedder, EmbeddingConfig

# Use default configuration
embedder = Embedder()

# Embed documents
texts = ["El presupuesto aprobado fue...", "La reunión se celebró..."]
embeddings = embedder.embed_documents(texts)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding shape: {embeddings.shape}")  # (2, 384)
```

### Custom Configuration
```python
# Custom config for specific needs
config = EmbeddingConfig(
    batch_size=32,           # Smaller batches if memory constrained
    show_progress=True,      # Show progress bar
    device="cuda"            # Use GPU if available
)

embedder = Embedder(config)
embeddings = embedder.embed_documents(texts)
```

### Embed Query
```python
# Always use embed_query for search queries
question = "¿Qué presupuesto se aprobó?"
query_embedding = embedder.embed_query(question)

print(f"Query embedding shape: {query_embedding.shape}")  # (384,)
```

### Get Model Info
```python
# Check embedder configuration
stats = embedder.get_stats()
print(stats)
# {
#   'model_name': 'paraphrase-multilingual-MiniLM-L12-v2',
#   'dimension': 384,
#   'device': 'cpu',
#   'batch_size': 64,
#   'normalized': True
# }
```

### Integration with Parsed Documents
```python
from rag_system.parser import DocumentParser
from rag_system.embeddings import Embedder

# Parse documents
parser = DocumentParser()
parsed_docs = parser.parse_multiple(pdf_paths)

# Extract all texts
all_chunks = []
for doc in parsed_docs:
    all_chunks.extend(doc.chunks)

texts = [chunk.text for chunk in all_chunks]

# Embed all chunks
embedder = Embedder()
embeddings = embedder.embed_documents(texts)

print(f"Embedded {len(texts)} chunks into {embeddings.shape[1]}-dim vectors")
```

## Best Practices

### 1. Model Initialization
```python
# Initialize once, reuse throughout session
embedder = Embedder()

# Good: Reuse the same instance
embeddings1 = embedder.embed_documents(batch1)
embeddings2 = embedder.embed_documents(batch2)

# Bad: Creating new instance each time (reloads model)
embedder1 = Embedder()
embeddings1 = embedder1.embed_documents(batch1)
embedder2 = Embedder()  # Reloads model unnecessarily
embeddings2 = embedder2.embed_documents(batch2)
```

### 2. Batch Processing
```python
# Good: Use embed_documents for multiple texts
embeddings = embedder.embed_documents(texts)

# Bad: One at a time with embed_query
embeddings = [embedder.embed_query(t) for t in texts]  # Much slower!
```

### 3. Document vs Query Embedding
```python
# Correct usage
doc_embeddings = embedder.embed_documents(chunk_texts)  # For indexing
query_embedding = embedder.embed_query(user_question)   # For searching

# Some models optimize differently for documents vs queries
# Always use the right method for the use case
```

### 4. Progress Tracking
```python
# Enable progress bar for long operations
config = EmbeddingConfig(show_progress=True)
embedder = Embedder(config)

# Progress bar shows during embedding
embeddings = embedder.embed_documents(large_text_list)
# Embedding documents: 100%|████████| 8/8 [00:05<00:00,  1.5batch/s]
```

### 5. Memory Management
```python
# For very large datasets, process in chunks
def embed_large_dataset(texts: List[str], embedder: Embedder, chunk_size: int = 1000):
    all_embeddings = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        chunk_embeddings = embedder.embed_documents(chunk)
        all_embeddings.append(chunk_embeddings)
    return np.vstack(all_embeddings)
```

## Testing Embedding Quality

### Semantic Similarity Test
```python
import numpy as np
from rag_system.embeddings import Embedder

embedder = Embedder()

# Embed related Spanish texts
text1 = "El presupuesto aprobado fue de 5 millones"
text2 = "Se aprobó un presupuesto de 5M de euros"
text3 = "El clima está soleado hoy"

emb1 = embedder.embed_query(text1)
emb2 = embedder.embed_query(text2)
emb3 = embedder.embed_query(text3)

# Cosine similarity (normalized vectors = dot product)
sim_12 = np.dot(emb1, emb2)  # Should be high (~0.8-0.9)
sim_13 = np.dot(emb1, emb3)  # Should be low (~0.2-0.4)

print(f"Related texts similarity:    {sim_12:.3f}")
print(f"Unrelated texts similarity:  {sim_13:.3f}")
```

**Expected results:**
- Related texts (same topic): 0.75 - 0.95
- Unrelated texts (different topics): 0.1 - 0.4

### Batch vs Single Embedding Consistency
```python
# Verify batch and single embedding produce same results
texts = ["texto 1", "texto 2", "texto 3"]

# Batch embedding
batch_embeddings = embedder.embed_documents(texts)

# Single embeddings
single_embeddings = np.array([embedder.embed_query(t) for t in texts])

# Should be nearly identical (minor floating point differences)
difference = np.abs(batch_embeddings - single_embeddings).max()
print(f"Max difference: {difference:.10f}")  # Should be < 1e-6
```

## Integration with FAISS
```python
from rag_system.embeddings import Embedder
from rag_system.parser import DocumentParser
import faiss

# 1. Parse documents and extract chunks
parser = DocumentParser()
parsed_docs = parser.parse_multiple(pdf_paths)
all_chunks = [chunk for doc in parsed_docs for chunk in doc.chunks]

# 2. Embed all chunks
embedder = Embedder()
chunk_texts = [chunk.text for chunk in all_chunks]
embeddings = embedder.embed_documents(chunk_texts)

# 3. Create FAISS index with correct dimension
dimension = embedder.dimension  # 384
index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

# 4. Add embeddings to index (already normalized)
index.add(embeddings)

print(f"Indexed {index.ntotal} vectors of dimension {dimension}")
```

## Troubleshooting

### Issue: Model download fails
**Solution**: Pre-download model
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
```

### Issue: Slow on first run
**Cause**: Model downloads on first import (~120MB)  
**Solution**: Expected behavior, subsequent runs are fast

### Issue: Out of memory
**Solution**: Reduce batch size
```python
embed_documents(texts, batch_size=32)  # Instead of 64
```

## Dependencies
```txt
sentence-transformers>=2.2.0
torch>=1.13.0  # Auto-installed with sentence-transformers
numpy>=1.24.0
```
