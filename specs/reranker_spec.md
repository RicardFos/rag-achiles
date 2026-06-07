# Re-ranker Specification

## Goal
Re-rank retrieved chunks using a cross-encoder model to improve relevance before passing to the LLM. This addresses the assignment requirement for re-ranking before LLM generation.

## Why Re-ranking?

### The Problem with Bi-encoders
The embedding model (bi-encoder) creates independent vectors for query and documents:
- **Fast**: Can pre-compute document vectors, search with dot product
- **Limited context**: Query and document never "see" each other during encoding
- **Semantic similarity**: Good at finding related topics, but may miss nuanced relevance

### How Cross-encoders Help
Cross-encoders process query + document **together**:
- **Better relevance**: Model sees both query and document simultaneously
- **Contextual scoring**: Can judge if document actually answers the query
- **Trade-off**: Slower (must score each pair individually, can't pre-compute)

### Two-Stage Strategy
```
Stage 1 (Fast): Bi-encoder retrieves top-20 candidates from ~620 chunks
Stage 2 (Accurate): Cross-encoder re-ranks 20 candidates → top-7 for LLM
```

**Result**: Better precision for the LLM without sacrificing recall.

## How Cross-Encoders Work Internally

### Bi-encoder vs Cross-encoder Architecture

#### Bi-encoder (Current embedding model)
```
Query:    "¿Qué compromisos incluye el plan?"
          ↓
      [Encoder]  (independent encoding)
          ↓
      [0.12, -0.43, 0.88, ...]  ← Query vector (384-dim)

Document: "El plan incluye cinco compromisos principales..."
          ↓
      [Encoder]  (independent encoding)
          ↓
      [0.18, -0.31, 0.72, ...]  ← Document vector (384-dim)

Similarity = dot(query_vec, doc_vec) = 0.75
```

**Problem**: Query and document are encoded separately. The model doesn't know what the document says when encoding the query, and vice versa.

#### Cross-encoder (Re-ranker)
```
Input pair:
"[CLS] ¿Qué compromisos incluye el plan? [SEP] El plan incluye cinco compromisos principales... [SEP]"
                                        ↓
                            [Transformer Encoder]
                          (processes both together)
                                        ↓
                            [CLS] representation
                                        ↓
                            [Classification head]
                                        ↓
                            Relevance score: 8.42
```

**Advantage**: The model sees the full query-document pair. It can use cross-attention between query tokens and document tokens to determine relevance.


### Why Cross-encoders Are More Accurate

#### Example: Nuanced Query Understanding

**Query**: "¿Qué compromisos se dirigen a personas mayores?"

**Document A** (Bi-encoder score: 0.76):
```
"El cuarto plan incluye cinco compromisos principales: comunicación clara,
espacio juvenil de participación, evaluación de accesibilidad, prevención
de soledad juvenil, y mediación tributaria."
```

**Document B** (Bi-encoder score: 0.73):
```
"El compromiso de comunicación clara se dirige especialmente al colectivo
de personas mayores, facilitando el derecho a entender mediante lenguaje
claro y accesible."
```

**Bi-encoder reasoning** (vector similarity):
- Document A: Mentions "compromisos" (3 times) + similar words → high score
- Document B: Only mentions "compromiso" (singular) → slightly lower score

**Cross-encoder reasoning** (contextual understanding):
- Document A: Lists compromises but doesn't mention "personas mayores" → score: 5.2
- Document B: Explicitly answers the query about elderly people → score: 9.1

**Result**: Cross-encoder correctly re-ranks Document B higher because it actually answers the specific question, not just shares keywords.


## Model Choice: `cross-encoder/ms-marco-MiniLM-L-12-v2`

### Why This Model?
- **Trained for ranking**: MS MARCO dataset (passage ranking task)
- **Multilingual**: Works well with Spanish despite being English-trained
- **Lightweight**: 12-layer MiniLM (~120MB)
- **Fast**: Can score 20 pairs in ~100-200ms on CPU
- **Free & Local**: No API keys required

### Alternative Considered
- **Cohere rerank-multilingual-v2.0**: Better quality but requires API key and costs money

## Architecture

Uses **Pydantic configuration** and **class-based design** for consistency:
- `RerankerConfig`: Pydantic model for configuration
- `RerankResult`: Pydantic model for re-ranked result
- `Reranker`: Main class handling cross-encoder scoring

## Data Models

### RerankerConfig
Configuration for cross-encoder re-ranker behavior.

**Fields:**
- `model_name` — Cross-encoder model (default: `cross-encoder/ms-marco-MiniLM-L-12-v2`)
- `top_n` — Results to return after re-ranking (default: 7, range: 1-20)
- `batch_size` — Pairs to score per batch (default: 32, range: 1-64)
- `device` — Compute device: `"cpu"` or `"cuda"` (default: `"cpu"`)

Config is **immutable** (frozen) after creation.

### RerankResult
Re-ranked search result with both bi-encoder and cross-encoder scores.

**Fields:**
- `chunk` — DocumentChunk object
- `retrieval_score` — Original bi-encoder (vector) similarity
- `rerank_score` — Cross-encoder relevance score
- `rank` — Final position after re-ranking (≥1)

**Model Config:**
        json_schema_extra={
            "example": {
                "chunk": {
                    "text": "El presupuesto aprobado fue...",
                    "document": "ResumenReunionGA_20231124.pdf",
                    "page": 3,
                    "chunk_index": 5
                },
                "retrieval_score": 0.75,
                "rerank_score": 8.42,
                "rank": 1
            }
        }
    )
```

**Note**: Cross-encoder scores are **not normalized** (typically range from -10 to +10). Higher is better.

## API Reference

### Key Dependencies
- **sentence-transformers** — CrossEncoder for query-document scoring
- **NumPy** — Score normalization
- Internal: `SearchResult`, `DocumentChunk` from `rag_system.models`

### Reranker Class
```python
class Reranker:
    """
    Cross-encoder re-ranker for improving retrieval relevance.
    
    Takes candidates from bi-encoder search and re-ranks them using
    a cross-encoder that processes query+document pairs jointly for
    better contextual relevance scoring.
    """
    
    def __init__(self, config: Optional[RerankerConfig] = None):
        """
        Initialize re-ranker with configuration.
        
        Loads the configured cross-encoder model on the specified device.
        Model loading takes 1-2 seconds on first initialization.
        
        Args:
            config: Re-ranker configuration (uses defaults if None)
            
        Example:
            >>> reranker = Reranker()  # Uses defaults
            Loading cross-encoder: cross-encoder/ms-marco-MiniLM-L-12-v2...
            ✓ Cross-encoder loaded
        """
    
    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_n: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Re-rank candidates using cross-encoder.
        
        Scores each query-candidate pair jointly using the cross-encoder,
        sorts by relevance score, and returns top-N results with updated ranks.
        
        Args:
            query: User query string
            candidates: List of SearchResult from vector store (typically 20)
            top_n: Number of top results to return (uses config default if None)
            
        Returns:
            List of RerankResult sorted by rerank_score (descending)
            
        Note:
            Cross-encoder scores are not normalized (typically -10 to +10).
            Higher scores indicate better relevance.
            
        Example:
            >>> candidates = vector_store.search(query_emb, top_k=20)
            >>> reranked = reranker.rerank(query, candidates, top_n=5)
            >>> for r in reranked:
            ...     print(f"Rank {r.rank}: score={r.rerank_score:.2f}")
        """
    
    def compare_rankings(
        self,
        query: str,
        candidates: List[SearchResult]
    ) -> dict:
        """
        Compare bi-encoder vs cross-encoder rankings.
        
        Analyzes how re-ranking changes the top-5 results compared to the
        original bi-encoder ranking. Useful for understanding re-ranker impact.
        
        Args:
            query: User query
            candidates: Search results from vector store
            
        Returns:
            Dictionary with comparison metrics:
            - query: Original query
            - total_candidates: Number of candidates evaluated
            - top_5_overlap: How many of top-5 appear in both rankings
            - overlap_percentage: Overlap as percentage
            - biencoder_top_5: Chunk indices from bi-encoder ranking
            - crossencoder_top_5: Chunk indices from cross-encoder ranking
            - rank_changes: List of chunks with biggest rank shifts
            
        Example:
            >>> comparison = reranker.compare_rankings(query, candidates)
            >>> print(f"Top-5 overlap: {comparison['top_5_overlap']}/5")
            >>> print("Biggest rank changes:")
            >>> for change in comparison['rank_changes'][:3]:
            ...     print(f"  Rank {change['old_rank']} → {change['new_rank']}")
        """
```

## Integration with RAG Pipeline

### Concept: Two-Stage Retrieval

The RAGGenerator should support an optional `use_reranking` flag that changes the retrieval strategy:

**Without re-ranking (single-stage)**:
1. Generate query embedding
2. Vector search returns top-K chunks directly
3. Pass chunks to LLM

**With re-ranking (two-stage)**:
1. Generate query embedding
2. Vector search returns top-20 candidates (cast wider net)
3. Cross-encoder re-ranks the 20 candidates → top-7 most relevant
4. Pass refined top-7 chunks to LLM

### Implementation Guidelines

**RAGGenerator initialization**:
- Add `use_reranking: bool` parameter (default: True)
- Initialize a `Reranker` instance when enabled
- Store it as an instance variable

**Retrieval method**:
- Branch on the `use_reranking` flag
- If enabled:
  - Retrieve top-20 candidates from vector store
  - Call `reranker.rerank(query, candidates, top_n=7)`
  - Convert RerankResult back to SearchResult format
- If disabled:
  - Retrieve top-K directly from vector store using config
  
**Key considerations**:
- The reranker needs both the query string and the candidates
- Preserve both retrieval_score and rerank_score in results
- The LLM receives the same number of chunks either way (5-7 typically)

## Usage Examples

### Basic Usage
```python
from rag_system.reranker import Reranker, RerankerConfig
from rag_system.vector_store import FAISSVectorStore
from rag_system.embeddings import Embedder

# Initialize components
vector_store = FAISSVectorStore()
embedder = Embedder()
reranker = Reranker()

# Get candidates from vector search
query = "¿Qué compromisos incluye el plan?"
query_emb = embedder.embed_query(query)
candidates = vector_store.search(query_emb, top_k=20)

print(f"Retrieved {len(candidates)} candidates")

# Re-rank to top-7
reranked = reranker.rerank(query, candidates, top_n=7)

print(f"\nTop-7 after re-ranking:")
for result in reranked:
    print(f"Rank {result.rank}: {result.chunk.document}, p.{result.chunk.page}")
    print(f"  Retrieval score: {result.retrieval_score:.3f}")
    print(f"  Rerank score:    {result.rerank_score:.3f}")
```

### Compare Rankings
```python
# Compare bi-encoder vs cross-encoder
comparison = reranker.compare_rankings(query, candidates)

print(f"\nComparison:")
print(f"  Total candidates: {comparison['total_candidates']}")
print(f"  Top-5 overlap:    {comparison['top_5_overlap']}/5 ({comparison['overlap_percentage']:.0f}%)")

print(f"\nBiggest rank changes:")
for change in comparison['rank_changes'][:3]:
    direction = "↑" if change['change'] > 0 else "↓"
    print(f"  {direction} Rank {change['old_rank']} → {change['new_rank']}: {change['document']}, p.{change['page']}")
    print(f"     Retrieval: {change['retrieval_score']:.3f}, Rerank: {change['rerank_score']:.3f}")
```

### Custom Configuration
```python
# Use more candidates and get top-10
config = RerankerConfig(
    model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
    top_n=10,
    batch_size=16
)

reranker = Reranker(config)
reranked = reranker.rerank(query, candidates, top_n=10)
```

### Integration with RAGGenerator
```python
from rag_system.llm import RAGGenerator, LLMConfig
from pydantic import SecretStr

# Initialize RAG with re-ranking enabled
config = LLMConfig(api_key=SecretStr(api_key))

rag = RAGGenerator(
    vector_store=vector_store,
    embedder=embedder,
    config=config,
    use_reranking=True  # Enable re-ranking
)

# Generate answer (uses re-ranking internally)
response = rag.generate_answer("¿Qué compromisos incluye el plan?")
print(response.answer)
```

### A/B Test: With vs Without Re-ranking
```python
# Without re-ranking
rag_no_rerank = RAGGenerator(
    vector_store, embedder, config,
    use_reranking=False
)
response_no_rerank = rag_no_rerank.generate_answer(query)

# With re-ranking
rag_with_rerank = RAGGenerator(
    vector_store, embedder, config,
    use_reranking=True
)
response_with_rerank = rag_with_rerank.generate_answer(query)

print("WITHOUT re-ranking:")
print(response_no_rerank.answer)
print(f"\nWITH re-ranking:")
print(response_with_rerank.answer)
```

## Visual Example: How Re-ranking Changes Results

### Before Re-ranking (Bi-encoder only)
```
Query: "¿Cuáles son los compromisos de comunicación clara?"

Top-5 by bi-encoder:
1. Score: 0.78 | "...comunicación clara...transparencia..." [Relevant ✓]
2. Score: 0.76 | "...compromisos del plan...estrategia..." [Generic]
3. Score: 0.75 | "...comunicación...redes sociales..."    [Somewhat related]
4. Score: 0.74 | "...claridad en documentos tributarios..." [Relevant ✓]
5. Score: 0.73 | "...comunicación institucional..."       [Generic]
```

### After Re-ranking (Cross-encoder)
```
Top-5 by cross-encoder:
1. Score: 9.2 | "...comunicación clara...transparencia..." [Relevant ✓] (was #1)
2. Score: 8.8 | "...claridad en documentos tributarios..." [Relevant ✓] (was #4 → moved up!)
3. Score: 7.9 | "...derecho a entender de la ciudadanía..." [Relevant ✓] (was #8 → big jump!)
4. Score: 7.1 | "...comunicación...redes sociales..."    [Somewhat related] (was #3)
5. Score: 6.5 | "...lenguaje claro en administración..." [Relevant ✓] (was #12 → huge jump!)
```

**Key improvement**: Chunks #4, #8, and #12 moved up because cross-encoder understood they better answer the specific question.

## Testing Re-ranker Quality

### Relevance Test
```python
query = "¿Qué es la Escuela de Gobierno Abierto?"
query_emb = embedder.embed_query(query)

# Get 20 candidates
candidates = vector_store.search(query_emb, top_k=20)

# Re-rank
reranked = reranker.rerank(query, candidates, top_n=5)

# Check if top result is actually about the school
top_chunk = reranked[0].chunk
assert "escuela" in top_chunk.text.lower()
assert "gobierno abierto" in top_chunk.text.lower()

print("✓ Top result is relevant")
```

### Rank Correlation
```python
from scipy.stats import spearmanr

# Compare bi-encoder and cross-encoder rankings
biencoder_scores = [c.score for c in candidates[:20]]
crossencoder_scores = [r.rerank_score for r in reranker.rerank(query, candidates, top_n=20)]

correlation, p_value = spearmanr(biencoder_scores, crossencoder_scores)

print(f"Rank correlation: {correlation:.3f}")
# Typical: 0.6-0.8 (some agreement but meaningful differences)
```

## Best Practices

### 1. Candidate Pool Size
```python
# Good: Retrieve 2-4x more than final top_n
candidates = vector_store.search(query_emb, top_k=20)  # Want top-7 after rerank
reranked = reranker.rerank(query, candidates, top_n=7)

# Too few: No room for re-ranker to improve
candidates = vector_store.search(query_emb, top_k=8)  # Only 1 extra candidate
reranked = reranker.rerank(query, candidates, top_n=7)  # Limited benefit

# Too many: Slower with diminishing returns
candidates = vector_store.search(query_emb, top_k=100)  # Overkill
```

### 2. When to Use Re-ranking
```python
# Always use for:
# - User-facing Q&A (quality matters)
# - Production RAG systems
# - When precision is critical

# Skip for:
# - Batch processing (speed matters)
# - When top-k is already very small (top-3)
# - Offline indexing tasks
```

### 3. Model Selection
```python
# For multilingual/Spanish:
reranker = Reranker(RerankerConfig(
    model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"  # Default, works well
))

# For better quality (slower):
reranker = Reranker(RerankerConfig(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"  # Faster
))

# Note: Most cross-encoders are English-trained but generalize to Spanish
```

### 4. Monitoring Re-ranker Impact
```python
def analyze_reranking_impact(query: str, candidates: List[SearchResult], reranked: List[RerankResult]):
    """Log how re-ranking changed results."""
    
    # Compare top-1
    top_biencoder = candidates[0].chunk.chunk_index
    top_crossencoder = reranked[0].chunk.chunk_index
    
    if top_biencoder != top_crossencoder:
        print(f"⚠ Re-ranker changed top result!")
        print(f"  Bi-encoder:   chunk #{top_biencoder}")
        print(f"  Cross-encoder: chunk #{top_crossencoder}")
    
    # Compute average rank change
    changes = reranker.compare_rankings(query, candidates)['rank_changes']
    if changes:
        avg_change = sum(abs(c['change']) for c in changes) / len(changes)
        print(f"  Avg rank shift: {avg_change:.1f} positions")
```

## Dependencies
```txt
sentence-transformers>=2.2.0
torch>=1.13.0
numpy>=1.24.0
scipy>=1.10.0  # For rank correlation testing
```

## Troubleshooting

### Issue: Re-ranking is slow
**Solution**: Reduce batch_size or use fewer candidates
```python
config = RerankerConfig(batch_size=16)  # Default is 32
candidates = vector_store.search(query_emb, top_k=10)  # Instead of 20
```

### Issue: Results worse after re-ranking
**Cause**: Query might be too short or ambiguous  
**Solution**: Use re-ranking selectively or increase candidate pool
```python
if len(query.split()) > 3:  # Only rerank for detailed queries
    reranked = reranker.rerank(query, candidates)
else:
    reranked = candidates[:5]  # Use bi-encoder directly
```

### Issue: Out of memory
**Solution**: Reduce batch_size
```python
config = RerankerConfig(batch_size=8)  # Lower memory usage
```
