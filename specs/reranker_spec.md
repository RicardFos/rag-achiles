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

### Internal Processing Steps

#### 1. **Tokenization & Concatenation**
```python
# Input texts
query = "¿Qué compromisos incluye el plan?"
document = "El plan incluye cinco compromisos principales: comunicación clara, ..."

# Tokenizer creates special format
tokens = tokenizer(
    query, 
    document, 
    padding=True, 
    truncation=True, 
    max_length=512
)

# Result (conceptual):
# [CLS] ¿ Qué compromisos incluye el plan ? [SEP] El plan incluye cinco compromisos principales ... [SEP] [PAD] [PAD]
#   0   1  2      3         4      5    6     7    8   9    10      11    12          13           ...  511  512
```

**Special tokens**:
- `[CLS]`: Classification token (start of sequence)
- `[SEP]`: Separator between query and document
- `[PAD]`: Padding to reach max_length

#### 2. **Self-Attention Across Query + Document**

The Transformer processes all tokens together using **self-attention**:

```
Query tokens can attend to document tokens:
"compromisos" (token 3) → attends to → "compromisos principales" (tokens 12-13)
     ↓
Cross-attention discovers that the document directly mentions "compromisos"

Document tokens can attend to query tokens:
"El plan incluye" (tokens 8-10) → attends to → "incluye el plan" (tokens 4-6)
     ↓
Cross-attention sees the question asks about what the plan "includes"
```

**Attention Matrix** (simplified for 10 tokens):
```
         Q1  Q2  Q3  Q4  Q5  D1  D2  D3  D4  D5
Query1   0.1 0.2 0.1 0.0 0.1 0.3 0.1 0.0 0.1 0.0  ← Q1 attends heavily to D1
Query2   0.2 0.3 0.1 0.2 0.0 0.1 0.1 0.0 0.0 0.0
Query3   0.0 0.1 0.2 0.1 0.1 0.0 0.2 0.4 0.0 0.0  ← Q3 attends to D3,D4
...
Doc1     0.2 0.1 0.0 0.1 0.0 0.3 0.2 0.1 0.0 0.0  ← D1 attends back to Q1
Doc2     0.0 0.0 0.1 0.3 0.1 0.1 0.2 0.1 0.0 0.0  ← D2 attends to Q4
```

This **bidirectional attention** lets the model understand:
- Does the document contain the entities/concepts from the query?
- Does the document structure match the query type (who/what/when/where)?
- Is the answer explicit or just tangentially related?

#### 3. **[CLS] Token Aggregation**

After 12 layers of Transformer processing, the `[CLS]` token has accumulated information about the **entire query-document relationship**:

```
[CLS] final representation:
  - Aggregates all cross-attention between query and document
  - Encodes: "Does this document answer this query?"
  - High-dimensional vector: [768-dim for base models, 384-dim for MiniLM]
```

#### 4. **Classification Head (Scoring)**

```python
# [CLS] representation → Linear layer → Sigmoid/tanh → Score
cls_vector = transformer_output[0]  # [CLS] token embedding (384-dim)
logits = linear_layer(cls_vector)   # Single value
score = activation(logits)          # Normalized score (-10 to +10 typical range)
```

**Score interpretation**:
- **> 8.0**: Highly relevant, document directly answers the query
- **6.0 - 8.0**: Relevant, document contains useful information
- **4.0 - 6.0**: Somewhat related, mentions similar topics
- **< 4.0**: Not relevant or off-topic

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

### Technical: Attention Scores Example

For the query-document pair:

```
Query: "¿Qué compromisos incluye el plan?"
Document: "El plan incluye cinco compromisos principales..."
```

**Cross-attention pattern** (simplified):

```
Query token "compromisos" attends to:
  - "compromisos" in document: 0.82  ← Very high! Direct match
  - "principales" in document: 0.15  ← Moderate, related concept
  - "plan" in document:        0.12  ← Lower, common word
  - Other tokens:              < 0.10

Query token "incluye" attends to:
  - "incluye" in document:     0.78  ← High! Direct match
  - "cinco" in document:       0.22  ← Moderate, answers "how many"
  - Other tokens:              < 0.10
```

The cross-encoder sees that critical query terms ("compromisos", "incluye") have strong matches in the document, leading to a high relevance score.

### Computational Difference

**Bi-encoder (one-time cost per document)**:
```
encode(query)    → 10ms  → cache this
encode(doc_1)    → 10ms  → cache this
encode(doc_2)    → 10ms  → cache this
...
encode(doc_619)  → 10ms  → cache this

At query time:
  similarity(query_vec, all_doc_vecs) → 1ms (dot products)
```

**Cross-encoder (per query-document pair)**:
```
encode([CLS] query [SEP] doc_1 [SEP])  → 10ms
encode([CLS] query [SEP] doc_2 [SEP])  → 10ms
...
encode([CLS] query [SEP] doc_20 [SEP]) → 10ms

Total for 20 pairs: 200ms (can't be cached!)
```

**This is why we use a two-stage pipeline**:
1. Bi-encoder filters ~620 → 20 (fast, cached)
2. Cross-encoder re-ranks 20 → 7 (slow, but only 20 pairs)

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

### Update RAGGenerator to Use Re-ranker

Modify `rag_system/llm.py`:

```python
from rag_system.reranker import Reranker, RerankerConfig

class RAGGenerator:
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: Embedder,
        config: LLMConfig,
        use_reranking: bool = True  # NEW
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.config = config
        self.use_reranking = use_reranking
        
        self._load_llm()
        
        # NEW: Initialize re-ranker if enabled
        if self.use_reranking:
            self.reranker = Reranker()
            print("✓ Re-ranker enabled")
    
    def _retrieve_context(self, question: str) -> List[SearchResult]:
        """Retrieve relevant chunks with optional re-ranking."""
        query_embedding = self.embedder.embed_query(question)
        
        # Retrieve more candidates if using re-ranking
        if self.use_reranking:
            # Get top-20 candidates
            candidates = self.vector_store.search(
                query_embedding,
                top_k=20
            )
            
            # Re-rank to top-7
            reranked = self.reranker.rerank(question, candidates, top_n=7)
            
            # Convert back to SearchResult format
            results = [
                SearchResult(chunk=r.chunk, score=r.retrieval_score)
                for r in reranked
            ]
            return results
        else:
            # Direct retrieval (no re-ranking)
            return self.vector_store.search(
                query_embedding,
                top_k=self.config.top_k
            )
```

## Performance Characteristics

### Speed (CPU, typical laptop)
- **Model loading**: 1-2 seconds (first time only)
- **Re-ranking 20 pairs**: 100-200ms
- **Bottleneck**: Still the LLM call (1-3 seconds)
- **Total overhead**: ~10-15% increase in latency

### Memory Usage
- **Cross-encoder model**: ~500MB
- **Combined with bi-encoder**: ~1GB total
- **Negligible for scoring**: Just forward passes

### Accuracy Improvement
Typical improvements with re-ranking:
- **MRR (Mean Reciprocal Rank)**: +10-20%
- **Precision@5**: +15-25%
- **Answer quality**: Fewer irrelevant contexts to LLM

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

## Expected Improvement Metrics

Based on typical RAG benchmarks:

| Metric | Without Re-ranking | With Re-ranking | Improvement |
|--------|-------------------|-----------------|-------------|
| MRR@5 | 0.65 | 0.78 | +20% |
| Precision@5 | 0.72 | 0.85 | +18% |
| NDCG@5 | 0.68 | 0.81 | +19% |
| Answer Quality | Baseline | +15-25% | Subjective |

**Note**: Actual improvements depend on query types and corpus quality.

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

## Integration Checklist

- [ ] Create `rag_system/reranker.py` with `Reranker` class
- [ ] Add `RerankResult` to `rag_system/models.py`
- [ ] Update `RAGGenerator` in `rag_system/llm.py` to support re-ranking
- [ ] Update `rag_system/__init__.py` to export re-ranker classes
- [ ] Add `sentence-transformers>=2.2.0` to `requirements.txt`
- [ ] Create comparison demo in notebook 03 (with/without re-ranking)
- [ ] Update specs with actual results after implementation
