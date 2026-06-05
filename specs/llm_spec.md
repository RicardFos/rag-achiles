# LLM Specification

## Goal
Generate natural language answers to user questions based on retrieved context chunks, with proper citation of sources (document name and page number).

## Model Choice: Google Gemini (Free Tier)

### Why Gemini?
- **Free tier**: 15 requests/min, 500 requests/day
- **No credit card required**: Easy signup for demo purposes
- **Good quality**: Competitive with GPT-3.5 for RAG tasks
- **LangChain support**: First-class integration
- **Context window**: 32K tokens (sufficient for RAG context)

### Alternative Considered
- **OpenAI GPT-3.5**: Better quality but requires payment setup
- **Local LLM (Ollama)**: Free but slower and lower quality

## Architecture

Uses **Pydantic configuration** and **class-based design** for consistency with other modules:
- `LLMConfig`: Pydantic model for configuration
- `Citation`: Pydantic model representing a source citation
- `LLMResponse`: Pydantic model for structured LLM output
- `RAGGenerator`: Main class orchestrating retrieval and generation

## Data Models

### Citation
Represents a citation to a specific page in a document, grouping all chunks from that page.

**Core Concept:** A citation references a **(document, page)** pair, not individual chunks. When multiple chunks from the same page are retrieved, they're grouped into a single citation.

**Fields:**
- `document` — Source PDF filename
- `page` — Page number (≥1)
- `chunks` — List of SearchResult objects from this page

**Computed Properties:**
- `num_chunks` — Count of chunks from this page
- `best_score` — Highest retrieval score among chunks
- `best_rerank_score` — Highest rerank score (if re-ranking used)
- `all_text` — Concatenated text from all chunks

**Example:**
```python
{
    "document": "ResumenReunionGA_20231124.pdf",
    "page": 3,
    "chunks": [
        {"chunk": {...}, "score": 0.85, "rerank_score": 0.92},
        {"chunk": {...}, "score": 0.78, "rerank_score": 0.88}
    ]
}
# → One citation for page 3 with 2 chunks
```

**Design Rationale:** Citations in academic/professional contexts reference pages, not arbitrary text snippets. This structure mirrors that convention while preserving full chunk-level traceability.

### LLMResponse
Complete RAG system output with answer, sources, and metadata.

**Fields:**
- `question` — Original user question
- `answer` — Generated answer with inline citations
- `citations` — List of Citation objects (grouped by page)
- `num_chunks_retrieved` — Total chunks retrieved before grouping
- `has_citations` — Boolean, whether answer cites sources
- `used_reranking` — Boolean, whether cross-encoder was used

**Computed Property:**
- `formatted_response` — Human-readable display format with:
  - Answer text
  - Numbered source list with scores
  - Text preview from each citation

### LLMConfig
Configuration for Gemini LLM and retrieval behavior.

**Fields:**
- `api_key` — Google API key (SecretStr, not logged)
- `model_name` — Gemini model (default: `gemini-2.0-flash-001`)
- `temperature` — Sampling randomness (default: 0.0, range: 0.0-2.0)
  - 0.0 = deterministic (best for factual Q&A)
  - 0.7 = balanced creativity
  - 2.0 = maximum variation
- `max_output_tokens` — Response length cap (default: 1024, range: 64-8192)
- `top_k` — Chunks to pass to LLM (default: 7, range: 1-20)

Config is **immutable** (frozen) after creation.

## API Reference

### RAGGenerator Class
```python
class RAGGenerator:
    """
    RAG system orchestrating retrieval and generation.
    
    Coordinates vector search, optional re-ranking, context formatting,
    and LLM generation to produce cited answers from document corpus.
    """
    
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: Embedder,
        config: LLMConfig,
        use_reranking: bool = True,
        reranker_config: Optional[RerankerConfig] = None
    ):
        """
        Initialize RAG generator.
        
        Loads Gemini LLM with configured parameters and optionally initializes
        cross-encoder re-ranker for improved retrieval quality.
        
        Args:
            vector_store: Populated FAISS vector store
            embedder: Embedder for query embedding
            config: LLM configuration
            use_reranking: Whether to use cross-encoder re-ranking (default: True)
            reranker_config: Re-ranker configuration (uses defaults if None)
            
        Example:
            >>> rag = RAGGenerator(vector_store, embedder, config)
            ✓ Loaded gemini-2.0-flash-001
            ✓ Re-ranker enabled (cross-encoder)
            
            >>> rag_no_rerank = RAGGenerator(vector_store, embedder, config, use_reranking=False)
            ✓ Loaded gemini-2.0-flash-001
            ℹ Re-ranker disabled (bi-encoder only)
        """
    
    def generate_answer(self, question: str) -> LLMResponse:
        """
        Generate answer to question using RAG pipeline.
        
        Orchestrates the complete pipeline: query embedding → retrieval →
        optional re-ranking → context formatting → LLM generation →
        citation extraction. Returns structured response with answer and sources.
        
        Args:
            question: User's natural language question
            
        Returns:
            LLMResponse with answer, citations, and metadata
            
        Note:
            Returns "No puedo responder..." if no relevant chunks retrieved.
            Always includes citation information when sources are available.
            
        Example:
            >>> response = rag.generate_answer("¿Qué presupuesto se aprobó?")
            >>> print(response.answer)
            El presupuesto aprobado fue de 5 millones [doc.pdf, p.3]
            >>> print(f"Citations: {len(response.citations)}")
            Citations: 1
        """
    
    def batch_generate(self, questions: List[str]) -> List[LLMResponse]:
        """
        Generate answers for multiple questions.
        
        Processes questions sequentially through the RAG pipeline. Reuses
        loaded models and vector store for efficiency.
        
        Args:
            questions: List of questions
            
        Returns:
            List of LLMResponse objects, one per question
            
        Note:
            Subject to API rate limits (15 requests/min for Gemini free tier).
            
        Example:
            >>> questions = ["¿Qué presupuesto?", "¿Cuándo fue la reunión?"]
            >>> responses = rag.batch_generate(questions)
            >>> for r in responses:
            ...     print(f"Q: {r.question}\nA: {r.answer}\n")
        """
```

## Usage Examples

### Basic Usage (with Re-ranking - Default)
```python
from rag_system.llm import RAGGenerator, LLMConfig
from rag_system.vector_store import FAISSVectorStore
from rag_system.embeddings import Embedder
from pydantic import SecretStr

# Load components
vector_store = FAISSVectorStore()  # Load existing index
embedder = Embedder()

# Configure LLM
config = LLMConfig(
    api_key=SecretStr("YOUR_GOOGLE_API_KEY"),
    model_name="gemini-2.0-flash-001",
    temperature=0.0,
    top_k=5
)

# Initialize RAG system (re-ranking enabled by default)
rag = RAGGenerator(vector_store, embedder, config)
# Output: ✓ Loaded gemini-2.0-flash-001
#         ✓ Re-ranker enabled (cross-encoder)

# Ask question
question = "¿Qué presupuesto se aprobó para el proyecto urbano?"
response = rag.generate_answer(question)

print(response.formatted_response)
```

### Disable Re-ranking (Faster, Less Accurate)
```python
# Initialize without re-ranking (bi-encoder only)
rag = RAGGenerator(
    vector_store, 
    embedder, 
    config,
    use_reranking=False  # Disable cross-encoder
)
# Output: ✓ Loaded gemini-2.0-flash-001
#         ℹ Re-ranker disabled (bi-encoder only)

response = rag.generate_answer(question)
```

### Custom Re-ranker Configuration
```python
from rag_system.reranker import RerankerConfig

# Custom re-ranker settings
reranker_config = RerankerConfig(
    top_n=10,  # Get top-10 after re-ranking (instead of default 5)
    batch_size=16  # Smaller batches for memory-constrained systems
)

rag = RAGGenerator(
    vector_store,
    embedder,
    config,
    use_reranking=True,
    reranker_config=reranker_config
)
```

### A/B Test: With vs Without Re-ranking
```python
# Without re-ranking
rag_no_rerank = RAGGenerator(
    vector_store, embedder, config,
    use_reranking=False
)
response_no_rerank = rag_no_rerank.generate_answer(question)

# With re-ranking
rag_with_rerank = RAGGenerator(
    vector_store, embedder, config,
    use_reranking=True
)
response_with_rerank = rag_with_rerank.generate_answer(question)

# Compare
print("WITHOUT re-ranking:")
print(response_no_rerank.answer)
print(f"Citations: {len(response_no_rerank.citations)}")

print("\nWITH re-ranking:")
print(response_with_rerank.answer)
print(f"Citations: {len(response_with_rerank.citations)}")
```

### Inspect Re-ranking Scores
```python
# Access both bi-encoder and cross-encoder scores
response = rag.generate_answer(question)

# The response internally used re-ranked results
print(f"Used re-ranking: {response.used_reranking}")

# To see the scores, we need to manually retrieve (for inspection)
query_emb = embedder.embed_query(question)
candidates = vector_store.search(query_emb, top_k=20)
reranked = rag.reranker.rerank(question, candidates, top_n=5)

print("\nScore comparison:")
for i, r in enumerate(reranked, 1):
    print(f"{i}. {r.chunk.document[:40]}... | Page {r.chunk.page}")
    print(f"   Bi-encoder score:    {r.retrieval_score:.3f}")
    print(f"   Cross-encoder score: {r.rerank_score:.3f}")
```

### With Environment Variable (.env file)
```python
import os
from pydantic import SecretStr
from dotenv import load_dotenv

# Load from .env file
load_dotenv()
api_key = SecretStr(os.getenv("GOOGLE_API_KEY"))

config = LLMConfig(api_key=api_key)
rag = RAGGenerator(vector_store, embedder, config)
```

### Adjust Temperature for Creativity
```python
# Deterministic (best for factual Q&A)
config = LLMConfig(
    api_key=api_key,
    temperature=0.0  # No randomness
)

# Creative (for summaries, interpretations)
config = LLMConfig(
    api_key=api_key,
    temperature=0.7  # More variation
)
```

### Retrieve More/Fewer Chunks
```python
# Retrieve top 10 chunks (more context, slower)
config = LLMConfig(
    api_key=api_key,
    top_k=10
)

# Retrieve only top 3 (faster, less context)
config = LLMConfig(
    api_key=api_key,
    top_k=3
)
```

### Batch Processing
```python
questions = [
    "¿Qué presupuesto se aprobó?",
    "¿Cuándo fue la reunión?",
    "¿Quién participó en la decisión?"
]

responses = rag.batch_generate(questions)

for resp in responses:
    print(f"\nQ: {resp.question}")
    print(f"A: {resp.answer}")
    print(f"Citations: {len(resp.citations)}")
```

### Complete End-to-End Pipeline
```python
from pathlib import Path
from pydantic import SecretStr
from rag_system.parser import DocumentParser
from rag_system.embeddings import Embedder
from rag_system.vector_store import FAISSVectorStore
from rag_system.llm import RAGGenerator, LLMConfig

def build_rag_system(pdf_dir: Path, api_key: str) -> RAGGenerator:
    """Build complete RAG system from PDFs."""
    
    # 1. Parse PDFs
    parser = DocumentParser()
    parsed_docs = parser.parse_multiple(pdf_dir.glob("*.pdf"))
    all_chunks = [chunk for doc in parsed_docs for chunk in doc.chunks]
    
    # 2. Generate embeddings
    embedder = Embedder()
    texts = [chunk.text for chunk in all_chunks]
    embeddings = embedder.embed_documents(texts)
    
    # 3. Build vector store
    vector_store = FAISSVectorStore()
    vector_store.create_index(dimension=embedder.dimension)
    vector_store.add_chunks(all_chunks, embeddings)
    vector_store.save()
    
    # 4. Initialize RAG generator
    config = LLMConfig(api_key=SecretStr(api_key))
    rag = RAGGenerator(vector_store, embedder, config)
    
    return rag

# Usage
api_key = os.getenv("GOOGLE_API_KEY")
rag = build_rag_system(Path("pdfs"), api_key)

# Ask questions
response = rag.generate_answer("¿Qué se decidió en la reunión?")
print(response.formatted_response)
```

### Access Response Components
```python
response = rag.generate_answer(question)

# Raw answer
print(response.answer)

# Citations
for cite in response.citations:
    print(f"Source: {cite.document}, p.{cite.page}")
    print(f"Number of chunks: {cite.num_chunks}")
    print(f"Best score: {cite.best_score:.3f}")
    
    # Access individual chunks
    for chunk_result in cite.chunks:
        print(f"  Chunk {chunk_result.chunk.chunk_index}: {chunk_result.score:.3f}")
        print(f"  Text: {chunk_result.chunk.text[:100]}...")

# Metadata
print(f"Retrieved {response.num_chunks_retrieved} chunks")
print(f"Has citations: {response.has_citations}")
print(f"Used re-ranking: {response.used_reranking}")

# Formatted output
print(response.formatted_response)
```

### Inspect Citations (Debugging Utility)
```python
from rag_system import inspect_citations

response = rag.generate_answer(question)

# Preview mode (truncated text)
inspect_citations(response, show_full_text=False)

# Full text mode (for verification)
inspect_citations(response, show_full_text=True)
```

**Output includes:**
- ❓ Original question
- 💬 Generated answer with inline citations
- 📊 Metadata (chunks retrieved, reranking used)
- 📚 Per-citation details:
  - Document and page
  - Number of chunks from that page
  - Best retrieval and rerank scores
  - Individual chunk details with scores
  - Full or preview text

**Use cases:**
- Debug citation extraction
- Verify chunk grouping by page
- Compare scores before/after reranking
- Validate answer grounding

### JSON Serialization (for API)
```python
# Serialize response to JSON
json_output = response.model_dump_json(indent=2)

# Or as dictionary
dict_output = response.model_dump()

# Example output:
# {
#   "question": "¿Qué presupuesto se aprobó?",
#   "answer": "El presupuesto aprobado fue de 5 millones [doc.pdf, p.3]",
#   "citations": [
#     {
#       "document": "doc.pdf",
#       "page": 3,
#       "text_snippet": "El presupuesto aprobado fue de 5 millones..."
#     }
#   ],
#   "num_chunks_retrieved": 5,
#   "has_citations": true
# }
```

## Prompt Engineering

### System Prompt Design
The system prompt has three critical goals:

1. **Grounding**: Force LLM to use only provided context
   - "ÚNICAMENTE en el contexto proporcionado"
   - "No inventes información"

2. **Citation enforcement**: Require source attribution
   - "SIEMPRE incluye al menos una cita"
   - Format: `[documento, p.X]`

3. **Conciseness**: Prevent rambling responses
   - "2-4 oraciones"
   - "Sé conciso y directo"

### Context Format Strategy
```
Fragmento 1 [documento.pdf, p.3]:
El presupuesto aprobado fue...

Fragmento 2 [documento.pdf, p.5]:
La reunión se celebró...
```

**Why this format?**
- **Numbered fragments**: Help LLM track multiple sources
- **Citation format shown**: LLM learns format from examples
- **Clear separation**: Each chunk is distinct

### Handling No Context
```python
if not results:
    return LLMResponse(
        question=question,
        answer="No puedo responder esta pregunta con la información disponible.",
        citations=[],
        num_chunks_retrieved=0,
        has_citations=False
    )
```

Better to return explicit "no info" than hallucinate.

## Citation Extraction Strategy

### Pattern Matching
```python
citation_ref = f"[{chunk.document}, p.{chunk.page}]"

if citation_ref in answer:
    # This chunk was cited
    citations.append(...)
```

**Why this works:**
- LLM is prompted to use exact format `[file.pdf, p.N]`
- We format context chunks with same pattern
- Simple string matching finds citations

**Alternative (more robust):**
```python
import re

# Extract all citations from answer
pattern = r'\[([^,]+),\s*p\.(\d+)\]'
matches = re.findall(pattern, answer)

for doc, page in matches:
    # Find corresponding chunk
    matching_chunks = [
        r for r in results 
        if r.chunk.document == doc and r.chunk.page == int(page)
    ]
    if matching_chunks:
        citations.append(...)
```

Use regex version for more reliable extraction.

## Performance

### Latency Breakdown
For typical question (~20 words):
- **Embed query**: 10-30ms
- **Vector search**: 1-5ms
- **Format context**: <1ms
- **LLM call**: 1,000-3,000ms ← bottleneck
- **Extract citations**: <1ms
- **Total**: ~1-3 seconds

### Cost (Gemini Free Tier)
- **Limits**: 15 requests/min, 1M/day
- **Cost**: $0 (free)
- **For demo**: ~50-100 eval questions ← well within limits

### Optimization Tips
```python
# 1. Use flash model (faster, cheaper)
config = LLMConfig(
    model_name="gemini-2.0-flash-001",  # Not "gemini-2.0-flash-001"
    max_output_tokens=512           # Shorter = faster
)

# 2. Reduce retrieval
config = LLMConfig(top_k=3)  # Fewer chunks = less context

# 3. Batch questions (reuse embedder/vector_store)
responses = rag.batch_generate(questions)  # Don't recreate RAG each time
```

## Error Handling

### API Key Errors
```python
from google.api_core.exceptions import Unauthenticated

try:
    rag = RAGGenerator(vector_store, embedder, config)
    response = rag.generate_answer(question)
except Unauthenticated:
    print("Invalid API key. Get one at: https://makersuite.google.com/app/apikey")
```

### Rate Limiting
```python
from google.api_core.exceptions import ResourceExhausted
import time

def generate_with_retry(rag: RAGGenerator, question: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return rag.generate_answer(question)
        except ResourceExhausted:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
```

### No Results Handling
Already handled in `generate_answer()`:
```python
if not results:
    return LLMResponse(
        question=question,
        answer="No puedo responder...",
        citations=[],
        num_chunks_retrieved=0,
        has_citations=False
    )
```

## Testing LLM Output

### Basic Quality Test
```python
response = rag.generate_answer("¿Qué presupuesto se aprobó?")

# Check answer generated
assert response.answer != ""
assert len(response.answer) > 20

# Check citations present
assert response.has_citations
assert len(response.citations) > 0

# Check citation format
assert "[" in response.answer
assert "p." in response.answer
```

### Citation Validity Test
```python
def validate_citations(response: LLMResponse) -> bool:
    """Check all citations are valid."""
    for cite in response.citations:
        # Check citation in answer
        ref = f"[{cite.document}, p.{cite.page}]"
        if ref not in response.answer:
            return False
        
        # Check snippet is reasonable
        if len(cite.text_snippet) < 10:
            return False
    
    return True

assert validate_citations(response)
```

### Grounding Test (No Hallucination)
```python
# Ask question NOT in corpus
response = rag.generate_answer("¿Cuál es la capital de Francia?")

# Should respond with "no info" not hallucinate
assert "no tengo información" in response.answer.lower() or \
       "no puedo responder" in response.answer.lower()
```

## Example Responses

### Good Response
```
Q: ¿Qué presupuesto se aprobó para el proyecto urbano?

A: El presupuesto aprobado fue de 5 millones de euros para el 
proyecto de renovación urbana en el distrito centro 
[ResumenReunionGA_20231124.pdf, p.3]. Este presupuesto incluye 
inversiones en infraestructura y espacios públicos 
[ResumenReunionGA_20231124.pdf, p.4].

Sources (2):
1. ResumenReunionGA_20231124.pdf, p.3
   "El presupuesto aprobado fue de 5 millones de euros para el 
   proyecto de renovación urbana..."

2. ResumenReunionGA_20231124.pdf, p.4
   "Este presupuesto incluye inversiones en infraestructura y 
   espacios públicos del distrito centro..."
```

**Quality indicators:**
✅ Direct answer to question  
✅ Uses only context info  
✅ Cites sources inline  
✅ Concise (2 sentences)  
✅ Multiple citations

### Bad Response (No Citations)
```
Q: ¿Qué presupuesto se aprobó?

A: Se aprobó un presupuesto para un proyecto urbano de renovación.

Sources (0):
*No sources cited*
```

**Issues:**
❌ No citation  
❌ Vague answer  
❌ Doesn't mention specific amount

### Bad Response (Hallucination)
```
Q: ¿Cuándo comenzará el proyecto?

A: El proyecto comenzará en enero de 2024 y se completará 
en diciembre de 2025.

Sources (0):
*No sources cited*
```

**Issues:**
❌ Invented dates not in context  
❌ No citations (red flag)  
❌ Should say "no tengo información"

## Dependencies

**Core Libraries:**
- `langchain` + `langchain-google-genai` — LLM integration and message handling
- `google-generativeai` — Gemini API client
- `pydantic` — Data validation and configuration
- Internal: `rag_system.vector_store`, `rag_system.embeddings`, `rag_system.reranker`

## Getting Google API Key

1. Go to: https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy key
4. Configure API key:

**Option 1: .env file (Recommended)**
```bash
# Create .env file in project root
GOOGLE_API_KEY=your-api-key-here
```

**Option 2: Environment variable**
```bash
# Linux/Mac
export GOOGLE_API_KEY="your-api-key-here"

# Windows PowerShell
$env:GOOGLE_API_KEY="your-api-key-here"
```

**Free tier limits:**
- 15 requests per minute
- 1 million requests per day
- No credit card required

Sufficient for demo and evaluation (~50-100 questions).
