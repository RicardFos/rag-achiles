# Evaluation Specification

## Goal
Evaluate the RAG system's performance on a test set with quantifiable metrics for answer quality and citation accuracy.

## Why Evaluation?

### Purpose
- **Measure quality**: Quantify how well the system answers questions
- **Validate citations**: Ensure answers are grounded in source documents
- **Compare approaches**: Benchmark with/without re-ranking
- **Identify weaknesses**: Find edge cases where the system fails
- **Assignment requirement**: "Métricas eval.jsonl a nivel de respuesta y % de respuestas con citas válidas"

### What We're Measuring
1. **Citation Accuracy**: Does the system provide source citations?
2. **Source Precision**: Are the cited documents correct?
3. **Source Recall**: Does it cite all relevant sources?
4. **Retrieval Effectiveness**: Average chunks retrieved per query

## Test Set: eval.jsonl

### Format
JSON Lines format (one JSON object per line):

```json
{
  "question": "¿Cuándo tuvo lugar la primera reunión del grupo motor?",
  "expected_answer": "La primera reunión del grupo motor tuvo lugar el 24 de noviembre de 2023.",
  "source_passages": [
    {
      "document": "ResumenReunionGA_20231124.pdf",
      "page": 1,
      "text": "El día 24 de noviembre de 2023 tiene lugar la primera reunión..."
    }
  ]
}
```

### Fields

**`question`** (string)
- Natural language question in Spanish
- Based on actual document content
- Various types: factual, multi-source, complex

**`expected_answer`** (string)
- Ground truth answer
- 1-3 sentences
- Human-written reference answer

**`source_passages`** (array of objects)
- Expected source documents and pages
- Each passage contains:
  - `document`: PDF filename
  - `page`: Page number (1-indexed)
  - `text`: Relevant text excerpt from source

### Test Set Characteristics

**Size**: 10 question-answer pairs

**Coverage**:
- Factual questions (dates, numbers, names)
- List questions (enumerations, phases)
- Conceptual questions (definitions, descriptions)
- Multi-source questions (require synthesizing multiple chunks)

**Difficulty Distribution**:
- Easy (30%): Single fact from one chunk
- Medium (50%): Requires understanding context
- Hard (20%): Synthesize across multiple sources

### Example Questions

```jsonl
{"question": "¿Cuándo tuvo lugar la primera reunión del grupo motor para el cuarto plan de gobierno abierto?", "expected_answer": "La primera reunión del grupo motor tuvo lugar el 24 de noviembre de 2023.", "source_passages": [{"document": "ResumenReunionGA_20231124.pdf", "page": 1, "text": "El día 24 de noviembre de 2023 tiene lugar la primera reunión del grupo motor para el diseño del cuarto plan de gobierno abierto del Ayuntamiento de Madrid"}]}
{"question": "¿Qué composición tiene el grupo motor?", "expected_answer": "El grupo motor tiene una composición paritaria con 10 representantes de la sociedad civil y 10 representantes del Ayuntamiento.", "source_passages": [{"document": "ResumenReunionGA_20231124.pdf", "page": 1, "text": "El grupo motor tiene una composición paritaria entre representantes de la sociedad civil y del ayuntamiento... Cuenta con 10 representantes de la sociedad civil y 10 del Ayuntamiento"}]}
{"question": "¿Cuáles son las fases del proceso de elaboración del cuarto plan?", "expected_answer": "El proceso consta de 6 fases: Fase 0 (establecimiento del plan de trabajo), Fase 1 (consulta pública previa), Fase 2 (debate y co-creación del borrador), Fase 3 (consulta pública sobre el borrador de compromisos), Fase 4 (análisis de propuestas y redacción), y Fase 5 (aprobación del Plan).", "source_passages": [{"document": "ResumenReunionGA_20231124.pdf", "page": 2, "text": "Fase 0: Establecimiento del plan de trabajo, Fase 1: Consulta pública previa, Fase 2: Debate y co-creación del borrador el IV Plan, Fase 3: Consulta pública sobre el borrador de compromisos, Fase 4: Análisis de las propuestas y redacción del Plan, Fase 5: Aprobación del Plan"}]}
```

## Evaluation Metrics

### 1. Citation Accuracy
**Definition**: Percentage of responses that include at least one valid citation.

**Formula**:
```
Citation Accuracy = (Responses with citations / Total responses) × 100%
```

**What it measures**: Whether the system grounds its answers in sources (not hallucinating).

**Target**: ≥90%

**Example**:
- 10 questions, 9 have citations → 90% citation accuracy
- If 10/10 have citations → 100% ✅

### 2. Source Precision
**Definition**: Of the documents cited by the system, what percentage are correct (match expected sources)?

**Formula**:
```
Precision = Correct citations / Total citations
```

**Per-question calculation**:
```python
expected_sources = {(doc, page) for doc, page in expected_sources}
generated_sources = {(doc, page) for doc, page in generated_citations}

correct = expected_sources ∩ generated_sources
precision = |correct| / |generated_sources|
```

**What it measures**: Citation relevance (are we citing the right documents?).

**Target**: ≥70%

**Example**:
- Expected sources: `{(doc1.pdf, 1), (doc1.pdf, 2)}`
- Generated citations: `{(doc1.pdf, 1), (doc2.pdf, 3)}`
- Correct: `{(doc1.pdf, 1)}`
- Precision: 1/2 = 50%

### 3. Source Recall
**Definition**: Of the expected source documents, what percentage were cited by the system?

**Formula**:
```
Recall = Correct citations / Expected sources
```

**Per-question calculation**:
```python
correct = expected_sources ∩ generated_sources
recall = |correct| / |expected_sources|
```

**What it measures**: Citation completeness (are we finding all relevant sources?).

**Target**: ≥60%

**Example**:
- Expected sources: `{(doc1.pdf, 1), (doc1.pdf, 2)}`
- Generated citations: `{(doc1.pdf, 1), (doc2.pdf, 3)}`
- Correct: `{(doc1.pdf, 1)}`
- Recall: 1/2 = 50%

### 4. Average Chunks Retrieved
**Definition**: Average number of relevant chunks retrieved per question.

**Formula**:
```
Avg Chunks = Σ(chunks_per_question) / Total questions
```

**What it measures**: Retrieval effectiveness (are we finding relevant context?).

**Target**: 3-5 chunks (with re-ranking)

## Implementation

### API Reference

```python
def load_eval_data(eval_file: Path) -> list:
    """
    Load evaluation dataset from JSONL file.
    
    Reads JSON Lines format where each line is a complete evaluation example
    with question, expected answer, and source passages.
    
    Args:
        eval_file: Path to eval.jsonl
        
    Returns:
        List of evaluation examples, each a dict with:
        - question: User question string
        - expected_answer: Ground truth answer
        - source_passages: List of dicts with document, page, text
        
    Example:
        >>> eval_data = load_eval_data(Path("eval.jsonl"))
        Loaded 10 evaluation examples
    """
```

```python
def run_evaluation(
    rag: RAGGenerator,
    eval_data: list
) -> List[dict]:
    """
    Run RAG system on evaluation dataset.
    
    Processes each question through the full RAG pipeline and collects
    results for metric computation. Preserves both expected and generated
    values for comparison.
    
    Args:
        rag: Initialized RAG generator
        eval_data: List of evaluation examples from load_eval_data
        
    Returns:
        List of result dicts, each containing:
        - question: Original question
        - expected_answer: Ground truth answer
        - generated_answer: RAG system answer
        - expected_sources: Ground truth source passages
        - generated_citations: Citations from RAG response
        - has_citations: Boolean citation presence flag
        - num_chunks_retrieved: Retrieval count
        
    Example:
        >>> results = run_evaluation(rag, eval_data)
        ✓ Evaluation complete: 10 questions
    """
```

```python
def compute_citation_accuracy(results: list) -> float:
    """
    Calculate percentage of responses with citations.
    
    Args:
        results: List of evaluation results from run_evaluation
        
    Returns:
        Citation accuracy as percentage (0-100)
        
    Example:
        >>> accuracy = compute_citation_accuracy(results)
        >>> print(f"Citation Accuracy: {accuracy:.1f}%")
    """

def compute_source_metrics(
    expected_sources: list,
    generated_citations: list
) -> tuple[float, float]:
    """
    Compute precision and recall for source citations.
    
    Compares (document, page) pairs between expected and generated citations.
    Returns both metrics as floats in [0, 1] range.
    
    Args:
        expected_sources: List of ground truth source passages
        generated_citations: List of RAG system citations
    
    Returns:
        (precision, recall) tuple:
        - precision: Correct citations / Total generated citations
        - recall: Correct citations / Total expected sources
        
    Note:
        Returns (0.0, 0.0) if no citations generated.
        
    Example:
        >>> prec, rec = compute_source_metrics(expected, generated)
        >>> print(f"Precision: {prec:.1%}, Recall: {rec:.1%}")
    """

def compute_all_metrics(results: list) -> dict:
    """
    Compute all evaluation metrics from results.
    
    Aggregates citation accuracy, source precision/recall, and average
    chunk retrieval count across all evaluation examples.
    
    Args:
        results: List of evaluation results from run_evaluation
        
    Returns:
        Dictionary with metrics:
        - citation_accuracy: Percentage with citations (0-100)
        - source_precision: Average precision (0-100)
        - source_recall: Average recall (0-100)
        - avg_chunks_retrieved: Mean chunks per query
        
    Example:
        >>> metrics = compute_all_metrics(results)
        Citation Accuracy: 90.0%
        Source Precision: 75.0%
        Source Recall: 65.0%
        Avg Chunks Retrieved: 4.5
    """
```

## Expected Results

Based on the corpus (9 PDFs, 619 chunks) and test set (10 questions):

| Metric | Target | Expected Range | Notes |
|--------|--------|----------------|-------|
| **Citation Accuracy** | ≥90% | 90-100% | System should consistently cite sources |
| **Source Precision** | ≥70% | 70-90% | Most citations should be correct |
| **Source Recall** | ≥60% | 60-80% | Should capture majority of relevant sources |
| **Avg Chunks Retrieved** | 3-5 | 4-5 | With re-ranking enabled |

### Interpreting Results

**High Citation Accuracy (90%+)**
- ✅ System grounds answers in sources
- ✅ Not hallucinating
- ✅ LLM follows prompt instructions

**High Source Precision (80%+)**
- ✅ Retrieval + re-ranking working well
- ✅ Cited documents are relevant
- ✅ Good bi-encoder + cross-encoder combination

**Lower Source Recall (60-70%)**
- ⚠️ May miss some relevant sources
- ℹ️ Often acceptable: LLM can answer from subset of sources
- ℹ️ Multi-source questions are harder

**Avg Chunks 4-5**
- ✅ Re-ranking reducing from 20 → 5
- ✅ Providing sufficient context to LLM
- ✅ Not overwhelming LLM with irrelevant chunks

## Evaluation in Notebook

Implementation in [03_llm_generation_and_eval.ipynb](../03_llm_generation_and_eval.ipynb):

### Structure
1. **Load evaluation data** from eval.jsonl
2. **Run RAG system** on all questions
3. **Compute metrics** (citation accuracy, precision, recall)
4. **Display results** with examples
5. **Analyze failures** (questions with low scores)

### Output Format
```
==============================================
EVALUATION METRICS
==============================================

1. Citation Accuracy: 90.0%
   (9/10 responses have citations)

2. Source Precision: 75.0%
   (Average % of cited documents matching expected sources)

3. Source Recall: 65.0%
   (Average % of expected sources that were cited)

4. Average Chunks Retrieved: 4.5
   (Average number of chunks retrieved per question)

==============================================
EXAMPLE RESULTS
==============================================

Example 1
📝 Question: ¿Cuándo tuvo lugar la primera reunión del grupo motor?
✅ Expected: La primera reunión tuvo lugar el 24 de noviembre de 2023.
🤖 Generated: Según el documento [ResumenReunionGA_20231124.pdf, p.1], la primera reunión del grupo motor tuvo lugar el 24 de noviembre de 2023.
📚 Citations: 1
   - ResumenReunionGA_20231124.pdf, p.1
📊 Source Metrics: Precision: 100% | Recall: 100%
```

## Comparison: With vs Without Re-ranking

### Methodology
Run evaluation twice:
1. With re-ranking enabled (`use_reranking=True`)
2. Without re-ranking (`use_reranking=False`)

Compare metrics to quantify re-ranking impact.

### Expected Differences

| Metric | Without Re-ranking | With Re-ranking | Improvement |
|--------|-------------------|-----------------|-------------|
| Citation Accuracy | 80-90% | 90-100% | +10% |
| Source Precision | 60-70% | 70-90% | +15% |
| Source Recall | 50-60% | 60-80% | +15% |
| Avg Chunks | 5 | 5 | Same |

**Key insight**: Re-ranking improves **which** documents are retrieved (precision/recall) without changing the number of documents (top-5 in both cases).

## Alternative Metrics (Not Implemented)

### ROUGE-L
Measures lexical overlap between generated and expected answers.

**Why not used**: 
- Language-specific (Spanish ROUGE less reliable)
- Doesn't capture semantic similarity
- Focus on citations, not exact wording

### BERTScore
Measures semantic similarity using contextualized embeddings.

**Why not used**:
- Requires additional model loading
- Computational overhead
- Assignment focuses on citation accuracy

### Exact Match
Binary: does generated answer exactly match expected?

**Why not used**:
- Too strict for free-form generation
- Equivalent answers may differ in wording
- Not practical for LLM evaluation

### Human Evaluation
Manual scoring of answer quality.

**Why not used**:
- Time-consuming
- Not scalable
- Automated metrics sufficient for assignment

## Best Practices

### Creating Good Test Sets

**DO**:
✅ Base questions on actual document content
✅ Include diverse question types (factual, list, conceptual)
✅ Mark expected sources explicitly
✅ Use natural language (how users actually ask)
✅ Cover both easy and hard questions

**DON'T**:
❌ Make questions too ambiguous
❌ Ask about content not in documents
❌ Use overly technical jargon
❌ Create trick questions
❌ Expect verbatim answers

### Interpreting Low Scores

**Low Citation Accuracy (<80%)**
- Check LLM prompt engineering
- Verify citation extraction logic
- Ensure retrieval is finding relevant chunks

**Low Source Precision (<60%)**
- Review bi-encoder quality
- Check similarity threshold (too low?)
- Tune re-ranking if enabled

**Low Source Recall (<50%)**
- Increase candidate pool (retrieve top-30 instead of top-20)
- Lower similarity threshold
- Review chunking strategy (chunks too small/large?)

## Usage Examples

### Run Full Evaluation

```bash
# Ensure API key is configured in .env file:
# GOOGLE_API_KEY=your-api-key

# Run evaluation notebook
jupyter notebook 03_llm_generation_and_eval.ipynb
```

### Programmatic Evaluation

```python
from pathlib import Path
import json
import os
from dotenv import load_dotenv
from rag_system import RAGGenerator, LLMConfig, FAISSVectorStore, Embedder
from pydantic import SecretStr

def evaluate_rag_system(eval_file: Path, use_reranking: bool = True) -> dict:
    """
    Run full evaluation on RAG system.
    
    Returns:
        Dictionary with metrics
    """
    # Load API key from .env
    load_dotenv()
    
    # Load data
    eval_data = []
    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            eval_data.append(json.loads(line))
    
    # Initialize RAG
    vector_store = FAISSVectorStore()
    embedder = Embedder()
    config = LLMConfig(api_key=SecretStr(os.getenv("GOOGLE_API_KEY")))
    
    rag = RAGGenerator(
        vector_store=vector_store,
        embedder=embedder,
        config=config,
        use_reranking=use_reranking
    )
    
    # Run evaluation
    results = []
    for item in eval_data:
        response = rag.generate_answer(item['question'])
        results.append({
            'has_citations': response.has_citations,
            'expected_sources': item['source_passages'],
            'generated_citations': [
                {'document': c.document, 'page': c.page}
                for c in response.citations
            ],
            'num_chunks': response.num_chunks_retrieved
        })
    
    # Compute metrics
    citation_accuracy = sum(1 for r in results if r['has_citations']) / len(results) * 100
    
    precisions, recalls = [], []
    for r in results:
        prec, rec = compute_source_metrics(r['expected_sources'], r['generated_citations'])
        precisions.append(prec)
        recalls.append(rec)
    
    return {
        'citation_accuracy': citation_accuracy,
        'source_precision': sum(precisions) / len(precisions) * 100,
        'source_recall': sum(recalls) / len(recalls) * 100,
        'avg_chunks': sum(r['num_chunks'] for r in results) / len(results),
        'num_questions': len(results)
    }

# Usage
metrics = evaluate_rag_system(Path("eval.jsonl"), use_reranking=True)
print(f"Citation Accuracy: {metrics['citation_accuracy']:.1f}%")
print(f"Source Precision: {metrics['source_precision']:.1f}%")
print(f"Source Recall: {metrics['source_recall']:.1f}%")
```

## Dependencies

No additional dependencies required beyond core RAG system:
```txt
# Already included in requirements.txt
pydantic>=2.0.0  # For data models
numpy>=1.24.0    # For metric calculations
```

Evaluation runs entirely in Python with JSON standard library.

## Files

### Core Files
- **eval.jsonl**: Test set with 10 Q&A pairs
- **03_llm_generation_and_eval.ipynb**: Evaluation notebook

### Generated During Evaluation
- None (results displayed in notebook, not persisted)

### Optional
Could extend to save evaluation results:
```python
# Save results to JSON
with open('evaluation_results.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

## Assignment Requirements Met

- ✅ **eval.jsonl**: ~10 preguntas y respuestas esperadas + pasajes fuente
- ✅ **Métricas a nivel de respuesta**: Citation accuracy, source precision/recall
- ✅ **% de respuestas con citas válidas**: Citation accuracy metric
- ✅ **Evaluación automatizada**: Implemented in notebook
- ✅ **Comparación con/sin re-ranking**: Methodology documented

## Troubleshooting

### Issue: All metrics are 0%
**Cause**: RAG system not retrieving or LLM not generating citations  
**Solution**: Check vector store loaded, API key set, retrieval working

### Issue: Low citation accuracy (<50%)
**Cause**: LLM not following citation format instructions  
**Solution**: Review prompt engineering in [specs/llm_spec.md](llm_spec.md)

### Issue: Low precision but high recall
**Cause**: Over-retrieving, citing too many documents  
**Solution**: Increase similarity threshold or improve re-ranking

### Issue: High precision but low recall
**Cause**: Under-retrieving, missing relevant sources  
**Solution**: Decrease similarity threshold or retrieve more candidates (top-30)

### Issue: Evaluation too slow
**Cause**: LLM API calls take time (~2s each)  
**Solution**: Expected for 10 questions (~20-30s total), can't parallelize with rate limits

## Future Enhancements

1. **Automatic answer scoring**: Use LLM-as-judge to score semantic similarity
2. **Error analysis**: Categorize failure modes (retrieval vs generation)
3. **Per-question difficulty**: Track metrics by question complexity
4. **Temporal evaluation**: Track metrics over time as system evolves
5. **A/B testing**: Compare different configurations systematically
6. **Cost tracking**: Log API costs per evaluation run
7. **Human validation**: Sample manual review of automated scores
