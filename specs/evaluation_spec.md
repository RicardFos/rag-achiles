# Evaluation Specification

## Goal
Evaluate the RAG system's performance on a test set with quantifiable metrics for answer quality and citation accuracy.

## Purpose
- **Measure quality**: Quantify how well the system answers questions
- **Validate citations**: Ensure answers are grounded in source documents
- **Compare approaches**: Benchmark with/without re-ranking
- **Identify weaknesses**: Find edge cases where the system fails

## Metrics to Implement

### 1. Citation Accuracy
**Definition**: Percentage of responses that include at least one valid citation.

**Formula**: `(Responses with citations / Total responses) × 100%`

### 2. Source Precision
**Definition**: Of the documents cited by the system, what percentage are correct (match expected sources)?

**Formula**: `Correct citations / Total citations`

**Calculation per question**:
- Compare (document, page) pairs between expected and generated
- `correct = expected_sources ∩ generated_sources`
- `precision = |correct| / |generated_sources|`

### 3. Source Recall
**Definition**: Of the expected source documents, what percentage were cited by the system?

**Formula**: `Correct citations / Expected sources`

**Calculation per question**:
- `correct = expected_sources ∩ generated_sources`
- `recall = |correct| / |expected_sources|`

### 4. Average Chunks Retrieved
**Definition**: Average number of relevant chunks retrieved per question.

**Formula**: `Σ(chunks_per_question) / Total questions`


## Test Set: eval.jsonl

### Format
JSON Lines format (one JSON object per line) with the following fields:

**`question`** (string)
- Natural language question in Spanish
- Based on actual document content

**`expected_answer`** (string)
- Ground truth answer (1-3 sentences)
- Human-written reference answer

**`source_passages`** (array of objects)
- Expected source documents and pages
- Each passage contains:
  - `document`: PDF filename
  - `page`: Page number (1-indexed)
  - `text`: Relevant text excerpt from source

### Test Set Guidelines

**Size**: ~10-15 question-answer pairs

**Question Types to Include**:
- Factual questions (dates, numbers, names)
- List questions (enumerations, phases)
- Conceptual questions (definitions, descriptions)
- Multi-source questions (require synthesizing multiple chunks)

**Difficulty Distribution**:
- Easy (30%): Single fact from one chunk
- Medium (50%): Requires understanding context
- Hard (20%): Synthesize across multiple sources


## Implementation Guidelines

### Required Functions

**`load_eval_data(eval_file: Path) -> list`**
- Load evaluation dataset from JSONL file
- Return list of dicts with question, expected_answer, source_passages

**`run_evaluation(rag: RAGGenerator, eval_data: list) -> list[dict]`**
- Run RAG system on all evaluation questions
- Return list of result dicts containing:
  - question, expected_answer, generated_answer
  - expected_sources, generated_citations
  - has_citations, num_chunks_retrieved

**`compute_citation_accuracy(results: list) -> float`**
- Calculate percentage of responses with citations
- Return percentage (0-100)

**`compute_source_metrics(expected_sources: list, generated_citations: list) -> tuple[float, float]`**
- Compare (document, page) pairs between expected and generated
- Return (precision, recall) as floats in [0, 1] range

**`compute_all_metrics(results: list) -> dict`**
- Aggregate all metrics across evaluation examples
- Return dict with citation_accuracy, source_precision, source_recall, avg_chunks_retrieved

## Evaluation Notebook Structure

Implementation should be in [03_llm_generation_and_eval.ipynb](../03_llm_generation_and_eval.ipynb):

1. **Load evaluation data** from eval.jsonl
2. **Run RAG system** on all questions
3. **Compute metrics** (citation accuracy, precision, recall, avg chunks)
4. **Display results** with examples
5. **Optional**: Compare with/without re-ranking

## Best Practices for Test Sets

**DO**:
- Base questions on actual document content
- Include diverse question types (factual, list, conceptual)
- Mark expected sources explicitly
- Use natural language (how users actually ask)
- Cover both easy and hard questions

**DON'T**:
- Make questions too ambiguous
- Ask about content not in documents
- Use overly technical jargon
- Create trick questions
- Expect verbatim answers

## Troubleshooting Metrics

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
