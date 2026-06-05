"""
Evaluation utilities for RAG system.

Provides metrics for assessing answer quality and citation accuracy.
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import util

from rag_system.embeddings import Embedder
from rag_system.llm import LLMResponse


def load_eval_data(eval_file: Path) -> List[Dict]:
    """
    Load evaluation dataset from JSONL file.

    Args:
        eval_file: Path to eval.jsonl file

    Returns:
        List of evaluation examples with question, expected_answer, source_passages
    """
    eval_data = []
    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            eval_data.append(json.loads(line))
    return eval_data


def compute_citation_accuracy(results: List[Dict]) -> float:
    """
    Calculate percentage of responses that include valid citations.

    Args:
        results: List of result dictionaries with 'has_citations' field

    Returns:
        Citation accuracy as percentage (0-100)
    """
    responses_with_citations = sum(1 for r in results if r.get('has_citations', False))
    return (responses_with_citations / len(results)) * 100 if results else 0.0


def compute_source_metrics(
    expected_sources: List[Dict],
    generated_citations: List[Dict]
) -> Tuple[float, float]:
    """
    Compute precision and recall for source citations.

    Precision: % of cited documents that are correct
    Recall: % of expected documents that were cited

    Args:
        expected_sources: List of dicts with 'document' and 'page' keys
        generated_citations: List of dicts with 'document' and 'page' keys

    Returns:
        (precision, recall) as floats in [0, 1]
    """
    if not generated_citations:
        return 0.0, 0.0

    # Extract document-page pairs
    expected_pairs = {
        (src['document'], src['page'])
        for src in expected_sources
    }

    generated_pairs = {
        (cite['document'], cite['page'])
        for cite in generated_citations
    }

    # Compute overlap
    correct_citations = expected_pairs & generated_pairs

    precision = len(correct_citations) / len(generated_pairs) if generated_pairs else 0.0
    recall = len(correct_citations) / len(expected_pairs) if expected_pairs else 0.0

    return precision, recall


def compute_answer_similarity(
    expected_answer: str,
    generated_answer: str,
    embedder: Embedder
) -> float:
    """
    Compute semantic similarity between expected and generated answers.

    Uses cosine similarity of sentence embeddings to measure how close
    the generated answer is to the expected answer semantically.

    Args:
        expected_answer: Ground truth answer
        generated_answer: System-generated answer
        embedder: Embedder instance for computing embeddings

    Returns:
        Similarity score in [0, 1] (1 = identical semantically)
    """
    # Embed both answers
    exp_embedding = embedder.embed_query(expected_answer)
    gen_embedding = embedder.embed_query(generated_answer)

    # Compute cosine similarity
    similarity = util.cos_sim(exp_embedding, gen_embedding).item()

    # Ensure in [0, 1] range (cosine similarity can be [-1, 1])
    return max(0.0, similarity)


def evaluate_rag_system(
    eval_data: List[Dict],
    responses: List[LLMResponse],
    embedder: Embedder
) -> Dict[str, float]:
    """
    Comprehensive evaluation of RAG system responses.

    Computes all metrics:
    - Citation accuracy
    - Source precision
    - Source recall
    - Answer semantic similarity
    - Average chunks retrieved

    Args:
        eval_data: List of evaluation examples from eval.jsonl
        responses: List of LLMResponse objects from RAG system
        embedder: Embedder instance for computing answer similarity

    Returns:
        Dictionary with all metrics as percentages (except avg_chunks)
    """
    if len(eval_data) != len(responses):
        raise ValueError(f"Mismatch: {len(eval_data)} questions but {len(responses)} responses")

    # Prepare results for citation metrics
    results = []
    for item, response in zip(eval_data, responses):
        results.append({
            'has_citations': response.has_citations,
            'expected_sources': item['source_passages'],
            'generated_citations': [
                {'document': c.document, 'page': c.page}
                for c in response.citations
            ],
            'num_chunks': response.num_chunks_retrieved,
            'expected_answer': item['expected_answer'],
            'generated_answer': response.answer
        })

    # 1. Citation accuracy
    citation_accuracy = compute_citation_accuracy(results)

    # 2. Source precision and recall
    precisions = []
    recalls = []
    for r in results:
        prec, rec = compute_source_metrics(
            r['expected_sources'],
            r['generated_citations']
        )
        precisions.append(prec)
        recalls.append(rec)

    avg_precision = (sum(precisions) / len(precisions)) * 100 if precisions else 0.0
    avg_recall = (sum(recalls) / len(recalls)) * 100 if recalls else 0.0

    # 3. Answer similarity
    similarities = []
    for r in results:
        sim = compute_answer_similarity(
            r['expected_answer'],
            r['generated_answer'],
            embedder
        )
        similarities.append(sim)

    avg_similarity = (sum(similarities) / len(similarities)) * 100 if similarities else 0.0

    # 4. Average chunks retrieved
    avg_chunks = sum(r['num_chunks'] for r in results) / len(results) if results else 0.0

    return {
        'citation_accuracy': citation_accuracy,
        'source_precision': avg_precision,
        'source_recall': avg_recall,
        'answer_similarity': avg_similarity,
        'avg_chunks_retrieved': avg_chunks,
        'num_questions': len(results)
    }


def print_evaluation_results(metrics: Dict[str, float]) -> None:
    """
    Pretty-print evaluation metrics.

    Args:
        metrics: Dictionary returned by evaluate_rag_system()
    """
    print("=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)
    print(f"\n1. Citation Accuracy: {metrics['citation_accuracy']:.1f}%")
    print(f"   ({int(metrics['citation_accuracy'] * metrics['num_questions'] / 100)}/{metrics['num_questions']} responses have citations)")

    print(f"\n2. Source Precision: {metrics['source_precision']:.1f}%")
    print(f"   (Average % of cited documents that match expected sources)")

    print(f"\n3. Source Recall: {metrics['source_recall']:.1f}%")
    print(f"   (Average % of expected sources that were cited)")

    print(f"\n4. Answer Similarity: {metrics['answer_similarity']:.1f}%")
    print(f"   (Semantic similarity between generated and expected answers)")

    print(f"\n5. Average Chunks Retrieved: {metrics['avg_chunks_retrieved']:.1f}")
    print(f"   (Average number of chunks retrieved per question)")
    print()


def analyze_result(
    question: str,
    expected_answer: str,
    generated_answer: str,
    expected_sources: List[Dict],
    generated_citations: List[Dict],
    embedder: Embedder
) -> None:
    """
    Print detailed analysis of a single evaluation result.

    Args:
        question: The question
        expected_answer: Ground truth answer
        generated_answer: System-generated answer
        expected_sources: Expected source passages
        generated_citations: Generated citations
        embedder: Embedder for computing similarity
    """
    print(f"\n📝 Question:")
    print(f"   {question}")

    print(f"\n✅ Expected Answer:")
    print(f"   {expected_answer}")

    print(f"\n🤖 Generated Answer:")
    print(f"   {generated_answer}")

    print(f"\n📚 Citations: {len(generated_citations)}")
    for cite in generated_citations:
        print(f"   - {cite['document']}, p.{cite['page']}")

    # Compute metrics for this result
    prec, rec = compute_source_metrics(expected_sources, generated_citations)
    sim = compute_answer_similarity(expected_answer, generated_answer, embedder)

    print(f"\n📊 Metrics:")
    print(f"   Source Precision: {prec*100:.0f}% | Source Recall: {rec*100:.0f}%")
    print(f"   Answer Similarity: {sim*100:.0f}%")
