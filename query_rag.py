"""
Query the RAG system from command line.

Usage:
    python query_rag.py "¿Cuál es la misión del grupo motor?"

Environment:
    GOOGLE_API_KEY: Required for Gemini LLM (loaded from .env file)
"""
import os
import sys
from pathlib import Path
from pydantic import SecretStr
from dotenv import load_dotenv

from rag_system.vector_store import FAISSVectorStore
from rag_system.embeddings import Embedder
from rag_system.llm import RAGGenerator, LLMConfig

# Load environment variables from .env file
load_dotenv()


def query_rag(question: str, use_reranking: bool = True) -> None:
    """
    Query the RAG system and display results.

    Args:
        question: Natural language question
        use_reranking: Whether to use cross-encoder re-ranking
    """
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found")
        print("\nCreate a .env file with:")
        print("  GOOGLE_API_KEY=your-api-key-here")
        print("\nOr set environment variable:")
        print("  PowerShell: $env:GOOGLE_API_KEY='your-api-key-here'")
        print("  Bash: export GOOGLE_API_KEY='your-api-key-here'")
        sys.exit(1)

    # Check if index exists
    index_path = Path("faiss_index/vector_index.faiss")
    if not index_path.exists():
        print("❌ Error: Vector index not found")
        print("\nRun indexing first:")
        print("  python index_documents.py pdfs/")
        sys.exit(1)

    print("=" * 80)
    print("RAG QUERY SYSTEM")
    print("=" * 80)
    print(f"\n📝 Question: {question}\n")

    # Load components
    print("Loading components...")
    vector_store = FAISSVectorStore()
    embedder = Embedder()

    config = LLMConfig(
        api_key=SecretStr(api_key),
        model_name="gemini-2.0-flash-001",
        temperature=0.0,
        top_k=5
    )

    rag = RAGGenerator(
        vector_store=vector_store,
        embedder=embedder,
        config=config,
        use_reranking=use_reranking
    )

    print(f"✓ Loaded (re-ranking: {'enabled' if use_reranking else 'disabled'})\n")

    # Generate answer
    print("=" * 80)
    print("GENERATING ANSWER")
    print("=" * 80)
    response = rag.generate_answer(question)

    # Display results
    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(f"\n{response.answer}\n")

    if response.citations:
        print("=" * 80)
        print(f"SOURCES ({len(response.citations)})")
        print("=" * 80)
        for i, cite in enumerate(response.citations, 1):
            print(f"\n{i}. {cite.document}, página {cite.page}")
            print(f'   "{cite.text_snippet}"')
    else:
        print("\n⚠ No citations found")

    # Display metadata
    print("\n" + "=" * 80)
    print("METADATA")
    print("=" * 80)
    print(f"  Chunks retrieved: {response.num_chunks_retrieved}")
    print(f"  Has citations: {response.has_citations}")
    print(f"  Used re-ranking: {response.used_reranking}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_rag.py <question> [--no-rerank]")
        print("\nExamples:")
        print('  python query_rag.py "¿Cuándo fue la primera reunión?"')
        print('  python query_rag.py "¿Qué es la Escuela de Gobierno Abierto?" --no-rerank')
        sys.exit(1)

    question_text = sys.argv[1]
    use_rerank = "--no-rerank" not in sys.argv

    query_rag(question_text, use_reranking=use_rerank)
