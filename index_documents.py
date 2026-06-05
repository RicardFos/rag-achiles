"""
End-to-end document indexing script.

Usage:
    python index_documents.py pdfs/

This script:
1. Parses all PDFs in the specified directory
2. Generates embeddings for document chunks
3. Creates and populates a FAISS vector store
4. Saves the index to disk
"""
import sys
from pathlib import Path
from rag_system.parser import DocumentParser
from rag_system.embeddings import Embedder
from rag_system.vector_store import FAISSVectorStore


def index_documents(pdf_dir: Path) -> None:
    """
    Index all PDFs in a directory.

    Args:
        pdf_dir: Directory containing PDF files
    """
    if not pdf_dir.exists():
        print(f"❌ Error: Directory '{pdf_dir}' does not exist")
        sys.exit(1)

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ Error: No PDF files found in '{pdf_dir}'")
        sys.exit(1)

    print(f"📁 Found {len(pdf_files)} PDF files in {pdf_dir}\n")

    # Step 1: Parse documents
    print("=" * 80)
    print("STEP 1: Parsing PDFs")
    print("=" * 80)
    parser = DocumentParser()
    parsed_docs = parser.parse_multiple(pdf_files)

    # Collect all chunks
    all_chunks = []
    for doc in parsed_docs:
        all_chunks.extend(doc.chunks)

    print(f"\n✓ Parsed {len(parsed_docs)} documents")
    print(f"✓ Generated {len(all_chunks)} chunks\n")

    # Step 2: Generate embeddings
    print("=" * 80)
    print("STEP 2: Generating Embeddings")
    print("=" * 80)
    embedder = Embedder()
    texts = [chunk.text for chunk in all_chunks]
    embeddings = embedder.embed_documents(texts)

    print(f"\n✓ Generated {len(embeddings)} embeddings")
    print(f"✓ Embedding dimension: {embedder.dimension}\n")

    # Step 3: Create and populate vector store
    print("=" * 80)
    print("STEP 3: Building Vector Index")
    print("=" * 80)
    vector_store = FAISSVectorStore()
    vector_store.create_index(dimension=embedder.dimension)
    vector_store.add_chunks(all_chunks, embeddings)

    # Step 4: Save to disk
    print("\n" + "=" * 80)
    print("STEP 4: Saving Index")
    print("=" * 80)
    vector_store.save()

    # Display stats
    print("\n" + "=" * 80)
    print("INDEX STATISTICS")
    print("=" * 80)
    stats = vector_store.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✅ Indexing complete! Ready for queries.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python index_documents.py <pdf_directory>")
        print("\nExample:")
        print("  python index_documents.py pdfs/")
        sys.exit(1)

    pdf_directory = Path(sys.argv[1])
    index_documents(pdf_directory)
