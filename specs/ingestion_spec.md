# Ingestion & Chunking Specification

## Goal
Convert PDF documents into searchable text chunks with page number tracking for citation purposes.

## Requirements
- **Chunking method**: Recursive by tokens with overlap (as per assignment)
- **Chunk size**: 512 tokens (default)
- **Overlap**: 50 tokens (default, ~10% overlap)
- **Page tracking**: Each chunk must preserve source page number
- **Text cleaning**: Remove PDF artifacts, normalize whitespace

## Architecture

Uses **Pydantic models** for type safety, validation, and easy serialization:
- `DocumentChunk`: Represents a single chunk with metadata
- `ParsedDocument`: Container for all chunks from a PDF
- `ParserConfig`: Configuration for chunking behavior
- `DocumentParser`: Main parser class orchestrating the pipeline

## Data Models

### DocumentChunk
Represents a single text chunk with metadata for citation purposes.

**Fields:**
- `text` — Chunk text content
- `document` — Source PDF filename
- `page` — Page number in original PDF (≥1)
- `chunk_index` — Sequential chunk number within document (≥0)

**Example:**
```python
{
    "text": "El presupuesto aprobado fue...",
    "document": "ResumenReunionGA_20231124.pdf",
    "page": 3,
    "chunk_index": 5
}
```

### ParsedDocument
Container for all chunks from a single PDF with processing metadata.

**Fields:**
- `filename` — PDF filename
- `total_pages` — Total pages in PDF
- `total_chunks` — Total chunks created
- `chunks` — List of DocumentChunk objects
- `file_size_bytes` _(optional)_ — File size
- `processing_time_seconds` _(optional)_ — Parse duration

**Utility Methods:**
- `chunks_per_page` — Average chunks/page
- `get_chunks_by_page(page)` — Filter chunks by page
- `get_chunk_texts()` — Extract text list for embedding

### ParserConfig
Chunking configuration parameters.

**Fields:**
- `chunk_size` — Tokens per chunk (default: 512, range: 50-2000)
- `chunk_overlap` — Overlap tokens (default: 50, range: 0-500)
- `separators` — Split boundaries (default: `["\n\n", "\n", " ", ""]`)

Config is **immutable** (frozen) after creation.

## API Reference

### DocumentParser Class
```python
class DocumentParser:
    """
    Parses PDF documents into chunks with metadata.
    
    Orchestrates the complete parsing pipeline: PDF extraction, text cleaning,
    recursive chunking, and metadata preservation.
    """
    
    def __init__(self, config: Optional[ParserConfig] = None):
        """
        Initialize parser with configuration.
        
        Loads the configured chunking strategy (RecursiveCharacterTextSplitter)
        with specified chunk size, overlap, and separator hierarchy.
        
        Args:
            config: Parser configuration (uses defaults if None)
            
        Example:
            >>> parser = DocumentParser()  # Uses defaults
            >>> parser = DocumentParser(ParserConfig(chunk_size=1024))
        """
    
    def parse_pdf(self, pdf_path: str | Path) -> ParsedDocument:
        """
        Parse a PDF file into chunks.
        
        Extracts text from all pages, applies cleaning transformations,
        chunks with recursive splitting, and preserves page metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ParsedDocument with all chunks and metadata
            
        Example:
            >>> parsed = parser.parse_pdf("pdfs/document.pdf")
            >>> print(f"Created {parsed.total_chunks} chunks from {parsed.total_pages} pages")
        """
    
    def parse_multiple(self, pdf_paths: List[str | Path]) -> List[ParsedDocument]:
        """
        Parse multiple PDFs.
        
        Processes each PDF independently and returns list of parsed documents.
        
        Args:
            pdf_paths: List of PDF file paths
            
        Returns:
            List of ParsedDocument objects
            
        Example:
            >>> pdf_files = Path("pdfs").glob("*.pdf")
            >>> parsed_docs = parser.parse_multiple(pdf_files)
        """
```

## Usage Examples

### Basic Usage
```python
from rag_system.parser import DocumentParser, ParserConfig

# Use default config
parser = DocumentParser()
parsed_doc = parser.parse_pdf("pdfs/ResumenReunionGA_20231124.pdf")

print(f"Document: {parsed_doc.filename}")
print(f"Pages: {parsed_doc.total_pages}")
print(f"Chunks: {parsed_doc.total_chunks}")
print(f"Processing time: {parsed_doc.processing_time_seconds}s")
```

### Custom Configuration
```python
# Custom chunk size and overlap
config = ParserConfig(chunk_size=1024, chunk_overlap=100)
parser = DocumentParser(config)
parsed_doc = parser.parse_pdf("pdfs/document.pdf")
```

### Multiple Documents
```python
pdf_files = list(Path("pdfs").glob("*.pdf"))
parsed_docs = parser.parse_multiple(pdf_files)

total_chunks = sum(doc.total_chunks for doc in parsed_docs)
print(f"Processed {len(parsed_docs)} documents, {total_chunks} total chunks")
```

### Extract Texts for Embedding
```python
# Get all chunk texts in correct order
texts = parsed_doc.get_chunk_texts()
embeddings = embed_documents(texts)  # Pass to embedding function
```

### JSON Serialization (for API)
```python
# Pydantic models serialize easily to JSON
json_output = parsed_doc.model_dump_json(indent=2)

# Or as dictionary
dict_output = parsed_doc.model_dump()

# Individual chunks
chunk_dict = parsed_doc.chunks[0].model_dump()
```

### Validation
```python
from pydantic import ValidationError

# Pydantic validates automatically
try:
    bad_chunk = DocumentChunk(
        text="Test",
        document="test.pdf",
        page=-1,  # Error! Must be >= 1
        chunk_index=0
    )
except ValidationError as e:
    print(e)
    # Shows: page must be >= 1
```

## Metadata Structure
Each DocumentChunk contains:
```python
{
    'text': str,           # Chunk content
    'document': str,       # Source PDF filename
    'page': int,           # Page number in original PDF (validated >= 1)
    'chunk_index': int     # Sequential chunk number (validated >= 0)
}
```

## How RecursiveCharacterTextSplitter Works

### The "Recursive" Algorithm

Unlike simple splitting that cuts text at fixed character positions, RecursiveCharacterTextSplitter uses a **hierarchical approach** to preserve semantic boundaries:

1. **Try separator 1** (`\n\n` - paragraph breaks):
   - Split text by double newlines
   - Check each chunk's size
   - If chunk is too large → recurse with next separator
   - If chunk is good size → keep it

2. **Try separator 2** (`\n` - line breaks):
   - If paragraph was too large, split by single newlines
   - Check each sub-chunk's size
   - If still too large → recurse with next separator
   - If good size → keep it

3. **Try separator 3** (` ` - spaces):
   - If line was too large, split by spaces (word boundaries)
   - Check each word-group's size
   - If still too large → recurse with next separator
   - If good size → keep it

4. **Final separator** (`""` - characters):
   - Last resort: split by individual characters
   - Guarantees chunks never exceed max size

### Visual Example

```python
text = """
This is a long paragraph about Madrid city council decisions.
It contains multiple sentences and important information.

This is a second paragraph with different content.
It also has multiple lines and details.
"""

# Simple splitting at position 100:
# "This is a long paragraph about Madrid city council decisions.\nIt contains multiple sentenc"
# (cuts mid-word, loses semantic meaning)

# RecursiveCharacterTextSplitter with chunk_size=100:
# 1. Try \n\n → Two paragraphs (both > 100 chars) → recurse
# 2. Try \n → Split each paragraph by lines
#    - "This is a long paragraph about Madrid city council decisions."  ✓ (64 chars)
#    - "It contains multiple sentences and important information."       ✓ (59 chars)
#    - "This is a second paragraph with different content."              ✓ (52 chars)
#    - "It also has multiple lines and details."                        ✓ (42 chars)
# Result: 4 clean chunks at natural sentence boundaries
```

### Why This Matters for RAG

**Bad chunking** (mid-sentence cut):
```
Chunk 1: "...el presupuesto aprobado fue de 5 millones de euros par"
Chunk 2: "a el proyecto de renovación urbana en el distrito..."
```
→ Context broken, incomplete information, poor retrieval

**Good chunking** (semantic boundary):
```
Chunk 1: "...el presupuesto aprobado fue de 5 millones de euros para el proyecto de renovación urbana."
Chunk 2: "El distrito centro recibirá inversiones adicionales..."
```
→ Complete thoughts, better embeddings, accurate retrieval

### Separator Order Strategy

```python
separators = ["\n\n", "\n", " ", ""]
```

**Priority order** (most to least semantic):
1. **`\n\n`** (paragraphs) - Highest semantic boundary, preserves complete topics
2. **`\n`** (lines/sentences) - Medium boundary, preserves complete statements  
3. **` `** (words) - Low boundary, preserves complete words
4. **`""`** (characters) - Last resort, force-splits to meet size constraint

### Chunk Overlap

The `chunk_overlap=50` parameter creates **sliding windows**:

```
Text: [AAAAAAAAAA][BBBBBBBBBB][CCCCCCCCCC][DDDDDDDDDD]

Without overlap:
Chunk 1: [AAAAAAAAAA][BBBBBBBBBB]
Chunk 2: [CCCCCCCCCC][DDDDDDDDDD]

With overlap=50 chars:
Chunk 1: [AAAAAAAAAA][BBBBBBBBBB]
Chunk 2:              [BBBBBBBBBB][CCCCCCCCCC]
Chunk 3:                           [CCCCCCCCCC][DDDDDDDDDD]
```

**Benefits**:
- Prevents context loss at chunk boundaries
- If relevant info spans boundary, both chunks capture it
- Improves retrieval recall (same concept retrieved from multiple angles)

## Key Implementation Notes

- **Page tracking is critical**: Must be preserved through entire pipeline for citations
- **Token counting**: For true token-based chunking, integrate `tiktoken` library; otherwise character-based approximation works well
- **Separators are configurable**: Adjust for different document types (code, tables, etc.)
- **Chunk size is a target**: Actual chunks may be slightly smaller to respect boundaries
- **Overlap creates redundancy**: Embedding cost increases ~10-20% but retrieval quality improves significantly
