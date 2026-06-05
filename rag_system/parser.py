import re
import time
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_system.models import DocumentChunk, ParsedDocument


class ParserConfig(BaseModel):
    """Configuration for document parsing."""

    chunk_size: int = Field(default=512, ge=50, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)
    separators: List[str] = Field(default=["\n\n", "\n", " ", ""])

    model_config = ConfigDict(frozen=True)  # Immutable config


class DocumentParser:
    """Parses PDF documents into chunks with metadata."""

    def __init__(self, config: Optional[ParserConfig] = None):
        """
        Initialize parser with configuration.

        Args:
            config: Parser configuration (uses defaults if None)
        """
        self.config = config or ParserConfig()
        self.text_splitter = self._create_splitter()

    def _create_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create text splitter from config."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            length_function=len,
        )

    def parse_pdf(self, pdf_path: str | Path) -> ParsedDocument:
        """
        Parse a PDF file into chunks.

        Args:
            pdf_path: Path to PDF file

        Returns:
            ParsedDocument with all chunks and metadata
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)

        # Extract pages
        pages = self._extract_pages(pdf_path)

        # Clean each page
        cleaned_pages = [(page_num, self._clean_text(text)) for page_num, text in pages]

        # Chunk pages
        chunks = self._chunk_pages(cleaned_pages, pdf_path.name)

        # Build result
        return ParsedDocument(
            filename=pdf_path.name,
            total_pages=len(pages),
            total_chunks=len(chunks),
            chunks=chunks,
            file_size_bytes=pdf_path.stat().st_size,
            processing_time_seconds=round(time.time() - start_time, 3),
        )

    def parse_multiple(self, pdf_paths: List[str | Path]) -> List[ParsedDocument]:
        """Parse multiple PDFs."""
        return [self.parse_pdf(path) for path in pdf_paths]

    def _extract_pages(self, pdf_path: Path) -> List[tuple[int, str]]:
        """
        Extract text per page from PDF.

        Returns:
            List of (page_number, text) tuples
        """
        reader = PdfReader(str(pdf_path))
        pages = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            pages.append((page_num, text))

        return pages

    def _clean_text(self, text: str) -> str:
        """
        Remove PDF artifacts and normalize whitespace.
        """
        # Remove page numbers (English and Spanish)
        text = re.sub(r"Page \d+ of \d+", "", text)
        text = re.sub(r"Página \d+ de \d+", "", text)

        # Normalize multiple newlines to double newlines (paragraph breaks)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove excessive spaces/tabs
        text = re.sub(r"[ \t]+", " ", text)

        # Remove leading/trailing whitespace per line
        text = "\n".join(line.strip() for line in text.split("\n"))

        return text.strip()

    def _chunk_pages(
        self, pages: List[tuple[int, str]], doc_name: str
    ) -> List[DocumentChunk]:
        """
        Chunk pages into DocumentChunk objects with metadata.

        Args:
            pages: List of (page_number, text) tuples
            doc_name: Document filename

        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        chunk_index = 0

        for page_num, page_text in pages:
            if not page_text.strip():
                continue

            # Split page into chunks
            page_chunks = self.text_splitter.split_text(page_text)

            # Create DocumentChunk for each
            for chunk_text in page_chunks:
                if chunk_text.strip():
                    chunks.append(
                        DocumentChunk(
                            text=chunk_text,
                            document=doc_name,
                            page=page_num,
                            chunk_index=chunk_index,
                        )
                    )
                    chunk_index += 1

        return chunks
