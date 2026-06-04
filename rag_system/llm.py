import os
from textwrap import dedent
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict, SecretStr

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from rag_system.models import DocumentChunk, Citation
from rag_system.vector_store import FAISSVectorStore, SearchResult
from rag_system.embeddings import Embedder
from rag_system.reranker import Reranker, RerankerConfig


class LLMConfig(BaseModel):
    """Configuration for LLM generation."""

    api_key: SecretStr = Field(..., description="Google API key")
    model_name: str = Field(default="gemini-1.5-flash", description="Gemini model name")
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0=deterministic, 2=creative)",
    )
    max_output_tokens: int = Field(
        default=1024, ge=64, le=8192, description="Maximum tokens in generated response"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve from vector store",
    )

    model_config = ConfigDict(frozen=True)


class LLMResponse(BaseModel):
    """Complete response from RAG system."""

    question: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(
        default_factory=list, description="Source citations"
    )
    num_chunks_retrieved: int = Field(
        ..., ge=0, description="Number of chunks retrieved"
    )
    has_citations: bool = Field(
        ..., description="Whether answer includes valid citations"
    )
    used_reranking: bool = Field(
        default=False, description="Whether cross-encoder re-ranking was used"
    )

    @property
    def formatted_response(self) -> str:
        """Format response with citations for display."""
        response = f"**Answer:**\n{self.answer}\n"

        if self.citations:
            response += f"\n**Sources ({len(self.citations)}):**\n"
            for i, cite in enumerate(self.citations, 1):
                response += f"{i}. {cite.document}, p.{cite.page}\n"
                response += f'   "{cite.text_snippet[:100]}..."\n'
        else:
            response += "\n*No sources cited*"

        return response

    model_config = ConfigDict(validate_assignment=True)


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
        reranker_config: Optional[RerankerConfig] = None,
    ):
        """
        Initialize RAG generator.

        Args:
            vector_store: Populated FAISS vector store
            embedder: Embedder for query embedding
            config: LLM configuration
            use_reranking: Whether to use cross-encoder re-ranking (default: True)
            reranker_config: Re-ranker configuration (uses defaults if None)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.config = config
        self.use_reranking = use_reranking

        self._load_llm()

        # Initialize re-ranker if enabled
        if self.use_reranking:
            self.reranker = Reranker(config=reranker_config)
            print("✓ Re-ranker enabled (cross-encoder)")
        else:
            self.reranker = None
            print("ℹ Re-ranker disabled (bi-encoder only)")

    def _load_llm(self) -> None:
        """Load the Gemini LLM with configured parameters."""
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.model_name,
            google_api_key=self.config.api_key.get_secret_value(),
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )
        print(f"✓ Loaded {self.config.model_name}")

    def generate_answer(self, question: str) -> LLMResponse:
        """
        Generate answer to question using RAG pipeline.

        Args:
            question: User's natural language question

        Returns:
            LLMResponse with answer and citations
        """
        # 1. Retrieve relevant chunks
        results = self._retrieve_context(question)

        if not results:
            return LLMResponse(
                question=question,
                answer="No puedo responder esta pregunta con la información disponible.",
                citations=[],
                num_chunks_retrieved=0,
                has_citations=False,
                used_reranking=self.use_reranking,
            )

        # 2. Format context for LLM
        context = self._format_context(results)

        # 3. Generate answer
        answer = self._call_llm(question, context)

        # 4. Extract citations
        citations = self._extract_citations(results, answer)

        # 5. Build response
        return LLMResponse(
            question=question,
            answer=answer,
            citations=citations,
            num_chunks_retrieved=len(results),
            has_citations=len(citations) > 0,
            used_reranking=self.use_reranking,
        )

    def _retrieve_context(self, question: str) -> List[SearchResult]:
        """
        Retrieve relevant chunks from vector store with optional re-ranking.

        Args:
            question: User question

        Returns:
            List of SearchResult objects (re-ranked if enabled)
        """
        query_embedding = self.embedder.embed_query(question)

        if self.use_reranking:
            # Two-stage retrieval: bi-encoder → cross-encoder
            # Get more candidates for re-ranking
            candidates = self.vector_store.search(
                query_embedding, top_k=20  # Retrieve 20 candidates
            )

            if not candidates:
                return []

            # Re-rank to top-N
            reranked = self.reranker.rerank(
                query=question, candidates=candidates, top_n=self.config.top_k
            )

            # Convert RerankResult back to SearchResult
            # Keep both scores: retrieval_score (bi-encoder) and rerank_score (cross-encoder)
            results = [
                SearchResult(
                    chunk=r.chunk,
                    score=r.retrieval_score,
                    rerank_score=r.rerank_score,
                )
                for r in reranked
            ]
            return results
        else:
            # Single-stage retrieval: bi-encoder only
            return self.vector_store.search(
                query_embedding, top_k=self.config.top_k
            )

    def _format_context(self, results: List[SearchResult]) -> str:
        """
        Format retrieved chunks into context string for LLM.

        Args:
            results: Search results with chunks

        Returns:
            Formatted context string with citations
        """
        context_parts = []

        for i, result in enumerate(results, 1):
            chunk = result.chunk
            citation_ref = f"[{chunk.document}, p.{chunk.page}]"
            context_parts.append(f"Fragmento {i} {citation_ref}:\n{chunk.text}\n")

        return "\n".join(context_parts)

    def _call_llm(self, question: str, context: str) -> str:
        """
        Call LLM to generate answer from context.

        Args:
            question: User question
            context: Formatted context from retrieval

        Returns:
            Generated answer string
        """
        # Build prompt
        prompt = self._build_prompt(question, context)

        # Call LLM
        messages = [
            SystemMessage(content=prompt["system"]),
            HumanMessage(content=prompt["user"]),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def _build_prompt(self, question: str, context: str) -> dict:
        """
        Build system and user prompts for LLM.

        Args:
            question: User question
            context: Retrieved context

        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        system_prompt = dedent("""
            Eres un asistente especializado en responder preguntas basándote ÚNICAMENTE en el contexto proporcionado.

            INSTRUCCIONES:
            1. Responde la pregunta usando SOLO la información del contexto
            2. Cita tus fuentes usando el formato [documento, p.X]
            3. Si el contexto no contiene información relevante, di claramente "No tengo información suficiente"
            4. Sé conciso y directo
            5. SIEMPRE incluye al menos una cita cuando respondas

            FORMATO DE RESPUESTA:
            - Responde en 2-4 oraciones
            - Incluye citas en el texto: "Según el documento [archivo.pdf, p.5], el presupuesto fue..."
            - No inventes información que no esté en el contexto
        """).strip()

        user_prompt = dedent(f"""
            CONTEXTO:
            {context}

            PREGUNTA:
            {question}

            RESPUESTA:
        """).strip()

        return {"system": system_prompt, "user": user_prompt}

    def _extract_citations(
        self, results: List[SearchResult], answer: str
    ) -> List[Citation]:
        """
        Extract citations from retrieved chunks that appear in answer.

        Args:
            results: Retrieved search results
            answer: Generated answer text

        Returns:
            List of Citation objects
        """
        citations = []
        seen = set()  # Deduplicate by (document, page)

        for result in results:
            chunk = result.chunk
            key = (chunk.document, chunk.page)

            # Check if this source is cited in the answer
            citation_ref = f"[{chunk.document}, p.{chunk.page}]"

            if citation_ref in answer and key not in seen:
                # Extract snippet (first 200 chars of chunk)
                snippet = chunk.text[:200].strip()
                if len(chunk.text) > 200:
                    snippet += "..."

                citations.append(
                    Citation(
                        document=chunk.document, page=chunk.page, text_snippet=snippet
                    )
                )

                seen.add(key)

        return citations

    def batch_generate(self, questions: List[str]) -> List[LLMResponse]:
        """
        Generate answers for multiple questions.

        Args:
            questions: List of questions

        Returns:
            List of LLMResponse objects
        """
        return [self.generate_answer(q) for q in questions]
