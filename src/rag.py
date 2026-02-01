"""
RAG (Retrieval-Augmented Generation) orchestrator.

This is the main entry point for querying the system.
It coordinates retrieval and generation to produce answers.
"""

import logging
from typing import Dict, Any, List

from src.retrieval import get_retrieval_service
from src.llm import get_llm_service
from src.prompts import no_results_prompt
from src.config import Config

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Complete RAG system orchestrator.

    Pipeline:
    1. User asks a question
    2. Retrieve relevant chunks from database
    3. Format chunks as context
    4. Generate answer using LLM
    5. Return answer with metadata
    """

    def __init__(self):
        """Initialize RAG system with retrieval and LLM services."""
        self.retrieval_service = get_retrieval_service()
        self.llm_service = get_llm_service()

        logger.info("Initialized RAG system")

    def query(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using RAG.

        Args:
            question: User's question

        Returns:
            Dict containing:
            - question: Original question
            - answer: Generated answer
            - sources: List of source documents used
            - chunks_retrieved: Number of chunks found
            - tokens_used: LLM tokens consumed
            - success: Whether query succeeded
        """
        logger.info(f"Processing query: {question[:100]}...")

        if not question or not question.strip():
            return {
                "question": question,
                "answer": "Please provide a question.",
                "sources": [],
                "chunks_retrieved": 0,
                "tokens_used": 0,
                "success": False,
            }

        try:
            # Step 1: Retrieve relevant chunks
            chunks = self.retrieval_service.retrieve(question)

            # Step 2: Handle no results
            if not chunks:
                logger.warning("No relevant chunks found")
                return {
                    "question": question,
                    "answer": no_results_prompt(question),
                    "sources": [],
                    "chunks_retrieved": 0,
                    "tokens_used": 0,
                    "success": True,
                    "no_results": True,
                }

            # Step 3: Format context
            context = self.retrieval_service.format_context(chunks)

            # Step 4: Generate answer
            result = self.llm_service.generate_answer(question, context)

            # Step 5: Extract sources
            sources = self._extract_sources(chunks)

            # Step 6: Return complete response
            response = {
                "question": question,
                "answer": result["answer"],
                "sources": sources,
                "chunks_retrieved": len(chunks),
                "chunks": chunks,  # Include full chunk data
                "tokens_used": result.get("tokens_used", 0),
                "model": result.get("model", Config.OPENAI_MODEL),
                "success": result["success"],
            }

            logger.info(
                f"Query completed: {len(chunks)} chunks, "
                f"{result.get('tokens_used', 0)} tokens"
            )

            return response

        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {
                "question": question,
                "answer": "An error occurred while processing your question. Please try again.",
                "sources": [],
                "chunks_retrieved": 0,
                "tokens_used": 0,
                "success": False,
                "error": str(e),
            }

    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract unique source documents from chunks.

        Args:
            chunks: Retrieved chunks

        Returns:
            List of unique sources with metadata
        """
        sources_dict = {}

        for chunk in chunks:
            filename = chunk["filename"]

            if filename not in sources_dict:
                sources_dict[filename] = {
                    "filename": filename,
                    "chunks_used": [],
                    "max_score": chunk["score"],
                }

            sources_dict[filename]["chunks_used"].append(chunk["chunk_index"])
            sources_dict[filename]["max_score"] = max(
                sources_dict[filename]["max_score"], chunk["score"]
            )

        # Convert to sorted list (by max score)
        sources = list(sources_dict.values())
        sources.sort(key=lambda x: x["max_score"], reverse=True)

        return sources


# Global singleton instance
_rag_system: RAGSystem = None


def get_rag_system() -> RAGSystem:
    """Get or create the global RAG system instance."""
    global _rag_system

    if _rag_system is None:
        _rag_system = RAGSystem()

    return _rag_system


def query(question: str) -> Dict[str, Any]:
    """
    Convenience function to query the RAG system.

    Args:
        question: User's question

    Returns:
        Dict with answer and metadata

    Example:
        >>> result = query("What is the remote work policy?")
        >>> print(result["answer"])
        >>> print(result["sources"])
    """
    rag = get_rag_system()
    return rag.query(question)
