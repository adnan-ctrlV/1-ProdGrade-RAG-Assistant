"""
Vector-based retrieval for RAG system.
"""

import logging
from typing import List, Dict, Any, Optional

from database import Database
from embeddings import get_embedding_service
from config import Config


logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Service for retrieving relevant document chunks.
    """

    def __init__(
        self,
        top_k: int = Config.RETRIEVAL_TOP_K,
        min_score: float = Config.RETRIEVAL_MIN_SCORE,
    ):
        """
        Initialize retrieval service.
        """
        self.top_k = top_k
        self.min_score = min_score
        self.embedding_service = get_embedding_service()

        logger.info(
            f"Initialized RetrievalService (top_k={top_k}, min_score={min_score})"
        )

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.

        Process:
        1. Embed the query (convert text → vector)
        2. Search database for similar vectors
        3. Filter by minimum score
        4. Return top-k results
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to retrieve()")
            return []

        try:
            # Step 1: Embed the query
            logger.debug(f"Embedding query: {query[:50]}...")
            query_embedding = self.embedding_service.embed_text(query)

            if query_embedding is None:
                logger.error("Failed to generate query embedding")
                return []

            # Step 2: Search database using vector similarity
            logger.debug(f"Searching for top {self.top_k} similar chunks")
            results = self._vector_search(query_embedding)

            # Step 3: Filter by minimum score
            filtered_results = [r for r in results if r["score"] >= self.min_score]

            logger.info(
                f"Retrieved {len(filtered_results)} chunks (score >= {self.min_score})"
            )

            return filtered_results

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    def _vector_search(self, query_embedding: List[float]) -> List[Dict[str, Any]]:
        """Perform vector similarity search in database."""

        logger.debug(f"Performing vector search (top_k={self.top_k})")

        pg_array = "[" + ",".join(map(str, query_embedding)) + "]"

        # Strip quotes from stored JSON strings before casting
        query = f"""
        SELECT 
            id,
            filename,
            chunk_index,
            content,
            token_count,
            (1 - (embedding <=> '{pg_array}'::vector)) AS score
        FROM documents
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> '{pg_array}'::vector
        LIMIT %s
        """

        try:
            results = Database.execute_query(
                query, (pg_array, pg_array, self.top_k), fetch=True
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

        if not results:
            logger.warning("No results found")
            return []

        chunks = []
        for row in results:
            chunks.append(
                {
                    "id": row["id"],
                    "filename": row["filename"],
                    "chunk_index": row["chunk_index"],
                    "content": row["content"],
                    "token_count": row["token_count"],
                    "score": float(row["score"]),
                }
            )

            logger.debug(f"  {row['filename']}: score={row['score']:.3f}")

        return chunks

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context for the LLM.

        Format:
        ---
        Source: filename.txt (chunk 0)
        Content: [chunk text]
        ---
        Source: filename.txt (chunk 1)
        Content: [chunk text]
        ---

        Args:
            chunks: Retrieved chunks from vector search

        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant information found."

        context_parts = []

        for chunk in chunks:
            context_parts.append(
                f"---\n"
                f"Source: {chunk['filename']} (chunk {chunk['chunk_index']})\n"
                f"Content: {chunk['content']}\n"
                f"---"
            )

        return "\n\n".join(context_parts)


# Global singleton instance
_retrieval_service: Optional[RetrievalService] = None


def get_retrieval_service() -> RetrievalService:
    """Get or create the global retrieval service instance."""
    global _retrieval_service

    if _retrieval_service is None:
        _retrieval_service = RetrievalService()

    return _retrieval_service
