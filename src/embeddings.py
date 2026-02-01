"""
OpenAI embedding service with retry logic and error handling.
"""

import logging
import time
from typing import List, Optional

from openai import OpenAI, APIError, RateLimitError, APITimeoutError

from src.config import Config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using OpenAI's API.

    Features:
    - Automatic retries with exponential backoff
    - Batch processing for efficiency
    - Error handling for common API failures
    """

    def __init__(
        self,
        api_key: str = Config.OPENAI_API_KEY,
        model: str = Config.OPENAI_EMBEDDING_MODEL,
        max_retries: int = Config.OPENAI_MAX_RETRIES,
        timeout: int = Config.OPENAI_TIMEOUT,
    ):
        # Initialize the embedding service.

        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.max_retries = max_retries

        logger.info(f"Initialized EmbeddingService (model={model})")

    def embed_text(self, text: str, retry_count: int = 0) -> Optional[List[float]]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            retry_count: Current retry attempt (used internally)

        Returns:
            List of floats (embedding vector) or None on failure
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to embed_text")
            return None

        try:
            logger.debug(f"Generating embedding for text (length={len(text)})")

            response = self.client.embeddings.create(model=self.model, input=text)

            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding (dimension={len(embedding)})")

            return embedding

        except RateLimitError as e:
            logger.warning(f"Rate limit hit: {e}")
            return self._retry_with_backoff(text, retry_count, e)

        except APITimeoutError as e:
            logger.warning(f"API timeout: {e}")
            return self._retry_with_backoff(text, retry_count, e)

        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            return self._retry_with_backoff(text, retry_count, e)

        except Exception as e:
            logger.error(f"Unexpected error generating embedding: {e}")
            return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts.

        OpenAI's API supports batch requests (up to 2048 texts).
        This is more efficient than individual requests.
        """
        if not texts:
            logger.warning("Empty text list provided to embed_batch")
            return []

        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")

            # Filter out empty texts but preserve indices
            valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
            valid_texts = [texts[i] for i in valid_indices]

            if not valid_texts:
                logger.warning("No valid texts in batch")
                return [None] * len(texts)

            response = self.client.embeddings.create(
                model=self.model, input=valid_texts
            )

            # Map embeddings back to original indices
            embeddings = [None] * len(texts)
            for i, idx in enumerate(valid_indices):
                embeddings[idx] = response.data[i].embedding

            logger.info(f"Successfully generated {len(valid_texts)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            # Fallback: Try embedding individually
            logger.info("Falling back to individual embedding requests")
            return [self.embed_text(text) for text in texts]

    def _retry_with_backoff(
        self, text: str, retry_count: int, error: Exception
    ) -> Optional[List[float]]:
        """
        Retry with exponential backoff.

        Backoff strategy:
        - Retry 1: Wait 1 second
        - Retry 2: Wait 2 seconds
        - Retry 3: Wait 4 seconds

        Args:
            text: Text to embed
            retry_count: Current retry attempt
            error: The error that triggered retry

        Returns:
            Embedding or None if max retries exceeded
        """
        if retry_count >= self.max_retries:
            logger.error(
                f"Max retries ({self.max_retries}) exceeded. Last error: {error}"
            )
            return None

        wait_time = 2**retry_count  # Exponential: 1, 2, 4, 8...
        logger.info(
            f"Retrying in {wait_time}s (attempt {retry_count + 1}/{self.max_retries})"
        )

        time.sleep(wait_time)
        return self.embed_text(text, retry_count + 1)

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for this model.

        Returns:
            1536 for text-embedding-3-small
            3072 for text-embedding-3-large
        """
        # Quick lookup (faster than API call)
        dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        return dimension_map.get(self.model, 1536)


# Global singleton instance (reuse across modules)
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """
    Get or create the global embedding service instance.

    Using a singleton prevents creating multiple OpenAI clients,
    which is wasteful and can cause connection issues.
    """
    global _embedding_service

    if _embedding_service is None:
        _embedding_service = EmbeddingService()

    return _embedding_service
