"""
OpenAI LLM service for answer generation.

This module handles:
- Prompt construction (system + user messages)
- LLM API calls with retry logic
- Response parsing and error handling
"""

import logging
from typing import Optional, Dict, Any

from openai import OpenAI, APIError, RateLimitError, APITimeoutError

from src.config import Config
from src.prompts import system_prompt

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for generating answers using OpenAI's chat completion API.

    Features:
    - Automatic retries on failures
    - Token usage tracking
    - Structured prompt management
    """

    def __init__(
        self,
        api_key: str = Config.OPENAI_API_KEY,
        model: str = Config.OPENAI_MODEL,
        temperature: float = Config.OPENAI_TEMPERATURE,
        max_retries: int = Config.OPENAI_MAX_RETRIES,
        timeout: int = Config.OPENAI_TIMEOUT,
    ):
        """
        Initialize the LLM service.

        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-4o-mini)
            temperature: Randomness (0 = deterministic, 1 = creative)
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

        logger.info(
            f"Initialized LLMService (model={model}, temperature={temperature})"
        )

    def generate_answer(
        self, question: str, context: str, retry_count: int = 0
    ) -> Dict[str, Any]:
        #  -> Dict[str, Any] = This function returns a dictionary with string keys and any type of values
        """
        Generate an answer based on question and context.

        Args:
            question: User's question
            context: Retrieved document chunks (concatenated)
            retry_count: Current retry attempt (internal)

        Returns:
            Dict containing:
            - answer: Generated response
            - tokens_used: Token count
            - model: Model used
            - success: Whether generation succeeded
        """
        if not question or not question.strip():
            logger.warning("Empty question provided")
            return {
                "answer": "Please provide a question.",
                "tokens_used": 0,
                "model": self.model,
                "success": False,
            }

        try:
            logger.debug(
                f"Generating answer (question_len={len(question)}, "
                f"context_len={len(context)})"
            )

            messages = [
                {"role": "system", "content": system_prompt()},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}",
                },
            ]

            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature
            )

            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

            logger.info(f"Answer generated successfully (tokens={tokens_used})")

            return {
                "answer": answer,
                "tokens_used": tokens_used,
                "model": response.model,
                "success": True,
            }

        except RateLimitError as e:
            logger.warning(f"Rate limit hit: {e}")
            return self._retry_generation(question, context, retry_count, e)

        except APITimeoutError as e:
            logger.warning(f"API timeout: {e}")
            return self._retry_generation(question, context, retry_count, e)

        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            return self._retry_generation(question, context, retry_count, e)

        except Exception as e:
            logger.error(f"Unexpected error generating answer: {e}")
            return {
                "answer": "I encountered an error while generating the answer. Please try again.",
                "tokens_used": 0,
                "model": self.model,
                "success": False,
                "error": str(e),
            }

    def _retry_generation(
        self, question: str, context: str, retry_count: int, error: Exception
    ) -> Dict[str, Any]:
        """
        Retry answer generation with exponential backoff.

        Args:
            question: User's question
            context: Retrieved context
            retry_count: Current retry attempt
            error: The error that triggered retry

        Returns:
            Dict with answer or error message

            # success flags are kept instead of raising an exception error so that UX doesn't fail and crash doesn't bubble up
            # Exceptions = Fire alarm (evacuate building!)
            # Success flags = Warning light (keep working, but be careful)

        """
        if retry_count >= self.max_retries:
            logger.error(
                f"Max retries ({self.max_retries}) exceeded. Last error: {error}"
            )
            return {
                "answer": (
                    "I'm experiencing technical difficulties. "
                    "Please try again in a moment."
                ),
                "tokens_used": 0,
                "model": self.model,
                "success": False,
                "error": str(error),
            }

        import time

        wait_time = 2**retry_count
        logger.info(
            f"Retrying in {wait_time}s (attempt {retry_count + 1}/{self.max_retries})"
        )

        time.sleep(wait_time)
        return self.generate_answer(question, context, retry_count + 1)

    def estimate_tokens(self, text: str) -> int:
        """
        Rough estimate of token count.

        Rule of thumb: 1 token ≈ 4 characters for English text

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // 4


# Global singleton instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create the global LLM service instance."""
    global _llm_service

    if _llm_service is None:
        _llm_service = LLMService()

    return _llm_service
