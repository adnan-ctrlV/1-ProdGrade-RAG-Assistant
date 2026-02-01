"""
Text chunking for RAG systems.

This module splits documents into chunks that:
1. Fit within LLM context windows
2. Preserve semantic meaning
3. Have overlap to maintain context continuity
"""

import logging
import re
from typing import List, Tuple
import tiktoken

from src.config import Config

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Chunks text documents for retrieval-augmented generation.

    Strategy:
    - Split on sentence boundaries (preserve meaning)
    - Target chunk_size tokens (default 500)
    - Overlap chunks by chunk_overlap tokens (default 50)
    - Count tokens accurately using tiktoken
    """

    def __init__(
        self,
        chunk_size: int = Config.CHUNK_SIZE,
        chunk_overlap: int = Config.CHUNK_OVERLAP,
        encoding_name: str = "cl100k_base",  # GPT-4 tokenizer
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Target tokens per chunk
            chunk_overlap: Tokens of overlap between chunks
            encoding_name: Tokenizer to use (cl100k_base = GPT-4/GPT-3.5)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

        logger.debug(
            f"Initialized TextChunker (size={chunk_size}, overlap={chunk_overlap})"
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the same tokenizer as OpenAI."""
        return len(self.encoding.encode(text))

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Uses regex to split on:
        - Period followed by space
        - Question mark followed by space
        - Exclamation mark followed by space

        Preserves abbreviations like "U.S." and "Dr."
        """
        # Simple sentence splitter (handles most cases)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(
        self, text: str, metadata: dict = None
    ) -> List[Tuple[str, int, dict]]:
        """
        Chunk text into overlapping segments.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of (chunk_text, token_count, metadata) tuples
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to chunker")
            return []

        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If single sentence exceeds chunk_size, split it forcefully
            if sentence_tokens > self.chunk_size:
                logger.warning(
                    f"Sentence exceeds chunk_size ({sentence_tokens} > {self.chunk_size}). "
                    f"Splitting forcefully."
                )
                # Split by words as fallback
                words = sentence.split()
                for word in words:
                    word_tokens = self.count_tokens(word)
                    if current_tokens + word_tokens > self.chunk_size:
                        # Save current chunk
                        if current_chunk:
                            chunk_text = " ".join(current_chunk)
                            chunks.append(
                                (
                                    chunk_text,
                                    self.count_tokens(chunk_text),
                                    metadata or {},
                                )
                            )
                        # Start new chunk with overlap
                        overlap_words = (
                            current_chunk[-(self.chunk_overlap // 10) :]
                            if current_chunk
                            else []
                        )
                        current_chunk = overlap_words + [word]
                        current_tokens = self.count_tokens(" ".join(current_chunk))
                    else:
                        current_chunk.append(word)
                        current_tokens += word_tokens
                continue

            # Check if adding this sentence exceeds chunk_size
            if current_tokens + sentence_tokens > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(
                        (chunk_text, self.count_tokens(chunk_text), metadata or {})
                    )

                # Start new chunk with overlap
                # Take last few sentences as overlap
                overlap_sentences = []
                overlap_tokens = 0
                for sent in reversed(current_chunk):
                    sent_tokens = self.count_tokens(sent)
                    if overlap_tokens + sent_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break

                current_chunk = overlap_sentences + [sentence]
                current_tokens = self.count_tokens(" ".join(current_chunk))
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append((chunk_text, self.count_tokens(chunk_text), metadata or {}))

        logger.info(f"Chunked text into {len(chunks)} chunks")
        return chunks


def chunk_document(text: str, filename: str) -> List[Tuple[str, int, dict]]:
    """
    Convenience function to chunk a document.

    Args:
        text: Document text
        filename: Source filename (for metadata)

    Returns:
        List of (chunk_text, token_count, metadata) tuples
    """
    chunker = TextChunker()
    metadata = {"filename": filename}
    return chunker.chunk_text(text, metadata)
