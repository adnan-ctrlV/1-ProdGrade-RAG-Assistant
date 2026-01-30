"""
Document ingestion pipeline.

This module handles the ETL process:
1. Extract: Load documents from files
2. Transform: Chunk text into semantic units
3. Load: Generate embeddings and store in database

Run this once to populate the database, then query with the RAG system.
"""

import logging
from pathlib import Path
from typing import List, Tuple

from database import Database
from chunking import TextChunker
from embeddings import get_embedding_service
from config import Config

logger = logging.getLogger(__name__)


class DocumentIngestion:
    """
    Pipeline for ingesting documents into the RAG system.

    Process:
    1. Load documents from data directory
    2. Chunk each document into semantic units
    3. Generate embeddings for each chunk
    4. Store chunks + embeddings in database
    """

    def __init__(
        self,
        data_dir: str = Config.DATA_DIR,
        chunk_size: int = Config.CHUNK_SIZE,
        chunk_overlap: int = Config.CHUNK_OVERLAP,
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            data_dir: Directory containing documents
            chunk_size: Target tokens per chunk
            chunk_overlap: Overlap between chunks
        """
        self.data_dir = Path(data_dir)
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.embedding_service = get_embedding_service()

        logger.info(f"Initialized ingestion pipeline (data_dir={data_dir})")

    def load_documents(self) -> List[Tuple[str, str]]:
        """
        Load all text documents from the data directory.

        Returns:
            List of (filename, content) tuples
        """
        if not self.data_dir.exists():
            logger.error(f"Data directory not found: {self.data_dir}")
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")

        documents = []

        # Load all .txt files
        for file_path in self.data_dir.glob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if content.strip():
                    documents.append((file_path.name, content))
                    logger.debug(f"Loaded {file_path.name} ({len(content)} chars)")
                else:
                    logger.warning(f"Empty file: {file_path.name}")

            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def ingest_documents(self) -> None:
        """
        Run the full ingestion pipeline.

        Steps:
        1. Load documents from disk
        2. Chunk each document
        3. Generate embeddings for chunks
        4. Store in database
        """
        logger.info("Starting document ingestion pipeline")

        # Step 1: Load documents
        documents = self.load_documents()
        if not documents:
            logger.warning("No documents found to ingest")
            return

        # Step 2: Create table if not exists
        Database.create_tables()

        # Step 3: Clear existing documents (fresh start)
        self._clear_existing_documents()

        # Step 4: Process each document
        total_chunks = 0

        for filename, content in documents:
            logger.info(f"Processing {filename}...")

            # Chunk the document
            chunks = self.chunker.chunk_text(content, metadata={"filename": filename})
            logger.info(f"  Created {len(chunks)} chunks")

            # Generate embeddings (batch for efficiency)
            chunk_texts = [chunk[0] for chunk in chunks]
            embeddings = self.embedding_service.embed_batch(chunk_texts)

            # Store in database
            stored_count = self._store_chunks(filename, chunks, embeddings)
            total_chunks += stored_count

            logger.info(f"  Stored {stored_count} chunks in database")

        logger.info(
            f"Ingestion complete! "
            f"Processed {len(documents)} documents, "
            f"stored {total_chunks} chunks"
        )

    def _clear_existing_documents(self) -> None:
        """Clear all existing documents from the database."""
        query = "DELETE FROM documents"
        Database.execute_query(query, fetch=False)
        logger.info("Cleared existing documents from database")

    def _store_chunks(
        self,
        filename: str,
        chunks: List[Tuple[str, int, dict]],
        embeddings: List[List[float]],
    ) -> int:
        """
        Store chunks and embeddings in the database.

        Args:
            filename: Source filename
            chunks: List of (text, token_count, metadata) tuples
            embeddings: List of embedding vectors

        Returns:
            Number of chunks stored successfully
        """
        stored_count = 0

        for chunk_index, ((chunk_text, token_count, metadata), embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            if embedding is None:
                logger.warning(
                    f"Skipping chunk {chunk_index} from {filename} (embedding failed)"
                )
                continue

            try:
                query = """
                INSERT INTO documents (filename, chunk_index, content, embedding, token_count)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (filename, chunk_index) 
                DO UPDATE SET 
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    token_count = EXCLUDED.token_count
                """

                Database.execute_query(
                    query,
                    (filename, chunk_index, chunk_text, embedding, token_count),
                    fetch=False,
                )

                stored_count += 1

            except Exception as e:
                logger.error(
                    f"Failed to store chunk {chunk_index} from {filename}: {e}"
                )

        return stored_count


def run_ingestion() -> None:
    """
    Convenience function to run the ingestion pipeline.

    Usage:
        python -c "from src.ingestion import run_ingestion; run_ingestion()"
    """
    pipeline = DocumentIngestion()
    pipeline.ingest_documents()


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=Config.LOG_LEVEL.value,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    run_ingestion()
