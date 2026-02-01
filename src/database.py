"""
This module handles:
- connection pooling
- pgvector extension setup
- Safe query execution + automatic connection management
"""

import logging
from contextlib import contextmanager
from typing import Generator, List, Tuple, Optional

from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector


from src.config import Config

logger = logging.getLogger(__name__)
# __name__ helps you understand where the log came from (debug errors easily)
# because __name__ is assigned to database in database file, config in config file, etc
# __name__ is assgined to main in main.py file
# That'a why you see __name__ == "__main__"


class Database:
    _pool: Optional[ConnectionPool] = None
    # _variable = private (internal use only)
    # variabe = public

    @classmethod
    def initialize(cls) -> None:
        # Initialize the connection pool.

        if cls._pool is not None:
            logger.warning("Database pool already initialized")
            return

        try:
            logger.info(
                f"Initializing database pool "
                f"(min={Config.DB_POOL_MIN_CONN}, max={Config.DB_POOL_MAX_CONN})"
            )

            cls._pool = ConnectionPool(
                conninfo=Config.DATABASE_URL,
                min_size=Config.DB_POOL_MIN_CONN,
                max_size=Config.DB_POOL_MAX_CONN,
            )

            cls._setup_pgvector()

            logger.info("Database pool initialized successfully")

        except Exception as e:
            logger.critical(f"Failed to initialize database pool: {e}")
            raise

    @classmethod
    def _setup_pgvector(cls) -> None:
        """
        Enable the pgvector extension if not already enabled.

        This allows us to store and search vector embeddings.
        """
        with cls.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
                logger.info("pgvector extension enabled")

    @classmethod
    @contextmanager
    def get_connection(cls) -> Generator:
        """
        Get a connection from the pool using a context manager.
        The connection is automatically returned to the pool when done.
        """

        if cls._pool is None:
            raise RuntimeError(
                "Database pool not initialized. Call Database.initialize() first."
            )

        conn = None
        try:
            with cls._pool.connection() as conn:
                register_vector(conn)
                yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise

    @classmethod
    def close_pool(cls) -> None:
        """
        Close all connections in the pool.
        Should be called at application shutdown.
        """
        if cls._pool:
            cls._pool.closeall()
            cls._pool = None
            logger.info("Database pool closed")

    @classmethod
    def execute_query(
        cls, query: str, params: Optional[Tuple] = None, fetch: bool = True
    ) -> Optional[List]:
        # Execute a SQL query safely.

        with cls.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, params)

                if fetch:
                    return cur.fetchall()
                else:
                    conn.commit()
                    return None

    @classmethod
    def create_tables(cls) -> None:
        """
        Create the necessary tables for document storage.

        Schema:
        - documents: Stores document chunks with embeddings
        - Includes: id, content, embedding (vector), metadata
        """
        schema = """
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding vector(1536),  -- text-embedding-3-small dimension
            token_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(filename, chunk_index)
        );
        
        -- Index for fast vector similarity search
        CREATE INDEX IF NOT EXISTS documents_embedding_idx 
        ON documents USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        """

        # ivvflat = Inverted File with Flat Structure (group similar vectors into clusters for faster retrievals)

        cls.execute_query(schema, fetch=False)
        logger.info("Database tables created successfully")


# Initialize on import (if not in testing mode)
if not Config.ENVIRONMENT.value == "testing":
    Database.initialize()
