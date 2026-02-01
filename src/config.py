"""
This module centralizes all environment configurations.
This makes it easier to switch between dev/prod/test environemnts
"""

import os
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Config:
    # Enviornment
    ENVIRONMENT: Environment = Environment(os.getenv("ENVIRONMENT", "development"))

    # Logging
    LOG_LEVEL: LogLevel = LogLevel(os.getenv("LOG_LEVEL", "INFO"))

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-40-mini")
    OPENAI_EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "test-embedding-3-small"
    )
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    OPENAI_MAX_RETRIES: int = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
    OPENAI_TIMEOUT: int = int(os.getenv("OPENAI_TIMEOUT", "30"))

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    DB_POOL_MIN_CONN: int = int(os.getenv("DB_POOL_MIN_CONN", "2"))
    DB_POOL_MAX_CONN: int = int(os.getenv("DB_POOL_MAX_CONN", "10"))

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Retrieval
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    RETRIEVAL_MIN_SCORE: float = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.3"))

    # Data paths
    DATA_DIR: str = os.getenv("DATA_DIR", "data/policies")

    @classmethod
    def validate(cls) -> None:
        errors = []

        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is not set")

        if not cls.DATABASE_URL:
            errors.append("DATABASE_URL is not set")

        if errors:
            raise RuntimeError(
                "Configuration validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    @classmethod
    def is_production(cls) -> bool:
        # Checks if running in production or not
        return cls.ENVIRONMENT == Environment.PRODUCTION


Config.validate()
# This is set to fail fast.
# The app won't even start without validating configurations.
# Or else, the app will start, user types questions, and then it might crash.
