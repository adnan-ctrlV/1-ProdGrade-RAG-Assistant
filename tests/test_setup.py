from src.config import Config
from src.database import Database
from src.chunking import TextChunker

# Test config
print("✅ Config loaded:")
print(f"  Environment: {Config.ENVIRONMENT}")
print(f"  Model: {Config.OPENAI_MODEL}")

# Test database
print("\n✅ Database connection:")
with Database.get_connection() as conn:
    print("  Connected successfully")

# Test chunking
chunker = TextChunker()
text = "This is sentence one. This is sentence two. This is sentence three."
chunks = chunker.chunk_text(text)
print(f"\n✅ Chunking works: {len(chunks)} chunks created")
