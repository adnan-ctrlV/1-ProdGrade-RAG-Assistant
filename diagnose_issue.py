import logging
from src.database import Database
from src.embeddings import get_embedding_service

logging.basicConfig(level="DEBUG")

print("=" * 80)
print("DIAGNOSING RAG SYSTEM")
print("=" * 80)

# Test 1: Check database
print("\n1. CHECKING DATABASE...")
with Database.get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents")
        total = cur.fetchone()[0]
        print(f"   Total documents: {total}")

        cur.execute("SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL")
        with_embeddings = cur.fetchone()[0]
        print(f"   Documents with embeddings: {with_embeddings}")

        if with_embeddings == 0:
            print("   ❌ PROBLEM: No embeddings in database!")
        else:
            print("   ✅ Embeddings exist")

# Test 2: Test embedding service
print("\n2. TESTING EMBEDDING SERVICE...")
embedding_service = get_embedding_service()
test_text = "What is the remote work policy?"

embedding = embedding_service.embed_text(test_text)

if embedding:
    print(f"   Generated embedding dimension: {len(embedding)}")
    print(
        f"   Is normalized? norm = {sum(x * x for x in embedding[:100]) ** 0.5:.4f}"
    )  # ≈1.0 = normalized

if embedding:
    print(f"   ✅ Embedding generated: {len(embedding)} dimensions")
    print(f"   First 5 values: {embedding[:5]}")
else:
    print("   ❌ PROBLEM: Embedding service failed!")

# Test 3: Test vector search manually
print("\n3. TESTING VECTOR SEARCH...")
if embedding:
    with Database.get_connection() as conn:
        with conn.cursor() as cur:
            pg_array = "[" + ",".join(map(str, embedding)) + "]"

            cur.execute(
                """
                SELECT filename, 
                       1 - (embedding <=> %s::vector) AS score
                FROM documents
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT 3
            """,
                (pg_array, pg_array),
            )

            results = cur.fetchall()

            if results:
                print("   ✅ Vector search works:")
                for row in results:
                    print(f"      {row[0]}: score={row[1]:.3f}")
            else:
                print("   ❌ PROBLEM: Vector search returned nothing!")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
