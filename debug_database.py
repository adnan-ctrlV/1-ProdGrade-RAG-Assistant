from src.database import Database

print("Checking database contents...")

with Database.get_connection() as conn:
    with conn.cursor() as cur:
        # Check total documents
        cur.execute("SELECT COUNT(*) FROM documents")
        total = cur.fetchone()[0]
        print(f"\n✅ Total documents: {total}")

        # Check if embeddings are NULL
        cur.execute("SELECT COUNT(*) FROM documents WHERE embedding IS NULL")
        null_embeddings = cur.fetchone()[0]
        print(f"❌ Documents with NULL embeddings: {null_embeddings}")

        # Check embedding dimensions
        cur.execute("""
            SELECT filename, 
                   vector_dims(embedding) AS dimension
            FROM documents 
            WHERE embedding IS NOT NULL
            LIMIT 5
        """)
        results = cur.fetchall()

        if results:
            print("\n📊 Embedding dimensions:")
            for row in results:
                print(f"  {row[0]}: {row[1]} dimensions")
        else:
            print("\n❌ NO EMBEDDINGS FOUND IN DATABASE!")

        # Show sample document
        cur.execute("""
            SELECT filename, chunk_index, content, 
                   embedding IS NOT NULL as has_embedding
            FROM documents 
            LIMIT 1
        """)
        sample = cur.fetchone()
        print("\n📄 Sample document:")
        print(f"  Filename: {sample[0]}")
        print(f"  Chunk index: {sample[1]}")
        print(f"  Content: {sample[2][:100]}...")
        print(f"  Has embedding: {sample[3]}")
