from src.database import Database

# Check how many chunks were stored
with Database.get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents")
        count = cur.fetchone()[0]
        print(f"✅ Total chunks in database: {count}")

        cur.execute("SELECT filename, COUNT(*) FROM documents GROUP BY filename")
        results = cur.fetchall()
        print("\n📄 Chunks per document:")
        for filename, chunk_count in results:
            print(f"  {filename}: {chunk_count} chunks")
