# test_rag.py

import logging
from src.rag import query

# Configure logging to see what's happening
logging.basicConfig(
    level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Test questions
questions = [
    "What is the remote work policy?",
    "How many days of leave do employees get?",
    "When are expense reports due?",
    "What is the company's stance on artificial intelligence?",  # Should find no results
]

print("=" * 80)
print("TESTING RAG SYSTEM")
print("=" * 80)

for i, question in enumerate(questions, 1):
    print(f"\n\n{'=' * 80}")
    print(f"QUESTION {i}: {question}")
    print("=" * 80)

    result = query(question)

    print("\n📝 ANSWER:")
    print(result["answer"])

    print("\n📊 METADATA:")
    print(f"  Chunks retrieved: {result['chunks_retrieved']}")
    print(f"  Tokens used: {result['tokens_used']}")
    print(f"  Success: {result['success']}")

    if result.get("sources"):
        print("\n📚 SOURCES:")
        for source in result["sources"]:
            print(
                f"  - {source['filename']} (chunks: {source['chunks_used']}, score: {source['max_score']:.3f})"
            )
    else:
        print("\n📚 SOURCES: None")

    print("\n" + "-" * 80)

print("\n\n" + "=" * 80)
print("ALL TESTS COMPLETE")
print("=" * 80)
