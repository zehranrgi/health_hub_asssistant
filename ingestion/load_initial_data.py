"""
Initial Data Loading Script
Loads sample healthcare documents into ChromaDB vector store
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rag.vector_store import initialize_vector_store


def load_documents():
    """Load all sample healthcare documents into vector store"""

    print("🏥 CVS HealthHub AI - Initial Data Loading")
    print("=" * 60)

    # Initialize vector store
    print("\n📊 Initializing vector store...")
    vs = initialize_vector_store()

    # Get current stats
    initial_stats = vs.get_collection_stats()
    print(f"Current chunks in database: {initial_stats['total_chunks']}")

    # Define document files
    doc_dir = Path(__file__).parent.parent / "data" / "documents"
    document_files = {
        "medications.txt": "medication",
        "vaccines.txt": "vaccines",
        "drug_interactions.txt": "interactions",
        "cvs_services.txt": "services",
        "insurance_coverage.txt": "insurance"
    }

    total_added = 0

    # Load each document
    for filename, category in document_files.items():
        filepath = doc_dir / filename

        if not filepath.exists():
            print(f"\n⚠️  File not found: {filename}")
            continue

        print(f"\n📄 Loading {filename}...")

        try:
            # Read document
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Add to vector store with metadata
            chunks_added = vs.add_documents(
                documents=[content],
                metadatas=[{
                    "category": category,
                    "source": filename,
                    "type": "healthcare_information"
                }],
                ids=[f"{category}_{filename}"]
            )

            total_added += chunks_added
            print(f"   ✅ Added {chunks_added} chunks")

        except Exception as e:
            print(f"   ❌ Error loading {filename}: {str(e)}")

    # Final stats
    print("\n" + "=" * 60)
    print("📈 Loading Complete!")
    final_stats = vs.get_collection_stats()
    print(f"Total chunks in database: {final_stats['total_chunks']}")
    print(f"Chunks added this session: {total_added}")
    print("=" * 60)

    # Test search
    print("\n🔍 Testing semantic search...")
    test_query = "What are the side effects of Lisinopril?"
    results = vs.similarity_search(test_query, k=2)

    if results:
        print(f"\nQuery: \"{test_query}\"")
        print(f"Found {len(results)} results:\n")
        for idx, result in enumerate(results, 1):
            print(f"Result {idx}:")
            print(f"  Category: {result['metadata'].get('category', 'N/A')}")
            print(f"  Distance: {result.get('distance', 'N/A'):.4f}")
            print(f"  Content preview: {result['content'][:150]}...")
            print()
    else:
        print("⚠️  No results found in test search")

    print("✅ Data loading complete! You can now run the application.")


if __name__ == "__main__":
    load_documents()
