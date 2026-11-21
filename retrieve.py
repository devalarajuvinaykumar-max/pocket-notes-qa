import chromadb  # Connects to our saved local vector database
from sentence_transformers import SentenceTransformer  # Turns the user's question into a vector


def retrieve_chunks(
    query: str,
    top_k: int = 3,
    db_dir: str = "chroma_db",
    model: SentenceTransformer | None = None,
) -> list[dict]:
    """Embed a user query and return the top_k most similar note chunks from ChromaDB."""
    # Load the embedding model if the caller didn't pass an already-loaded one
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    # Connect to the existing ChromaDB folder we created in Milestone 3
    client = chromadb.PersistentClient(path=db_dir)

    # Open our saved collection (will raise an error if index_notes.py hasn't been run yet)
    collection = client.get_collection(name="pocket_notes")

    # Embed the query as a 1-item list and convert NumPy array -> standard Python list
    query_embedding = model.encode([query]).tolist()

    # Search ChromaDB for the top_k nearest vectors to our query vector
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],  # Ask for text, filename, and distance score
    )

    # ChromaDB returns nested lists (one inner list per query), so [0] grabs our single query's results
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    matches = []  # Will store clean, structured match dictionaries

    # Pair up all 4 parallel lists so each chunk has its ID, source, similarity, and text together
    for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
        # Convert cosine distance (0 = identical) into similarity (1 = identical)
        similarity = round(1 - dist, 4)

        matches.append(
            {
                "id": chunk_id,  # e.g., 'python_venv_chunk_0'
                "source": meta["source"],  # Original note filename
                "similarity": similarity,  # Higher = closer meaning (typically 0.0 to 1.0)
                "text": doc,  # The matched chunk text
            }
        )

    return matches  # Return matches sorted from most similar to least similar


if __name__ == "__main__":
    # Load the model ONCE here so we don't reload it for every test query
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Test 1: Synonym query (should score high) | Test 2: Off-topic query (should score low)
    test_queries = [
        "How do I keep Python libraries separate?",
        "What is the recipe for chicken biryani?",
    ]

    for q in test_queries:
        print("\n" + "=" * 65)
        print(f"QUERY: {q}")
        print("=" * 65)

        hits = retrieve_chunks(query=q, top_k=3, model=embedding_model)

        # Print each retrieved chunk with its rank, similarity score, and source file
        for rank, hit in enumerate(hits, start=1):
            print(f"\n[Rank {rank}] Similarity: {hit['similarity']} | Source: {hit['source']} ({hit['id']})")
            print("-" * 65)
            print(hit["text"])