import chromadb  # Local vector database that stores embeddings on disk
from sentence_transformers import SentenceTransformer  # Runs local embedding models on CPU

from chunk_notes import load_and_chunk_notes  # Reuse our Milestone 2 chunker (no duplicated code)


def build_index(notes_dir: str = "notes", db_dir: str = "chroma_db") -> None:
    """Chunk all notes, generate local embeddings, and persist them in ChromaDB."""
    # 1. Load and split all .txt/.md files into overlapping chunk dictionaries
    chunks = load_and_chunk_notes(notes_dir=notes_dir, chunk_size=150, overlap=30)

    # Stop early if the notes folder is empty so we don't pass empty lists to ChromaDB
    if not chunks:
        print("No notes found to index.")
        return

    # 2. Load the free, lightweight local embedding model (downloads ~90 MB on first run only)
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. Unpack our chunk dicts into the 3 parallel lists ChromaDB requires
    ids = [c["id"] for c in chunks]  # Unique string ID per chunk (e.g., 'python_venv_chunk_0')
    documents = [c["text"] for c in chunks]  # Raw text strings to embed and store
    metadatas = [{"source": c["source"]} for c in chunks]  # Filename metadata for citations later

    # 4. Convert every chunk string into a 384-dimensional vector and convert NumPy array -> Python list
    print(f"Generating embeddings for {len(documents)} chunks...")
    embeddings = model.encode(documents).tolist()

    # 5. Connect to a persistent ChromaDB database stored inside the local 'chroma_db/' folder
    client = chromadb.PersistentClient(path=db_dir)

    # Create or load our collection; 'cosine' measures the angle (meaning similarity) between vectors
    collection = client.get_or_create_collection(
        name="pocket_notes",
        metadata={"hnsw:space": "cosine"},
    )

    # 6. Save everything to disk ('upsert' inserts new IDs or overwrites existing ones safely)
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # 7. Print concrete verification stats so we can inspect what actually got stored
    vector_dim = len(embeddings[0])  # Check how many numbers represent a single chunk
    sample_values = [round(v, 4) for v in embeddings[0][:5]]  # Grab first 5 numbers of Chunk 0

    print("\n--- Indexing Complete ---")
    print(f"Chunks indexed       : {len(chunks)}")
    print(f"Embedding dimensions : {vector_dim}")
    print(f"Chunk 0 first 5 nums : {sample_values}")
    print(f"Total in ChromaDB    : {collection.count()}")
    print(f"Database saved to    : ./{db_dir}/")


if __name__ == "__main__":
    build_index()