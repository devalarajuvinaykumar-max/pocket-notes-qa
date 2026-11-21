import chromadb  # Provides EphemeralClient for disposable in-memory experiments
from dotenv import load_dotenv  # Loads GEMINI_API_KEY from .env
from google import genai  # Official Google Gemini SDK
from google.genai import types  # Configures system_instruction and temperature
from sentence_transformers import SentenceTransformer  # Local embedding model

from ask_notes import SYSTEM_PROMPT  # Reuse our strict grounded prompt from Milestone 5
from chunk_notes import split_text_with_overlap  # Reuse our Milestone 2 chunker

load_dotenv()  # Load GEMINI_API_KEY into the environment


def run_experiment(
    title: str,
    collection_name: str,
    notes: dict[str, str],
    query: str,
    chunk_size: int,
    overlap: int,
    embed_model: SentenceTransformer,
    genai_client: genai.Client,
) -> None:
    """Run an isolated in-memory RAG test to observe how a specific setting breaks or behaves."""
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {title}")
    print(f"Settings  : chunk_size={chunk_size}, overlap={overlap}")
    print(f"Query     : {query}")
    print("=" * 70)

    # 1. Create a temporary in-memory ChromaDB client (never touches your real chroma_db/ folder)
    temp_client = chromadb.EphemeralClient()
    collection = temp_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # 2. Chunk every test note using the experiment's chunk_size and overlap
    ids, docs, metas = [], [], []
    for filename, text in notes.items():
        pieces = split_text_with_overlap(text, chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            ids.append(f"{filename}_{i}")  # Unique ID per test chunk
            docs.append(piece)  # Chunk text
            metas.append({"source": filename})  # Source filename for citation

    # 3. Embed and store the test chunks in our temporary collection
    embeddings = embed_model.encode(docs).tolist()
    collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)

    # 4. Retrieve top 3 chunks for the test query
    query_vec = embed_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_vec,
        n_results=min(3, len(docs)),  # Avoid asking for more results than exist
        include=["documents", "metadatas", "distances"],
    )

    # 5. Print what the retriever actually fetched so we can diagnose Retrieval vs. Generation
    context_blocks = []
    print("\n--- Retrieved Chunks ---")
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        sim = round(1 - dist, 4)  # Convert cosine distance to similarity score
        print(f"[{meta['source']} | sim={sim}] -> {doc!r}")  # !r shows exact string & newlines
        context_blocks.append(f"[{meta['source']} | Similarity: {sim}]\n{doc}")

    # 6. Send the retrieved chunks to Gemini using our strict grounded prompt
    user_prompt = f"NOTES:\n" + "\n\n---\n\n".join(context_blocks) + f"\n\nQUESTION:\n{query}"
    response = genai_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
        ),
    )

    print("\n--- Gemini's Answer ---")
    print(response.text.strip())


if __name__ == "__main__":
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = genai.Client()

    # EXPERIMENT 1: Shred the note with tiny chunks and zero overlap
    venv_note = {
        "python_venv.txt": (
            "A Python virtual environment isolates project dependencies from the global system Python.\n"
            "To create one on Windows, run: python -m venv .venv\n"
            "To activate it in PowerShell, run: .\\.venv\\Scripts\\Activate.ps1\n"
            "Always add .venv/ to .gitignore so you do not upload gigabytes of libraries to GitHub."
        )
    }
    run_experiment(
        title="1. Bad Chunk Size (Shredded Context)",
        collection_name="exp_bad_chunks",
        notes=venv_note,
        query="How do I activate a virtual environment on Windows?",
        chunk_size=35,  # Way too small! Splits the command away from the word "activate"
        overlap=0,  # Zero overlap guarantees broken boundaries
        embed_model=embed_model,
        genai_client=client,
    )

    # EXPERIMENT 2: Two notes that directly contradict each other
    conflicting_notes = {
        "study_schedule_old.md": "- Monday: 60 minutes of Python and vector databases.",
        "study_schedule_new.md": "- Monday: 90 minutes of System Design and Docker.",
    }
    run_experiment(
        title="2. Conflicting Notes",
        collection_name="exp_conflicts",
        notes=conflicting_notes,
        query="What do I study on Monday and for how long?",
        chunk_size=150,
        overlap=30,
        embed_model=embed_model,
        genai_client=client,
    )