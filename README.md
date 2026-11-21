# Pocket Notes Q&A (Pure-Python RAG)

A zero-framework Retrieval-Augmented Generation (RAG) CLI app built in pure Python to query local `.txt` and `.md` notes with strict grounding and source citations.

## Architecture
1. **Chunking (`chunk_notes.py`):** Splits local notes into overlapping character windows (`chunk_size=150`, `overlap=30`) with filename metadata.
2. **Indexing (`index_notes.py`):** Generates 384-dim local embeddings using `all-MiniLM-L6-v2` and persists them to ChromaDB (`cosine` distance) via `upsert()`.
3. **Retrieval (`retrieve.py`):** Embeds user queries and fetches Top-K chunks with similarity scores (`1 - cosine_distance`).
4. **Grounded Generation (`ask_notes.py`):** Two-layer hallucination defense:
   - **Layer 1 (Local Gate):** Blocks queries below `min_similarity=0.25` locally (0 API calls).
   - **Layer 2 (Strict Prompt):** Calls `gemini-3.5-flash-lite` at `temperature=0.0` with a strict system instruction to answer only from retrieved excerpts and cite sources.
5. **Failure-Mode Testing (`debug_rag.py`):** Isolated in-memory (`EphemeralClient`) experiments testing shredded chunk sizes and conflicting notes.

## Quick Start (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt