from dotenv import load_dotenv  # Loads GEMINI_API_KEY from .env into environment variables
from google import genai  # Official Google Gemini Python SDK
from google.genai import types  # Lets us configure system instructions and temperature
from sentence_transformers import SentenceTransformer  # Local embedding model

from retrieve import retrieve_chunks  # Reuse our Milestone 4 retrieval function

# Load environment variables from .env before creating the Gemini client
load_dotenv()

# Strict rules that force Gemini to act like an open-book exam taker
SYSTEM_PROMPT = """You are a strict personal notes assistant.
Answer the user's question using ONLY the provided note excerpts.
Rules:
1. Never use outside knowledge, even if you know the answer.
2. If the provided notes do not contain the exact answer, reply ONLY with: "I couldn't find that in your notes."
3. Keep the answer concise and cite the source filename in brackets, e.g. [python_venv.txt]."""


def answer_question(
    query: str,
    embedding_model: SentenceTransformer,
    genai_client: genai.Client,
    top_k: int = 3,
    min_similarity: float = 0.25,
) -> tuple[str, list[dict]]:
    """Retrieve relevant chunks and generate a strictly grounded answer using Gemini."""
    # 1. Fetch the top_k most similar chunks from ChromaDB
    matches = retrieve_chunks(query=query, top_k=top_k, model=embedding_model)

    # 2. Defense Layer 1: Filter out low-similarity chunks so junk never reaches the LLM
    relevant_chunks = [m for m in matches if m["similarity"] >= min_similarity]

    # Stop early and save an API call if nothing in our notes is relevant enough
    if not relevant_chunks:
        return "I couldn't find that in your notes. (Blocked locally: low similarity)", []

    # 3. Combine the surviving chunks into one labeled context block for the prompt
    context_blocks = [
        f"[Source: {c['source']} | Similarity: {c['similarity']}]\n{c['text']}"
        for c in relevant_chunks
    ]
    context_text = "\n\n---\n\n".join(context_blocks)  # Separate chunks visually with dividers

    # 4. Build the user prompt pairing the retrieved notes with the actual question
    user_prompt = f"NOTES:\n{context_text}\n\nQUESTION:\n{query}"

    # 5. Defense Layer 2: Call Gemini with our strict system prompt and temperature=0.0
    response = genai_client.models.generate_content(
        model="gemini-3.5-flash-lite",  # Fast, free-tier Gemini model
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,  # Enforces "notes only, no outside knowledge"
            temperature=0.0,  # Zero randomness -> more factual and repeatable answers
        ),
    )

    # Return both the generated answer text and the chunks used so we can inspect sources
    return response.text.strip(), relevant_chunks


if __name__ == "__main__":
    # Load both heavy objects ONCE before starting the interactive loop
    print("Starting Pocket Notes Q&A...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = genai.Client()  # Automatically reads GEMINI_API_KEY from the environment

    print("Ready! Ask a question about your notes (type 'exit' to quit).\n")

    # Simple CLI loop so you can test multiple questions in one run
    while True:
        question = input("You: ").strip()  # Read user input and strip extra spaces

        if not question:  # Ignore accidental empty Enter presses
            continue

        if question.lower() in {"exit", "quit"}:  # Clean exit condition
            print("Goodbye!")
            break

        # Run the full RAG pipeline (Retrieve -> Filter -> Generate)
        answer, used_chunks = answer_question(question, embed_model, client)

        print(f"\nAssistant:\n{answer}")

        # Show which chunks were passed to the LLM for transparency
        if used_chunks:
            sources = ", ".join(f"{c['source']} ({c['similarity']})" for c in used_chunks)
            print(f"[Retrieved: {sources}]")

        print("\n" + "-" * 60 + "\n")