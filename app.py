import streamlit as st  # Turns this Python script into an interactive local web app
from dotenv import load_dotenv  # Loads GEMINI_API_KEY from .env
from google import genai  # Gemini SDK client
from sentence_transformers import SentenceTransformer  # Local embedding model

from ask_notes import answer_question  # Reuse our Milestone 5 RAG function
from index_notes import build_index  # Reuse our Milestone 3 indexer for the refresh button

load_dotenv()  # Ensure API key is loaded before initializing Gemini

st.set_page_config(page_title="Pocket Notes Q&A", page_icon="📓")
st.title("📓 Pocket Notes Q&A")
st.caption("Pure Python RAG: all-MiniLM-L6-v2 + ChromaDB + Gemini")


# Cache heavy resources so Streamlit doesn't reload the embedding model on every message
@st.cache_resource
def load_resources():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    genai_client = genai.Client()
    return embed_model, genai_client


embed_model, genai_client = load_resources()

# Sidebar controls for re-indexing notes and tuning retrieval live
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K Chunks", min_value=1, max_value=5, value=3)
    min_sim = st.slider("Min Similarity Threshold", 0.0, 0.8, 0.25, step=0.05)

    # Lets you add/edit files in notes/ and re-index without leaving the browser
    if st.button("🔄 Re-index Notes Folder"):
        with st.spinner("Chunking and embedding notes..."):
            build_index()
        st.success("Notes re-indexed!")

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render all previous messages on screen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box at the bottom of the page
if question := st.chat_input("Ask something from your notes..."):
    # Display and save the user's message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Run our Milestone 5 RAG pipeline and display the grounded answer
    with st.chat_message("assistant"):
        with st.spinner("Searching your notes..."):
            answer, used_chunks = answer_question(
                query=question,
                embedding_model=embed_model,
                genai_client=genai_client,
                top_k=top_k,
                min_similarity=min_sim,
            )
        st.markdown(answer)

        # Show retrieved chunks inside a collapsible drawer for easy debugging
        if used_chunks:
            with st.expander(f"Retrieved {len(used_chunks)} source chunk(s)"):
                for c in used_chunks:
                    st.markdown(f"**{c['source']}** (`{c['id']}` | similarity: `{c['similarity']}`)")
                    st.code(c["text"])

    st.session_state.messages.append({"role": "assistant", "content": answer})