import os  # Gives access to environment variables and operating system paths
from pathlib import Path  # Modern, clean way to handle Windows/Linux file paths
from dotenv import load_dotenv  # Reads key=value pairs from our .env file

# Load variables from .env into the current Python process's environment
load_dotenv()

# 1. Verify our required third-party libraries can be imported from .venv
import chromadb  # Local vector database library
import sentence_transformers  # Local text-to-embedding model library
from google import genai  # Official Google Gemini Python SDK

# 2. Fetch the API key from the environment (returns None if missing)
api_key = os.getenv("GEMINI_API_KEY")

# Check whether the key exists and is not still the placeholder text
key_loaded = bool(api_key and "paste_your_actual_key" not in api_key)

# 3. Point to the 'notes' directory and collect all .txt and .md files
notes_dir = Path("notes")  # Represents the ./notes folder relative to this script
note_files = [
    f.name  # Store just the filename (e.g., 'git_workflow.md')
    for f in notes_dir.iterdir()  # Loop through every item inside notes/
    if f.is_file() and f.suffix in {".txt", ".md"}  # Keep only .txt and .md files
]

# 4. Print a clear status report so we know Milestone 1 is solid
print("Libraries imported : OK (chromadb, sentence-transformers, google-genai)")
print(f"API key loaded     : {'OK' if key_loaded else 'MISSING - check .env'}")
print(f"Notes found ({len(note_files)})    : {sorted(note_files)}")