from pathlib import Path  # Handles Windows/Linux file paths cleanly without slash bugs


def split_text_with_overlap(text: str, chunk_size: int = 150, overlap: int = 30) -> list[str]:
    """Split a string into overlapping character chunks."""
    # Safety check: if overlap >= chunk_size, the window would never move forward (infinite loop)
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []  # Will hold all finished text pieces for this single file
    start = 0  # Tracks the starting character index of the current window
    text_length = len(text)  # Total number of characters in the note

    # Keep sliding the window until we have covered the entire text
    while start < text_length:
        end = start + chunk_size  # Calculate where this chunk ends
        chunk = text[start:end].strip()  # Slice the text and remove leading/trailing whitespace

        if chunk:  # Only keep non-empty chunks
            chunks.append(chunk)  # Add the cleaned chunk to our list

        # If this slice already reached (or passed) the end of the file, stop looping
        if end >= text_length:
            break

        # Advance the start pointer, stepping back by 'overlap' characters to preserve context
        start += chunk_size - overlap

    return chunks  # Return all chunks created from this note


def load_and_chunk_notes(notes_dir: str = "notes", chunk_size: int = 150, overlap: int = 30) -> list[dict]:
    """Read all .txt/.md files in notes_dir and return structured chunk dictionaries."""
    folder = Path(notes_dir)  # Convert the folder string into a Path object
    all_chunks = []  # Master list holding chunks across all files

    # Find every .txt and .md file, sorted alphabetically for consistent IDs
    note_files = sorted(
        f for f in folder.iterdir() if f.is_file() and f.suffix in {".txt", ".md"}
    )

    # Loop through each note file one by one
    for file_path in note_files:
        content = file_path.read_text(encoding="utf-8")  # Read full file text safely as UTF-8
        file_chunks = split_text_with_overlap(content, chunk_size, overlap)  # Split into overlapping chunks

        # Attach metadata to each chunk so ChromaDB (Milestone 3) knows where it came from
        for index, chunk_text in enumerate(file_chunks):
            all_chunks.append(
                {
                    "id": f"{file_path.stem}_chunk_{index}",  # Unique ID, e.g., 'python_venv_chunk_0'
                    "source": file_path.name,  # Original filename, e.g., 'python_venv.txt'
                    "text": chunk_text,  # The actual text content of this chunk
                }
            )

    return all_chunks  # Return the full list of chunk dictionaries


# Only run the inspection printout when this file is executed directly (not when imported in Milestone 3)
if __name__ == "__main__":
    chunks = load_and_chunk_notes(notes_dir="notes", chunk_size=150, overlap=30)

    print(f"Total chunks created: {len(chunks)}\n")  # Show overall chunk count

    # Print each chunk clearly so we can verify the overlap with our own eyes
    for c in chunks:
        print("=" * 60)
        print(f"ID: {c['id']} | Source: {c['source']} | Length: {len(c['text'])} chars")
        print("-" * 60)
        print(c["text"])
        print()