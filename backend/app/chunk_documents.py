from pathlib import Path


def load_document(file_path):
    path = Path(file_path)
    return path.read_text(encoding="utf-8")


def chunk_markdown(text):
    sections = []
    current_section = []

    for line in text.splitlines():
        if line.startswith("## ") and current_section:
            sections.append("\n".join(current_section))
            current_section = []

        current_section.append(line)

    if current_section:
        sections.append("\n".join(current_section))

    return sections


if __name__ == "__main__":
    document_path = "knowledge/rag.md"

    text = load_document(document_path)
    chunks = chunk_markdown(text)

    print(f"Document length: {len(text)} characters")
    print(f"Number of chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)