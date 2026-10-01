import os
import math
from pathlib import Path

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found.")

client = genai.Client(api_key=api_key)


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


def create_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return response.embeddings[0].values


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (magnitude_a * magnitude_b)


if __name__ == "__main__":
    document_path = "knowledge/rag.md"

    text = load_document(document_path)
    chunks = chunk_markdown(text)

    print(f"Loaded {len(chunks)} chunks.")

    print("\nCreating embeddings...")

    chunk_data = []

    for i, chunk in enumerate(chunks):
        embedding = create_embedding(chunk)

        chunk_data.append({
            "chunk_id": i + 1,
            "text": chunk,
            "embedding": embedding
        })

        print(f"Embedded chunk {i + 1}")

    question = "What are the main components of a RAG system?"

    print(f"\nQuestion: {question}")

    question_embedding = create_embedding(question)

    results = []

    for chunk in chunk_data:
        score = cosine_similarity(
            question_embedding,
            chunk["embedding"]
        )

        results.append({
            "chunk_id": chunk["chunk_id"],
            "score": score,
            "text": chunk["text"]
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    print("\nMost relevant chunks:")

    for result in results[:3]:
        print("\n------------------------------")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Similarity: {result['score']:.4f}")
        print(result["text"])