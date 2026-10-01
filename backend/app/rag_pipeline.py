import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file."
    )

client = genai.Client(api_key=api_key)


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENT_PATH = Path("knowledge/rag.md")

EMBEDDINGS_PATH = Path("data/embeddings.json")

EMBEDDING_MODEL = "gemini-embedding-001"

GENERATION_MODEL = "gemini-3.5-flash-lite"


# ============================================================
# DOCUMENT LOADING
# ============================================================

def load_document(file_path):
    """
    Load a text or Markdown document.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    return path.read_text(
        encoding="utf-8"
    )


# ============================================================
# MARKDOWN CHUNKING
# ============================================================

def chunk_markdown(text):
    """
    Split Markdown into sections using ## headings.
    """

    sections = []

    current_section = []

    for line in text.splitlines():

        if line.startswith("## ") and current_section:

            sections.append(
                "\n".join(current_section).strip()
            )

            current_section = []

        current_section.append(line)

    if current_section:

        sections.append(
            "\n".join(current_section).strip()
        )

    return sections


# ============================================================
# EMBEDDINGS
# ============================================================

def create_embedding(text):
    """
    Create an embedding using Gemini.
    """

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    return response.embeddings[0].values


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors.
    """

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(
            a * a
            for a in vector_a
        )
    )

    magnitude_b = math.sqrt(
        sum(
            b * b
            for b in vector_b
        )
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )


# ============================================================
# CREATE AND SAVE EMBEDDINGS
# ============================================================

def create_and_save_embeddings(chunks):
    """
    Create embeddings for all chunks and save them.
    """

    print(
        "Creating embeddings for knowledge base..."
    )

    embedding_records = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        print(
            f"Embedding chunk {index}/{len(chunks)}..."
        )

        lines = chunk.splitlines()

        section = (
            lines[0].replace("## ", "").strip()
            if lines
            else f"Section {index}"
        )

        embedding = create_embedding(
            chunk
        )

        embedding_records.append(
            {
                "chunk_id": index,
                "document_name": DOCUMENT_PATH.name,
                "section": section,
                "text": chunk,
                "embedding": embedding,
            }
        )

    EMBEDDINGS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_data = {
        "document_modified": DOCUMENT_PATH.stat().st_mtime,
        "chunks": embedding_records,
    }

    with open(
        EMBEDDINGS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            cache_data,
            file,
            indent=2,
        )

    print(
        f"Saved embeddings to: {EMBEDDINGS_PATH}"
    )

    return cache_data


# ============================================================
# LOAD SAVED EMBEDDINGS
# ============================================================

def load_saved_embeddings():
    """
    Load embeddings from the JSON cache.
    """

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            "Embeddings cache does not exist."
        )

    with open(
        EMBEDDINGS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        cache_data = json.load(file)

    return cache_data


# ============================================================
# CACHE VALIDATION
# ============================================================

def cache_is_valid():
    """
    Check whether the saved embeddings cache is valid.

    Returns:
        True  -> cache can be reused
        False -> cache must be rebuilt
    """

    if not EMBEDDINGS_PATH.exists():
        return False

    try:

        cache_data = load_saved_embeddings()

        # The cache must be a dictionary.
        if not isinstance(
            cache_data,
            dict,
        ):
            return False

        cached_time = cache_data.get(
            "document_modified"
        )

        cached_chunks = cache_data.get(
            "chunks"
        )

        if cached_time is None:
            return False

        if not isinstance(
            cached_chunks,
            list,
        ):
            return False

        current_time = DOCUMENT_PATH.stat().st_mtime

        return cached_time == current_time

    except Exception:

        return False


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_chunks(
    question,
    chunk_data,
    top_k=3,
):
    """
    Retrieve the most relevant chunks for a question.
    """

    question_embedding = create_embedding(
        question
    )

    chunks = chunk_data["chunks"]

    scored_chunks = []

    for chunk in chunks:

        score = cosine_similarity(
            question_embedding,
            chunk["embedding"],
        )

        result = chunk.copy()

        result["score"] = score

        scored_chunks.append(
            result
        )

    scored_chunks.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored_chunks[:top_k]


# ============================================================
# GENERATION
# ============================================================

def generate_answer(
    question,
    retrieved_chunks,
):
    """
    Generate a grounded answer using Gemini.
    """

    source_text = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):

        source_text.append(
            f"""
SOURCE {index}

Document:
{chunk["document_name"]}

Section:
{chunk["section"]}

Similarity:
{chunk["score"]:.4f}

Content:
{chunk["text"]}
"""
        )

    context = "\n".join(
        source_text
    )

    prompt = f"""
You are an AI learning assistant.

Answer the user's question using the provided
knowledge-base sources.

Important rules:

1. Prefer the provided sources over general knowledge.
2. Do not invent information that is not supported
   by the sources.
3. Explain the concept clearly for someone learning AI.
4. Use simple language.
5. Include a practical example when useful.
6. At the end, include a short "Sources" section.
7. In the Sources section, mention the source numbers
   that support the answer.

User question:

{question}

Knowledge-base sources:

{context}

Now provide a clear, educational answer.
"""

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    return response.text


# ============================================================
# CLI PIPELINE
# ============================================================

def run_pipeline(question):
    """
    Run the complete RAG pipeline.
    """

    text = load_document(
        DOCUMENT_PATH
    )

    chunks = chunk_markdown(
        text
    )

    if cache_is_valid():

        print(
            f"Loading saved embeddings from: "
            f"{EMBEDDINGS_PATH}"
        )

        chunk_data = load_saved_embeddings()

        print(
            f"Loaded "
            f"{len(chunk_data['chunks'])} "
            f"saved embeddings."
        )

    else:

        print(
            "Embedding cache is missing or outdated."
        )

        chunk_data = (
            create_and_save_embeddings(
                chunks
            )
        )

    print(
        "\nSearching knowledge base..."
    )

    retrieved = retrieve_chunks(
        question,
        chunk_data,
        top_k=3,
    )

    print(
        "\nTop sources:"
    )

    for index, chunk in enumerate(
        retrieved,
        start=1,
    ):

        print(
            f"{index}. "
            f"{chunk['section']} "
            f"(similarity: "
            f"{chunk['score']:.4f})"
        )

    print(
        "\nGenerating answer..."
    )

    answer = generate_answer(
        question,
        retrieved,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ANSWER"
    )

    print(
        "=" * 70
    )

    print(
        answer
    )

    return answer


# ============================================================
# INTERACTIVE CLI
# ============================================================

if __name__ == "__main__":

    print(
        "\nAI Concept Explorer"
    )

    print(
        "Type 'exit' to quit."
    )

    while True:

        question = input(
            "\nAsk an AI question: "
        ).strip()

        if question.lower() == "exit":

            print(
                "Goodbye!"
            )

            break

        if not question:

            print(
                "Please enter a question."
            )

            continue

        try:

            run_pipeline(
                question
            )

        except Exception as error:

            print(
                "\nError:"
            )

            print(
                error
            )