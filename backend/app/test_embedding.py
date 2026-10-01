import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found.")

client = genai.Client(api_key=api_key)

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents="What is Retrieval-Augmented Generation?"
)

embedding = response.embeddings[0].values

print(f"Embedding dimensions: {len(embedding)}")
print("First 10 values:")
print(embedding[:10])