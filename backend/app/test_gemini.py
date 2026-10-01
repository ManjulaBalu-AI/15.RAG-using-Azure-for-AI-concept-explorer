import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found.")

client = genai.Client(api_key=api_key)

for attempt in range(3):
    try:
        print(f"Attempt {attempt + 1}...")

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents="Explain RAG in one simple paragraph."
        )

        print("\nGemini response:\n")
        print(response.text)
        break

    except Exception as e:
        print(f"\nRequest failed: {e}")

        if attempt < 2:
            wait_time = 5 * (2 ** attempt)
            print(f"Retrying in {wait_time} seconds...\n")
            time.sleep(wait_time)
        else:
            print("\nGemini request failed after 3 attempts.")