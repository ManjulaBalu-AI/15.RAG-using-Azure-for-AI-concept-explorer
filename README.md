# ✦ AI Concept Explorer

> A Retrieval-Augmented Generation (RAG) application for exploring and learning AI concepts through grounded, source-aware explanations.

AI Concept Explorer combines **semantic retrieval**, **Gemini embeddings**, and **Gemini generation** to answer questions using a curated AI knowledge base.

The application retrieves the most relevant knowledge before generating an explanation, helping keep responses grounded in the available source material.

---

## 🚀 Overview

AI Concept Explorer is designed as a learning and career-development workspace for understanding concepts across:

- Artificial Intelligence
- Machine Learning
- Generative AI
- Large Language Models
- Retrieval-Augmented Generation
- Embeddings
- Vector Search

Instead of sending a question directly to an LLM, the application first searches the knowledge base for relevant information.

### RAG Flow

```text
                    ┌──────────────────┐
                    │   User Question  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Question         │
                    │ Embedding        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Semantic         │
                    │ Retrieval        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Relevant         │
                    │ Knowledge Chunks │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Gemini           │
                    │ Generation       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Grounded AI      │
                    │ Explanation      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Knowledge        │
                    │ Sources          │
                    └──────────────────┘