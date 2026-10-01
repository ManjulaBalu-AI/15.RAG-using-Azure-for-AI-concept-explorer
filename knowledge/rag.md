# Retrieval-Augmented Generation (RAG)

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with a generative AI model.

Instead of asking a language model to answer a question using only the knowledge stored in its parameters, a RAG system first retrieves relevant information from an external knowledge source. The retrieved information is then provided to the language model as context.

The model uses that context to generate an answer.

## Why is RAG useful?

RAG is useful when an application needs to answer questions using specific or frequently changing information.

Examples include:

- Company documentation
- Product manuals
- Research papers
- Internal knowledge bases
- Technical documentation
- Educational materials

## Basic RAG workflow

A typical RAG system follows these steps:

1. The user asks a question.
2. The question is converted into a representation suitable for searching.
3. The system searches a knowledge base.
4. Relevant pieces of information are retrieved.
5. The retrieved information is added to the prompt.
6. The language model generates an answer using the retrieved context.

## RAG versus a normal LLM request

In a normal LLM request, the application might send:

User question → Language model → Answer

In a RAG application:

User question → Retrieval system → Relevant information → Language model → Grounded answer

## Example

Suppose a company has a document explaining its employee vacation policy.

A user asks:

"What is the maximum number of vacation days an employee can carry over?"

A RAG system searches the company's documents, retrieves the relevant section of the vacation policy, and provides that section to the language model.

The language model then generates an answer based on the retrieved policy.

## Important RAG components

A RAG system commonly contains:

- Documents
- Document chunks
- Embeddings
- A vector or hybrid search system
- Retrieved context
- A language model
- Source information

The retrieval component helps the language model access information that may not have been included in its original training data.

RAG systems can use vector similarity to find relevant information.
RAG retrieval helps connect a user question with relevant knowledge.