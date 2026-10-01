import streamlit as st

from rag_pipeline import (
    load_document,
    chunk_markdown,
    cache_is_valid,
    load_saved_embeddings,
    create_and_save_embeddings,
    retrieve_chunks,
    generate_answer,
    DOCUMENT_PATH,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Concept Explorer",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SIMPLE AI THEME
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            linear-gradient(
                135deg,
                #07111f 0%,
                #0b1628 50%,
                #111827 100%
            );
        color: #e5e7eb;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
    }

    p, label {
        color: #cbd5e1 !important;
    }

    /* Input */

    .stTextInput input {
        background-color: #111c2e !important;
        color: white !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }

    .stTextInput input:focus {
        border-color: #60a5fa !important;
    }

    /* Buttons */

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        background: linear-gradient(
            90deg,
            #2563eb,
            #4f46e5
        );
        color: white;
        font-weight: 700;
        padding: 0.7rem;
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #3b82f6,
            #6366f1
        );
    }

    /* Info boxes */

    [data-testid="stAlert"] {
        background-color: #111c2e;
        border: 1px solid #334155;
        border-radius: 10px;
    }

    /* Expanders */

    [data-testid="stExpander"] {
        background-color: #0f1b2d;
        border: 1px solid #334155;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.title("✦ AI Concept Explorer")

st.subheader(
    "Artificial Intelligence · Knowledge Exploration"
)

st.write(
    "Learn AI concepts through intelligent retrieval, "
    "grounded explanations, and practical examples."
)

st.divider()


# ============================================================
# CAPABILITIES
# ============================================================

st.markdown("### ◈ What this application does")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("◈\n\n**AI Concepts**")

with col2:
    st.info("⌁\n\n**Semantic Retrieval**")

with col3:
    st.info("✦\n\n**Gemini Generation**")

with col4:
    st.info("◇\n\n**Source Grounding**")


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

@st.cache_resource
def load_knowledge_base():

    text = load_document(DOCUMENT_PATH)

    chunks = chunk_markdown(text)

    if cache_is_valid():
        return load_saved_embeddings()

    return create_and_save_embeddings(chunks)


with st.spinner("Preparing AI knowledge base..."):
    chunk_data = load_knowledge_base()


# ============================================================
# SEARCH AREA
# ============================================================

st.divider()

st.header("⌕ Explore an AI Concept")

st.write(
    "Ask a question and the RAG system will retrieve "
    "relevant knowledge before generating the answer."
)


question = st.text_input(
    "Your question",
    placeholder=(
        "Example: How does retrieval work in a RAG system?"
    ),
)


# ============================================================
# TOPICS
# ============================================================

st.markdown("#### Explore Topics")

topic_cols = st.columns(3)

with topic_cols[0]:
    st.write("◉ Retrieval-Augmented Generation")
    st.write("◈ Embeddings")

with topic_cols[1]:
    st.write("◇ Large Language Models")
    st.write("⌁ Vector Search")

with topic_cols[2]:
    st.write("✦ Generative AI")
    st.write("△ Machine Learning")


# ============================================================
# ASK BUTTON
# ============================================================

ask = st.button(
    "✦  Ask AI Concept Explorer",
    type="primary",
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask:

    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

    else:

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        with st.spinner(
            "Searching the knowledge base..."
        ):

            retrieved = retrieve_chunks(
                question,
                chunk_data,
                top_k=3,
            )


        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        with st.spinner(
            "Generating your AI explanation..."
        ):

            answer = generate_answer(
                question,
                retrieved,
            )


        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        st.divider()

        st.header("✦ AI Explanation")

        st.markdown(answer)


        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        st.header("◇ Knowledge Sources")

        st.caption(
            "These knowledge-base sections were retrieved "
            "to support the generated answer."
        )


        for i, chunk in enumerate(
            retrieved,
            start=1,
        ):

            with st.expander(
                f"📄 Source {i} · {chunk['section']}"
            ):

                col1, col2 = st.columns(2)

                with col1:

                    st.write(
                        f"**Document:** "
                        f"{chunk['document_name']}"
                    )

                with col2:

                    st.write(
                        f"**Similarity:** "
                        f"{chunk['score']:.4f}"
                    )

                st.divider()

                st.write(
                    chunk["text"]
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "✦ AI Concept Explorer · "
    "Retrieval-Augmented Generation · Gemini"
)