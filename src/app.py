"""
app.py
Streamlit frontend for dynamic YouTube Transcript Q&A MVP.
Runs full pipeline: transcript → embedding → retrieval → LLM answer.

Author: Sourav Agrawal
"""

import os
import json
import tempfile
import streamlit as st

# Import pipeline modules
from ingest import get_transcript
from embed import embed_transcript
from retrieve import build_faiss_index, search_index
from qa import answer_question


# -----------------------------------------------------------
# 🔹 App Config
# -----------------------------------------------------------
st.set_page_config(page_title="VideoQnA", layout="wide")
st.title("🎥 YouTube Video Q&A (Local RAG Demo)")
st.caption("Ask questions directly from any YouTube video transcript — all processed locally!")


# -----------------------------------------------------------
# 🔹 Sidebar Settings
# -----------------------------------------------------------
st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.text_input("Embedding model", "all-MiniLM-L6-v2")
llm_model = st.sidebar.text_input("LLM model (Ollama)", "llama3:8b")
top_k = st.sidebar.slider("Top chunks to retrieve", 1, 10, 5)
threshold = st.sidebar.slider("Semantic chunk similarity threshold", 0.5, 0.9, 0.65)

st.sidebar.markdown("---")
st.sidebar.markdown("💡 *Runs locally — no API keys required!*")


# -----------------------------------------------------------
# 🔹 Main Input Section
# -----------------------------------------------------------
video_url = st.text_input("📺 Enter YouTube Video URL:")
question = st.text_input("❓ Ask your question about the video:")
run_button = st.button("🔍 Generate Answer")


# -----------------------------------------------------------
# 🔹 Helper Functions
# -----------------------------------------------------------
def display_chunks(chunks):
    """Show retrieved chunks with expanders."""
    st.markdown("### 🔎 Top Retrieved Context")
    for i, (chunk, score) in enumerate(chunks, start=1):
        with st.expander(f"Chunk {i} — Score {score:.3f}"):
            st.write(chunk)


# -----------------------------------------------------------
# 🔹 Main Workflow
# -----------------------------------------------------------
if run_button:
    if not question or not video_url:
        st.warning("Please enter both a video URL and a question.")
        st.stop()

    # Step 1: Transcript Generation
    with st.spinner("🎙 Extracting transcript from video... (may take a few minutes)"):
        try:
            transcript_text = get_transcript(video_url, model_size="base")

            # ✅ Ensure transcript_text is a string
            if isinstance(transcript_text, bytes):
                transcript_text = transcript_text.decode("utf-8")
        except Exception as e:
            st.error(f"❌ Failed to transcribe video: {e}")
            st.stop()

    # Step 2: Semantic Chunking + Embedding
    with st.spinner("🧠 Performing semantic chunking and embedding..."):
        try:
            data = embed_transcript(
                transcript_text,
                model_name=model_name,
                similarity_threshold=threshold,
            )
            # Save temporary embeddings for FAISS
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json", encoding="utf-8") as temp_file:
                json.dump(
                    [
                        {
                            "id": d["id"],
                            "text": d["text"],
                            "embedding": d["embedding"].tolist(),
                        }
                        for d in data
                    ],
                    temp_file,
                    ensure_ascii=False,
                    indent=2,
                )
                temp_file.close()
                embeddings_file = temp_file.name
        except Exception as e:
            st.error(f"❌ Embedding failed: {e}")
            st.stop()

    # Step 3: Retrieval
    with st.spinner("🔍 Retrieving relevant chunks..."):
        try:
            index, texts = build_faiss_index(embeddings_file)
            top_chunks = search_index(question, index, texts, top_k=top_k)
        except Exception as e:
            st.error(f"❌ Retrieval failed: {e}")
            st.stop()

    display_chunks(top_chunks)

    # Step 4: Answer Generation
    with st.spinner("🧠 Generating grounded answer..."):
        try:
            answer = answer_question(question, top_chunks, model=llm_model)
        except Exception as e:
            st.error(f"❌ Answer generation failed: {e}")
            st.stop()

    st.markdown("### 🧾 Answer")
    st.success(answer)

    st.markdown("---")
    st.markdown("##### ⚙️ Technical Details")
    st.json(
        {
            "video_url": video_url,
            "embedding_model": model_name,
            "llm_model": llm_model,
            "top_k": top_k,
            "threshold": threshold,
        }
    )

    # Clean up temp file
    try:
        os.remove(embeddings_file)
    except Exception:
        pass


# -----------------------------------------------------------
# 🔹 Footer
# -----------------------------------------------------------
st.markdown("---")
st.caption("Built by Sourav Agrawal — 100% local, open-source, and free.")
