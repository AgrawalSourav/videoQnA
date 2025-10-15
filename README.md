---
title: VideoQnA
emoji: 📹
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.30.0"
app_file: src/app.py
pinned: false
---

# VideoQnA: Ask Questions of a YouTube Video Locally

> Turn any YouTube video into a searchable Q&A — no cloud needed, 100% open source + free.

---

## What Is This?

VideoQnA is a demonstration project that lets you ask natural language questions about a YouTube video’s content, and (on your local machine) it will:

1. Transcribe or fetch the video’s transcript,  
2. Break it into semantic chunks & embed them,  
3. Build a lightweight search index (FAISS),  
4. Retrieve relevant snippets for your query,  
5. Generate a grounded answer via a local small LLM, with timestamped citations.

It showcases a full **transcript → search → Q&A** pipeline using open-source tools. The entire system runs **locally**, no paid APIs required.

---

## Why It Matters / What It Demonstrates

- Many videos have lengthy unsearchable transcripts — this shows how to turn them into interactive Q&A.  
- Demonstrates knowledge of: ASR (speech-to-text), embeddings & semantic search, retrieval-augmented generation, prompt engineering, and local LLM orchestration.  
- A clean, modular, reproducible codebase you can walk through in ~10 minutes.

---

## Tech Stack & Design Highlights

| Layer | Component | Rationale |
|---|---|---|
| **Transcription / ASR** | `faster-whisper` (CPU-friendly) | Significantly faster and lower memory usage compared to original Whisper. :contentReference[oaicite:0]{index=0} |
| **Embeddings** | `sentence-transformers` → `all-MiniLM-L6-v2` | A compact, fast embedding model (384 dims) that balances speed and semantic quality. :contentReference[oaicite:1]{index=1} |
| **Vector Index / Retrieval** | **FAISS** (local) | Lightweight, efficient, no external service dependency |
| **LLM / Generation** | `llama.cpp` (quantized small model) or equivalent local small model | Enough to show generation + citation logic without needing GPU |
| **UI / Demo** | `Streamlit` (optional) | Simple interactive interface (question → answer) to impress recruiters |

---

## How to Run (Quick Start)

1. **Clone the repo**  
   ```bash
   git clone https://github.com/AgrawalSourav/VideoQnA.git
   cd VideoQnA

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

3. **Run locally**
   streamlit run src/app.py

> First run will download Whisper and embedding models.
> You can use smaller models (like tiny or base) for faster performance.

🔗 Live demo example:


**Features**

- Full offline pipeline — no API keys or paid LLMs
- Semantic chunking (context-aware transcript segmentation)
- Local embedding + FAISS retrieval
- Real-time Q&A via Streamlit
- Supports any public YouTube video URL
- Works locally and on Hugging Face Spaces

**Workflow Summary**

| Step       | Description                              | File              |
| ---------- | ---------------------------------------- | ----------------- |
| **Step 1** | Extract & transcribe audio from YouTube  | `src/ingest.py`   |
| **Step 2** | Clean text, semantic chunking, embedding | `src/embed.py`    |
| **Step 3** | Build FAISS index & retrieve context     | `src/retrieve.py` |
| **Step 4** | Context-grounded Q&A (LLM prompt)        | `src/qa.py`       |
| **Step 5** | Streamlit frontend                       | `src/app.py`      |
| **Step 6** | Free deployment on Spaces                | –                 |

**Key Learnings**

| Concept                 | What You Learn                                      |
| ----------------------- | --------------------------------------------------- |
| **Semantic chunking**   | Breaking text by meaning instead of length          |
| **FAISS retrieval**     | Vector-based semantic search                        |
| **RAG**                 | Combine retrieval + generation for grounded answers |
| **Local ML deployment** | Hosting full ML pipeline without external APIs      |
| **Streamlit UX**        | Build simple, powerful data apps fast               |

**Future Improvements**

🔹 Add caching for repeated URLs (Streamlit st.cache_data)
🔹 Display timestamps with retrieved chunks
🔹 Add history & export feature
🔹 Optionally integrate small local models (e.g., phi3 via Ollama)
🔹 Deploy also on Streamlit Cloud

**TL;DR**
A fully local, open-source YouTube Transcript Q&A app — powered by Whisper, FAISS, SentenceTransformers, and Streamlit.
Deployable on Hugging Face Spaces — zero cost, zero API keys.