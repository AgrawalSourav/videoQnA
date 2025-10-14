# VideoQnA: Ask Questions of a YouTube Video Locally

> **Tagline:** Turn any YouTube video into a searchable Q&A — no cloud needed, 100% open source + free.

---

## 🚀 What Is This?

VideoQnA is a demonstration project that lets you ask natural language questions about a YouTube video’s content, and (on your local machine) it will:

1. Transcribe or fetch the video’s transcript,  
2. Break it into semantic chunks & embed them,  
3. Build a lightweight search index (FAISS),  
4. Retrieve relevant snippets for your query,  
5. Generate a grounded answer via a local small LLM, with timestamped citations.

It showcases a full **transcript → search → Q&A** pipeline using open-source tools. The entire system runs **locally**, no paid APIs required.

---

## 🧩 Why It Matters / What It Demonstrates

- Many videos have lengthy unsearchable transcripts — this shows how to turn them into interactive Q&A.  
- Demonstrates knowledge of: ASR (speech-to-text), embeddings & semantic search, retrieval-augmented generation, prompt engineering, and local LLM orchestration.  
- A clean, modular, reproducible codebase you can walk through in ~10 minutes — perfect to show technical depth to recruiters.

---

## 🔧 Tech Stack & Design Highlights

| Layer | Component | Rationale |
|---|---|---|
| **Transcription / ASR** | `faster-whisper` (CPU-friendly) | Significantly faster and lower memory usage compared to original Whisper. :contentReference[oaicite:0]{index=0} |
| **Embeddings** | `sentence-transformers` → `all-MiniLM-L6-v2` | A compact, fast embedding model (384 dims) that balances speed and semantic quality. :contentReference[oaicite:1]{index=1} |
| **Vector Index / Retrieval** | **FAISS** (local) | Lightweight, efficient, no external service dependency |
| **LLM / Generation** | `llama.cpp` (quantized small model) or equivalent local small model | Enough to show generation + citation logic without needing GPU |
| **UI / Demo** | `Streamlit` (optional) | Simple interactive interface (question → answer) to impress recruiters |

---

## ⚙️ How to Run (Quick Start)

1. **Clone the repo**  
   ```bash
   git clone https://github.com/AgrawalSourav/VideoQnA.git
   cd VideoQnA
