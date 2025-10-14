"""
qa.py

Handles context-grounded question answering using retrieved chunks.

Author: Sourav Agrawal
"""

import textwrap
from typing import List, Tuple
import subprocess
import tempfile
import os


# -----------------------------------------------------------
# 🔹 Prompt Construction
# -----------------------------------------------------------
def build_prompt(question: str, retrieved_chunks: List[Tuple[str, float]]) -> str:
    """
    Builds a grounded prompt from top retrieved chunks and the question.
    """
    context_text = "\n\n".join(
        [f"[Chunk {i+1} | Score={score:.3f}]\n{chunk}" for i, (chunk, score) in enumerate(retrieved_chunks)]
    )

    prompt = textwrap.dedent(f"""
    You are a helpful assistant answering questions based only on the provided transcript segments.
    If the answer is not clearly found in the text, reply with "I'm not sure based on the video content."

    ---------------------------
    CONTEXT:
    {context_text}
    ---------------------------

    QUESTION:
    {question}

    ANSWER (based only on CONTEXT):
    """).strip()

    return prompt


# -----------------------------------------------------------
# 🔹 LLM Call (Local)
# -----------------------------------------------------------
def run_llm(prompt: str, model: str = "llama3:8b") -> str:
    """
    Runs the prompt on a local LLM via Ollama (must be installed).
    For CPU-only users, you can switch to smaller models like llama3:instruct or phi3.
    """
    print(f"🦙 Running local LLM ({model})...")

    try:
        # Use Ollama CLI (no API keys required)
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        output = result.stdout.decode("utf-8").strip()
        return output
    except Exception as e:
        print(f"❌ LLM generation failed: {e}")
        print("💡 Tip: Ensure Ollama is installed and the model is pulled.")
        return "⚠️ Unable to generate an answer (LLM unavailable)."


# -----------------------------------------------------------
# 🔹 Main QA Pipeline
# -----------------------------------------------------------
def answer_question(
    query: str,
    retrieved_chunks: List[Tuple[str, float]],
    model: str = "llama3:8b"
) -> str:
    """
    Builds prompt → runs LLM → returns generated answer.
    """
    prompt = build_prompt(query, retrieved_chunks)
    answer = run_llm(prompt, model=model)
    return answer


# -----------------------------------------------------------
# 🔹 Example Run
# -----------------------------------------------------------
if __name__ == "__main__":
    from retrieve import build_faiss_index, search_index

    embeddings_file = "embeddings_semantic.json"
    query = "What are the main uses of linear algebra in machine learning?"

    index, texts = build_faiss_index(embeddings_file)
    top_chunks = search_index(query, index, texts, top_k=5)

    answer = answer_question(query, top_chunks, model="llama3:8b")

    print("\n==============================")
    print(f"❓ QUESTION: {query}")
    print("==============================")
    print(answer)
