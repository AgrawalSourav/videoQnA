"""
Handles semantic chunking and embedding of transcript text.
"""

import re
import time
from typing import List, Dict
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download("punkt_tab", quiet=True)


# -----------------------------------------------------------
# 🔹 Text Preprocessing
# -----------------------------------------------------------
def clean_text(text: str) -> str:
    """Normalize spaces and remove redundant newlines."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------------------------------------
# 🔹 Semantic Chunking
# -----------------------------------------------------------

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_chunk_text(
    text: str,
    model: SentenceTransformer,
    similarity_threshold: float = 0.65
) -> List[str]:
    """
    Create semantically coherent chunks based on cosine similarity
    between consecutive sentence embeddings.
    """
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)

    if len(sentences) < 2:
        return [text]

    sentence_embeddings = model.encode(sentences, batch_size=16, convert_to_tensor=False, show_progress_bar=True)
    chunks, current_chunk = [], [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity([sentence_embeddings[i - 1]], [sentence_embeddings[i]])
        if sim < similarity_threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"✅ Semantic chunking created {len(chunks)} chunks (threshold={similarity_threshold})")
    return chunks


# -----------------------------------------------------------
# 🔹 Embedding Generation
# -----------------------------------------------------------
def create_embeddings(chunks: List[str], model: SentenceTransformer):
    """Generate embeddings for given chunks."""
    data = []
    for idx, chunk in enumerate(tqdm(chunks, desc="Embedding chunks", ncols=80)):
        chunk = str(chunk)  # <-- ensure chunk is string

        try:
            emb = model.encode(chunk, convert_to_numpy=True, convert_to_tensor=False, show_progress_bar=False)
            if not isinstance(emb, np.ndarray):
                emb = np.array(emb, dtype=np.float32)  # fallback
            else:
                emb = emb.astype(np.float32) 
        except Exception as e:
            print(f"❌ Failed to embed chunk {idx}: {e}")
            continue
        
        data.append({
            "id": idx,
            "text": chunk,
            "embedding": emb
        })
    return data


# -----------------------------------------------------------
# 🔹 Master Pipeline
# -----------------------------------------------------------
def embed_transcript(
    text: str,
    model_name: str = "all-MiniLM-L6-v2",
    similarity_threshold: float = 0.65
) -> List[Dict]:
    """
    Full pipeline: clean → semantic chunk → embed.
    """
    text = str(text)
    print(f"\n🧠 Semantic chunking with model: {model_name}")
    start_time = time.time()

    text = clean_text(text)
    
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {e}")


    chunks = semantic_chunk_text(text, model=model, similarity_threshold=similarity_threshold)
    print(f"🧩 {len(chunks)} chunks created.")

    data = create_embeddings(chunks, model)
    duration = time.time() - start_time
    print(f"✅ Embedding pipeline complete in {duration:.2f}s")
    return data


# -----------------------------------------------------------
# 🔹 Example Run
# -----------------------------------------------------------
if __name__ == "__main__":
    import json
    import os

    transcript_file = "transcript.txt"
    if not os.path.exists(transcript_file):
        raise FileNotFoundError(
            "❌ transcript.txt not found. Run the ingest step first."
        )

    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript_text = f.read()

    chunks = embed_transcript(
        transcript_text,
        model_name="all-MiniLM-L6-v2",
        similarity_threshold=0.65
    )

    file_name = "embeddings_semantic.json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "id": c["id"],
                    "text": c["text"],
                    "embedding": c["embedding"].tolist(),
                }
                for c in chunks
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"💾 Saved embeddings to {file_name}")
