"""
Handles building a FAISS index from transcript embeddings
and retrieving the most semantically similar chunks for a query.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from typing import List, Dict, Tuple


# -----------------------------------------------------------
# 🔹 Build FAISS Index
# -----------------------------------------------------------
def build_faiss_index(embeddings_file: str) -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    """
    Loads precomputed embeddings and builds a FAISS index.

    Returns:
        (index, metadata_list)
    """
    print(f"📂 Loading embeddings from {embeddings_file} ...")
    with open(embeddings_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = np.array([np.array(d["embedding"], dtype=np.float32) for d in data])
    texts = [d["text"] for d in data]

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # IP (Inner Product) works with normalized vectors
    index.add(embeddings)

    print(f"✅ FAISS index built with {index.ntotal} vectors (dim={dim})")
    return index, texts


# -----------------------------------------------------------
# 🔹 Semantic Search
# -----------------------------------------------------------
def search_index(
    query: str,
    index: faiss.IndexFlatIP,
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Perform semantic search to retrieve top_k most similar chunks.
    """
    model = SentenceTransformer(model_name)
    query_emb = model.encode([query], convert_to_tensor=False, show_progress_bar=False)
    query_emb = np.array(query_emb, dtype=np.float32)
    faiss.normalize_L2(query_emb)

    scores, indices = index.search(query_emb, top_k)
    scores = scores[0]
    indices = indices[0]

    results = [(texts[i], float(scores[idx])) for idx, i in enumerate(indices)]
    return results


# -----------------------------------------------------------
# 🔹 Example Run
# -----------------------------------------------------------
if __name__ == "__main__":
    # Load your semantic embeddings file from Step 2
    embeddings_file = "embeddings_semantic.json"
    query = "What are the main concepts of linear algebra?"

    index, texts = build_faiss_index(embeddings_file)
    results = search_index(query, index, texts, top_k=5)

    print("\n🔎 Top Matches:")
    for i, (chunk, score) in enumerate(results, start=1):
        print(f"\n{i}. (score={score:.3f})\n{chunk[:300]}...")
