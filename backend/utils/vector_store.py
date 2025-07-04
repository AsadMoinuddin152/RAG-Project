import os
import json
import faiss
import numpy as np

# In-memory store of each chunk’s text and its source metadata
_CONTENTS = []

def init_index(dim: int) -> faiss.IndexFlatIP:
    """Create a new FAISS IndexFlatIP of the given dimension."""
    return faiss.IndexFlatIP(dim)

def load_index(path: str, dim: int = None) -> faiss.IndexFlatIP:
    """
    If an index exists at `path`, load it and its metadata.
    Otherwise initialize a new index of dimension `dim`.
    """
    meta_path = path + ".meta.json"

    if os.path.exists(path):
        # 1) Load the FAISS index
        index = faiss.read_index(path)
        # 2) Reload metadata list
        global _CONTENTS
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                _CONTENTS = json.load(f)
        else:
            _CONTENTS = []
        return index

    # No existing index: create new
    if dim is None:
        raise ValueError("Index not found and no dimension provided to initialize.")
    _CONTENTS.clear()
    return init_index(dim)

def save_index(index: faiss.IndexFlatIP, path: str):
    """
    Persist both the FAISS index and the metadata list.
    Writes:
      - `path`             (the .faiss index file)
      - `path + ".meta.json"` (the JSON metadata)
    """
    faiss.write_index(index, path)
    meta_path = path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_CONTENTS, f, ensure_ascii=False, indent=2)

def add_to_index(index: faiss.IndexFlatIP, embedding: np.ndarray, metadata: dict):
    """
    Add a single embedding + its metadata to the index.
    `metadata` should include:
      - "text": the chunk’s text
      - "source": the source metadata
    """
    index.add(embedding.reshape(1, -1))
    _CONTENTS.append(metadata)

def search_index(index: faiss.IndexFlatIP, query_emb: np.ndarray, top_k: int):
    """
    Search the index and return top_k results as:
      [{"score": float, "source": {...}, "text": "..."}]
    """
    D, I = index.search(query_emb.reshape(1, -1), top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        entry = _CONTENTS[idx]
        results.append({
            "score": float(score),
            "source": entry.get("source", {}),
            "text":   entry.get("text", "")
        })
    return results
