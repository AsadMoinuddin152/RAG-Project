# backend/utils/embeddings.py

from sentence_transformers import SentenceTransformer

# load once at import
_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    """
    texts: List[str]
    returns: List[np.ndarray] (one per text)
    """
    embeddings = _EMBED_MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings
