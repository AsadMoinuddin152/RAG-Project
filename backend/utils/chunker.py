# backend/utils/chunker.py

from typing import List, Dict
import tiktoken 

# adjust these as needed
_CHUNK_SIZE    = 500
_CHUNK_OVERLAP = 50

def _tokenize(text):
    # Example using tiktoken; swap in your tokenizer of choice
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.encode(text)

def _detokenize(tokens):
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(tokens)

def chunk_text(text: str, source: Dict) -> List[Dict]:
    """
    Split `text` into overlapping chunks of _CHUNK_SIZE_ tokens.
    Attach the passed-in `source` metadata to each chunk.
    """
    tokens = _tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + _CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _detokenize(chunk_tokens)
        chunks.append({
            "text": chunk_text,
            "source": source
        })
        start += _CHUNK_SIZE - _CHUNK_OVERLAP
    return chunks
