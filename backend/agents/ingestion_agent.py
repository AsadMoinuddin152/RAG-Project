# backend/agents/ingestion_agent.py

import os
import uuid
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from backend.utils.parsers import parse_file
from backend.utils.chunker import chunk_text
from backend.utils.embeddings import embed_texts
from backend.utils.vector_store import load_index, save_index, add_to_index
from backend.utils.mcp import make_message, log_message

from backend.agents.file_agent import _load_registry, _save_registry

ingest_bp = Blueprint('ingest_bp', __name__)

@ingest_bp.route('/', methods=['POST'])
def ingest():
    # 1. Receive uploaded file
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    upload_id = str(uuid.uuid4())
    upload_dir = os.path.join('data', 'uploads', upload_id)
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)
    file.save(file_path)

    # MCP: log upload
    msg = make_message(
        sender="IngestionAgent", receiver="IngestionAgent",
        msg_type="UPLOAD_RECEIVED", trace_id=upload_id,
        payload={"filename": filename}
    )
    log_message(msg)

    # 2. Parse into text + metadata
    docs = parse_file(file_path)
    # 3. Chunk text
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc["text"], doc["source"]))

    # 4. Embed
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)  # shape: (n_chunks, emb_dim)

    # 5. Prepare per-file index path
    idx_dir = os.path.join('data', 'indexes')
    os.makedirs(idx_dir, exist_ok=True)
    index_path = os.path.join(idx_dir, f"{upload_id}.faiss")

    # 6. Load or init FAISS index
    emb_dim = embeddings.shape[1]
    index = load_index(index_path, dim=emb_dim)

    # 7. Add embeddings + metadata
    for emb, chunk in zip(embeddings, chunks):
        metadata = {
            "text":     chunk["text"],
            "source":   chunk["source"],
            "filename": filename,
            "file_id":  upload_id
        }
        add_to_index(index, emb, metadata)

    # 8. Save index and metadata
    save_index(index, index_path)

    # 9. Update global registry
    registry = _load_registry()
    registry.append({
        "id":         upload_id,
        "name":       filename,
        "index_path": index_path,
        "upload_dir": upload_dir
    })
    _save_registry(registry)

    # MCP: log completion
    msg = make_message(
        sender="IngestionAgent", receiver="VectorStore",
        msg_type="INDEX_UPDATED", trace_id=upload_id,
        payload={"num_chunks": len(chunks)}
    )
    log_message(msg)

    return jsonify({
        "status":  "success",
        "uploaded": filename,
        "chunks":  len(chunks),
        "file_id": upload_id
    }), 200