# backend/agents/retrieval_agent.py

import uuid
from flask import Blueprint, request, jsonify

from backend.utils.embeddings import embed_texts
from backend.utils.vector_store import load_index, search_index
from backend.utils.mcp import make_message, log_message
from backend.agents.file_agent import _load_registry

retrieve_bp = Blueprint('retrieve_bp', __name__)

@retrieve_bp.route('/', methods=['POST'])
def retrieve():
    data     = request.get_json() or {}
    query    = data.get('query')
    top_k    = data.get('top_k', 5)
    trace_id = data.get('trace_id', str(uuid.uuid4()))
    file_ids = data.get('file_ids') or []

    if not query or not file_ids:
        return jsonify({"error": "Missing 'query' or 'file_ids'"}), 400

    statuses = []
    statuses.append("Received query")

    log_message(make_message(
        sender="RetrievalAgent",
        receiver="RetrievalAgent",
        msg_type="QUERY_RECEIVED",
        trace_id=trace_id,
        payload={"query": query, "file_ids": file_ids}
    ))
    statuses.append("Logged QUERY_RECEIVED")

    q_emb = embed_texts([query])[0]
    statuses.append("Query embedded")

    log_message(make_message(
        sender="RetrievalAgent",
        receiver="VectorStore",
        msg_type="QUERY_EMBEDDED",
        trace_id=trace_id,
        payload={"embedding_dim": len(q_emb)}
    ))
    statuses.append("Logged QUERY_EMBEDDED")

    registry = _load_registry()
    all_hits = []

    for fid in file_ids:
        entry = next((e for e in registry if e["id"] == fid), None)
        if not entry:
            statuses.append(f"Skipping unknown file_id {fid}")
            continue

        statuses.append(f"Loading index for {entry['name']}")
        idx = load_index(entry["index_path"])
        statuses.append(f"Searching index for {entry['name']}")

        hits = search_index(idx, q_emb, top_k)
        statuses.append(f"Found {len(hits)} hits in {entry['name']}")

        for h in hits:
            h["file_id"]  = fid
            h["filename"] = entry["name"]
        all_hits.extend(hits)

    statuses.append("Merging and sorting all hits")
    all_hits.sort(key=lambda x: x["score"], reverse=True)
    results = all_hits[:top_k]
    statuses.append(f"Selected top {len(results)} results overall")

    log_message(make_message(
        sender="RetrievalAgent",
        receiver="LLMResponseAgent",
        msg_type="RETRIEVAL_COMPLETE",
        trace_id=trace_id,
        payload={"top_k": top_k, "file_ids": file_ids}
    ))
    statuses.append("Logged RETRIEVAL_COMPLETE")

    return jsonify({
        "trace_id": trace_id,
        "statuses": statuses,
        "results":  results
    }), 200
