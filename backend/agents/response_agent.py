# backend/agents/response_agent.py

import uuid
from flask import Blueprint, request, jsonify, current_app
from llama_cpp import Llama

from backend.utils.mcp import make_message, log_message

respond_bp = Blueprint('respond_bp', __name__)

@respond_bp.route('/', methods=['POST'])
def respond():
    """
    1. Receive JSON with:
         - trace_id (str)
         - query (str)
         - results (List[{"score": float, "source": {...}, "text": str}])
    2. Log receipt via MCP.
    3. Assemble prompt from contexts + question.
    4. Load & call the local Llama 2 model.
    5. Log completion via MCP.
    6. Return { trace_id, answer, sources }.
    """
    data = request.get_json()
    query    = data.get('query')
    trace_id = data.get('trace_id', str(uuid.uuid4()))
    contexts = data.get('results', [])

    # MCP: log that we're starting generation
    log_message(make_message(
        sender="LLMResponseAgent",
        receiver="LLMResponseAgent",
        msg_type="RESPONSE_RECEIVED",
        trace_id=trace_id,
        payload={"query": query, "num_contexts": len(contexts)}
    ))

    # Build the prompt
    prompt_parts = []
    for i, ctx in enumerate(contexts, start=1):
        src_meta = ctx.get('source', {})
        text     = ctx.get('text', '')
        prompt_parts.append(f"[Context {i}] Source: {src_meta}\n{text}")
    prompt = "\n\n".join(prompt_parts)
    prompt += f"\n\nQuestion: {query}\nAnswer:"

    # Load the model (will use the .gguf file in /models)
    llm = Llama(
        model_path=current_app.config["MODEL_PATH"],
        n_ctx=current_app.config.get("N_CTX", 2048),
        n_gpu_layers=current_app.config.get("N_GPU_LAYERS", 32)
    )

    # Perform generation
    resp = llm(
        prompt,
        max_tokens=current_app.config.get("MAX_TOKENS", 256),
        stop=None
    )
    answer = resp["choices"][0]["text"].strip()

    # MCP: log that generation is complete
    log_message(make_message(
        sender="LLMResponseAgent",
        receiver="UI",
        msg_type="RESPONSE_COMPLETE",
        trace_id=trace_id,
        payload={"answer": answer}
    ))

    return jsonify({
        "trace_id": trace_id,
        "answer": answer,
        "sources": contexts
    }), 200
