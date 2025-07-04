# backend/agents/file_agent.py

import os
import json
import shutil
from flask import Blueprint, request, jsonify, current_app

file_bp = Blueprint("file_bp", __name__)

REGISTRY = os.path.join("data", "uploads", "files.json")

def _load_registry():
    if os.path.exists(REGISTRY):
        with open(REGISTRY, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def _save_registry(files):
    os.makedirs(os.path.dirname(REGISTRY), exist_ok=True)
    with open(REGISTRY, "w", encoding="utf-8") as f:
        json.dump(files, f, indent=2)

@file_bp.route("/", methods=["GET"])
def list_files():
    """
    Returns a list of uploaded files with their IDs and paths.
    """
    return jsonify(_load_registry())

@file_bp.route("/", methods=["DELETE"])
def delete_files():
    """
    Deletes either:
      - all files: {"files": "all"}
      - a subset:  {"files": ["file1.pdf","file2.pptx", ...]}
    Removes them from disk, clears their vectors, and updates the registry.
    """
    data = request.get_json() or {}
    to_del = data.get("files", "all")
    registry = _load_registry()
    kept = []

    # delete matching entries
    for entry in registry:
        name = entry["name"]
        if to_del == "all" or name in to_del:
            # remove upload dir
            if os.path.exists(entry["dir"]):
                shutil.rmtree(entry["dir"])
            # TODO: also remove vectors for that file from FAISS
        else:
            kept.append(entry)

    _save_registry(kept)
    # Optionally: completely rebuild the vector index from remaining files
    return jsonify({"deleted": to_del}), 200
