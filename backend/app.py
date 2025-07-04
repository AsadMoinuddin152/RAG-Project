# backend/app.py

from flask import Flask
from backend.agents.ingestion_agent import ingest_bp
from backend.agents.retrieval_agent  import retrieve_bp
from backend.agents.response_agent   import respond_bp
from backend.agents.file_agent     import file_bp
import os

def create_app():
    app = Flask(__name__)

    # Config: you can later load from env or a config file
    app.config["MODEL_PATH"] = "models/llama-2-7b.Q4_K_M.gguf"
    app.config["VECTOR_STORE_PATH"] = "data/vector_index.faiss"
    os.makedirs(os.path.dirname(app.config["VECTOR_STORE_PATH"]), exist_ok=True)


    # Register agent blueprints
    app.register_blueprint(ingest_bp,    url_prefix="/ingest")
    app.register_blueprint(retrieve_bp,  url_prefix="/ask")       
    app.register_blueprint(respond_bp,   url_prefix="/respond")
    app.register_blueprint(file_bp,     url_prefix="/files")

    return app

if __name__ == "__main__":
    # For quick local runs
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
