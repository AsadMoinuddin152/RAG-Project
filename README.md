# RAG Chatbot

A Retrieval-Augmented Generation (RAG) demo that lets you upload local documents (PDF, PPTX, CSV, DOCX, TXT/MD), ingest them into per-file vector stores, and ask natural-language questions—getting back concise, source-grounded answers via a 4-bit quantized Llama 2 model running locally on your GPU (or via the OpenAI GPT API).

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Quickstart](#quickstart)
   1. [Clone & set up](#1-clone--set-up)
   2. [Install prerequisites](#2-install-prerequisites)
   3. [Model Configuration](#3-model-configuration)
   4. [Run the backend](#4-run-the-backend)
   5. [Run the UI](#5-run-the-ui)
4. [Project Layout](#project-layout)
5. [How It Works](#how-it-works)
6. [Dependencies](#dependencies)
7. [Resources & Scaling](#resources--scaling)

---

## Features

- **Multi-format ingestion**: PDF, PowerPoint, CSV, Word, plain-text, Markdown
- **Per-file FAISS indexes**: upload, delete, list, and query individual documents
- **Local LLM inference**: 4-bit quantized Llama 2 7B (GGUF) with GPU offload via `llama-cpp-python`
- **Optional OpenAI GPT fallback**: use your `OPENAI_API_KEY` for cloud inference
- **Streamlit UI**: drag-and-drop upload, file management, chat interface with live status updates
- **Flask API**: three blueprints for ingestion, retrieval, and response agents
- **MCP-style logs** for end-to-end traceability

---

## Architecture

```text
+-------------+       +---------------+       +---------------------+
|  Streamlit  | <---> |    Flask      | <---> |  Llama 2 / OpenAI   |
|    UI       |       |  (API Layer)  |       |  ChatCompletion     |
+-------------+       +------+--------+       +----------+----------+
                             |                           |
                             |                           |
                    +--------v--------+          +-------v-------+
                    |  IngestionAgent |          | ResponseAgent |
                    |  (/ingest/)     |          | (/respond/)   |
                    +--------+--------+          +-------+-------+
                             |                           |
                  +----------v-----------+               |
                  | VectorStore Module   |               |
                  | (FAISS + metadata)   |               |
                  +----------+-----------+               |
                             |                           |
                    +--------v---------+         +-------v-------+
                    | RetrievalAgent   |         |  Llama 2      |
                    | (/ask/)          |         | inference     |
                    +------------------+         +---------------+
```

1. **Streamlit UI**

   - Upload & manage files
   - Select file(s) & ask questions

2. **Flask API**

   - **IngestionAgent** (`/ingest/`): parse → chunk → embed → index
   - **RetrievalAgent** (`/ask/`): embed query → search FAISS
   - **ResponseAgent** (`/respond/`): assemble prompt → LLM call

3. **VectorStore**

   - Per-file FAISS indexes + persisted metadata

4. **LLM**

   - Local quantized Llama 2 or OpenAI GPT via API

---

## Quickstart

### 1. Clone & set up

```bash
git clone git@github.com:<your-username>/rag-chatbot.git
cd rag-chatbot

python3 -m venv .venv
# On macOS/Linux
source .venv/bin/activate
# On Windows PowerShell
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Install prerequisites

- **CUDA 11.8+** & NVIDIA driver (for GPU offload)
- **Visual C++ Build Tools** (Windows) or `build-essential` (Linux/macOS)
- **Git**
- **Python 3.10+**

### 3. Model Configuration

#### Option A: Local Quantized Llama 2

1. Download the 4-bit quantized GGUF file (≥ 4 GB) from a trusted source (e.g. Hugging Face).

2. Place it in `models/`:

   ```bash
   mv ~/Downloads/llama-2-7b.Q4_K_M.gguf models/llama-2-7b.Q4_K_M.gguf
   ```

3. Ensure your GPU driver and CUDA are configured so that `llama-cpp-python` can offload layers to the GPU.

#### Option B: OpenAI GPT API

1. Sign up at [https://platform.openai.com](https://platform.openai.com) and generate an API key.

2. Set the environment variable:

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. In `backend/app.py`, switch the response agent to use OpenAI’s Python SDK:

   ```python
   import os, openai

   openai.api_key = os.getenv("OPENAI_API_KEY")

   def call_llm_with_openai(prompt):
       resp = openai.ChatCompletion.create(
           model="gpt-3.5-turbo",
           messages=[{"role":"user","content": prompt}],
           max_tokens=256
       )
       return resp.choices[0].message.content
   ```

4. Modify the `/respond/` blueprint to call `call_llm_with_openai()` instead of instantiating `llama_cpp.Llama`.

---

### 4. Run the backend

```bash
export FLASK_APP=backend.app    # Windows PowerShell: $env:FLASK_APP="backend.app"
flask run
```

---

### 5. Run the UI

```bash
streamlit run ui/streamlit_app.py
```

Browse to [http://localhost:8501](http://localhost:8501), ingest a file, select it, and ask a question!

---

## Project Layout

```
rag-chatbot/
├── backend/
│   ├── app.py
│   ├── agents/
│   │   ├── ingestion_agent.py
│   │   ├── retrieval_agent.py
│   │   ├── response_agent.py
│   │   └── file_agent.py
│   └── utils/
│       ├── parsers.py
│       ├── chunker.py
│       ├── embeddings.py
│       ├── vector_store.py
│       └── mcp.py
├── data/
│   ├── uploads/
│   └── indexes/
├── models/
│   └── llama-2-7b.Q4_K_M.gguf
├── ui/
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## How It Works

1. **Ingest**

   - **Parse** each uploaded file (PDF, PPTX, DOCX, CSV, TXT/MD) into raw text and metadata (file name, page/slide numbers).
   - **Chunk** the text into manageable pieces (e.g. 500-token windows with overlap) so that even large documents can be processed incrementally.
   - **Embed** each chunk using a Sentence-Transformers model (all-MiniLM-L6-v2), producing a fixed-length dense vector (e.g. 384 dimensions).
   - **Index** those vectors in a per-file FAISS FlatIP index, and persist a parallel JSON list of metadata (text, source info) for lookup at query time.

2. **Retrieve**

   - **Embed** the user’s natural-language question into the same vector space.
   - **Load** only the FAISS indexes for the file(s) the user selected.
   - **Search** each index for the top-K nearest neighbor chunks by inner product, returning both the similarity scores and the original text metadata.
   - **Merge & sort** results from multiple files, then take the overall top-K snippets to use as context.

3. **Respond**

   - **Assemble** a prompt that interleaves each retrieved snippet with its source annotation, for example:

     ```
     [Context 1 – fileA.pdf, page 3]
     <chunk text>

     [Context 2 – fileB.pptx, slide 5]
     <chunk text>

     Question: <user’s question>
     Answer:
     ```

   - **Call** the LLM: either a local 4-bit quantized Llama 2 (via `llama-cpp-python`) or the OpenAI GPT API, streaming the generated tokens back to the UI.
   - **Return** the final answer plus a list of “filename, page/slide, score” citations so the user can verify the source of each fact.

This three-stage RAG pipeline ensures that answers are both relevant (thanks to vector search) and grounded in your own document collection.

---

## Dependencies

- **Flask**, **Streamlit**, **requests**
- **sentence-transformers** (all-MiniLM-L6-v2)
- **llama-cpp-python** (quantized Llama 2, GPU offload)
- **openai** (for GPT API fallback)
- **faiss-cpu**
- **pdfplumber**, **python-pptx**, **python-docx**
- **tiktoken**

---

## Resources & Scaling

- **Local**: 16 GB RAM, ≥ 6 GB VRAM (RTX 3050/4050/4060), Python 3.10+
- **Cloud (GPT API)**: minimal local resources; costs billed per token

```

```
