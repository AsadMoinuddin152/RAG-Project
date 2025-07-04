import streamlit as st
import requests
from streamlit.runtime.secrets import StreamlitSecretNotFoundError

try:
    BACKEND_URL = st.secrets["backend_url"]
except (KeyError, StreamlitSecretNotFoundError):
    BACKEND_URL = "http://localhost:5000"

def fetch_files():
    try:
        return requests.get(f"{BACKEND_URL}/files/").json()
    except requests.RequestException:
        return []

st.title("📚 RAG Chatbot")

# Sidebar: file list, delete, upload
st.sidebar.header("🔧 Files")
files = fetch_files()
file_map = {f["name"]: f["id"] for f in files}
file_names = list(file_map.keys())

selected_name = st.sidebar.selectbox("Which file to search?", file_names)
file_id = file_map.get(selected_name)

if st.sidebar.button("🗑️ Delete selected file"):
    if file_id:
        resp = requests.delete(f"{BACKEND_URL}/files/", json={"files": [selected_name]})
        if resp.ok:
            st.sidebar.success(f"Deleted {selected_name}")
            files = fetch_files()
            file_map = {f["name"]: f["id"] for f in files}
            file_names = list(file_map.keys())
        else:
            st.sidebar.error(f"Error: {resp.text}")
    else:
        st.sidebar.warning("No file selected to delete")

if st.sidebar.button("🗑️ Delete all files"):
    resp = requests.delete(f"{BACKEND_URL}/files/", json={"files": "all"})
    if resp.ok:
        st.sidebar.success("Deleted all files")
        files = fetch_files()
        file_map = {f["name"]: f["id"] for f in files}
        file_names = list(file_map.keys())
    else:
        st.sidebar.error(f"Error: {resp.text}")

st.sidebar.header("⬆️ Upload Documents")
uploads = st.sidebar.file_uploader(
    "Select files to upload",
    type=["pdf", "pptx", "csv", "docx", "txt", "md"],
    accept_multiple_files=True
)

if uploads:
    for up in uploads:
        requests.post(f"{BACKEND_URL}/ingest/", files={"file": (up.name, up.getvalue())})
    st.sidebar.success("Uploads complete")
    files = fetch_files()
    file_map = {f["name"]: f["id"] for f in files}
    file_names = list(file_map.keys())

# Main: select files for querying
st.header("💬 Ask a Question")
selected_files = st.multiselect(
    "Select file(s) to search:",
    options=file_names,
    default=file_names[:1] if file_names else []
)

query = st.text_input("Your question:")
top_k = st.slider("Number of contexts to retrieve:", 1, 10, 5)

if st.button("Ask"):
    if not query:
        st.warning("Please enter a question.")
    elif not selected_files:
        st.warning("Please select at least one file.")
    else:
        file_ids = [file_map[name] for name in selected_files]
        payload = {"query": query, "top_k": top_k, "file_ids": file_ids}
        ret = requests.post(f"{BACKEND_URL}/ask/", json=payload, timeout=60)

        if not ret.ok:
            st.error(f"Retrieval error: {ret.status_code} {ret.text}")
        else:
            data = ret.json()
            trace = data.get("trace_id")
            contexts = data.get("results", [])
            with st.spinner("Generating answer, please be patient…"):
                try:
                    gen = requests.post(
                        f"{BACKEND_URL}/respond/",
                        json={
                            "trace_id": trace,
                            "query":    query,
                            "results":  contexts
                        },
                        timeout=300
                    )
                    gen.raise_for_status()
                    out = gen.json()
                except requests.ReadTimeout:
                    st.error(
                        "Generation is taking too long—"
                        "try reducing ‘Number of contexts’ or raising the timeout."
                    )
                    st.stop()
                except requests.RequestException as e:
                    st.error(f"Generation error: {e}")
                    st.stop()

            # display the answer & sources as before
            st.markdown("### 🤖 Answer")
            st.write(out.get("answer", "No answer returned."))

            st.markdown("### 📑 Sources")
            for src in out.get("sources", []):
                name   = src.get("filename", "unknown")
                meta   = src.get("source", {})
                # try page, then slide
                loc    = meta.get("page") or meta.get("slide") or "?"
                score  = src.get("score", 0.0)
                st.write(f"- **{name}**, page/slide {loc} (score: {score:.3f})")