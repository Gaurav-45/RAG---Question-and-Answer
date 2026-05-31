RAG based PDF Q&A with Qdrant

## Brief

This is a small Retrieval-Augmented Generation project that lets you upload PDF documents and ask questions. The system indexes document embeddings into a Qdrant vector database, retrieves relevant passages, and returns answers via a FastAPI backend and a Streamlit front-end.

## Key features

- Upload PDF documents and index them into Qdrant
- Semantic search over document content using embeddings
- Question-answering API and a Streamlit UI for interactive use

## Tech stack

- Python + FastAPI — backend API
- Qdrant — vector database for embeddings and retrieval
- Inngest — middleware / observability (optional)
- Streamlit — simple UI for uploads and queries

## Quickstart (local)

Prerequisites: Docker, Python 3.10+, and Node.js if you run Inngest locally.

1. Create a virtual environment and install dependencies (examples):

```bash
python -m venv .venv
source .venv/bin/activate
uv sync
```

2. Start Qdrant (data persisted to `qdrant_storage`):

```bash
docker run -d --name qdrantRagDb -p 6333:6333 -v "./qdrant_storage:/qdrant/storage" qdrant/qdrant
```

3. Start Inngest dev server for observability:

```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

4. Run the backend API:

```bash
uv run uvicorn main:app --reload
```

5. Start the Streamlit UI (default port 8501):

```bash
uv run streamlit run streamlit_app.py
```

## Usage

- Open the Streamlit app at `http://localhost:8501` to upload PDF files and ask questions interactively.
- Alternatively, use the FastAPI endpoints on `http://localhost:8000` to programmatically upload documents and query the system.

## Project layout

- `main.py` — FastAPI application entrypoint
- `streamlit_app.py` — Streamlit front-end for uploads and queries
- `vector_db.py` — Qdrant integration and vector helpers
- `data_loader.py` — PDF parsing and document chunking
- `custom_types.py` — shared pydantic models and types
- `qdrant_storage/` — local Qdrant persistence (mounted into Docker)
- `uploads/` — uploaded PDF files (local storage)
