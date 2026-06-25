# CogniVerdict Backend — Cognee Cloud Integration Layer (Phase 1)

This is the backend service for CogniVerdict, a legal reasoning copilot. In this phase, it provides a clean API integration layer for Cognee Cloud, exposing memory management, ingestion, and querying endpoints.

## Tech Stack
- **Python** (3.14.x)
- **FastAPI**
- **Uvicorn**
- **HTTPX** (Async HTTP Client)
- **Pydantic** (V2 for Request/Response Schemas)

---

## 1. Setup

### Prerequisites
Make sure Python 3.10+ is installed on your system.

### Configuration
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Populate your Cognee Cloud credentials in `.env`:
   ```env
   COGNEE_API_KEY=your_cognee_cloud_api_key
   COGNEE_API_URL=https://api.cognee.ai
   ```

### Installation
Install the required packages:
```bash
pip install -r requirements.txt
```
*(If on Debian/Ubuntu systems with PEP 668 restrictions, use the `--break-system-packages` flag or run inside a virtual environment).*

---

## 2. Running the Server

Start the development server:
```bash
python run.py
```
By default, the server runs on `http://localhost:8000`.

You can view the interactive Swagger API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 3. API Endpoints

### Case Upload Ingestion
* **`POST /cases/upload`**: Uploads a legal document (`.pdf`, `.docx`, or `.txt`) and triggers Cognee remember/ingestion pipeline.
  - **Form Data**:
    - `file`: The legal document file.
    - `case_name` (optional): Name of the case dataset.
    - `run_in_background` (optional, default: `true`): Runs chunking and graph building in background.

### Ingestion Status
* **`GET /cases/{id}/status`**: Returns the background processing/cognify status of the case (`pending`, `running`, `completed`, `failed`).

### Memory Queries & Search
* **`POST /cases/recall`**: Global semantic memory search across datasets.
* **`POST /cases/{id}/recall`**: Scoped semantic recall within a single case.
  - **Form/JSON parameters**:
    - `query`: The question or query string.
    - `search_type` (default: `GRAPH_COMPLETION`): Search mode (e.g. `GRAPH_COMPLETION`, `CHUNKS`, `RAG_COMPLETION`).
    - `top_k` (default: `10`): Max context retrieved.

### Memory & Graph Operations
* **`GET /cases/{id}/graph`**: Returns the full structured case graph (nodes and edges).
* **`GET /cases/{id}/nodes`**: Returns entities/nodes only.
* **`GET /cases/{id}/edges`**: Returns relationships/edges only.
* **`GET /cases/{id}/chunks`**: Returns segmented text chunks of the document.
* **`GET /cases/{id}/provenance`**: Returns document file and dataset lineage metadata.
* **`GET /cases/{id}/citations`**: Returns document-level citation references.
* **`POST /cases/{id}/improve`**: Triggers improvement/memify graph enrichment pipeline.
* **`POST /cases/{id}/forget`**: Deletes data or resets memory representations for the case.
