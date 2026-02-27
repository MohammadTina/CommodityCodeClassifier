# Commodity Code Classifier
A RAG system for classifying product descriptions into national commodity codes using hybrid retrieval and local LLM reasoning.

## Overview

This system automates tariff code assignment by combining semantic search, keyword matching, and LLM-based reasoning. It handles ~13K hierarchical commodity codes and provides confidence signals for routing uncertain classifications to human review.

**Key Features:**
- Hybrid retrieval (FAISS + BM25 + RRF fusion)
- Local LLM deployment (Qwen2.5-7B via Ollama)
- Clustering-based confidence signals
- Legal notes integration for exclusion rules
- Fully configurable (swap LLMs, embeddings, retrieval params via YAML)
- Production-ready REST API with auto-generated docs
- SQLite-based classification history and audit logging

---

## Architecture
```
User Query
    ↓
[Preprocessing] ← configurable pipeline (strip phrases, expand abbreviations)
    ↓
[Hybrid Retrieval]
    ├── Semantic Search  → FAISS vector index (top-20)
    ├── BM25 Search      → keyword index (top-20)
    └── RRF Fusion       → merged candidates (top-10)
    ↓
[Hierarchy Analysis] ← clustering signal (confidence calibration)
    ↓
[LLM Classification] ← structured prompting for reliable output
    ↓
[Response Validation] ← ensure code exists in candidates
    ↓
REST API Response
```

---

## Prerequisites

### System Requirements
- Python 3.10+
- 8GB+ RAM (for embeddings + LLM)
- Ollama installed (for local LLM serving)

### Database Requirements

The system works with any hierarchical commodity code database containing:

**Required Tables:**
1. **Section/Chapter level** (1-2 digit codes)
2. **Chapter level** (2 digit codes)
3. **Heading level** (4 digit codes)
4. **Subheading level** (6 digit codes)
5. **National code level** (8-10 digit codes, target classification)

**Required Columns per Table:**
- Code identifier (e.g., `HS2_COD`, `HS4_COD`)
- Description (e.g., `HS2_DSC`)
- Valid from/to dates (for filtering active codes)
- Parent code reference (for hierarchy joins)

**Optional but Recommended:**
- Legal notes/exclusion text (e.g., `HS2_TXT`)

---

## Installation

### 1. Install Dependencies
```bash
# Clone the repository
git clone this repo
cd commodity_classifier

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Install and Configure Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the LLM model
ollama pull qwen2.5:7b

# Keep model loaded (optional but recommended for faster inference)
export OLLAMA_KEEP_ALIVE=-1
```

### 3. Configure Database Connection

**Option A: Environment Variable (Recommended)**
```bash
export DB_CONNECTION_STRING="your_database_connection_string"
# Examples:
# Oracle: "oracle+oracledb://user:pass@host:port/?service_name=dbname"
# PostgreSQL: "postgresql://user:pass@host:port/dbname"
```

**Option B: Direct Configuration**
Edit `config/database.yaml` and replace the connection string placeholder.

### 4. Configure Table and Column Mappings

Edit `config/database.yaml` to match your database schema:
```yaml
database:
  tables:
    section:    "your_section_table_name"
    chapter:    "your_chapter_table_name"
    heading:    "your_heading_table_name"
    subheading: "your_subheading_table_name"
    national:   "your_national_table_name"
  
  columns:
    section:
      code:        "your_code_column"
      description: "your_description_column"
      # ... map all required columns
```

See `config/database.yaml` for the complete column mapping template.

---

## Setup

### Build Search Indexes

**Run this once before starting the API** (and again whenever commodity codes change):
```bash
python scripts/build_index.py
```

This will:
1. Connect to your database
2. Fetch all active commodity codes with full hierarchy
3. Build FAISS vector index (~13K embeddings)
4. Build BM25 keyword index
5. Save indexes to `data/indexes/`

Expected output:
```
Fetched 13000 active national commodity codes
Building FAISS index: 13000 records, 1024 dimensions
FAISS index saved: 13000 vectors
BM25 index saved
```

### Validate Installation
```bash
python scripts/validate_index.py --query "wheat flour"
```

Expected output:
```
Index loaded: 13000 records
Test query: 'wheat flour'
Top 3 candidates:
  1. [11010090] Wheat Or Meslin Flour, Excl. wrapped/canned...
  2. [11010010] Wheat Or Meslin Flour, wrapped/canned...
  3. [11022000] Maize (Corn) Flour...
Validation PASSED
```

---

## Running the API

### Start the Server
```bash
python main.py
```

The API will be available at:
- **API endpoint:** `http://localhost:8000`
- **Interactive docs:** `http://localhost:8000/docs` (Swagger UI)
- **Alternative docs:** `http://localhost:8000/redoc`

### API Endpoints

#### Classify a Product
```bash
POST /api/v1/classify
Content-Type: application/json

{
    "description": "wheat flour for baking",
    "request_id": "optional-tracking-id"
}
```

**Response:**
```json
{
    "request_id": "abc-123",
    "confidence": "HIGH",
    "national_code": "11010090",
    "description": "Wheat Or Meslin Flour, Excl. wrapped/canned upto 2.5 kg",
    "reasoning": "The description clearly indicates wheat flour...",
    "hierarchy_path": {
        "section": "Vegetable products",
        "chapter": "Products of the milling industry...",
        "heading": "Wheat or meslin flour",
        "subheading": "Wheat or meslin flour",
        "national": "Wheat Or Meslin Flour, Excl. wrapped/canned..."
    },
    "processing_time_ms": 3200
}
```

#### Health Check
```bash
GET /api/v1/health
```

#### Trigger Index Rebuild
```bash
POST /api/v1/reindex
```

#### Get Classification History
```bash
GET /api/v1/history?page=1&page_size=20&confidence=HIGH
```

---

## Configuration

All behavior is controlled via YAML files in `config/`:

### Swap LLM Model
`config/llm.yaml`:
```yaml
llm:
  model: "qwen2.5:14b"  # upgrade from 7b to 14b
```

### Change Embedding Model
`config/embedding.yaml`:
```yaml
embedding:
  model: "intfloat/e5-large-v2"  # swap embedding model
```
Then rebuild indexes: `python scripts/build_index.py`

### Adjust Retrieval Parameters
`config/retrieval.yaml`:
```yaml
retrieval:
  semantic_top_k: 30     # increase candidates
  llm_candidates: 5      # pass more to LLM
```

### Add Preprocessing Rules
`config/preprocessing.yaml`:
```yaml
- name: abbreviation_expansion
  enabled: true
  mappings:
    "PC": "personal computer"
    "YOUR_ABBR": "your expansion"  # add custom mappings
```

---


---

## Performance

**First Request (Cold Start):**
- Model loading: 10-15 seconds
- Classification: 3-5 seconds
- **Total: 15-20 seconds**

**Subsequent Requests (Warm):**
- Model already loaded
- **Total: 3-5 seconds**

**Optimization:**
Set `OLLAMA_KEEP_ALIVE=-1` to keep model loaded permanently.

---

## Troubleshooting

### "No module named 'oracledb'"
```bash
pip install oracledb
```
Update connection string to use `oracle+oracledb://` format.

### "Failed to load indexes"
```bash
python scripts/build_index.py
```
Indexes must be built before starting the API.

### "LLM request timed out"
Increase timeout in `config/llm.yaml`:
```yaml
llm:
  timeout: 120  # increase from 60 seconds
```

### First request is slow
This is normal for local LLM deployment. Set:
```bash
export OLLAMA_KEEP_ALIVE=-1
```
Then restart Ollama.

---

## Technologies Used

- **Embeddings:** sentence-transformers (PyTorch)
- **Vector Search:** FAISS
- **Keyword Search:** rank-bm25
- **LLM:** Qwen2.5-7B-Instruct (via Ollama)
- **API:** FastAPI
- **Database:** SQLAlchemy (supports Oracle, PostgreSQL, MySQL, etc.)
- **History:** SQLite
- **Config:** PyYAML

---

## License

MIT License - See LICENSE file for details

---

## Citation

If you use this project in your work, please cite:
```bibtex
@software{commodity_classifier_2026,
  author = Mohammad Tina,
  title = {Commodity Code Classifier: A RAG System for Tariff Classification},
  year = {2026},
  url = {https://github.com/MohammadTina/CommodityCodeClassifier/}
}
```

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## Contact

For questions or issues, please open a GitHub issue or contact [your email].
