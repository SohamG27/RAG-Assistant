# RAG Assistant — Groq + HuggingFace Embeddings

A **Retrieval-Augmented Generation (RAG)** chatbot that turns any PDF into an interactive knowledge base.  
Built with **Groq ultra-fast inference**, **local HuggingFace embeddings**, and **FAISS vector search**.

---

## Features

- Ask questions in plain English — get accurate, sourced answers from your PDFs
- Fully **local embeddings** — no external embedding API calls or costs
- **FAISS** semantic search for fast, relevant retrieval
- Answers include **page citations** and source snippets
- Modular design — swap models or vector stores easily
- Interactive PDF picker at ingest time — choose exactly which document to index

---

## Architecture

```
ingest.py  →  Loads PDF → Splits text → Embeds → Saves FAISS index
main.py    →  Loads FAISS index → Retrieves chunks → Groq LLM → Answer + Citations
```

### `ingest.py` — Knowledge Base Builder
- Scans folder for PDFs, shows a selection menu
- Splits text with `RecursiveCharacterTextSplitter` (600 token chunks, 60 overlap)
- Embeds with `BAAI/bge-small-en-v1.5` (runs locally on CPU)
- Saves to `faiss_index/`

### `main.py` — Interactive Chatbot
- Loads pre-built FAISS index instantly
- Retrieves top-3 relevant chunks per query
- Sends context + question to Groq LLM
- Prints answer with page numbers and text snippets

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM Inference | [Groq Cloud](https://console.groq.com) (`openai/gpt-oss-20b`) |
| Embeddings | `BAAI/bge-small-en-v1.5` via HuggingFace (local CPU) |
| Vector Store | FAISS |
| Orchestration | LangChain 0.3.x |
| PDF Loader | PyPDF |
| Env Management | python-dotenv |

---

## Getting Started

### Prerequisites
- Python 3.9+
- A free [Groq API Key](https://console.groq.com)

### 1. Clone the repo
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
Copy `.env.example` to `.env` and fill in your Groq API key:
```bash
cp .env.example .env
```

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

---

## Usage

### Step 1 — Build the knowledge base
Place your PDF(s) in the project folder, then run:
```bash
python ingest.py
```
An interactive menu will appear — pick the PDF you want to index:
```
Available PDFs:
  [1] resume.pdf
  [2] report.pdf
  [R] Refresh list
  [Q] Quit

Select a PDF to ingest (1-2, R to refresh, Q to quit): 1
```

### Step 2 — Start the assistant
```bash
python main.py
```
```
---  Assistant Active  ---

Question: What are the key projects mentioned?

AI: The document highlights three main projects: ...

Sources used:
  - [Page 0]: Soham Ghosh — Data Science Graduate...
  - [Page 1]: IPL 2026 Impact Player Prediction...
```

Type `exit` or `quit` to stop.

---

## Project Structure

```
├── main.py                  # Chatbot interface
├── ingest.py                # PDF ingestion & FAISS index builder
├── requirements.txt         # Pinned dependencies
├── .env.example             # Environment variable template
├── .gitignore               # Excludes .env, .venv, faiss_index/
└── your_document.pdf        # Your PDF(s) go here
```

> **Note:** `faiss_index/` and `.env` are excluded from git — never commit your API key.

---

## Example Use Cases

- **Resume Q&A** — Ask about skills, experience, and projects from a CV
- **Document Search** — Query reports, manuals, or legal contracts in natural language
- **Study Assistant** — Chat with your textbooks or research papers
- **Support Bot** — Turn product documentation into an AI helpdesk

---

## License

MIT License — see `LICENSE` for details.

---

Built with Python, LangChain, Groq, and HuggingFace.
