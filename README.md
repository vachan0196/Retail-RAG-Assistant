# 🛍️ Retail RAG Assistant

> An **AI-powered Q&A assistant** for retail companies, built with **RAG (Retrieval-Augmented Generation)**, **FAISS vector search**, and **LLMs (OpenAI/Cohere)**.  
> It can answer questions about **returns, refunds, warranty, shipping, and product catalogues** by retrieving from your internal docs — with citations.

---

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ Features
- **Document ingestion & chunking** (policies, FAQs, product catalogue, sales summaries)  
- **Semantic search** with embeddings + FAISS  
- **Reranking** using cross-encoder for higher precision  
- **LLM-powered answers** (OpenAI / Cohere, with offline fallback)  
- **Evaluation** (Hit@K, MRR, faithfulness)  
- **Streamlit UI** with citations & clear sources  
- **Dockerized** for easy deployment  
- **Environment template** via `.env.example` (no secrets committed)

---



## ▶️ Usage

### 1. Run ingestion (chunk documents)
```bash
python ingest.py
```

### 2. Build embeddings + FAISS index
```bash
python index.py embed
python index.py build
```

### 3. Smoke test retrieval
```bash
python index.py search --q "What is the return window for returns?" --k 5
```

### 4. Ask questions (CLI)
```bash
python rag.py --cmd answer --q "What is the return window for returns?" --k 10 --rk 5
```

### 5. Launch Streamlit app
```bash
streamlit run app.py
```
Then open 👉 [http://localhost:8501](http://localhost:8501)

---

## 📂 Project Structure
```
retail-rag-assistant/
│
├── app.py                  # Streamlit UI
├── rag.py                  # Retrieval + generation pipeline
├── ingest.py               # Chunk & ingest documents
├── index.py                # Embeddings + FAISS index
├── eval.py                 # Evaluation scripts
├── utils/                  # Helper functions
│   ├── io.py
│   ├── prompts.py
│   └── text.py
│
├── requirements.txt        # Dependencies
├── Dockerfile              # Docker setup
├── .dockerignore           # Docker ignore rules
├── .gitignore              # Git ignore rules
└── .env.example            # Env template (no secrets)
```

---

## 🐳 Docker

### 📦 Build image
```bash
docker build -t retail-rag .
```

### ▶️ Run container
```bash
docker run --rm -p 8501:8501 --env-file .env retail-rag
```

Then open 👉 [http://localhost:8501](http://localhost:8501)

---

## 🔑 Environment Variables
See `.env.example` for template.

```ini
OPENAI_API_KEY=your_OpenAI_key_here
COHERE_API_KEY=your_cohere_key_here

DOCS_DIR=/app/docs
USE_OPENAI=false
USE_COHERE=true

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

TOP_K=8
RERANK_TOP_K=5

PINECONE_API_KEY=your_key_here
PINECONE_INDEX=retail-rag

WEAVIATE_URL=your_url_here
WEAVIATE_API_KEY=your_key_here

HF_CACHE_DIR=/app/artifacts/models
```

---

## 📸 Demo (Optional)
Add a screenshot or GIF of your Streamlit app running here for extra impact. Example:

![Retail RAG Assistant Demo](demo-screenshot.png)

---

