🛍️ Retail RAG Assistant

An AI-powered Q&A assistant for retail knowledge bases.
Built with RAG (Retrieval-Augmented Generation), FAISS vector search, and modern LLMs (Cohere / OpenAI).

This project demonstrates real-world AI skills companies expect in 2025:

✅ RAG pipelines (retriever + reranker + generator)

✅ Vector databases (FAISS locally, Pinecone/Weaviate for production)

✅ Semantic search with embeddings

✅ Prompt/context engineering

✅ LLMOps basics (evaluation, feedback logging)

✅ Deployment via Streamlit + Docker

📂 Project Structure
retail-rag-assistant/
│
├── app.py                  # Streamlit UI
├── rag.py                  # Retrieval + generation pipeline
├── ingest.py               # Chunk & ingest documents
├── index.py                # Embeddings + FAISS index
├── eval.py                 # Simple evaluation scripts
├── utils/                  # Helper functions
├── requirements.txt        # Dependencies
├── Dockerfile              # Docker setup
├── .dockerignore
├── .gitignore
└── .env.example            # Environment template (no secrets)

⚡ Setup
1) Clone the repo
git clone https://github.com/vachan0196/Retail-RAG-Assistant.git
cd Retail-RAG-Assistant

2) Create environment
conda create -n retail-rag python=3.11 -y
conda activate retail-rag
pip install -r requirements.txt

3) Configure environment variables

Copy the example env and fill in your keys:

cp .env.example .env

▶️ Usage
Run ingestion (chunk documents)
python ingest.py

Build embeddings & FAISS index
python index.py embed
python index.py build

Test retrieval + RAG pipeline
python rag.py --cmd answer --q "What is the return window for returns?" --k 8 --rk 5

Launch Streamlit app
streamlit run app.py

🐳 Run with Docker
docker build -t retail-rag .
docker run -p 8501:8501 retail-rag


App will be live at http://localhost:8501
.

📊 Evaluation (LLMOps-lite)
python eval.py


Outputs:

Retrieval metrics (Hit@k, MRR)

Faithfulness check samples → artifacts/eval/results.json

🌟 Features

Semantic search with sentence-transformers/all-MiniLM-L6-v2

Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)

Flexible generation (OpenAI GPT-4o-mini / Cohere command-r / offline fallback)

Streamlit chat UI with citations and sources

Clean modular design (easy to extend with Pinecone, Weaviate, or LangChain)

🚀 Business Impact

This project shows how retail companies can:

Reduce support load by automating FAQs (returns, warranty, delivery)

Provide faster answers grounded in policies & product catalogues

Extend to sales insights, recommendations, and seasonal analysis

📌 Notes

Never commit .env with real keys. Only use .env.example.

Models are cached locally (HF_CACHE_DIR), not stored in GitHub.

👤 Author

Vachan Sardar
Data Science, ML Engineering, AI Agents

LinkedIn www.linkedin.com/in/vachan-sardar
