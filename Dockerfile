# ---------- Base ----------
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Make pip fast & robust
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_CACHE_DIR=/app/artifacts/models \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501

# Copy requirements first (better caching)
COPY requirements.txt .

# 1) Upgrade pip tooling
# 2) Install Torch CPU wheel from official index (no compile)
# 3) Install FAISS CPU wheel (pinned to wheel build)
# 4) Install the rest of your deps
RUN pip install --upgrade pip setuptools wheel && \
    pip install --prefer-binary --no-cache-dir \
        torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --prefer-binary --no-cache-dir faiss-cpu==1.8.0.post1 && \
    pip install --prefer-binary --no-cache-dir -r requirements.txt

# App code
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.serverAddress=localhost"]

 