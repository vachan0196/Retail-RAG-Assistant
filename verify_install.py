import os, sys, time
import numpy as np, pandas as pd

print("== Environment ==")
print("Python:", sys.version)

# Torch
try:
    import torch
    print("Torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("Torch import failed ->", repr(e))

# FAISS
try:
    import faiss
    print("FAISS:", faiss.__version__)
except Exception as e:
    print("FAISS import failed ->", repr(e))

print("\n== Embeddings test ==")
from sentence_transformers import SentenceTransformer
start = time.time()
emb_model_name = "sentence-transformers/all-MiniLM-L6-v2"
emb_model = SentenceTransformer(emb_model_name)
vec = emb_model.encode(["hello retail"], convert_to_numpy=True, normalize_embeddings=True)
print("Model:", emb_model_name, "| Embeddings shape:", vec.shape, "| norm≈", float(np.linalg.norm(vec[0])))
print("Embed time: %.2fs" % (time.time() - start))

print("\n== Reranker test ==")
from sentence_transformers import CrossEncoder
reranker_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(reranker_name)
pairs = [("return policy", "You have 30 days to return items."), ("return policy", "Free shipping on orders over £50.")]
scores = reranker.predict(pairs)
print("Model:", reranker_name, "| Scores:", [round(float(s), 4) for s in scores])

print("\n== Quick FAISS search smoke test ==")
# Build a tiny FAISS index to ensure it works end-to-end
dim = vec.shape[1]  # 384 for MiniLM
index = faiss.IndexFlatIP(dim)  # inner product for normalized vectors
# Create 3 dummy vectors
texts = ["return policy allows 30 days", "warranty covers 12 months", "standard delivery is 3-5 business days"]
X = emb_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
index.add(X)
D, I = index.search(vec, k=3)  # query is "hello retail" embedding; arbitrary smoke test
print("Nearest indices:", I.tolist(), "| distances:", [round(float(d), 4) for d in D[0]])
print("Nearest texts:", [texts[i] for i in I[0]])

print("\nAll good ✅")
