import json, numpy as np

E = "artifacts/faiss/embeddings.npy"
M = "artifacts/faiss/meta.json"

X = np.load(E)
with open(M, "r", encoding="utf-8") as f:
    meta = json.load(f)

assert X.shape[0] == len(meta["ids"]) == meta["num_vectors"]
print("Vectors & IDs aligned ✅")
print("Shape:", X.shape, "| dim:", meta["dim"], "| num_vectors:", meta["num_vectors"])
