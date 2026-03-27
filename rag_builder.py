"""
rag_builder.py
--------------
Builds and saves a FAISS vector index from PPK COMPILE.xlsx.
Run this once before starting the chatbot:
    py -3 rag_builder.py

The index is saved to ./faiss_index/ and reloaded by app.py at startup.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── Configuration ────────────────────────────────────────────────────────────
EXCEL_PATH   = os.path.join(os.path.dirname(__file__), "PPK COMPILE.xlsx")
INDEX_DIR    = os.path.join(os.path.dirname(__file__), "faiss_index")
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"   # fast & light
# ─────────────────────────────────────────────────────────────────────────────


def load_ppk_data(excel_path: str) -> list[dict]:
    """
    Parse PPK COMPILE.xlsx into structured records.

    The Excel has 3 columns: Disease_key, Section, Content.
    Disease_key is only filled in the first row of each disease block;
    subsequent rows inherit the same disease name.
    """
    df = pd.read_excel(excel_path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    records = []
    current_disease = None

    for _, row in df.iterrows():
        disease = str(row.get("Disease_key", "")).strip()
        section = str(row.get("Section", "")).strip()
        content = str(row.get("Content", "")).strip()

        if disease and disease.lower() not in ("nan", ""):
            current_disease = disease

        if current_disease and section and content and content.lower() != "nan":
            records.append({
                "disease":  current_disease,
                "section":  section,
                "content":  content,
                "doc_text": f"Disease: {current_disease}\nSection: {section}\n{content}"
            })

    print(f"[RAG Builder] Loaded {len(records)} PPK records for "
          f"{len(set(r['disease'] for r in records))} diseases.")
    return records


def build_index(records: list[dict], embed_model_name: str, index_dir: str):
    """Embed all records and save FAISS index + metadata."""
    import faiss

    os.makedirs(index_dir, exist_ok=True)

    print(f"[RAG Builder] Loading embedding model: {embed_model_name}")
    model = SentenceTransformer(embed_model_name)

    texts = [r["doc_text"] for r in records]
    print(f"[RAG Builder] Embedding {len(texts)} documents…")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)          # Inner-product = cosine similarity
    index.add(embeddings)

    faiss.write_index(index, os.path.join(index_dir, "ppk.index"))

    with open(os.path.join(index_dir, "ppk_records.pkl"), "wb") as f:
        pickle.dump(records, f)

    print(f"[RAG Builder] Index saved to {index_dir}/ "
          f"({index.ntotal} vectors, dim={dim})")


def retrieve(query: str, model, index, records: list[dict],
             top_k: int = 5, threshold: float = 0.30) -> list[dict]:
    """
    Retrieve the top-k PPK records most relevant to `query`.
    Returns records whose cosine similarity exceeds `threshold`.
    """
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype=np.float32)

    scores, indices = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and score >= threshold:
            results.append({**records[idx], "score": float(score)})
    return results


if __name__ == "__main__":
    records = load_ppk_data(EXCEL_PATH)
    build_index(records, EMBED_MODEL, INDEX_DIR)
    print("\n[RAG Builder] ✅  Done! You can now run: py -3 app.py")
