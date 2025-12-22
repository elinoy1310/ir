# exe3/stage4_llm_rag.py
import json
import re
from pathlib import Path

import numpy as np
from scipy import sparse
import requests
from sentence_transformers import SentenceTransformer

ROOT = Path("exe3")

# choose which retrieval store to use
BM25_DIR = ROOT / "bm25_vectors_v2"
DENSE_DIR = ROOT / "st_vectors_v2"

FIXED_UK_DIR = ROOT / "fixed-chunked-text" / "UK"
FIXED_US_DIR = ROOT / "fixed-chunked-text" / "US"
FIXED_UK_INDEX = FIXED_UK_DIR / "chunk_index.json"
FIXED_US_INDEX = FIXED_US_DIR / "chunk_index.json"

# Embedding model (same as your stage3)
MODEL_NAME = "embaas/sentence-transformers-multilingual-e5-base"

# Ollama settings
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "gemma:2b"

def tokenize_for_bm25(q: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", q.lower())

def read_text_file(p: Path, max_chars: int = 1200) -> str:
    txt = p.read_text(encoding="utf-8", errors="ignore")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt[:max_chars]

def norm_path(s: str) -> str:
    return s.replace("\\", "/").strip()

def load_chunkpath_to_source() -> dict[str, str]:
    m = {}
    def ingest(index_path: Path):
        data = json.loads(index_path.read_text(encoding="utf-8"))
        for source_file, chunk_paths in data.items():
            for ch in chunk_paths:
                m[norm_path(ch)] = source_file
    ingest(FIXED_UK_INDEX)
    ingest(FIXED_US_INDEX)
    return m

# -------- BM25 --------
def load_bm25_store():
    X = sparse.load_npz(BM25_DIR / "bm25_vectors.npz").tocsr()
    vocab = json.loads((BM25_DIR / "bm25_vectors_vocabulary.json").read_text(encoding="utf-8"))
    files = json.loads((BM25_DIR / "bm25_vectors_files.json").read_text(encoding="utf-8"))["files"]
    names = [norm_path(str(Path(f)).replace("\\", "/")) for f in files]
    return X, vocab, names

def bm25_retrieve(query: str, X_bm25, vocab: dict, chunk_names: list[str], top_k: int):
    tokens = tokenize_for_bm25(query)
    counts = {}
    for t in tokens:
        if t in vocab:
            counts[t] = counts.get(t, 0) + 1
    if not counts:
        return []

    idxs = [vocab[t] for t in counts.keys()]
    vals = list(counts.values())
    q_vec = sparse.csr_matrix((vals, ([0]*len(idxs), idxs)), shape=(1, X_bm25.shape[1]))
    scores = (X_bm25 @ q_vec.T).toarray().ravel()
    top_idx = np.argsort(-scores)[:top_k]
    return [(int(i), chunk_names[int(i)], float(scores[int(i)])) for i in top_idx if scores[int(i)] > 0]

# -------- Dense --------
def load_dense_store():
    X = np.load(DENSE_DIR / "embeddings.npy")
    names = (DENSE_DIR / "filenames.txt").read_text(encoding="utf-8").splitlines()
    names = [norm_path(n) for n in names]  # UK/chunk_123.txt
    return X, names

def cosine_topk(X: np.ndarray, q: np.ndarray, k: int):
    q = q / (np.linalg.norm(q) + 1e-12)
    sims = X @ q
    top_idx = np.argsort(-sims)[:k]
    return top_idx, sims

def dense_retrieve(query: str, X_emb: np.ndarray, chunk_names: list[str], model: SentenceTransformer, top_k: int):
    q_emb = model.encode("query: " + query, convert_to_numpy=True, normalize_embeddings=True)
    top_idx, sims = cosine_topk(X_emb, q_emb, top_k)
    return [(int(i), chunk_names[int(i)], float(sims[int(i)])) for i in top_idx]

# -------- Paths / enrichment --------
def chunk_path_from_relname(relname: str) -> Path:
    relname = norm_path(relname)
    if relname.startswith("UK/"):
        return FIXED_UK_DIR / relname.split("/", 1)[1]
    if relname.startswith("US/"):
        return FIXED_US_DIR / relname.split("/", 1)[1]
    p1 = FIXED_UK_DIR / relname
    if p1.exists():
        return p1
    return FIXED_US_DIR / relname

def enrich(results, chunkpath_to_source, max_chars=800):
    out = []
    for row_i, relname, score in results:
        p = chunk_path_from_relname(relname)
        chunk_key = norm_path(str(p))
        source = chunkpath_to_source.get(chunk_key, "UNKNOWN_SOURCE")
        text = read_text_file(p, max_chars=max_chars) if p.exists() else ""
        out.append({
            "row": row_i,
            "chunk": relname,
            "score": score,
            "chunk_path": str(p),
            "source_file": source,
            "text": text,
        })
    return out

# -------- LLM call --------
def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 300
        }
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "").strip()

def build_prompt(query: str, contexts: list[dict]) -> str:
    # Force grounded answer + sources
    ctx_lines = []
    for i, c in enumerate(contexts, 1):
        ctx_lines.append(
            f"[{i}] SOURCE_FILE: {c['source_file']}\n"
            f"CHUNK: {c['chunk']}\n"
            f"TEXT: {c['text']}\n"
        )

    return (
        "You are a QA assistant. Answer ONLY using the CONTEXT below.\n"
        "If the answer is not in the context, say: \"I don't know based on the provided context.\" \n"
        "At the end, list the sources you used as: SOURCES: [1], [2], ...\n\n"
        f"QUESTION: {query}\n\n"
        "CONTEXT:\n"
        + "\n".join(ctx_lines)
    )

def run_rag(query: str, method: str = "hybrid", k: int = 5):
    chunkpath_to_source = load_chunkpath_to_source()

    X_bm25, vocab, bm25_names = load_bm25_store()
    X_emb, dense_names = load_dense_store()
    st_model = SentenceTransformer(MODEL_NAME)

    if method == "bm25":
        retrieved = bm25_retrieve(query, X_bm25, vocab, bm25_names, top_k=k)
        # convert names to rel format best-effort: expect already like UK/chunk_x
        retrieved = [(r, norm_path(n), s) for (r, n, s) in retrieved]

    elif method == "dense":
        retrieved = dense_retrieve(query, X_emb, dense_names, st_model, top_k=k)

    else:
        # simplest hybrid: BM25 candidates then rerank by dense over SAME names
        bm25_cands = bm25_retrieve(query, X_bm25, vocab, bm25_names, top_k=200)
        # map by basename inside UK/ or US/ if missing
        dense_map = {name: i for i, name in enumerate(dense_names)}
        cand_rows = []
        cand_names = []
        for _, name, _ in bm25_cands:
            n = norm_path(name)
            if n in dense_map:
                cand_rows.append(dense_map[n]); cand_names.append(n); continue
            base = Path(n).name
            for pref in ("UK/", "US/"):
                key = pref + base
                if key in dense_map:
                    cand_rows.append(dense_map[key]); cand_names.append(key); break

        q_emb = st_model.encode("query: " + query, convert_to_numpy=True, normalize_embeddings=True)
        X_cand = X_emb[np.array(cand_rows, dtype=int)]
        sims = X_cand @ q_emb
        top_local = np.argsort(-sims)[:k]
        retrieved = [(int(cand_rows[j]), cand_names[j], float(sims[j])) for j in top_local]

    enriched = enrich(retrieved, chunkpath_to_source, max_chars=900)

    prompt = build_prompt(query, enriched)
    answer = call_ollama(prompt)

    print("\n" + "="*90)
    print(f"METHOD={method}  K={k}")
    print("QUESTION:", query)
    print("\nANSWER:\n", answer)
    print("\nTOP CONTEXT SOURCES:")
    for i, c in enumerate(enriched, 1):
        print(f"[{i}] {c['source_file']}  ({c['chunk']})  score={c['score']:.4f}")

if __name__ == "__main__":
    q = "What was the main argument regarding the immigration bill that was presented?"
    run_rag(q, method="hybrid", k=5)
