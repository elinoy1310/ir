# exe3/stage3_retrieval.py
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse
from sentence_transformers import SentenceTransformer

ROOT = Path("exe3")
BM25_DIR = ROOT / "bm25_vectors"
DENSE_DIR = ROOT / "st_vectors"

FIXED_UK_DIR = ROOT / "fixed-chunked-text" / "UK"
FIXED_US_DIR = ROOT / "fixed-chunked-text" / "US"
FIXED_UK_INDEX = FIXED_UK_DIR / "chunk_index.json"
FIXED_US_INDEX = FIXED_US_DIR / "chunk_index.json"

MODEL_NAME = "embaas/sentence-transformers-multilingual-e5-base"


# ---------------- Utils ----------------
def tokenize_for_bm25(q: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", q.lower())


def read_text_file(p: Path, max_chars: int = 2000) -> str:
    txt = p.read_text(encoding="utf-8", errors="ignore")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt[:max_chars]


def norm_path(s: str) -> str:
    return s.replace("\\", "/").strip()


def uk_count() -> int:
    # number of chunks in UK folder; used to disambiguate UK vs US by row index
    return sum(1 for _ in FIXED_UK_DIR.glob("chunk_*.txt"))


def load_chunkpath_to_source() -> Dict[str, str]:
    """
    Builds map: normalized_chunk_path -> source_file
    from both UK/US chunk_index.json.
    """
    m: Dict[str, str] = {}

    def ingest(index_path: Path):
        data = json.loads(index_path.read_text(encoding="utf-8"))
        for source_file, chunk_paths in data.items():
            for ch in chunk_paths:
                m[norm_path(ch)] = source_file

    ingest(FIXED_UK_INDEX)
    ingest(FIXED_US_INDEX)
    return m


def chunk_path_from_row(row_i: int, chunk_name: str, uk_n: int) -> Path:
    # UK rows are [0..uk_n-1], US rows are [uk_n..]
    if row_i < uk_n:
        return FIXED_UK_DIR / chunk_name
    return FIXED_US_DIR / chunk_name


def enrich_results(
    results: List[Tuple[int, str, float]],
    chunkpath_to_source: Dict[str, str],
    uk_n: int,
    max_chars: int = 600,
):
    out = []
    for row_i, chunk_name, score in results:
        p = chunk_path_from_row(row_i, chunk_name, uk_n)
        chunk_key = norm_path(str(p))
        source = chunkpath_to_source.get(chunk_key, "UNKNOWN_SOURCE")
        text = read_text_file(p, max_chars=max_chars) if p.exists() else ""
        out.append(
            {
                "row": int(row_i),
                "chunk": chunk_name,
                "score": float(score),
                "chunk_path": str(p),
                "source_file": source,
                "text": text,
            }
        )
    return out


# ---------------- BM25 ----------------
def load_bm25_store():
    X = sparse.load_npz(BM25_DIR / "bm25_vectors.npz").tocsr()
    vocab = json.loads((BM25_DIR / "bm25_vectors_vocabulary.json").read_text(encoding="utf-8"))
    files = json.loads((BM25_DIR / "bm25_vectors_files.json").read_text(encoding="utf-8"))["files"]
    names = [Path(f).name for f in files]  # only basename
    return X, vocab, names


def bm25_retrieve(query: str, X_bm25, vocab: dict, chunk_names: List[str], top_k: int):
    tokens = tokenize_for_bm25(query)
    counts: Dict[str, int] = {}
    for t in tokens:
        if t in vocab:
            counts[t] = counts.get(t, 0) + 1
    if not counts:
        return []

    idxs = [vocab[t] for t in counts.keys()]
    vals = list(counts.values())

    q_vec = sparse.csr_matrix((vals, ([0] * len(idxs), idxs)), shape=(1, X_bm25.shape[1]))
    scores = (X_bm25 @ q_vec.T).toarray().ravel()

    top_idx = np.argsort(-scores)[:top_k]
    return [(int(i), chunk_names[int(i)], float(scores[int(i)])) for i in top_idx if scores[int(i)] > 0]


# ---------------- Dense ----------------
def load_dense_store():
    # NOTE: stored as sparse but loaded here to dense ndarray for speed in cosine ops
    X = sparse.load_npz(DENSE_DIR / "embeddings_sparse.npz").toarray()
    names = [Path(x).name for x in (DENSE_DIR / "filenames.txt").read_text(encoding="utf-8").splitlines()]
    return X, names


def cosine_topk(X: np.ndarray, q: np.ndarray, k: int):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    qn = q / (np.linalg.norm(q) + 1e-12)
    sims = Xn @ qn
    top_idx = np.argsort(-sims)[:k]
    return top_idx, sims


def dense_retrieve(query: str, X_emb: np.ndarray, chunk_names: List[str], model: SentenceTransformer, top_k: int):
    q_emb = model.encode("query: " + query)
    top_idx, sims = cosine_topk(X_emb, q_emb, top_k)
    return [(int(i), chunk_names[int(i)], float(sims[int(i)])) for i in top_idx]


# ---------------- Hybrid (FIXED) ----------------
def build_dense_lookup(dense_names: List[str], uk_n: int) -> Dict[Tuple[str, str], int]:
    """
    Map (region, chunk_name) -> dense_row_index
    Region is determined from the dense row index using uk_n.
    """
    lookup: Dict[Tuple[str, str], int] = {}
    for i, name in enumerate(dense_names):
        region = "UK" if i < uk_n else "US"
        lookup[(region, name)] = int(i)
    return lookup


def hybrid_retrieve(
    query: str,
    X_bm25,
    vocab: dict,
    bm25_names: List[str],
    X_emb: np.ndarray,
    dense_lookup: Dict[Tuple[str, str], int],
    uk_n: int,
    model: SentenceTransformer,
    top_k: int,
    bm25_candidates: int = 200,
):
    """
    Correct Hybrid:
    1) BM25 -> top-N candidates (in BM25 index space)
    2) For each candidate, infer region (UK/US) using BM25 row index and uk_n
    3) Map (region, chunk_name) to the correct dense row via dense_lookup
    4) Dense rerank only mapped candidates (returns DENSE row indices)
    """
    bm25_res = bm25_retrieve(query, X_bm25, vocab, bm25_names, top_k=bm25_candidates)
    if not bm25_res:
        return []

    cand_dense_rows: List[int] = []
    cand_chunk_names: List[str] = []

    for bm25_row, chunk_name, _score in bm25_res:
        region = "UK" if bm25_row < uk_n else "US"
        drow = dense_lookup.get((region, chunk_name))
        if drow is None:
            continue
        cand_dense_rows.append(drow)
        cand_chunk_names.append(chunk_name)

    if not cand_dense_rows:
        return []

    q_emb = model.encode("query: " + query)

    X_cand = X_emb[cand_dense_rows, :]
    Xn = X_cand / (np.linalg.norm(X_cand, axis=1, keepdims=True) + 1e-12)
    qn = q_emb / (np.linalg.norm(q_emb) + 1e-12)
    sims = Xn @ qn

    top_local = np.argsort(-sims)[:top_k]

    out: List[Tuple[int, str, float]] = []
    for j in top_local:
        dense_row = cand_dense_rows[int(j)]
        name = cand_chunk_names[int(j)]
        out.append((int(dense_row), name, float(sims[int(j)])))
    return out


# ---------------- Runner ----------------
def run_query(query: str, K_values=(3, 5, 8)):
    uk_n = uk_count()
    chunkpath_to_source = load_chunkpath_to_source()

    # load stores
    X_bm25, vocab, bm25_names = load_bm25_store()
    X_emb, dense_names = load_dense_store()

    # build mapping for hybrid (no re-embedding needed)
    dense_lookup = build_dense_lookup(dense_names, uk_n)

    model = SentenceTransformer(MODEL_NAME)

    print(f"UK chunks count (used for disambiguation): {uk_n}")

    for K in K_values:
        print("\n" + "=" * 90)
        print(f"QUERY: {query}")
        print(f"K={K}")

        # BM25
        bm25_res = bm25_retrieve(query, X_bm25, vocab, bm25_names, top_k=K)
        bm25_enriched = enrich_results(bm25_res, chunkpath_to_source, uk_n)
        print("\n--- BM25 top-K ---")
        for i, r in enumerate(bm25_enriched, 1):
            print(
                f"[{i}] score={r['score']:.4f} row={r['row']} source={r['source_file']} chunk_path={r['chunk_path']}"
            )
            print(f"    {r['text']}\n")

        # Dense
        dense_res = dense_retrieve(query, X_emb, dense_names, model, top_k=K)
        dense_enriched = enrich_results(dense_res, chunkpath_to_source, uk_n)
        print("\n--- Dense (Embeddings) top-K ---")
        for i, r in enumerate(dense_enriched, 1):
            print(
                f"[{i}] sim={r['score']:.4f} row={r['row']} source={r['source_file']} chunk_path={r['chunk_path']}"
            )
            print(f"    {r['text']}\n")

        # Hybrid (BM25 candidates -> Dense rerank) -- fixed mapping
        hyb_res = hybrid_retrieve(
            query=query,
            X_bm25=X_bm25,
            vocab=vocab,
            bm25_names=bm25_names,
            X_emb=X_emb,
            dense_lookup=dense_lookup,
            uk_n=uk_n,
            model=model,
            top_k=K,
            bm25_candidates=200,
        )
        hyb_enriched = enrich_results(hyb_res, chunkpath_to_source, uk_n)
        print("\n--- HYBRID (BM25 candidates -> Dense rerank) top-K ---")
        for i, r in enumerate(hyb_enriched, 1):
            print(
                f"[{i}] sim={r['score']:.4f} row={r['row']} source={r['source_file']} chunk_path={r['chunk_path']}"
            )
            print(f"    {r['text']}\n")


if __name__ == "__main__":
    q = "What organizations were mentioned by the speakers as supporting the proposed reform of the health system?"
    run_query(q, K_values=(3, 5, 8))
