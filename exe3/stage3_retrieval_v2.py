# exe3/stage3_retrieval_v2.py
import json
import re
from pathlib import Path

import numpy as np
from scipy import sparse
from sentence_transformers import SentenceTransformer

ROOT = Path("exe3")

# BM25 (existing)
BM25_DIR = ROOT / "bm25_vectors_v2"

# Dense v2 (NEW)
DENSE_DIR = ROOT / "st_vectors_v2"

# chunks + indexes
FIXED_UK_DIR = ROOT / "fixed-chunked-text" / "UK"
FIXED_US_DIR = ROOT / "fixed-chunked-text" / "US"
FIXED_UK_INDEX = FIXED_UK_DIR / "chunk_index.json"
FIXED_US_INDEX = FIXED_US_DIR / "chunk_index.json"

MODEL_NAME = "embaas/sentence-transformers-multilingual-e5-base"


def tokenize_for_bm25(q: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", q.lower())


def read_text_file(p: Path, max_chars: int = 1200) -> str:
    txt = p.read_text(encoding="utf-8", errors="ignore")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt[:max_chars]


def norm_path(s: str) -> str:
    return s.replace("\\", "/").strip()


def load_chunkpath_to_source() -> dict[str, str]:
    """
    Builds map: normalized_chunk_path -> source_file
    from both UK/US chunk_index.json.
    """
    m = {}

    def ingest(index_path: Path):
        data = json.loads(index_path.read_text(encoding="utf-8"))
        for source_file, chunk_paths in data.items():
            for ch in chunk_paths:
                m[norm_path(ch)] = source_file

    ingest(FIXED_UK_INDEX)
    ingest(FIXED_US_INDEX)
    return m


# ---------------- BM25 ----------------
def load_bm25_store():
    X = sparse.load_npz(BM25_DIR / "bm25_vectors.npz").tocsr()
    vocab = json.loads((BM25_DIR / "bm25_vectors_vocabulary.json").read_text(encoding="utf-8"))
    files = json.loads((BM25_DIR / "bm25_vectors_files.json").read_text(encoding="utf-8"))["files"]

    # IMPORTANT: keep relative path if exists, not only basename
    # expected: .../UK/chunk_123.txt or .../US/chunk_123.txt, but if your file list is only basenames, we'll handle fallback.
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

    q_vec = sparse.csr_matrix((vals, ([0] * len(idxs), idxs)), shape=(1, X_bm25.shape[1]))
    scores = (X_bm25 @ q_vec.T).toarray().ravel()

    top_idx = np.argsort(-scores)[:top_k]
    return [(int(i), chunk_names[int(i)], float(scores[int(i)])) for i in top_idx if scores[int(i)] > 0]


# ---------------- Dense v2 ----------------
def load_dense_store_v2():
    X = np.load(DENSE_DIR / "embeddings.npy")  # shape: (N, dim), already normalized
    names = (DENSE_DIR / "filenames.txt").read_text(encoding="utf-8").splitlines()
    names = [norm_path(n) for n in names]  # 'UK/chunk_123.txt'
    return X, names


def cosine_topk(X: np.ndarray, q: np.ndarray, k: int):
    # X is normalized; q should be normalized too -> cosine = dot
    q = q / (np.linalg.norm(q) + 1e-12)
    sims = X @ q
    top_idx = np.argsort(-sims)[:k]
    return top_idx, sims


def dense_retrieve(query: str, X_emb: np.ndarray, chunk_names: list[str], model: SentenceTransformer, top_k: int):
    q_emb = model.encode("query: " + query, convert_to_numpy=True, normalize_embeddings=True)
    top_idx, sims = cosine_topk(X_emb, q_emb, top_k)
    return [(int(i), chunk_names[int(i)], float(sims[int(i)])) for i in top_idx]


def build_dense_name_to_row(dense_names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(dense_names)}


def bm25_name_to_dense_key(bm25_name: str) -> str:
    """
    Make a best-effort to convert BM25 'name' to our dense key format: 'UK/chunk_123.txt' or 'US/chunk_123.txt'
    If BM25 stores only 'chunk_123.txt' -> we'll return that and handle ambiguity later.
    """
    s = norm_path(bm25_name)

    # try to detect UK/US inside path
    if "/UK/" in s:
        return "UK/" + s.split("/UK/")[-1]
    if "/US/" in s:
        return "US/" + s.split("/US/")[-1]

    # fallback: basename only (ambiguous)
    return Path(s).name


def hybrid_retrieve_v2(
    query: str,
    X_bm25,
    vocab: dict,
    bm25_names: list[str],
    X_emb: np.ndarray,
    dense_names: list[str],
    model: SentenceTransformer,
    top_k: int,
    bm25_candidates: int = 200,
):
    """
    BM25 candidates -> Dense rerank, BUT we map by chunk name (not by row index alignment).
    """
    bm25_res = bm25_retrieve(query, X_bm25, vocab, bm25_names, top_k=bm25_candidates)
    if not bm25_res:
        return []

    dense_map = build_dense_name_to_row(dense_names)

    cand_dense_rows = []
    cand_display_names = []
    cand_bm25_rows = []

    # map each BM25 candidate to dense row
    for bm25_row, bm25_name, bm25_score in bm25_res:
        key = bm25_name_to_dense_key(bm25_name)

        if key in dense_map:
            cand_dense_rows.append(dense_map[key])
            cand_display_names.append(key)
            cand_bm25_rows.append(bm25_row)
        else:
            # try ambiguity resolution if only basename exists: check both UK/ and US/
            base = Path(key).name
            uk_key = f"UK/{base}"
            us_key = f"US/{base}"
            if uk_key in dense_map:
                cand_dense_rows.append(dense_map[uk_key])
                cand_display_names.append(uk_key)
                cand_bm25_rows.append(bm25_row)
            elif us_key in dense_map:
                cand_dense_rows.append(dense_map[us_key])
                cand_display_names.append(us_key)
                cand_bm25_rows.append(bm25_row)

    if not cand_dense_rows:
        return []

    q_emb = model.encode("query: " + query, convert_to_numpy=True, normalize_embeddings=True)
    X_cand = X_emb[np.array(cand_dense_rows, dtype=int)]
    sims = X_cand @ q_emb  # dot == cosine, embeddings normalized

    top_local = np.argsort(-sims)[:top_k]
    out = []
    for j in top_local:
        j = int(j)
        dense_row = int(cand_dense_rows[j])
        name = cand_display_names[j]
        out.append((dense_row, name, float(sims[j])))
    return out


# ---------------- Enrichment ----------------
def chunk_path_from_relname(relname: str) -> Path:
    # relname: 'UK/chunk_123.txt' or 'US/chunk_123.txt'
    relname = norm_path(relname)
    if relname.startswith("UK/"):
        return FIXED_UK_DIR / relname.split("/", 1)[1]
    if relname.startswith("US/"):
        return FIXED_US_DIR / relname.split("/", 1)[1]
    # fallback: treat as UK first then US if exists
    p1 = FIXED_UK_DIR / relname
    if p1.exists():
        return p1
    return FIXED_US_DIR / relname


def enrich_results_v2(results, chunkpath_to_source, max_chars=600):
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


def run_query(query: str, K_values=(3, 5, 8)):
    chunkpath_to_source = load_chunkpath_to_source()

    X_bm25, vocab, bm25_names = load_bm25_store()
    X_emb, dense_names = load_dense_store_v2()
    model = SentenceTransformer(MODEL_NAME)

    print(f"Dense v2 chunks: {len(dense_names)}")

    for K in K_values:
        print("\n" + "=" * 90)
        print(f"QUERY: {query}")
        print(f"K={K}")

        bm25_res = bm25_retrieve(query, X_bm25, vocab, bm25_names, top_k=K)
        # try to convert bm25 displayed name to rel format for printing
        bm25_res_rel = [(r, bm25_name_to_dense_key(n), s) for (r, n, s) in bm25_res]
        bm25_enriched = enrich_results_v2(bm25_res_rel, chunkpath_to_source)
        print("\n--- BM25 top-K ---")
        for i, r in enumerate(bm25_enriched, 1):
            print(f"[{i}] score={r['score']:.4f} row={r['row']} source={r['source_file']} chunk_path={r['chunk_path']}")
            print(f"    {r['text']}\n")

        dense_res = dense_retrieve(query, X_emb, dense_names, model, top_k=K)
        dense_enriched = enrich_results_v2(dense_res, chunkpath_to_source)
        print("\n--- Dense (Embeddings v2) top-K ---")
        for i, r in enumerate(dense_enriched, 1):
            print(f"[{i}] sim={r['score']:.4f} row={r['row']} source={r['source_file']} chunk_path={r['chunk_path']}")
            print(f"    {r['text']}\n")

        hyb_res = hybrid_retrieve_v2(query, X_bm25, vocab, bm25_names, X_emb, dense_names, model, top_k=K, bm25_candidates=200)
        hyb_enriched = enrich_results_v2(hyb_res, chunkpath_to_source)
        print("\n--- HYBRID (BM25 candidates -> Dense rerank) top-K ---")
        for i, r in enumerate(hyb_enriched, 1):
            print(f"[{i}] sim={r['score']:.4f} row={r['row']} source={r['source_file']} chunk_path={r['chunk_path']}")
            print(f"    {r['text']}\n")


if __name__ == "__main__":
    q = "What was the main argument regarding the immigration bill that was presented?"
    run_query(q, K_values=(3, 5, 8))
