# exe3/stage3_retrieval.py
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import json
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path

import numpy as np
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

CHANKING_METHOD="fixed" # or "parent-son" or "fixed"
ROOT = Path("exe3")
if CHANKING_METHOD=="fixed":
    BM25_DIR = ROOT / "bm25_vectors" 
    DENSE_DIR = ROOT / "st_vectors_fixed_chunks"

    FIXED_UK_DIR = ROOT / "chunked-text" / "UK"
    FIXED_US_DIR = ROOT / "chunked-text" / "US"
    FIXED_UK_INDEX = FIXED_UK_DIR / "reverse_chunk_index.json"
    FIXED_US_INDEX = FIXED_US_DIR / "reverse_chunk_index.json"
else:
    BM25_DIR = ROOT / "bm25_vectors_parentSon_chunks" 
    DENSE_DIR = ROOT / "st_vectors_parentSon_chunks"

    FIXED_UK_DIR = ROOT / "parent-child-chunked-text" / "UK" / "children"
    FIXED_US_DIR = ROOT / "parent-child-chunked-text" / "US" / "children"
    FIXED_UK_INDEX = FIXED_UK_DIR / "child_index.json"
    FIXED_US_INDEX = FIXED_US_DIR / "child_index.json"



MODEL_NAME = "intfloat/multilingual-e5-small"


# ---------------- Utils ----------------
def change_chanking_method(new_method: str):
    global CHANKING_METHOD, BM25_DIR, DENSE_DIR, FIXED_UK_DIR, FIXED_US_DIR, FIXED_UK_INDEX, FIXED_US_INDEX
    CHANKING_METHOD=new_method
    if CHANKING_METHOD=="fixed":
        BM25_DIR = ROOT / "bm25_vectors" 
        DENSE_DIR = ROOT / "st_vectors_fixed_chunks"

        FIXED_UK_DIR = ROOT / "chunked-text" / "UK"
        FIXED_US_DIR = ROOT / "chunked-text" / "US"
        FIXED_UK_INDEX = FIXED_UK_DIR / "reverse_chunk_index.json"
        FIXED_US_INDEX = FIXED_US_DIR / "reverse_chunk_index.json"
    else:
        BM25_DIR = ROOT / "bm25_vectors_parentSon_chunks" 
        DENSE_DIR = ROOT / "st_vectors_parentSon_chunks"

        FIXED_UK_DIR = ROOT / "parent-child-chunked-text" / "UK" / "children"
        FIXED_US_DIR = ROOT / "parent-child-chunked-text" / "US" / "children"
        FIXED_UK_INDEX = FIXED_UK_DIR / "child_index.json"
        FIXED_US_INDEX = FIXED_US_DIR / "child_index.json"
def tokenize_for_bm25(q: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", q.lower())


def read_text_file(p: Path, max_chars: int = 1200) -> str:
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
        if CHANKING_METHOD=="fixed":
            for  chunk_path,source_file in data.items():
             m[norm_path(chunk_path)] = norm_path(source_file)
        else:
            for  chunk_path,sources in data.items():
             m[norm_path(chunk_path)] = norm_path(sources["original_file"])

    ingest(FIXED_UK_INDEX)
    ingest(FIXED_US_INDEX)
    return m

import json
from pathlib import Path

def find_parent_chunk_path( chunk_path: str) -> str:
    # קריאה לקובץ ה-JSON
    index_path = FIXED_UK_INDEX if "UK" in chunk_path else FIXED_US_INDEX
    data = json.loads(index_path.read_text(encoding="utf-8"))

    # for k in list(data.keys()):

    normalized_data = {norm_path(key): value for key, value in data.items()}
    # בדיקה אם המפתח קיים
    if chunk_path in normalized_data:
        return normalized_data[chunk_path].get("parent_file", None)
    else:
        # אם המפתח לא קיים
        return None
    
def check_region(chunk_path):
    if "UK" in chunk_path:
        return "UK"
    elif "US" in chunk_path:
        return "US"
    else:
        return "UNKNOWN"


def chunk_path_from_row(row_i: int, chunk_name: str, uk_n: int) -> Path:
    # UK rows are [0..uk_n-1], US rows are [uk_n..]
    if row_i < uk_n:
        return FIXED_UK_DIR / chunk_name
    return FIXED_US_DIR / chunk_name


def enrich_results(
    results: List[Tuple[int, str, float]],
    chunkpath_to_source: Dict[str, str],
    max_chars: int = 900,
):
    out = []
    for row_i, chunk_name, score in results:
        p = Path(chunk_name) if CHANKING_METHOD=="fixed" else Path(find_parent_chunk_path(chunk_name))
        chunk_key = norm_path(chunk_name)
        source = chunkpath_to_source.get(chunk_key, "UNKNOWN_SOURCE")
        text = read_text_file(p,max_chars=max_chars) if p.exists() else ""
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
    names = [norm_path(f) for f in files] # only basename
    return X, vocab, names


def bm25_retrieve(query: str, X_bm25, vocab: dict, chunk_names: List[str], top_k: int):
    q_vec = transform_query_to_bm25(query, BM25_DIR, "bm25_vectors")
    scores = (X_bm25 @ q_vec.T).toarray().ravel()

    top_idx = np.argsort(-scores)[:top_k]
    return [(int(i), chunk_names[int(i)], float(scores[int(i)])) for i in top_idx if scores[int(i)] > 0]
def transform_query_to_bm25(query_text, config_dir: Path=BM25_DIR, base_name="bm25_vectors", k1=1.5, b=0.75):
    """
    ממירה טקסט שאילתה לווקטור BM25 שתואם למטריצה שנשמרה.
    """
    # 1. טעינת אוצר המילים והגדרת ה-Vectorizer
    vocab_path = config_dir / f"{base_name}_vocabulary.json"
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocabulary = json.load(f)
    
    # יוצרים Vectorizer חדש שמשתמש במילון הקיים בלבד
    count_vec = CountVectorizer(vocabulary=vocabulary, stop_words="english")
    
    # 2. טעינת הסטטיסטיקות (IDF ו-avgdl)
    idf = np.load(config_dir / f"{base_name}_idf.npy")
    avgdl = np.load(config_dir / f"{base_name}_avgdl.npy")[0]
    
    # 3. המרת השאילתה ל-Counts (Raw TF)
    # התוצאה היא מטריצה דלילה בשורה אחת (1, vocab_size)
    query_counts = count_vec.transform([query_text]).tocsr()
    
    # 4. חישוב אורך השאילתה (dl)
    q_dl = query_counts.sum()
    
    # 5. חישוב וקטור ה-BM25 לשאילתה
    q_bm25 = query_counts.copy().astype(np.float32)
    
    # נוסחת הנרמול לשאילתה
    q_norm = k1 * (1 - b + b * (q_dl / avgdl))
    
    # עדכון ערכי ה-data בתוך המטריצה הדלילה
    if q_bm25.nnz > 0:
        # idf[q_bm25.indices] מושך רק את ה-IDF של המילים שמופיעות בשאילתה
        q_bm25.data = idf[q_bm25.indices] * (q_bm25.data * (k1 + 1)) / (q_bm25.data + q_norm)
    
    return q_bm25

# ---------------- Dense ----------------
def load_dense_store():
    # NOTE: stored as sparse but loaded here to dense ndarray for speed in cosine ops
    X = np.load(DENSE_DIR / "embeddings.npy")

    names = [norm_path(x) for x in (DENSE_DIR / "filenames.txt").read_text(encoding="utf-8").splitlines()]
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



       

if __name__ == "__main__":
    q = "On what dates did the British Prime Minister deliver his speech on the defense budget?"
    run_query(q, K_values=(3,))
    # 