import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from sklearn.feature_extraction.text import  CountVectorizer
from stage_0_load import load_corpus
from pathlib import Path


# ---------- BM25 (Okapi) על בסיס CountVectorizer ----------
def bm25_matrix(counts_csr: sparse.csr_matrix, k1=1.5, b=0.75):
    """
    קלט: מטריצת ספירות (מסמך×מונח), מסוג CSR
    פלט: מטריצת BM25 באותו גודל, CSR (דלילה)
    נוסחת ה-IDF: ln((N - df + 0.5)/(df + 0.5) + 1)
    """
    N, _ = counts_csr.shape
    # אורך כל מסמך (סכום ספירות), ממוצע אורכים
    dl = np.asarray(counts_csr.sum(axis=1)).ravel()
    avgdl = dl.mean() if N > 0 else 0.0
    print(f"  ממוצע אורך מסמך: {avgdl:.2f} מילים")
    print(f"  סה׳׳כ מסמכים: {N}")

    # שכיחות מסמכים למונח (df)
    df = counts_csr.getnnz(axis=0)
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
    print(f" first 5 idf values: {idf[:5]}")

    # נבנה מטריצה חדשה ערך-ערך (יעיל ל-CSR דרך data/indices/indptr)
    bm25 = counts_csr.tolil(copy=True).astype(np.float32)  # עבודה נוחה לטפל בכל שורה
    for i in range(N):
        row = bm25.rows[i]
        data = bm25.data[i]
        doc_len = dl[i] if dl[i] > 0 else 1.0
        norm = k1 * (1 - b + b * (doc_len / (avgdl if avgdl > 0 else doc_len)))
        if i<5:
            print(f"  מסמך {i}: אורך={doc_len}, נורמליזציה={norm:.2f}")
        for j in range(len(row)):
            term_idx = row[j]
            tf = data[j]
            # משקל BM25
            score = idf[term_idx] * (tf * (k1 + 1)) / (tf + norm)
            data[j] = score
            if i<5 and j<5:
                print(f"    מונח {term_idx}: tf={tf}, BM25={score:.4f}")
    return bm25.tocsr()

def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def build_tfidf_vectors(base_dir, out_dir: Path, output_name:str):
    from sklearn.feature_extraction.text import CountVectorizer
    texts, labels, filenames = load_corpus(base_dir)
    print("Loaded documents:", len(texts))

    count_vec = CountVectorizer(
        input="content",
        analyzer="word",
        stop_words="english",
        min_df=5,
        token_pattern=r'\b[a-zA-Z0-9]+\b'
    )

    X_counts = count_vec.fit_transform(texts).tocsr()
    print(f"Counts shape: {X_counts.shape}")

    X_bm25 = bm25_matrix(X_counts, k1=1.5, b=0.75)
    print("TF-IDF shape:", X_bm25.shape)
    # שמירה
    out_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out_dir / f"{output_name}.npz", X_bm25)
    save_json(count_vec.vocabulary_, out_dir / f"{output_name}_vocabulary.json")
    save_json({"files": filenames}, out_dir / f"{output_name}_files.json")
    if isinstance(labels, np.ndarray):
        labels_list = labels.tolist()
    else:
        labels_list = labels # אם זה כבר רשימה רגילה, השתמש בה

    save_json({"labels": labels_list}, out_dir / f"{output_name}_labels.json")


    print(f"\n נשמרו קבצים בתיקייה: {out_dir.resolve()}")
    print(f"{output_name}.npz  + vocab + files")


if __name__ == "__main__":
    base_dir = r"exe2"

    # שלב 1 – טעינת המסמכים
    
    

    # שלב 2 – בניית TF-IDF
    build_tfidf_vectors(base_dir, out_dir=Path(base_dir) / "vectors_tfidf", output_name="TFIDF-Documents")

