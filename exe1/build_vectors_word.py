import argparse
import os
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


"""pip install scikit-learn scipy numpy
python build_vectors_word.py --input tokens --outdir vectors_word
"""
# ---------- עזר: קריאת קבצים מהתיקייה ושמירת מיפויים ----------
def read_corpus(input_dir: Path):
    files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower()==".txt"])
    texts = []
    names = []
    for p in files:
        try:
            texts.append(p.read_text(encoding="utf-8", errors="ignore"))
            names.append(p.name)
        except Exception as e:
            print(f"⚠️ דילגתי על {p.name}: {e}")
    if not texts:
        raise RuntimeError("לא נמצאו קובצי TXT בתיקייה")
    return texts, names

def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

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

    # שכיחות מסמכים למונח (df)
    df = counts_csr.getnnz(axis=0)
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)

    # נבנה מטריצה חדשה ערך-ערך (יעיל ל-CSR דרך data/indices/indptr)
    bm25 = counts_csr.tolil(copy=True)  # עבודה נוחה לטפל בכל שורה
    for i in range(N):
        row = bm25.rows[i]
        data = bm25.data[i]
        doc_len = dl[i] if dl[i] > 0 else 1.0
        norm = k1 * (1 - b + b * (doc_len / (avgdl if avgdl > 0 else doc_len)))
        for j in range(len(row)):
            term_idx = row[j]
            tf = data[j]
            # משקל BM25
            score = idf[term_idx] * (tf * (k1 + 1)) / (tf + norm)
            data[j] = score
    return bm25.tocsr()

def main():
    ap = argparse.ArgumentParser(description="בניית מטריצות TF-IDF ו-BM25 לקבצי tokens/ (Word-level).")
    ap.add_argument("--input", "-i", default="tokens", help="תיקיית קלט עם קבצי טוקנים (ברירת מחדל: tokens)")
    ap.add_argument("--outdir", "-o", default="vectors_word", help="תיקיית פלט לשמירת המטריצות (ברירת מחדל: vectors_word)")
    # פרמטרי צמצום מאפיינים
    ap.add_argument("--min_df", type=int, default=5, help="סף הופעה מינימלי במספר מסמכים (ברירת מחדל: 5)")
    args = ap.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) קריאת קורפוס ---
    texts, file_names = read_corpus(input_dir)
    print(f"נקראו {len(texts)} מסמכים מ-{input_dir}")

    # # --- 2) TF-IDF (דליל) ---
    # tfidf_vec = TfidfVectorizer(
    #     input="content",
    #     analyzer="word",
    #     stop_words="english",       # הסרת stopwords באנגלית
    #     min_df=args.min_df,         # מסמך מינימלי
    #     max_df=args.max_df,         # מסמך מקסימלי (יחסי)
    #     max_features=args.max_features,
    #     norm="l2",                  # נרמול וקטורים
    #     sublinear_tf=True           # tf לוגרי"ת - לעתים משפר
    # )
    # X_tfidf = tfidf_vec.fit_transform(texts)  # CSR
    # print(f"TF-IDF shape: {X_tfidf.shape}, צפיפות: {X_tfidf.nnz / (X_tfidf.shape[0]*X_tfidf.shape[1] + 1e-9):.6f}")

    # # שמירה
    # sparse.save_npz(out_dir / "TFIDF_Word.npz", X_tfidf)
    # save_json(tfidf_vec.vocabulary_, out_dir / "TFIDF_Word_vocabulary.json")
    # save_json({"files": file_names}, out_dir / "TFIDF_Word_files.json")

    # --- 2) BM25 (Okapi) בלבד ---
    count_vec = CountVectorizer(
        input="content",
        analyzer="word",
        stop_words="english",
        min_df=args.min_df
    )

    X_counts = count_vec.fit_transform(texts).tocsr()
    print(f"Counts shape: {X_counts.shape}")

    # חישוב BM25
    X_bm25 = bm25_matrix(X_counts, k1=1.5, b=0.75)
    source_type="Word" if args.input=="tokens" else "Lemm"

    # שמירה
    sparse.save_npz(out_dir / f"TFIDF-{source_type}.npz", X_bm25)
    save_json(count_vec.vocabulary_, out_dir / f"TFIDF-{source_type}_vocabulary.json")
    save_json({"files": file_names}, out_dir / f"TFIDF-{source_type}_files.json")

    print(f"\n נשמרו קבצים בתיקייה: {out_dir.resolve()}")
    print(f"  - BM25_{source_type}.npz  + vocab + files")

if __name__ == "__main__":
    main()
