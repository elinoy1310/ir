import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path("exe3")
FIXED_CHUNKS = ROOT / "fixed-chunked-text"
OUTDIR = ROOT / "bm25_vectors_v2"

OUTDIR.mkdir(parents=True, exist_ok=True)

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


def load_chunks():
    """
    מחזיר:
    - texts: רשימת טקסטים
    - names: רשימת מזהים ייחודיים (UK/chunk_x.txt או US/chunk_x.txt)
    """
    texts = []
    names = []

    for country in ["UK", "US"]:
        chunk_dir = FIXED_CHUNKS / country
        for p in sorted(chunk_dir.glob("chunk_*.txt")):
            txt = p.read_text(encoding="utf-8", errors="ignore")
            txt = re.sub(r"\s+", " ", txt).strip()
            texts.append(txt)
            names.append(f"{country}/{p.name}")  # ⭐ מזהה ייחודי

    return texts, names


def build_bm25(texts):
    """
    משתמשים ב-TF-IDF עם התאמה ל-BM25
    """
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        lowercase=True,
        norm=None,
        smooth_idf=False
    )

    X = vectorizer.fit_transform(texts)

    # התאמות BM25
    k1 = 1.5
    b = 0.75

    dl = X.sum(axis=1).A1
    avgdl = dl.mean()

    denom = X + k1 * (1 - b + b * (dl[:, None] / avgdl))
    X_bm25 = X.multiply(k1 + 1).multiply(1 / denom)

    return sparse.csr_matrix(X_bm25), vectorizer.vocabulary_


def main():
    print("Loading chunks...")
    texts, names = load_chunks()
    print(f"Total chunks: {len(texts)}")

    print("Building BM25...")
    X_bm25, vocab = build_bm25(texts)

    print("Saving outputs...")
    sparse.save_npz(OUTDIR / "bm25_vectors.npz", X_bm25)

    with open(OUTDIR / "bm25_vectors_vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)

    with open(OUTDIR / "bm25_vectors_files.json", "w", encoding="utf-8") as f:
        json.dump({"files": names}, f, indent=2)

    print("✅ Done.")
    print(f"- vectors   : {OUTDIR / 'bm25_vectors.npz'}")
    print(f"- vocabulary: {OUTDIR / 'bm25_vectors_vocabulary.json'}")
    print(f"- files     : {OUTDIR / 'bm25_vectors_files.json'}")


if __name__ == "__main__":
    main()
