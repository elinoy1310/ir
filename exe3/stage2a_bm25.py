#exe3/stage2a_bm25.py
import json
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import  CountVectorizer
from pathlib import Path
from load_files import load_text

def bm25_matrix(counts_csr: sparse.csr_matrix, k1=1.5, b=0.75):
    """
    חישוב BM25 בצורה וקטורית מלאה לביצועים מקסימליים.
    """
    # 1. הכנות בסיסיות
    N = counts_csr.shape[0]
    dl = np.asarray(counts_csr.sum(axis=1)).ravel()
    avgdl = dl.mean() if N > 0 else 0.0
    
    # 2. חישוב IDF (וקטור באורך אוצר המילים)
    df = counts_csr.getnnz(axis=0)
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
    
    # 3. יצירת עותק לעבודה
    bm25 = counts_csr.copy().astype(np.float32)
    
    # 4. חישוב ה-Norm לכל מסמך: k1 * (1 - b + b * (dl / avgdl))
    # זהו וקטור באורך N
    doc_norms = k1 * (1 - b + b * (dl / avgdl))
    
    # 5. עדכון ערכי ה-BM25 בצורה וקטורית
    # counts_csr.indices אומר לנו איזה מונח (term) נמצא בכל מיקום במערך ה-data
    # counts_csr.indptr עוזר לנו לדעת אילו ערכים שייכים לאיזה מסמך
    
    # יצירת מערך של ה-norms שמתאים למבנה ה-data של המטריצה הדלילה
    # חוזרים על ה-norm של כל מסמך כמספר האיברים הלא-אפסים בו
    repeat_counts = np.diff(bm25.indptr)
    data_norms = np.repeat(doc_norms, repeat_counts)
    
    # חישוב הציון: idf * (tf * (k1 + 1)) / (tf + norm)
    # idf[bm25.indices] מחזיר את ה-IDF הנכון לכל מונח שמופיע במטריצה
    bm25.data = idf[bm25.indices] * (bm25.data * (k1 + 1)) / (bm25.data + data_norms)
    
    return bm25.tocsr(),idf, avgdl

def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def build_tfidf_vectors(texts, filenames, out_dir: Path, output_name:str):
    
    count_vec = CountVectorizer(
        input="content",
        analyzer="word",
        stop_words="english",
        min_df=5
    )

    X_counts = count_vec.fit_transform(texts).tocsr()
    print(f"Counts shape: {X_counts.shape}")

    X_bm25,idf, avgdl = bm25_matrix(X_counts, k1=1.5, b=0.75)
    print("TF-IDF shape:", X_bm25.shape)
    # שמירה
    out_dir.mkdir(parents=True, exist_ok=True)
    # שמירת IDF ו-avgdl כקובץ npy (פורמט מהיר של numpy)
    np.save(out_dir / f"{output_name}_idf.npy", idf)
    np.save(out_dir / f"{output_name}_avgdl.npy", np.array([avgdl]))
    sparse.save_npz(out_dir / f"{output_name}.npz", X_bm25)
    save_json(count_vec.vocabulary_, out_dir / f"{output_name}_vocabulary.json")
    save_json({"files": filenames}, out_dir / f"{output_name}_files.json")
    

    print(f"\n נשמרו קבצים בתיקייה: {out_dir.resolve()}")
    print(f"{output_name}.npz  + vocab + files")

if __name__ == "__main__":
    # uk_chanks_texts, uk_filenames = load_text(r"exe3\fixed-chunked-text\UK")
    # us_chanks_texts, us_filenames = load_text(r"exe3\fixed-chunked-text\US")
    uk_chanks_texts, uk_filenames = load_text(r"exe3\parent-child-chunked-text\UK\children")
    us_chanks_texts, us_filenames = load_text(r"exe3\parent-child-chunked-text\US\children")
    # uk_chanks_texts, uk_filenames = load_text(r"exe3\chunked-text\UK")
    # us_chanks_texts, us_filenames = load_text(r"exe3\chunked-text\US")
    print(f"Loaded {len(uk_chanks_texts)} UK chunks and {len(us_chanks_texts)} US chunks.")
    all_texts = uk_chanks_texts + us_chanks_texts
    all_filenames = uk_filenames + us_filenames
    print(f"Total chunks: {len(all_texts)}")
    out_directory = Path(r"exe3\bm25_vectors_parentSon_chunks")
    #out_directory = Path(r"exe3\bm25_vectors")
    build_tfidf_vectors(all_texts, all_filenames, out_directory, "bm25_vectors")
