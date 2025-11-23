import argparse, os, json, re
from pathlib import Path
import numpy as np

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

PUNCT = set(list('.,;:?!()[]{}"\'`“”‘’-–—…'))

def read_docs(folder: Path):
    files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower()==".txt"])
    texts, names = [], []
    for p in files:
        try:
            texts.append(p.read_text(encoding="utf-8", errors="ignore"))
            names.append(p.name)
        except Exception as e:
            print(f"⚠️ דילגתי על {p.name}: {e}")
    if not texts:
        raise RuntimeError(f"לא נמצאו קבצים ב{folder}")
    return texts, names

def tokenize_and_filter(text: str, drop_stop=False):
    # הקבצים כבר מטוקננים (מילה/פיסוק מופרדים ברווח). נשתמש ב-split פשוט.
    tokens = text.split()
    clean = []
    for tok in tokens:
        # סינון פיסוק, מרכאות, מקפים
        if tok in PUNCT: 
            continue
        # סינון טוקנים שמכילים ספרות (מספרים/תאריכים)
        if any(ch.isdigit() for ch in tok):
            continue
        # סינון טוקנים "קצרים" חסרי משמעות
        if len(tok) == 1 and tok.lower() not in {"i","a"}:
            continue
        # סינון stop-words לפי הצורך
        if drop_stop and tok.lower() in ENGLISH_STOP_WORDS:
            continue
        clean.append(tok)
    return clean

def build_sentences(texts, drop_stop):
    return [tokenize_and_filter(t, drop_stop=drop_stop) for t in texts]

def train_w2v(sentences, vector_size=300, window=5, min_count=5, workers=4, epochs=10):
    model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    return model

def docs_to_matrix(sentences, model):
    dim = model.wv.vector_size
    mat = np.zeros((len(sentences), dim), dtype=np.float32)
    for i, sent in enumerate(sentences):
        vecs = [model.wv[w] for w in sent if w in model.wv]
        if vecs:
            mat[i] = np.mean(vecs, axis=0)
        # אחרת נשאר וקטור אפסים (מסמך בלי מילים אחרי סינון)
    return mat

def save_outputs(outdir: Path, tag: str, X: np.ndarray, file_names):
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / f"W2V_{tag}.npy", X)
    (outdir / f"W2V_{tag}_files.json").write_text(json.dumps({"files": file_names}, ensure_ascii=False, indent=2), encoding="utf-8")

def process_split(input_dir: Path, outdir: Path, label: str, model=None):
    texts, file_names = read_docs(input_dir)
    # basic
    s_basic = build_sentences(texts, drop_stop=False)
    # nostop
    s_no   = build_sentences(texts, drop_stop=True)

    # אם לא סופק מודל — נאמן על איחוד המשפטים (basic+nostop) להשגת ווקב גדול
    if model is None:
        union = s_basic + s_no
        model = train_w2v(union)

    X_basic = docs_to_matrix(s_basic, model)
    X_no    = docs_to_matrix(s_no,    model)

    save_outputs(outdir, f"{label}_basic", X_basic, file_names)
    save_outputs(outdir, f"{label}_nostop", X_no,    file_names)
    return model, (X_basic.shape, X_no.shape)

def main():
    ap = argparse.ArgumentParser(description="Document embeddings via Word2Vec (avg word vectors) for Word & Lemma corpora.")
    ap.add_argument("--word_dir", default="tokens", help="תיקיית קבצי Word-level (ברירת מחדל: tokens)")
    ap.add_argument("--lemma_dir", default="lemmatized_text", help="תיקיית קבצי Lemma-level (ברירת מחדל: lemmatized_text)")
    ap.add_argument("--outdir", default="embeddings_w2v", help="תיקיית פלט (ברירת מחדל: embeddings_w2v)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    # Word-level
    model, shapes_w = process_split(Path(args.word_dir), outdir, label="Word")

    # נשמור גם את המודל לשימוש עתידי
    model.save(str(outdir / "w2v_model_word.model"))

    # Lemma-level: נשתמש באותו מודל או נאמן מחדש (נבחר לאמן מחדש—לעתים מועיל ללמה)
    model_lemm, shapes_l = process_split(Path(args.lemma_dir), outdir, label="Lemm", model=None)
    model_lemm.save(str(outdir / "w2v_model_lemm.model"))

    print("\n נשמרו הייצוגים בתיקייה:", outdir.resolve())
    print("Word  shapes (basic, nostop):", shapes_w)
    print("Lemm  shapes (basic, nostop):", shapes_l)

if __name__ == "__main__":
    main()
