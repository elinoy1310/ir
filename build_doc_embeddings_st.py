import argparse, os, json, re
from pathlib import Path
import numpy as np
from tqdm import tqdm
from lxml import etree
from sentence_transformers import SentenceTransformer

# --- קריאת טקסט מה-XML (או TXT אם נתת) ---
def read_docs_from_folder(folder: Path):
    files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".xml", ".txt"}])
    texts, names = [], []
    for p in files:
        try:
            if p.suffix.lower() == ".xml":
                root = etree.parse(str(p)).getroot()
                text = " ".join(" ".join(root.itertext()).split())
            else:
                text = " ".join(p.read_text(encoding="utf-8", errors="ignore").split())
            texts.append(text)
            names.append(p.name)
        except Exception as e:
            print(f"⚠️ דילגתי על {p.name}: {e}")
    if not texts:
        raise RuntimeError(f"לא נמצאו קבצים מתאימים תחת {folder}")
    return texts, names

# --- פיצול למסות קצרות (בערך <= 256 טוקנים) כדי לא להיחתך ע"י המודל ---
def chunk_text(text, max_words=220):
    words = text.split()
    if not words:
        return [""]
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def encode_docs(texts, model_name, batch_size=32):
    model = SentenceTransformer(model_name)
    doc_vecs = []
    for t in tqdm(texts, desc=f"Encoding with {model_name}"):
        chunks = chunk_text(t, max_words=220)
        # embed כל chunk ואז ממוצע למסמך
        emb = model.encode(chunks, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        doc_vecs.append(emb.mean(axis=0) if len(emb) else np.zeros((model.get_sentence_embedding_dimension(),), dtype=np.float32))
    X = np.vstack(doc_vecs).astype(np.float32)
    return X

def main():
    ap = argparse.ArgumentParser(description="SimCSE & SBERT embeddings for original documents (XML/TXT).")
    ap.add_argument("--input", default="debates_xml", help="תיקיית קלט של המקור (XML מהאתר, או TXT אם יש)")
    ap.add_argument("--outdir", default="embeddings_st", help="תיקיית פלט")
    ap.add_argument("--simcse", default="princeton-nlp/unsup-simcse-bert-base-uncased", help="שם מודל SimCSE")
    ap.add_argument("--sbert",  default="sentence-transformers/all-MiniLM-L6-v2", help="שם מודל SBERT")
    ap.add_argument("--batch", type=int, default=32, help="Batch size לקידוד")
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    texts, names = read_docs_from_folder(Path(args.input))

    # SimCSE על קבצי המקור
    X_simcse = encode_docs(texts, args.simcse, batch_size=args.batch)
    np.save(out / "SimCSE_Origen.npy", X_simcse)
    (out / "SimCSE_Origen_files.json").write_text(json.dumps({"files": names}, ensure_ascii=False, indent=2), encoding="utf-8")

    # SBERT על קבצי המקור
    X_sbert = encode_docs(texts, args.sbert, batch_size=args.batch)
    np.save(out / "SBERT_Origen.npy", X_sbert)
    (out / "SBERT_Origen_files.json").write_text(json.dumps({"files": names}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n✅ נשמרו ייצוגים בתיקייה:", out.resolve())
    print("SimCSE_Origen.npy  shape:", X_simcse.shape)
    print("SBERT_Origen.npy   shape:", X_sbert.shape)

if __name__ == "__main__":
    main()
