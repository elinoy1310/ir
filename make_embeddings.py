
"""
make_embeddings_fast.py

גרסה משופרת:
- חיתוך אחד לכל מסמך
- חישוב embeddings ב-batch גדול לכל המסמכים
- ממוצע chunks לכל document
- SimCSE + SBERT
"""

import os
import json
import argparse
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# ----- ברירת מחדל מודלים -----
SIMCSE_MODEL = "princeton-nlp/unsup-simcse-bert-base-uncased"
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ----- פונקציה לחיתוך הטקסט -----
def chunk_text_by_tokenizer(tokenizer, text, max_tokens=512, words_per_chunk=200):
    if not text:
        return []

    words = text.split()
    chunks = []

    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i:i+words_per_chunk]
        chunk_text = " ".join(chunk_words)
        enc = tokenizer.encode(chunk_text, add_special_tokens=False)
        ids = enc if isinstance(enc, list) else list(enc)

        start = 0
        while start < len(ids):
            end = start + max_tokens
            slice_ids = ids[start:end]
            chunk_decoded = tokenizer.decode(slice_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            chunks.append(chunk_decoded)
            start = end

    return chunks

# ----- Main -----
def main(input_folder, out_dir, simcse_model_name=SIMCSE_MODEL, sbert_model_name=SBERT_MODEL, max_tokens=512, batch_size=64):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading models...")
    simcse = SentenceTransformer(simcse_model_name)
    sbert = SentenceTransformer(sbert_model_name)
    tokenizer = AutoTokenizer.from_pretrained(sbert_model_name, use_fast=True)

    # קריאת כל המסמכים
    files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(".txt")])
    texts = []
    file_names = []
    errors = []

    for fname in tqdm(files, desc="Reading files"):
        path = os.path.join(input_folder, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                errors.append({"file": fname, "error": "empty_file"})
                continue
            texts.append(text)
            file_names.append(fname)
        except Exception as e:
            errors.append({"file": fname, "error": str(e)})

    if not texts:
        print("No valid documents found.")
        return

    # ----- חיתוך למסמכים (chunks) -----
    all_chunks = []
    doc_chunk_map = []  # מספר chunks לכל document
    for idx, text in enumerate(tqdm(texts, desc="Chunking documents")):
        chunks = chunk_text_by_tokenizer(tokenizer, text, max_tokens=max_tokens)
        if len(chunks) == 0:
            errors.append({"file": file_names[idx], "error": "chunking_failed"})
            continue
        all_chunks.extend(chunks)
        doc_chunk_map.append((idx, len(chunks)))

    # ----- חישוב Embeddings ל-SimCSE -----
    print("Encoding all chunks with SimCSE...")
    simcse_embs = simcse.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True, batch_size=batch_size)
    print("Computing document embeddings by averaging chunks...")
    # ממוצע chunks לכל document
    simcse_vectors = []
    start = 0
    for idx, num_chunks in doc_chunk_map:
        end = start + num_chunks
        doc_emb = np.mean(simcse_embs[start:end], axis=0)
        simcse_vectors.append(doc_emb)
        start = end
    simcse_matrix = np.vstack(simcse_vectors)

    # ----- חישוב Embeddings ל-SBERT -----
    print("Encoding all chunks with SBERT...")
    sbert_embs = sbert.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True, batch_size=batch_size)

    sbert_vectors = []
    start = 0
    for idx, num_chunks in doc_chunk_map:
        end = start + num_chunks
        doc_emb = np.mean(sbert_embs[start:end], axis=0)
        sbert_vectors.append(doc_emb)
        start = end
    sbert_matrix = np.vstack(sbert_vectors)

    # ----- שמירה -----
    simcse_out = os.path.join(out_dir, "SimCSE-Origen.npz")
    sbert_out = os.path.join(out_dir, "SBERT-Origen.npz")
    mapping_out = os.path.join(out_dir, "files_mapping.json")
    error_log = os.path.join(out_dir, "errors.json")

    np.savez_compressed(simcse_out, vectors=simcse_matrix)
    np.savez_compressed(sbert_out, vectors=sbert_matrix)

    with open(mapping_out, "w", encoding="utf-8") as f:
        json.dump({"files": file_names}, f, ensure_ascii=False, indent=2)

    with open(error_log, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    print(f"Done. Processed {len(file_names)} documents. Errors: {len(errors)}")

# ----- Entry Point -----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create SimCSE and SBERT embeddings efficiently.")
    parser.add_argument("--input", required=True, help="Input folder with clean TXT files")
    parser.add_argument("--out_dir", default="embeddings_out", help="Output folder")
    parser.add_argument("--sbert_model", default=SBERT_MODEL, help="SBERT model")
    parser.add_argument("--simcse_model", default=SIMCSE_MODEL, help="SimCSE model")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens per chunk")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for encoding")
    args = parser.parse_args()

    main(input_folder=args.input, out_dir=args.out_dir,
         simcse_model_name=args.simcse_model,
         sbert_model_name=args.sbert_model,
         max_tokens=args.max_tokens,
         batch_size=args.batch_size)
