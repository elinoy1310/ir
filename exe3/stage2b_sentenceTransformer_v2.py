# # exe3/stage2b_sentenceTransformer_v2.py
# import json
# import re
# from pathlib import Path

# import numpy as np
# from sentence_transformers import SentenceTransformer

# ROOT = Path("exe3")

# # input: fixed chunks
# FIXED_UK_DIR = ROOT / "fixed-chunked-text" / "UK"
# FIXED_US_DIR = ROOT / "fixed-chunked-text" / "US"

# # output: NEW folder (does not touch existing st_vectors)
# OUT_DIR = ROOT / "st_vectors_v2"

# MODEL_NAME = "embaas/sentence-transformers-multilingual-e5-base"

# # tuning knobs
# BATCH_SIZE = 64
# MAX_CHARS_PER_CHUNK = 6000  # safety cap to avoid insanely long inputs
# SAVE_DTYPE = np.float16     # smaller + faster IO


# def read_text_file(p: Path) -> str:
#     txt = p.read_text(encoding="utf-8", errors="ignore")
#     txt = re.sub(r"\s+", " ", txt).strip()
#     if len(txt) > MAX_CHARS_PER_CHUNK:
#         txt = txt[:MAX_CHARS_PER_CHUNK]
#     return txt


# def list_chunks(base_dir: Path, prefix: str) -> list[tuple[str, Path]]:
#     """
#     Returns list of (relative_name, absolute_path)
#     relative_name example: 'UK/chunk_123.txt'
#     """
#     out = []
#     for p in sorted(base_dir.glob("chunk_*.txt"), key=lambda x: x.name):
#         rel = f"{prefix}/{p.name}"
#         out.append((rel, p))
#     return out


# def main():
#     OUT_DIR.mkdir(parents=True, exist_ok=True)

#     # gather all chunks with unique relative identifiers
#     items = []
#     items += list_chunks(FIXED_UK_DIR, "UK")
#     items += list_chunks(FIXED_US_DIR, "US")

#     if not items:
#         raise SystemExit("No chunks found. Check exe3/fixed-chunked-text/(UK|US)/chunk_*.txt")

#     names = [rel for (rel, _) in items]

#     print(f"Total chunks: {len(items)}")
#     print(f"Writing to: {OUT_DIR}")

#     # load model
#     model = SentenceTransformer(MODEL_NAME)

#     # read texts
#     texts = []
#     for rel, p in items:
#         t = read_text_file(p)
#         texts.append("passage: " + t)  # IMPORTANT for E5

#     # encode with batching; normalize -> cosine becomes dot-product friendly
#     emb = model.encode(
#         texts,
#         batch_size=BATCH_SIZE,
#         show_progress_bar=True,
#         convert_to_numpy=True,
#         normalize_embeddings=True,
#     )

#     # save
#     emb = emb.astype(SAVE_DTYPE, copy=False)
#     np.save(OUT_DIR / "embeddings.npy", emb)

#     (OUT_DIR / "filenames.txt").write_text("\n".join(names), encoding="utf-8")

#     meta = {
#         "model": MODEL_NAME,
#         "count": len(names),
#         "dim": int(emb.shape[1]),
#         "dtype": str(emb.dtype),
#         "normalized": True,
#         "note": "embeddings are for E5 with 'passage:' prefix; queries should use 'query:' prefix",
#     }
#     (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

#     print("✅ Done.")
#     print(f"- embeddings: {OUT_DIR / 'embeddings.npy'}")
#     print(f"- filenames : {OUT_DIR / 'filenames.txt'}")
#     print(f"- meta      : {OUT_DIR / 'meta.json'}")


# if __name__ == "__main__":
#     main()
