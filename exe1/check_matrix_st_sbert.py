import numpy as np
import pandas as pd
import json
from pathlib import Path

# ==== Paths ====
OUTDIR = Path("embeddings_out")      # change if needed
MATRIX_FILE = OUTDIR / "SimCSE-Origen.npz"   # or SBERT-Origen.npz
#MATRIX_FILE = OUTDIR / "SBERT-Origen.npz"   # or SBERT-Origen.npz
FILES_FILE = OUTDIR / "files_mapping.json"

# ==== Load matrix ====
data = np.load(MATRIX_FILE)
X = data["vectors"]    # (num_docs, dim)
print("Matrix shape:", X.shape)

# ==== Load file names ====
with open(FILES_FILE, "r", encoding="utf-8") as f:
    file_names = json.load(f)["files"]

# ==== Select first 5 rows and first 10 dimensions ====
num_rows = min(5, X.shape[0])
num_cols = min(10, X.shape[1])

preview_matrix = X[:num_rows, :num_cols]

row_names = file_names[:num_rows]
col_names = [f"dim_{i}" for i in range(num_cols)]

df_preview = pd.DataFrame(preview_matrix, index=row_names, columns=col_names)

# ==== Display ====
print("\n=== Preview: First 5 documents × First 10 dimensions ===")
print(df_preview)

# ==== Optional: Save to CSV ====
CSV_OUTPUT = OUTDIR / "preview_first5x10.csv"
df_preview.to_csv(CSV_OUTPUT, encoding="utf-8-sig")
print(f"\nSaved preview to: {CSV_OUTPUT.resolve()}")
