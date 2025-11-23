import numpy as np
import pandas as pd
import json
from pathlib import Path

# --- נתונים לדוגמה: שנה את זה לפי הפלט שלך ---
OUTDIR = Path("embeddings_w2v")
TAG = "Lemm_nostop"
#TAG = "Lemm_basic"
#TAG = "Word_nostop"
#TAG = "Word_basic"
X_FILE = OUTDIR / f"W2V_{TAG}.npy"
FILES_FILE = OUTDIR / f"W2V_{TAG}_files.json"

# --- טעינת מטריצה ושמות קבצים ---
X = np.load(X_FILE)
with open(FILES_FILE, "r", encoding="utf-8") as f:
    files_data = json.load(f)
file_names = files_data["files"]

print(f"צורה של המטריצה: {X.shape}")

# --- בדיקת שורות ריקות (רק אפסים) ---
empty_rows_count = np.sum(np.all(X == 0, axis=1))
print(f"\nמספר השורות הריקות במטריצה (רק אפסים): {empty_rows_count}")

# --- הצגה ויזואלית של 5 השורות הראשונות ו-10 המאפיינים הראשונים ---
num_rows = min(5, X.shape[0])
num_cols = min(10, X.shape[1])
preview_matrix = X[:num_rows, :num_cols]

row_names = file_names[:num_rows]
col_names = [f"dim{i}" for i in range(num_cols)]

df_preview = pd.DataFrame(preview_matrix, index=row_names, columns=col_names)

print("\n--- Preview 5 מסמכים × 10 ממדים ראשונים ---")
print(df_preview)

# --- שמירה ל-CSV ---
CSV_OUTPUT = OUTDIR / f"W2V_{TAG}_preview.csv"
df_preview.to_csv(CSV_OUTPUT, encoding="utf-8-sig")
print(f"\n✅ נשמרה ויזואליזציה ל-CSV: {CSV_OUTPUT.resolve()}")
