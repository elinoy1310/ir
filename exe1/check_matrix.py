# import json
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from scipy import sparse

# # תיקיית הפלט שבה נשמרו המטריצות
# OUTDIR = Path(r"C:\Users\elino\Desktop\לימודים\שנה ד\סמסטר א\איחזור מידע\ex\ir\exe3\bm25_vectors")  # שנה לפי הצורך
# BM25_FILE = OUTDIR / "bm25_vectors.npz"
# VOCAB_FILE = OUTDIR / "bm25_vectors_vocabulary.json"
# FILES_FILE = OUTDIR / "bm25_vectors_files.json"
# LABELS_FILE =   Path("exe2/vectors_tfidf") / "TFIDF-Documents_labels.json"

# # --- טעינת המטריצה ---
# X_bm25 = sparse.load_npz(BM25_FILE)
# print(f"צורה של המטריצה: {X_bm25.shape}")

# # --- בדיקת שורות ריקות (רק אפסים) ---
# empty_rows_count = (X_bm25.getnnz(axis=1) == 0).sum()
# print(f"\nמספר השורות הריקות במטריצה (רק אפסים): {empty_rows_count}")


# # --- טעינת שמות הקבצים והמילים ---
# with open(FILES_FILE, "r", encoding="utf-8") as f:
#     files_data = json.load(f)
# file_names = files_data["files"]

# with open(LABELS_FILE, "r", encoding="utf-8") as f:
#     labels_data = json.load(f)
# labels_names = labels_data["labels"]

# with open(VOCAB_FILE, "r", encoding="utf-8") as f:
#     vocab = json.load(f)
# # ניפוך המילון: אינדקס → מילה
# inv_vocab = {v: k for k, v in vocab.items()}

# # --- חישוב סכום עמודות --- 
# col_sums = np.array(X_bm25.sum(axis=0)).ravel()

# # --- בחירת 10 המאפיינים עם סכום העמודה הכי גבוה --- 
# top10_cols = np.argsort(-col_sums)[:10]

# # המרת המטריצה הדלילה למטריצה דחוסה (dense)
# dense_matrix = X_bm25[:, top10_cols].toarray()

# # בודק אם יש ערך שונה מ-0 בעמודות שנבחרו
# non_zero_condition = (dense_matrix != 0).any(axis=1)

# # --- בחר את השורות שמתאימות לתנאי --- 
# selected_rows = np.where(non_zero_condition)[0][:5]  # בחר 5 השורות הראשונות שמתאימות

# # --- סכומים של העמודות שנבחרו --- 
# top10_sums = col_sums[top10_cols]
# labels_selected = [labels_names[i] for i in selected_rows]

# # --- חילוץ תת-מטריצה --- 
# # נבחר את השורות והעמודות המתאימות במטריצה
# preview_matrix = X_bm25[selected_rows[:, None], top10_cols].toarray()

# # הוספת עמודת תוויות לתוך המטריצה
# labels_column = np.array(labels_selected).reshape(-1, 1)

# # הוספת העמודה למטריצה
# preview_matrix_with_labels = np.hstack([preview_matrix, labels_column])

# # שמות המאפיינים ושמות הקבצים
# feature_names = [inv_vocab[idx] for idx in top10_cols] + ["label (0=UK, 1=US)"]
# row_names = [file_names[i] for i in selected_rows]

# # --- בניית DataFrame עם שורה של סכומים --- 
# df_preview = pd.DataFrame(preview_matrix_with_labels, index=row_names, columns=feature_names)

# # הוספת שורה עם סכומי העמודות
# df_preview.loc["Σ (sum)"] = np.hstack([top10_sums, [""]])  # הוספת סכום עמודות בלי תווית בסוף

# # הצגת ה-DataFrame
# print("\n--- Preview BM25: 5 מסמכים × 10 מאפיינים הכי חשובים + סכומים ---")
# print(df_preview)

# # שמירה ל-CSV
# #df_preview.to_csv(CSV_OUTPUT, encoding="utf-8-sig")
# #print(f"\n✅ נשמרה ויזואליזציה ל-CSV עם סכומים: {CSV_OUTPUT.resolve()}")

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse

def load_bm25_data(OUTDIR,base_foldedr, LABELS_FILE=None):
    # תיקיית הפלט שבה נשמרו המטריצות
    BM25_FILE = OUTDIR / f"{base_foldedr}.npz"
    VOCAB_FILE = OUTDIR / f"{base_foldedr}_vocabulary.json"
    FILES_FILE = OUTDIR / f"{base_foldedr}_files.json"

    # --- טעינת המטריצה ---
    X_bm25 = sparse.load_npz(BM25_FILE)
    print(f"צורה של המטריצה: {X_bm25.shape}")

    # --- בדיקת שורות ריקות (רק אפסים) ---
    empty_rows_count = (X_bm25.getnnz(axis=1) == 0).sum()
    print(f"\nמספר השורות הריקות במטריצה (רק אפסים): {empty_rows_count}")

    # --- טעינת שמות הקבצים והמילים ---
    with open(FILES_FILE, "r", encoding="utf-8") as f:
        files_data = json.load(f)
    file_names = files_data["files"]

    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    # ניפוך המילון: אינדקס → מילה
    inv_vocab = {v: k for k, v in vocab.items()}

    # --- חישוב סכום עמודות ---
    col_sums = np.array(X_bm25.sum(axis=0)).ravel()

    # --- בחירת 10 המאפיינים עם סכום העמודה הכי גבוה ---
    top10_cols = np.argsort(-col_sums)[:10]

    # המרת המטריצה הדלילה למטריצה דחוסה (dense)
    dense_matrix = X_bm25[:, top10_cols].toarray()

    # בודק אם יש ערך שונה מ-0 בעמודות שנבחרו
    non_zero_condition = (dense_matrix != 0).any(axis=1)

    # --- בחר את השורות שמתאימות לתנאי ---
    selected_rows = np.where(non_zero_condition)[0][:5]  # בחר 5 השורות הראשונות שמתאימות

    # --- סכומים של העמודות שנבחרו ---
    top10_sums = col_sums[top10_cols]

    # --- חילוץ תת-מטריצה ---
    # נבחר את השורות והעמודות המתאימות במטריצה
    preview_matrix = X_bm25[selected_rows[:, None], top10_cols].toarray()

    # אם יש תוויות, נטען גם את התוויות
    if LABELS_FILE:
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            labels_data = json.load(f)
        labels_names = labels_data["labels"]
        labels_selected = [labels_names[i] for i in selected_rows]
        # הוספת עמודת תוויות לתוך המטריצה
        labels_column = np.array(labels_selected).reshape(-1, 1)
        preview_matrix_with_labels = np.hstack([preview_matrix, labels_column])
    else:
        preview_matrix_with_labels = preview_matrix

    # שמות המאפיינים ושמות הקבצים
    feature_names = [inv_vocab[idx] for idx in top10_cols] + (["label (0=UK, 1=US)"] if LABELS_FILE else [])
    row_names = [file_names[i] for i in selected_rows]

    # --- בניית DataFrame עם שורה של סכומים ---
    df_preview = pd.DataFrame(preview_matrix_with_labels, index=row_names, columns=feature_names)

    # --- הוספת שורה עם סכומי העמודות ---
    # בונים את מערך הסכומים: מוסיפים מחרוזת ריקה רק אם קיימת עמודת Label
    padding = [""] if LABELS_FILE else []
    sum_row = np.hstack([top10_sums, padding])

    # הוספת השורה ל-DataFrame
    df_preview.loc["Σ (sum)"] = sum_row
    # הוספת שורה עם סכומי העמודות
    #df_preview.loc["Σ (sum)"] = np.hstack([top10_sums, [""]])  # הוספת סכום עמודות בלי תווית בסוף

    # הצגת ה-DataFrame
    print("\n--- Preview BM25: 5 מסמכים × 10 מאפיינים הכי חשובים + סכומים ---")
    print(df_preview)

    # שמירה ל-CSV (אם רוצים לשמור)
    # df_preview.to_csv(CSV_OUTPUT, encoding="utf-8-sig")
    # print(f"\n✅ נשמרה ויזואליזציה ל-CSV עם סכומים: {CSV_OUTPUT.resolve()}")

# קריאה לפונקציה עם פרמטרים
OUTDIR = Path(r"C:\Users\elino\Desktop\לימודים\שנה ד\סמסטר א\איחזור מידע\ex\ir\exe3\bm25_vectors")
LABELS_FILE = Path("exe2/vectors_tfidf") / "TFIDF-Documents_labels.json"  # אפשר לשים None אם לא רוצים תוויות
load_bm25_data(OUTDIR,"bm25_vectors")

