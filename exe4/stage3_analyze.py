# exe4\stage3_analyze.py
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Set

# --- פונקציות עזר ---

def extract_chunk_path(file_str: str) -> str:
    """מחץ את הנתיב שנמצא בתוך הסוגריים עבור קובץ ה-Baseline"""
    if pd.isna(file_str): return ""
    match = re.search(r'\((.*?)\)', file_str)
    return match.group(1) if match else file_str

def get_overlap_stats(s1: Set, s2: Set, s3: Set) -> Dict:
    """חישוב חיתוכי הקבוצות עבור הפלוט"""
    return {
        "base_hard_soft": len(s1 & s2 & s3),
        "base_hard": len(s1 & s2),
        "base_soft": len(s1 & s3),
        "hard_soft": len(s2 & s3)
    }

def create_comparison_logic():
    from exe4.utils import get_queries
    queries = get_queries()
    
    # הגדרות הרצה
    k_val = 5
    embeddings = ["dense", "bm25"]
    chunking_methods = ["fixed", "parentSon"]
    

    for corpus in ("uk", "us"):
        all_overlap_data = []
        # נתיבי בסיס
        processed_csv_path = Path(f"exe4/outputs/stage1/{corpus}/for_plot/processed_sources.csv")
        hard_dir = Path(f"exe4/outputs/stage3_tables/hard_filter/{corpus}")
        soft_dir = Path(f"exe4/outputs/stage3_tables/soft_decay/{corpus}")
        out_tables_root = Path(f"exe4/outputs/stage3_tables/comparison_tables") / corpus
        out_plots = Path(f"exe4/outputs/stage3_tables/plots/{corpus}")
        
        out_plots.mkdir(parents=True, exist_ok=True)

        # טעינת Baseline
        if not processed_csv_path.exists():
            print(f"Warning: Baseline CSV not found at {processed_csv_path}")
            continue
        
        baseline_df = pd.read_csv(processed_csv_path)
        # ניקוי נתיבים ב-Baseline
        baseline_df['chunk_path_clean'] = baseline_df['file'].apply(extract_chunk_path)
        # print(baseline_df['chunk_path_clean'])

        for q_idx, query_text in enumerate(queries):
            query_folder = out_tables_root / f"query{q_idx}"
            query_folder.mkdir(parents=True, exist_ok=True)
            
            all_combos_txt = []

            for emb in embeddings:
                emb_key = "st" if emb == "dense" else "bm25"
                for chk in chunking_methods:
                    if chk == "parentSon":
                        chk_baseline="parent-son"
                    else:
                        chk_baseline=chk
                    
                    # 1. שליפת נתוני Baseline
                    b_sub = baseline_df[
                        (baseline_df['query'] == query_text) & 
                        (baseline_df['method']== emb.upper()) & 
                        (baseline_df['chunking'] == chk_baseline) &
                        (baseline_df['k'] == k_val)
                    ].sort_values('rank')


                    # 2. שליפת נתוני Hard Filter
                    hard_file = hard_dir / f"hard_filter_results_{q_idx}_{chk}_{emb_key}_k={k_val}.csv"
                    h_sub = pd.read_csv(hard_file) if hard_file.exists() else pd.DataFrame()

                    # 3. שליפת נתוני Soft Decay
                    # מחפש קובץ שמתחיל בתבנית הנכונה (מתעלם מפרמטרים של אלפא/למדא)
                    soft_files = list(soft_dir.glob(f"soft_decay_results_{q_idx}_{chk}_{emb_key}_k={k_val}_*.csv"))
                    s_sub = pd.read_csv(soft_files[0]) if soft_files else pd.DataFrame()

                    # בניית טבלה מאוחדת לפי Rank (1 עד K)
                    rows = []
                    b_paths, h_paths, s_paths = set(), set(), set()
                    
                    for r in range(1, k_val + 1):
                        # Baseline data
                        b_row = b_sub[b_sub['rank'] == r]
                       
                        b_p = b_row['chunk_path_clean'].values[0] if not b_row.empty else "no data"
                        b_s = b_row['norm_score'].values[0] if not b_row.empty else 0.0
                        if b_p != "no data": b_paths.add(b_p)

                        # Hard data (by row index)
                        h_p = h_sub.iloc[r-1]['chunk_path'] if len(h_sub) >= r else "no data"
                        h_s = h_sub.iloc[r-1]['score'] if len(h_sub) >= r else 0.0
                        if h_p != "no data": h_paths.add(h_p)

                        # Soft data (by row index)
                        s_p = s_sub.iloc[r-1]['chunk_path'] if len(s_sub) >= r else "no data"
                        s_s = s_sub.iloc[r-1]['final_score'] if len(s_sub) >= r else 0.0
                        if s_p != "no data": s_paths.add(s_p)

                        rows.append({
                            "rank": r,
                            "baseline_path": b_p, "baseline_score": b_s,
                            "hard_path": h_p, "hard_score": h_s,
                            "soft_path": s_p, "soft_score": s_s
                        })

                    # שמירת CSV לקומבינציה
                    df_combo = pd.DataFrame(rows)
                    csv_name = f"{emb}_{chk}_k={k_val}.csv"
                    df_combo.to_csv(query_folder / csv_name, index=False)

                    # הוספה ל-TXT
                    combo_header = f"--- k={k_val} --- embedding={emb} --- chunking method: {chk} ---\n"
                    table_str = df_combo.to_string(index=False)
                    all_combos_txt.append(f"=== query: {query_text} ===\n" + combo_header + table_str + "\n\n")

                    # איסוף נתונים לפלוט
                    overlaps = get_overlap_stats(b_paths, h_paths, s_paths)
                    overlaps.update({"query_idx": q_idx, "emb": emb, "chk": chk, "corpus": corpus})
                    all_overlap_data.append(overlaps)

            # שמירת TXT מרכז לשאילתה
            with open(query_folder / "all_combinations.txt", "w", encoding="utf-8") as f:
                f.writelines(all_combos_txt)

        if all_overlap_data:
            # עכשיו out_plots מכיל כבר את הנתיב הנכון (כולל ה-corpus)
            # פשוט קוראים לפונקציה פעם אחת עבור הקורפוס הנוכחי
            plot_overlaps(all_overlap_data, out_plots, corpus)

def plot_overlaps(data: List[Dict], save_path: Path, corpus_name: str):
    df_all = pd.DataFrame(data)
    if df_all.empty: return 

    # רשימת השילובים האפשריים
    embeddings = df_all['emb'].unique()
    chunking_methods = df_all['chk'].unique()

    for emb in embeddings:
        for chk in chunking_methods:
            # סינון הנתונים רק לשילוב הספציפי
            df_plot = df_all[(df_all['emb'] == emb) & (df_all['chk'] == chk)]
            if df_plot.empty: continue

            # "התכת" הנתונים לצורך הציור
            df_melted = df_plot.melt(id_vars=["query_idx"], 
                                     value_vars=["base_hard_soft", "base_hard", "base_soft", "hard_soft"],
                                     var_name="Overlap_Type", value_name="Count")
            
            plt.figure(figsize=(12, 6))
            sns.barplot(data=df_melted, x="query_idx", y="Count", hue="Overlap_Type", errorbar=None)
            
            plt.title(f"Overlap: {corpus_name.upper()} | {emb} | {chk}")
            plt.xlabel("Query Index")
            plt.ylabel("Number of Overlapping Chunks")
            plt.ylim(0, 6) # מקבע את הציר ל-5 (K) + מרווח
            plt.legend(title="Combination", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # שם קובץ מפורט שמונע דריסה
            filename = f"{corpus_name}_{emb}_{chk}_overlap.png"
            plt.savefig(save_path / filename)
            plt.close()
            print(f"Saved plot: {filename}")

if __name__ == "__main__":
    create_comparison_logic()