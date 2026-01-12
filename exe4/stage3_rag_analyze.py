import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
from pathlib import Path
import numpy as np
from .utils import get_queries,get_type

# ==========================================
# 1. הגדרות וסיווג שאילתות
# ==========================================

BASE_DIR = Path("exe4/outputs/stage3_tables")
OUTPUT_DIR = Path("exe4/outputs/stage3_tables/rag_analysis_report")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEPARATOR = '=' * 120

# ==========================================
# 2. טעינת נתונים (Parsing)
# ==========================================
def parse_txt_files():
    all_answers = []
    all_sources = []

    # מעבר רקורסיבי על התיקיות: approach -> corpus
    for approach_dir in BASE_DIR.iterdir():
        if not approach_dir.is_dir(): continue
        approach_name = approach_dir.name  # hard_filter / soft_decay

        for corpus_dir in approach_dir.iterdir():
            if not corpus_dir.is_dir(): continue
            corpus_name = corpus_dir.name # uk / us

            answers_path = corpus_dir / "answers.txt"
            sources_path = corpus_dir / "sources.txt"

            # --- Parse Answers ---
            if answers_path.exists():
                with open(answers_path, encoding='utf-8') as f:
                    blocks = f.read().split(SEPARATOR)
                
                for block in blocks:
                    if not block.strip(): continue
                    
                    # חילוץ נתונים בעזרת Regex
                    q_match = re.search(r'QUERY:\s*(.*?)\n', block)
                    k_match = re.search(r'K\s*=\s*(\d+)', block)
                    
                    # === התיקון כאן: הוספנו את ST לרשימת האפשרויות ===
                    config_match = re.search(r'(DENSE|BM25|ST) with ([\w-]+) chunking', block, re.IGNORECASE)
                    
                    if q_match and k_match and config_match:
                        query_text = q_match.group(1).strip()
                        
                        # המרה: אם כתוב ST נהפוך את זה ל-DENSE לטובת האחידות בגרפים
                        raw_method = config_match.group(1).upper()
                        method = "DENSE" if raw_method == "ST" else raw_method
                        
                        chunking = config_match.group(2)
                        
                        # מציאת התשובה עצמה
                        header_end = config_match.end()
                        content = block[header_end:].strip()
                        
                        # בדיקה אם התשובה היא "לא ידוע"
                        is_idk = 1 if ("I don't know" in content or "was not mentioned" in content) else 0
                        
                        all_answers.append({
                            'approach': approach_name,
                            'corpus': corpus_name,
                            'query': query_text,
                            'type': get_type(query_text),
                            'k': int(k_match.group(1)),
                            'method': method,
                            'chunking': chunking,
                            'answer_length': len(content),
                            'is_idk': is_idk,
                            'answer_text': content
                        })

            # --- Parse Sources ---
            if sources_path.exists():
                with open(sources_path, encoding='utf-8') as f:
                    blocks = f.read().split(SEPARATOR)
                
                for block in blocks:
                    if not block.strip(): continue
                    
                    q_match = re.search(r'QUERY:\s*(.*?)\n', block)
                    k_match = re.search(r'K\s*=\s*(\d+)', block)
                    
                    # === התיקון גם כאן ===
                    config_match = re.search(r'(DENSE|BM25|ST) with ([\w-]+) chunking', block, re.IGNORECASE)
                    
                    if q_match and k_match and config_match:
                        query_text = q_match.group(1).strip()
                        
                        raw_method = config_match.group(1).upper()
                        method = "DENSE" if raw_method == "ST" else raw_method
                        
                        # חילוץ שורות המקורות
                        srcs = re.findall(r'-\s*(.*?)\s+score=([\d.-]+)', block)
                        
                        for src_path, score in srcs:
                            # חילוץ שנה מהשם קובץ (הנחה: פורמט YYYY-MM-DD)
                            date_match = re.search(r'(\d{4})-\d{2}-\d{2}', src_path)
                            year = int(date_match.group(1)) if date_match else None
                            
                            all_sources.append({
                                'approach': approach_name,
                                'corpus': corpus_name,
                                'query': query_text,
                                'type': get_type(query_text),
                                'k': int(k_match.group(1)),
                                'method': method,
                                'chunking': config_match.group(2),
                                'source_file': src_path.strip(),
                                'score': float(score),
                                'year': year
                            })

    return pd.DataFrame(all_answers), pd.DataFrame(all_sources)

# ==========================================
# 3. הפקת גרפים (Visualization)
# ==========================================
def plot_all_graphs(df_ans, df_src):
    # --- הגדרות עיצוב ---
    # צבעים: Hard Filter = אדום, Soft Decay = כחול
    custom_palette = {"hard_filter": "#E24A33", "soft_decay": "#348ABD"}
    
    # פלטת צבעים לסוגי השאילתות (עבור גרף 13)
    type_palette = {
        "recency": "green", 
        "hard_filter": "red", 
        "evolution": "purple", 
        "ambiguity": "orange",
        "Other": "gray"
    }

    sns.set_theme(style="whitegrid")
    print("Generating Analysis Plots...")

    # ==========================================
    # 0. נרמול ציונים (Normalization)
    # ==========================================
    # מנרמלים את הציון לטווח 0-1 בנפרד לכל שיטה (BM25 vs Dense)
    # כדי שנוכל להשוות ביניהן בגרף אחד
    df_src['score_norm'] = df_src.groupby('method')['score'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0
    )

    # ==========================================
    # 1. התפלגות ציונים מנורמלים (עודכן)
    # ==========================================
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_src, x='method', y='score_norm', hue='approach', palette=custom_palette)
    plt.title('Normalized Score Distribution: Hard Filter vs Soft Decay')
    plt.ylabel('Normalized Score (0-1)')
    plt.savefig(OUTPUT_DIR / 'score_norm_dist_by_approach.png')
    plt.close()

    # ==========================================
    # גרפים 2-10 (ללא שינוי מהותי, רק שימוש בפלטה)
    # ==========================================

    # --- 2. Chunking Impact ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(data=df_src, x='chunking', y='score_norm', hue='approach', ax=axes[0], palette=custom_palette)
    axes[0].set_title('Normalized Score by Chunking & Approach')
    sns.barplot(data=df_ans, x='chunking', y='answer_length', hue='approach', ax=axes[1], palette=custom_palette)
    axes[1].set_title('Answer Length by Chunking & Approach')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'chunking_impact_by_approach.png')
    plt.close()

    # --- 3. Query Type Analysis ---
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df_src, x='type', y='score_norm', hue='approach', palette=custom_palette)
    plt.title('Normalized Score by Query Type & Approach')
    plt.savefig(OUTPUT_DIR / 'score_by_query_type_approach.png')
    plt.close()

    # --- 4. Score vs K ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_src, x='k', y='score_norm', hue='approach', style='method', markers=True, dashes=False, palette=custom_palette)
    plt.title('Avg Normalized Score vs K')
    plt.savefig(OUTPUT_DIR / 'score_vs_k_approach.png')
    plt.close()

    # --- 5. Overlap ---
    # (חישוב חפיפות נשאר זהה לקוד הקודם, רק הציור משתמש בפלטה)
    df_src['unique_id'] = df_src['query'] + "_" + df_src['k'].astype(str) + "_" + df_src['source_file']
    overlap_data = []
    grouped = df_src.groupby(['query', 'k', 'approach'])
    for name, group in grouped:
        bm25_srcs = set(group[group['method'] == 'BM25']['source_file'])
        dense_srcs = set(group[group['method'] == 'DENSE']['source_file'])
        if len(bm25_srcs) > 0 and len(dense_srcs) > 0:
            jaccard = len(bm25_srcs.intersection(dense_srcs)) / len(bm25_srcs.union(dense_srcs))
            overlap_data.append({'k': name[1], 'approach': name[2], 'jaccard_index': jaccard})
            
    if overlap_data:
        df_overlap = pd.DataFrame(overlap_data)
        plt.figure(figsize=(8, 6))
        sns.barplot(data=df_overlap, x='k', y='jaccard_index', hue='approach', palette=custom_palette)
        plt.title('Jaccard Overlap (BM25 vs Dense)')
        plt.savefig(OUTPUT_DIR / 'overlap_by_approach.png')
        plt.close()

    # --- 6. "I don't know" Rate ---
    plt.figure(figsize=(10, 6))
    idk_counts = df_ans.groupby(['approach', 'k'])['is_idk'].mean().reset_index()
    sns.barplot(data=idk_counts, x='k', y='is_idk', hue='approach', palette=custom_palette)
    plt.title('"I dont know" Rate by Approach')
    plt.savefig(OUTPUT_DIR / 'idk_rate_approach.png')
    plt.close()

    # --- 7. Violin Plot ---
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_src, x='corpus', y='score_norm', hue='approach', split=True, palette=custom_palette)
    plt.title('Normalized Score Density: UK vs US')
    plt.savefig(OUTPUT_DIR / 'corpus_approach_violin.png')
    plt.close()

    # --- 8. Answer Length ---
    plt.figure(figsize=(10, 6))
    sns.pointplot(data=df_ans, x='k', y='answer_length', hue='approach', palette=custom_palette)
    plt.title('Answer Length vs K')
    plt.savefig(OUTPUT_DIR / 'answer_length_approach.png')
    plt.close()

    # --- 9. Corpus Comparison ---
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_src, x='corpus', y='score_norm', hue='approach', palette=custom_palette)
    plt.title('Avg Normalized Score by Corpus')
    plt.savefig(OUTPUT_DIR / 'corpus_comparison_approach.png')
    plt.close()

    # --- 10. Temporal Footprint (Basic) ---
    if 'year' in df_src.columns and df_src['year'].notna().any():
        plt.figure(figsize=(14, 8))
        df_years = df_src.dropna(subset=['year'])
        sns.stripplot(data=df_years, x='year', y='query', hue='approach', 
                      dodge=True, jitter=True, alpha=0.6, palette=custom_palette)
        plt.title('Temporal Footprint Distribution')
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'temporal_footprint_approach.png')
        plt.close()

    # ==========================================
    # גרפים חדשים (11, 12, 13)
    # ==========================================

    df_years = df_src.dropna(subset=['year']).copy()
    if df_years.empty:
        print("Skipping plots 11-13 (No year data found)")
        return

    # --- 11. כמות צ'אנקים שאוחזרו בכל שנה (Soft vs Hard) ---
    plt.figure(figsize=(12, 6))
    # מסננים שנים לא הגיוניות אם יש
    df_years_filtered = df_years[(df_years['year'] >= 2000) & (df_years['year'] <= 2030)]
    
    sns.countplot(data=df_years_filtered, x='year', hue='approach', palette=custom_palette)
    plt.title('Retrieved Chunks Count by Year (Soft Decay vs Hard Filter)')
    plt.xlabel('Document Year')
    plt.ylabel('Number of Chunks Retrieved')
    plt.legend(title='Approach')
    plt.savefig(OUTPUT_DIR / 'chunks_count_by_year.png')
    plt.close()

    # --- 12. הקשר בין שנת מסמך לציון (Recency Bias Check) ---
    # אנו רוצים לראות אם יש ירידה או עליה בציון כתלות בשנה
    # נשתמש ב-FaceGrid כדי להפריד בין BM25 ל-Dense כי ההתנהגות שונה
    g = sns.FacetGrid(df_years_filtered, col="method", height=6, aspect=1.2, sharey=True)
    
    # שימוש ב-lineplot עם err_style='band' מראה את המגמה הממוצעת
    g.map_dataframe(sns.lineplot, x="year", y="score_norm", hue="approach", 
                    style="approach", markers=True, dashes=False, palette=custom_palette)
    
    g.add_legend(title="Approach")
    g.set_axis_labels("Document Year", "Normalized Relevance Score")
    g.fig.suptitle('Relevance Score vs Document Year (Checking Recency Bias)', y=1.02)
    plt.savefig(OUTPUT_DIR / 'score_vs_year_trend.png')
    plt.close()

    # --- 13. Query vs Year Hit Map (2023-2025) ---
    plt.figure(figsize=(15, 12)) # גרף גבוה כדי להכיל את כל השאילתות
    
    # סינון לשנים 2023-2025 בלבד
    df_map = df_years[df_years['year'].isin([2023, 2024, 2025])].copy()
    
    if not df_map.empty:
        # אנו משתמשים ב-scatterplot/stripplot
        # Y = Query Text
        # X = Year
        # Hue = Query Type (צבע לפי סוג השאילתה)
        # Style (Marker) = Approach (איקס או עיגול לפי השיטה)
        
        # כדי שהטקסט בציר Y לא יהיה ארוך מדי, נחתוך אותו קצת אם צריך, אבל ביקשת טקסט מלא
        # אז נשאיר מלא ונקטין פונט
        
        sns.scatterplot(
            data=df_map, 
            x='year', 
            y='query', 
            hue='type',       # צבע לפי סוג השאילתה
            style='approach', # צורה לפי soft/hard
            markers={"hard_filter": "X", "soft_decay": "o"}, 
            palette=type_palette,
            s=100,            # גודל הנקודה
            alpha=0.8
        )
        
        plt.title('Retrieval Hits Map (2023-2025): Query Type Colors, Approach Markers')
        plt.xlabel('Year')
        plt.ylabel('Query')
        plt.xticks([2023, 2024, 2025]) # להכריח הצגה של שנים שלמות בלבד
        plt.grid(True, axis='x', linestyle='--')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'query_year_hit_map.png')
    else:
        print("Skipping graph 13: No data found for years 2023-2025.")
    
    # ==========================================
    # 14. Optimization Heatmaps: Split by Approach
    # ==========================================
    # יצירת 2 קבצי תמונה נפרדים: אחד ל-Hard Filter ואחד ל-Soft Decay.
    # בכל תמונה נראה את הביצועים לכל סוג שאילתה ולכל K.
    
    # 1. הכנת הנתונים (Merge בין ציונים להצלחות)
    src_grouped = df_src.groupby(['approach', 'type', 'method', 'chunking', 'k'])['score_norm'].mean().reset_index()
    src_grouped.rename(columns={'score_norm': 'avg_score'}, inplace=True)
    
    ans_grouped = df_ans.groupby(['approach', 'type', 'method', 'chunking', 'k'])['is_idk'].mean().reset_index()
    ans_grouped.rename(columns={'is_idk': 'idk_rate'}, inplace=True)
    
    merged = pd.merge(src_grouped, ans_grouped, on=['approach', 'type', 'method', 'chunking', 'k'], how='left')
    merged['idk_rate'] = merged['idk_rate'].fillna(1.0) # אם אין מידע, נניח שנכשל
    
    # חישוב הציון המשוקלל
    merged['effective_score'] = merged['avg_score'] * (1 - merged['idk_rate'])
    merged['config'] = merged['method'] + "\n" + merged['chunking']

    target_types = ["hard_filter", "recency", "evolution", "ambiguity"]

    target_approaches = ["hard_filter", "soft_decay"]

    # לולאה ראשית: יצירת תמונה לכל גישה בנפרד
    for approx in target_approaches:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # סינון הנתונים לגישה הנוכחית בלבד
        subset_approx = merged[merged['approach'] == approx]
        
        for i, q_type in enumerate(target_types):
            ax = axes[i]
            
            # סינון לפי סוג שאילתה
            data_q = subset_approx[subset_approx['type'] == q_type]
            
            if data_q.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax.set_title(f"Type: {q_type}")
                continue

            # יצירת Pivot Table: שורות=קונפיגורציה, עמודות=K
            heatmap_data = data_q.pivot_table(
                index='config', 
                columns='k', 
                values='effective_score', 
                aggfunc='mean'
            )
            
            # בחירת פלטת צבעים: אדומים ל-Hard, כחולים/ירוקים ל-Soft (לבידול ויזואלי)
            cmap = "Reds" if approx == "hard_filter" else "GnBu"
            
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=cmap, ax=ax, linewidths=.5, vmin=0, vmax=1)
            
            ax.set_title(f"Query Type: {q_type}")
            ax.set_xlabel("K Value")
            ax.set_ylabel("Config (Method + Chunking)")
        
        # כותרת ראשית לתמונה
        readable_name = approx.replace("_", " ").title()
        plt.suptitle(f'Optimization Heatmap for: {readable_name}\n(Effective Score = Score * Success Rate)', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'heatmap_{approx}.png')
        plt.close()

    # ==========================================
    # 15. LLM Source Utilization (Split by Approach & K)
    # ==========================================
    
    def count_citations(row):
        if row['is_idk']: return 0
        matches = re.findall(r'\[(\d+)\]', row['answer_text'])
        return len(set(matches))

    df_ans['citations_count'] = df_ans.apply(count_citations, axis=1)
    df_ans['config_full'] = df_ans['method'] + "\n" + df_ans['chunking']

    # יצירת גרף מפוצל (Side by Side)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    
    for i, approx in enumerate(target_approaches):
        ax = axes[i]
        subset = df_ans[df_ans['approach'] == approx]
        
        if subset.empty:
            continue
            
        # בחירת סקאלת צבעים שונה לכל גישה כדי להבדיל ביניהן ויזואלית
        # Soft Decay = גווני כחול/ירוק, Hard Filter = גווני אדום
        current_palette = "mako" if approx == "soft_decay" else "flare"

        sns.barplot(
            data=subset, 
            x='config_full', 
            y='citations_count', 
            hue='k',                # <--- השינוי החשוב: פיצול לפי K
            ax=ax,
            palette=current_palette, 
            errorbar=None
        )
        
        readable_name = approx.replace("_", " ").title()
        ax.set_title(f"Approach: {readable_name}", fontsize=14)
        ax.set_xlabel("Configuration")
        
        # סידור ה-Legend שיהיה ברור
        ax.legend(title='K Value', loc='upper left')

        if i == 0:
            ax.set_ylabel("Avg. Unique Sources Cited")
        else:
            ax.set_ylabel("") 
        
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle('Source Utilization by K: Hard Filter vs Soft Decay', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'llm_source_utilization_split_k.png')
    plt.close()

    # # ==========================================
    # # 15. LLM Source Utilization (Split by Approach)
    # # ==========================================
    
    # def count_citations(row):
    #     if row['is_idk']: return 0
    #     matches = re.findall(r'\[(\d+)\]', row['answer_text'])
    #     return len(set(matches))

    # df_ans['citations_count'] = df_ans.apply(count_citations, axis=1)
    # df_ans['config_full'] = df_ans['method'] + "\n" + df_ans['chunking']

    # # יצירת גרף מפוצל (Side by Side)
    # fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    
    # for i, approx in enumerate(target_approaches):
    #     ax = axes[i]
    #     subset = df_ans[df_ans['approach'] == approx]
        
    #     if subset.empty:
    #         continue
            
    #     sns.barplot(
    #         data=subset, 
    #         x='config_full', 
    #         y='citations_count', 
    #         ax=ax,
    #         palette="viridis" if approx == "soft_decay" else "magma",
    #         errorbar=None # מסיר את קווי השגיאה למראה נקי יותר
    #     )
        
    #     readable_name = approx.replace("_", " ").title()
    #     ax.set_title(f"Approach: {readable_name}", fontsize=14)
    #     ax.set_xlabel("Configuration")
    #     if i == 0:
    #         ax.set_ylabel("Avg. Unique Sources Cited")
    #     else:
    #         ax.set_ylabel("") # הסתרת ציר Y בגרף הימני למניעת כפילות
        
    #     ax.grid(axis='y', linestyle='--', alpha=0.5)

    # plt.suptitle('Source Utilization: Hard Filter vs Soft Decay\n(Comparison of configurations within each approach)', y=1.02, fontsize=16)
    # plt.tight_layout()
    # plt.savefig(OUTPUT_DIR / 'llm_source_utilization_split.png')
    # plt.close()


# ==========================================
# 4. Main Execution
# ==========================================

if __name__ == "__main__":
    print("Parsing data from:", BASE_DIR)
    df_answers, df_sources = parse_txt_files()
    
    if df_answers.empty or df_sources.empty:
        print("❌ No data found! Check paths and ensure answers.txt/sources.txt exist.")
    else:
        print(f"Loaded {len(df_answers)} answers and {len(df_sources)} source chunks.")
        
        # שמירת הנתונים הגולמיים המעובדים ל-CSV לנוחות
        df_answers.to_csv(OUTPUT_DIR / "processed_answers.csv", index=False)
        df_sources.to_csv(OUTPUT_DIR / "processed_sources.csv", index=False)
        
        print("Generating plots...")
        plot_all_graphs(df_answers, df_sources)
        print(f"✅ Analysis complete! Reports saved to: {OUTPUT_DIR}")