# exe4/stage3_hard_filtering.py
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

import numpy as np
from sentence_transformers import SentenceTransformer
from exe3.stage3_retrieval import (
    load_bm25_store,
    load_dense_store,
    bm25_retrieve,
    dense_retrieve,
    MODEL_NAME,
    change_chanking_method
)
from .utils import resolve_chunk_metadata, get_queries

# -------------------- Utils --------------------
def return_keys_from_json_file(json_path:str):
    import json
    from pathlib import Path

    # נתיב לקובץ JSON
    json_path = Path(json_path)

    # טען את התוכן
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    # המפתחות בלבד
    keys = list(data.keys())
    return keys

def extract_year_range_from_query(query: str) -> Tuple[int, int]:
    """
    מחזיר טווח שנים (min_year, max_year) לפי הביטויים בשאילתה.
    ברירת מחדל:
        לפני שנה → min_year=2022
        אחרי שנה → max_year=2026
    """
    default_min = 2022
    default_max = 2025

    query = query.lower()
    
    # 1. בין שתי שנים מפורשות
    match = re.search(r"from (\d{4}) to (\d{4})", query)
    if match:
        return int(match.group(1)), int(match.group(2))

    match = re.search(r"between (\d{4}) and (\d{4})", query)
    if match:
        return int(match.group(1)), int(match.group(2))

    match = re.search(r"from (\d{4})[-\s]+(\d{4})", query)
    if match:
        return int(match.group(1)), int(match.group(2))

    # 2. שנה ספציפית
    match = re.search(r"\bin (\d{4})\b", query)
    if match:
        y = int(match.group(1))
        return y, y

    match = re.search(r"in the year (\d{4})", query)
    if match:
        y = int(match.group(1))
        return y, y


    # 3. מגבלות חד-צדדיות
    match = re.search(r"since (\d{4})", query)
    if match:
        y = int(match.group(1))
        return y, default_max

    match = re.search(r"after (\d{4})", query)
    if match:
        y = int(match.group(1))
        return y, default_max

    match = re.search(r"from (\d{4})", query)
    if match:
        y = int(match.group(1))
        return y, default_max

    match = re.search(r"before (\d{4})", query)
    if match:
        y = int(match.group(1))
        return default_min, y

    match = re.search(r"until (\d{4})", query)
    if match:
        y = int(match.group(1))
        return default_min, y

    # 4. שנה בודדת לא מזוהה בביטויים אחרים
    match = re.search(r"(19|20)\d{2}", query)
    if match:
        y = int(match.group())
        return y, y
    match = re.search(r"current", query)
    if match:
        return default_max, default_max

    # אם לא נמצא כלום, החזר טווח ברירת מחדל מלא
    return default_min, default_max


def filter_chunks_by_year_and_nation(chunk_names: List[str], min_year: int, max_year: int, chunk_index_path: Path, metadata_index_path: Path, method: str,nation:str="both") -> List[str]:
    filtered = []
    nation=nation.lower()
    if not(nation=="both" or nation=="uk" or nation=="us"):
        raise ValueError("not valid nation")

    for chunk_path in chunk_names:
        try:
            corpus, timestamp = resolve_chunk_metadata(
                chunk_path=str(chunk_path),
                chunk_index_path=str(chunk_index_path),
                metadata_index_path=str(metadata_index_path),
                chunking_method=method
            )
            if nation=="both" or corpus.lower()==nation:
                year = datetime.fromisoformat(timestamp).year
                if min_year <= year <= max_year:
                    filtered.append(chunk_path)
        except KeyError:
            continue
    return filtered


# -------------------- Main Hard Filtering Retrieval --------------------
def run_hard_filter_query(
    query: str,
    chunks_index_path: Path,
    chunking_method,
    
    top_k: int = 5,
    use_dense: bool = True,
    nation:str ="both"
):
    """
    הפעלת שאילתה עם סינון קשיח לפי שנה.
    """
    # --- Shared Preparation ---
    change_chanking_method(chunking_method)
    if not use_dense:
        X_bm25, vocab, names = load_bm25_store()
    else:
        X_emb, names = load_dense_store()
        model = SentenceTransformer(MODEL_NAME)
    # --- Step B: Extract year from query ---
    min_year, max_year = extract_year_range_from_query(query)
    if min_year is None:
        print(f"No min year found in query: {query} for Hard Filtering")
        return
    if max_year is None:
        print(f"No max year found in query: {query} for Hard Filtering")
        return
    # --- Step C: Filter chunks ---
    # הסינון נעשה על כל הצ'אנקים הקיימים
    filtered_chunk_names = filter_chunks_by_year_and_nation(
        chunk_names=names,
        min_year=min_year,
        max_year=max_year,
        chunk_index_path=chunks_index_path,  # הפונקציה resolve_chunk_metadata תשתמש בקובץ המתאים בתוך chunkpath_to_source
        metadata_index_path="exe4/metadata_index.json",
        method=chunking_method,
        nation=nation
    )

    if not filtered_chunk_names:
        print(f"No chunks found for year in ranget {min_year} -> {max_year}")
        return []

    # --- Step D: Compute similarity ---
    if use_dense:
        # 1. יצירת mapping: chunk_name -> row_index במטריצה
        dense_name_to_idx = {name: i for i, name in enumerate(names)}

        # 2. רשימת אינדקסים של הצ'אנקים שנותרו אחרי הסינון
        filtered_idx = [dense_name_to_idx[n] for n in filtered_chunk_names if n in dense_name_to_idx]
        # 3. סינון מטריצת Embeddings ורשימת השמות
        X_emb_filtered = X_emb[filtered_idx, :]
        chunk_names_filtered = [names[i] for i in filtered_idx]

        # 4. קריאה ל־dense_retrieve עם מטריצה חדשה ושמות מסוננים
        results = dense_retrieve(query, X_emb_filtered, chunk_names_filtered, model, top_k)
    
    else:
        # Mapping: chunk_name -> row_index
        bm25_name_to_idx = {name: i for i, name in enumerate(names)}

        # אינדקסים של הצ'אנקים שנותרו
        filtered_idx = [bm25_name_to_idx[n] for n in filtered_chunk_names if n in bm25_name_to_idx]

        # סינון מטריצה ורשימת שמות
        X_bm25_filtered = X_bm25[filtered_idx, :]
        chunk_names_filtered = [names[i] for i in filtered_idx]

        # קריאה ל־bm25_retrieve עם מטריצה ושמות מסוננים
        results = bm25_retrieve(query, X_bm25_filtered, vocab, chunk_names_filtered, top_k)
    
    return results,min_year,max_year

def save_results( query: str,
    chunks_index_path: Path,
    chunking_method,
    save_path,
    query_index=0,
    top_k: int = 5,
    use_dense: bool = True,
    nation:str ="both"):

    results,min_year,max_year=run_hard_filter_query(query,chunks_index_path,chunking_method, top_k, use_dense, nation)
    # Print table
    SCORE_INDEX=2
    CHUNK_PATH_INDEX=1
    embedding_method="st" if use_dense else "bm25"
    print(f"\n--- Hard Filtering Top-{top_k} for year in ranget {min_year} -> {max_year} ---")
    print(f"for query: {query} with chunking method: {chunking_method} with embedding: {embedding_method}")
    for i, r in enumerate(results, 1):
        print(f"[{i}] Score: {r[SCORE_INDEX]:.4f} | Chunk: {r[CHUNK_PATH_INDEX]}")
    
        # המרה ל-DataFrame
    df = pd.DataFrame(results, columns=["row_index", "chunk_path", "score"])
    df["query"] = query
    df["chunking_method"] = chunking_method
    df["embedding_method"] = "st" if use_dense else "bm25"
    df["year_range"] = f"{min_year}-{max_year}"
    # שמירה עם שם דינמי

    csv_filename = f"hard_filter_results_{query_index}_{chunking_method}_{'st' if use_dense else 'bm25'}_k={top_k}.csv"
    df.to_csv(Path(save_path)/Path(csv_filename), index=False, encoding="utf-8")
    print(f"Results saved to {csv_filename}")


        
    


# -------------------- Example Queries --------------------
example_queries = [
    "On what dates did the British Prime Minister deliver his speech on the defense budget in 2020?",
    "How did the COVID-19 pandemic affect UK policies from 2019 to 2020?",
    "US presidential elections results in 2016",
    "Debates on Brexit in the year 2018",
    "Economic stimulus plans passed since 2019",
    "Climate change discussions in US congress before 2021",
    "Parliamentary sessions about healthcare in UK 2017",
    "Defense budget decisions made until 2021",
    "Education reforms in US during 2015",
    "UK unemployment statistics reported from 2018-2019"
]


# -------------------- Runner --------------------
if __name__ == "__main__":

    queries = get_queries()
    print(f"Running {len(queries)} queries")

    k_lst = [5]
    # k_lst = [3, 5, 8]
    #embedding_methods = ["dense"]
    embedding_methods = ["dense", "bm25"]
    #chunking_methods = [ "parentSon"]
    chunking_methods = ["fixed", "parentSon"]

    for q_idx, q in enumerate(queries):
        for k in k_lst:
            for embedding_method in embedding_methods:
                for chunking_method in chunking_methods:
                    print(f"QUERY NUMBER: {q_idx} | k: {k}, embedding: {embedding_method} | chunking: {chunking_method}")

                    # --- embedding flag ---
                    use_dense = embedding_method == "dense"

                    # --- chunk index path לפי chunking ---
                   
                    chunks_index_path = Path(f"exe4/united_{chunking_method}_chunk_index.json" )
                   

                    try:
                        print("UK:")
                        save_results(
                            query=q,
                            chunks_index_path=chunks_index_path,
                            chunking_method=chunking_method,
                            save_path=Path("exe4/outputs/stage3_tables/hard_filter/uk"),
                            query_index=q_idx,
                            top_k=k,
                            use_dense=use_dense,
                            nation="uk"
                        )
                        print("US:")
                        save_results(
                            query=q,
                            chunks_index_path=chunks_index_path,
                            chunking_method=chunking_method,
                            save_path=Path("exe4/outputs/stage3_tables/hard_filter/us"),
                            query_index=q_idx,
                            top_k=k,
                            use_dense=use_dense,
                            nation="us"
                        )

                    except ValueError as e:
                        print(
                            f"Query skipped | "
                            f"q_idx={q_idx}, "
                            f"k={k}, "
                            f"embedding={embedding_method}, "
                            f"chunking={chunking_method} | "
                            f"Reason: {e}"
                        )

