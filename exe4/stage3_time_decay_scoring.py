from datetime import datetime
from pathlib import Path
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

import numpy as np
from sentence_transformers import SentenceTransformer
from exe3.stage3_retrieval import (
    load_chunkpath_to_source,
    load_bm25_store,
    load_dense_store,
    bm25_retrieve,
    dense_retrieve,
    MODEL_NAME,
    change_chanking_method
)
from .utils import resolve_chunk_metadata, get_queries


def compute_time_score(
    chunk_path: str,
    chunk_index_path: str,
    metadata_index_path: str,
    chunking_method: str,
    query_date: datetime,
    lambda_decay: float
) -> float:
    try:
        corpus, timestamp = resolve_chunk_metadata(
            chunk_path=chunk_path,
            chunk_index_path=chunk_index_path,
            metadata_index_path=metadata_index_path,
            chunking_method=chunking_method
        )
        doc_date = datetime.fromisoformat(timestamp)
        delta_t = (query_date - doc_date).days / 30  # חודשים
        return 1 / (1 + lambda_decay * delta_t), corpus
    except Exception:
        return 0.0
    
def run_soft_decay_query(
    query: str,
    chunks_index_path: Path,
    chunking_method: str,
    top_k: int = 5,
    use_dense: bool = True,
    alpha: float = 0.3,
    lambda_decay: float = 0.6,
    nation: str = "both"
):
    """
    Soft Decay / Recency Weighting Retrieval
    """

    # ---------- Shared Preparation ----------
    chunkpath_to_source = load_chunkpath_to_source()
    query_date = datetime.today() #maybe change this
    change_chanking_method(chunking_method)
    
    if use_dense:
        X_emb, names = load_dense_store()
        model = SentenceTransformer(MODEL_NAME)
        # similarity על כל הקורפוס
        sim_results = dense_retrieve(query, X_emb, names, model, top_k=len(names))
    else:
        X_bm25, vocab, names = load_bm25_store()
        sim_results = bm25_retrieve(query, X_bm25, vocab, names, top_k=len(names))

    # ---------- Combine similarity + time ----------
    final_results = []

    for row_idx, chunk_path, sim_score in sim_results:
        time_score,corpus = compute_time_score(
            chunk_path=chunk_path,
            chunk_index_path=str(chunks_index_path),
            metadata_index_path="exe4/metadata_index.json",
            chunking_method=chunking_method,
            query_date=query_date,
            lambda_decay=lambda_decay
        )

        final_score = (1 - alpha) * sim_score + alpha * time_score

        final_results.append((
            row_idx,
            chunk_path,
            corpus,
            sim_score,
            time_score,
            final_score
        ))

    # ---------- Ranking ----------
    if nation.lower() in ("uk", "us"):
        final_results = [
            x for x in final_results
            if x[2].lower() == nation.lower()
        ]

    final_results.sort(key=lambda x: x[-1] , reverse=True)

    final_results_clean = [
    (row_idx, chunk_path, sim_score, time_score, final_score)
    for (row_idx, chunk_path, corpus, sim_score, time_score, final_score)
    in final_results[:top_k]
]

    return final_results_clean


def save_soft_decay_results(
    query: str,
    chunks_index_path: Path,
    chunking_method: str,
    save_path: Path,
    query_index: int = 0,
    top_k: int = 5,
    use_dense: bool = True,
    alpha: float = 0.3,
    lambda_decay: float = 0.6,
    nation="both"
):
    results = run_soft_decay_query(
        query=query,
        chunks_index_path=chunks_index_path,
        chunking_method=chunking_method,
        top_k=top_k,
        use_dense=use_dense,
        alpha=alpha,
        lambda_decay=lambda_decay,
        nation=nation
    )

    df = pd.DataFrame(
        results,
        columns=["row_index", "chunk_path", "sim_score", "time_score", "final_score"]
    )

    df["query"] = query
    df["chunking_method"] = chunking_method
    df["embedding_method"] = "st" if use_dense else "bm25"
    df["alpha"] = alpha
    df["lambda"] = lambda_decay

    csv_filename = (
        f"soft_decay_results_{query_index}_"
        f"{chunking_method}_{'st' if use_dense else 'bm25'}_"
        f"k={top_k}_a={alpha}_l={lambda_decay}.csv"
    )

    df.to_csv(Path(save_path) / Path(csv_filename), index=False, encoding="utf-8")
    print(f"Results saved to {csv_filename}")



# -------------------- Runner --------------------
if __name__ == "__main__":

    queries = get_queries()
    print(f"Running {len(queries)} queries")

    k_lst = [5]
    # embedding_methods = ["dense"]
    # chunking_methods = ["parentSon"]
    # k_lst = [3, 5, 8]
    embedding_methods = ["dense", "bm25"]
    chunking_methods = ["fixed", "parentSon"]

    for q_idx, q in enumerate(queries):
        for k in k_lst:
            for embedding_method in embedding_methods:
                for chunking_method in chunking_methods:

                    # --- embedding flag ---
                    use_dense = embedding_method == "dense"

                    # --- chunk index path לפי chunking ---
                   
                    chunks_index_path = Path(f"exe4/united_{chunking_method}_chunk_index.json" )
                   

                    try:
                        save_soft_decay_results(
                            query=q,
                            chunks_index_path=chunks_index_path,
                            chunking_method=chunking_method,
                            save_path=Path("exe4/outputs/stage3_tables/soft_decay/uk"),
                            query_index=q_idx,
                            top_k=k,
                            use_dense=use_dense,
                            nation="uk"
                        )
                        save_soft_decay_results(
                            query=q,
                            chunks_index_path=chunks_index_path,
                            chunking_method=chunking_method,
                            save_path=Path("exe4/outputs/stage3_tables/soft_decay/us"),
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
