# exe4/stage3_time_decay_scoring.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import numpy as np
from exe3.stage3_retrieval import (

    change_chanking_method
)
from .utils import resolve_chunk_metadata, get_queries


def compute_time_score(
    *,
    chunk_path: str,
    chunk_index_path: str,
    metadata_index_path: str,
    chunking_method: str,
    query_date: datetime,
    lambda_decay: float
) -> Tuple[float, str, datetime]:
    """
    Stage 3 (Exercise 4): time-decay score
    Returns:
        (time_score, corpus, doc_date)

    time_score = 1 / (1 + lambda * Δt)
    where Δt is measured in months (approx days/30)
    """
    corpus, timestamp = resolve_chunk_metadata(
        chunk_path=chunk_path,
        chunk_index_path=chunk_index_path,
        metadata_index_path=metadata_index_path,
        chunking_method=chunking_method,
    )
    doc_date = datetime.fromisoformat(timestamp)
    delta_t_months = max((query_date - doc_date).days / 30.0, 0.0)
    time_score = 1.0 / (1.0 + float(lambda_decay) * delta_t_months)
    return float(time_score), corpus, doc_date


def run_soft_decay_query(
    query: str,
    chunks_index_path: Path,
    chunking_method: str,
    top_k: int = 5,
    use_dense: bool = True,
    alpha: float = 0.3,
    lambda_decay: float = 0.6,
    nation: str = "both",
    metadata_index_path: Path = Path("exe4/metadata_index.json"),
    query_date: datetime | None = None,
):
    """
    Stage 3 (Exercise 4): Soft Decay retrieval
    NOTE: heavy imports are inside to avoid pandas/numpy dependency for Stage 4 usage.
    """
    # heavy imports only here:
    from sentence_transformers import SentenceTransformer
    from exe3.stage3_retrieval import (
        load_bm25_store,
        load_dense_store,
        bm25_retrieve,
        dense_retrieve,
        MODEL_NAME,
    )

    # ---------- Shared Preparation ----------
    query_date = estimate_query_date(query)

    change_chanking_method(chunking_method)
    
    if use_dense:
        X_emb, names = load_dense_store()
        model = SentenceTransformer(MODEL_NAME)
        sim_results = dense_retrieve(query, X_emb, names, model, top_k=len(names))
    else:
        X_bm25, vocab, names = load_bm25_store()
        sim_results = bm25_retrieve(query, X_bm25, vocab, names, top_k=len(names))

    final_results = []
    for row_idx, chunk_path, sim_score in sim_results:
        time_score, corpus, _ = compute_time_score(
            chunk_path=chunk_path,
            chunk_index_path=str(chunks_index_path),
            metadata_index_path=str(metadata_index_path),
            chunking_method=chunking_method,
            query_date=query_date,
            lambda_decay=lambda_decay,
        )
        final_score = (1.0 - alpha) * float(sim_score) + alpha * float(time_score)
        final_results.append((row_idx, chunk_path, corpus, float(sim_score), float(time_score), float(final_score)))

    if nation.lower() in ("uk", "us"):
        final_results = [x for x in final_results if x[2].lower() == nation.lower()]

    final_results.sort(key=lambda x: x[-1], reverse=True)
    # return without corpus (like you did before)
    return [(r, p, s, t, f) for (r, p, _c, s, t, f) in final_results[:top_k]]


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
    """
    Optional utility: saves a CSV (requires pandas)
    """
    import pandas as pd  # heavy import only here

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

    df = pd.DataFrame(results, columns=["row_index", "chunk_path", "sim_score", "time_score", "final_score"])
    df["query"] = query
    df["chunking_method"] = chunking_method
    df["embedding_method"] = "st" if use_dense else "bm25"
    df["alpha"] = alpha
    df["lambda"] = lambda_decay

    save_path.mkdir(parents=True, exist_ok=True)
    csv_filename = (
        f"soft_decay_results_{query_index}_"
        f"{chunking_method}_{'st' if use_dense else 'bm25'}_"
        f"k={top_k}_a={alpha}_l={lambda_decay}.csv"
    )
    df.to_csv(save_path / csv_filename, index=False, encoding="utf-8")
    print(f"Results saved to {save_path / csv_filename}")

from datetime import datetime, timedelta
import re

def estimate_query_date(query: str) -> datetime:
    q = query.lower()
    today = datetime.today()

    # --- explicit year ---
    m = re.search(r"(19|20)\d{2}", q)
    if m:
        year = int(m.group())
        return datetime(year, 6, 30)

    # --- quarters ---
    if "last quarter of 2023" in q:
        return datetime(2023, 11, 15)

    if "late 2024" in q:
        return datetime(2024, 11, 15)

    # --- recency ---
    if any(x in q for x in ["current", "latest", "now"]):
        return today - timedelta(days=30)

    # --- evolution ---
    if "between" in q and "and" in q:
        # fallback: midpoint
        return today - timedelta(days=180)

    # --- default fallback ---
    return today - timedelta(days=60)


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
