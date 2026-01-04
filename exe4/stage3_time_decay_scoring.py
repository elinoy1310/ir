# exe4/stage3_time_decay_scoring.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Tuple, List

from .utils import resolve_chunk_metadata


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

    qd = query_date or datetime.today()

    # similarity over all chunks
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
            query_date=qd,
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
    metadata_index_path: Path = Path("exe4/metadata_index.json"),
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
        metadata_index_path=metadata_index_path,
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


if __name__ == "__main__":
    # quick sanity run (optional)
    q = "What was the specific budget allocated to security in 2024?"
    res = run_soft_decay_query(
        query=q,
        chunks_index_path=Path("exe4/united_fixed_chunk_index.json"),
        chunking_method="fixed",
        top_k=5,
        use_dense=True,
        alpha=0.3,
        lambda_decay=0.6,
        nation="both",
    )
    for row in res:
        print(row)
