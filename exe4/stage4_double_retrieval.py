# stage4_double_retrieval.py
# Performs EARLY / LATE retrieval using Exercise 3 retrieval pipeline

from dataclasses import dataclass
from typing import Literal
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from exe3.stage3_retrieval import (
    MODEL_NAME,
    change_chanking_method,
    enrich_results,
    load_bm25_store,
    load_dense_store,
    load_chunkpath_to_source,
    bm25_retrieve,
    dense_retrieve,
)

from exe4.stage4_windows import TimeWindow, get_doc_date

Method = Literal["bm25", "dense"]
ChunkMethod = Literal["fixed", "parentSon"]


@dataclass
class RetrievalBackend:
    # Holds retrieval resources so they are loaded once
    method: Method
    chunk_method: ChunkMethod
    chunkpath_to_source: dict
    X_bm25: object = None
    vocab: dict = None
    bm25_names: list = None
    X_emb: object = None
    dense_names: list = None
    st_model: SentenceTransformer = None


def prepare_backend(method: Method, chunk_method: ChunkMethod) -> RetrievalBackend:
    # Initialize retrieval backend according to method and chunking
    change_chanking_method(chunk_method)
    backend = RetrievalBackend(
        method=method,
        chunk_method=chunk_method,
        chunkpath_to_source=load_chunkpath_to_source()
    )

    if method == "bm25":
        backend.X_bm25, backend.vocab, backend.bm25_names = load_bm25_store()
    else:
        backend.X_emb, backend.dense_names = load_dense_store()
        backend.st_model = SentenceTransformer(MODEL_NAME)

    return backend


def retrieve_in_window(
    backend: RetrievalBackend,
    meta: dict,
    window: TimeWindow,
    query: str,
    k: int,
    oversample: int = 120,
):
    # Run retrieval and keep only results inside the given time window
    if backend.method == "bm25":
        raw = bm25_retrieve(query, backend.X_bm25, backend.vocab, backend.bm25_names, oversample)
    else:
        raw = dense_retrieve(query, backend.X_emb, backend.dense_names, backend.st_model, oversample)

    enriched = enrich_results(raw, backend.chunkpath_to_source)

    filtered = []
    for r in enriched:
        dt = get_doc_date(meta, r["source_file"])
        if dt and window.contains(dt):
            r["doc_date"] = dt.isoformat()
            filtered.append(r)

    return filtered[:k]


def double_retrieve(
    backend: RetrievalBackend,
    meta: dict,
    early: TimeWindow,
    late: TimeWindow,
    query: str,
    k: int,
):
    # Retrieve top-K contexts from EARLY and LATE windows
    early_ctx = retrieve_in_window(backend, meta, early, query, k)
    late_ctx = retrieve_in_window(backend, meta, late, query, k)
    return early_ctx, late_ctx
