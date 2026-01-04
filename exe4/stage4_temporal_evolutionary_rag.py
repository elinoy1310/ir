# exe4/stage4_temporal_evolution_rag.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

# -----------------------------
# Imports from EXE3 (baseline RAG)
# -----------------------------
from exe3.stage3_retrieval import (
    load_bm25_store,
    load_dense_store,
    transform_query_to_bm25,
    uk_count,
    MODEL_NAME,
    change_chanking_method,
)
from exe3.stage4_llm_rag import call_ollama  # same LLM wrapper you already use
from sentence_transformers import SentenceTransformer

# -----------------------------
# Imports from EXE4 (your temporal metadata)
# -----------------------------
from exe4.utils import resolve_chunk_metadata


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class ChunkEvidence:
    row: int
    chunk: str               # chunk name/path as stored in exe3 vectors
    score: float             # bm25 score or cosine sim
    timestamp_iso: str       # ISO date string
    corpus: str              # "UK"/"US"
    text_path: str           # file path that we will read for prompt
    text: str                # loaded text snippet


@dataclass
class TemporalWindows:
    early: Tuple[datetime, datetime]
    late: Tuple[datetime, datetime]


# -----------------------------
# Helpers
# -----------------------------
def _read_text(path: str, max_chars: int = 2000) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    txt = p.read_text(encoding="utf-8", errors="ignore")
    txt = " ".join(txt.split())
    return txt[:max_chars]


def _month_delta(months: int) -> timedelta:
    # simplified: 1 month ~= 30 days (matches course-style simplification)
    return timedelta(days=30 * int(months))


def _compute_windows(all_dates: List[datetime], window_months: int = 8) -> TemporalWindows:
    if not all_dates:
        raise ValueError("No dates found to compute temporal windows.")
    min_d = min(all_dates)
    max_d = max(all_dates)

    d = _month_delta(window_months)
    early = (min_d, min(min_d + d, max_d))
    late = (max(max_d - d, min_d), max_d)
    return TemporalWindows(early=early, late=late)


def _build_time_cache(
    *,
    names: List[str],
    chunking_method: str,
    # these are the same files you already have in exe4
    chunks_index_path: Path,
    metadata_index_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """
    cache[name] = {
      "timestamp_iso": "...",
      "timestamp_dt": datetime,
      "corpus": "UK"/"US",
      "source_file": "... (optional)",
    }
    """
    cache: Dict[str, Dict[str, Any]] = {}
    for chunk_path in names:
        corpus, ts = resolve_chunk_metadata(
            chunk_path=chunk_path,
            chunk_index_path=str(chunks_index_path),
            metadata_index_path=str(metadata_index_path),
            chunking_method=chunking_method,
        )
        ts_dt = datetime.fromisoformat(ts)
        cache[chunk_path] = {
            "timestamp_iso": ts,
            "timestamp_dt": ts_dt,
            "corpus": corpus.upper(),
        }
    return cache


def _dense_scores_all(X: np.ndarray, q: np.ndarray) -> np.ndarray:
    # cosine similarity for all rows
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    qn = q / (np.linalg.norm(q) + 1e-12)
    return (Xn @ qn).astype("float32")


def _bm25_scores_all(query: str, X_bm25, names: List[str]) -> np.ndarray:
    q_vec = transform_query_to_bm25(query)
    scores = (X_bm25 @ q_vec.T).toarray().ravel().astype("float32")
    return scores


def _pick_topk_in_window(
    *,
    names: List[str],
    scores: np.ndarray,
    time_cache: Dict[str, Dict[str, Any]],
    window: Tuple[datetime, datetime],
    nation: str,     # "uk"/"us"
    k: int,
) -> List[Tuple[int, str, float]]:
    start, end = window
    nation_u = nation.upper()

    # filter indices that belong to window AND nation
    valid_idx = []
    for i, name in enumerate(names):
        meta = time_cache.get(name)
        if not meta:
            continue
        if meta["corpus"] != nation_u:
            continue
        dt = meta["timestamp_dt"]
        if start <= dt <= end:
            valid_idx.append(i)

    if not valid_idx:
        return []

    # take topk among filtered indices
    sub_scores = scores[valid_idx]
    order = np.argsort(-sub_scores)[:k]
    picked = [(int(valid_idx[j]), names[int(valid_idx[j])], float(sub_scores[j])) for j in order]
    return picked


def _chunk_text_path_for_prompt(chunking_method: str, chunk_name: str) -> str:
    """
    In exe3:
    - fixed: chunk_name is already a path to chunk file -> use it directly
    - parent-son: chunk_name is a child chunk path; BUT in exe3 enrich they read parent_file for context.
      For stage4 we want maximum context too, so we will do the same: read parent_file from child_index.json.

    We reuse your already-generated united_parentSon_chunk_index.json in exe4 for the mapping.
    """
    if chunking_method == "fixed":
        return chunk_name

    # parent-son: use exe4/united_parentSon_chunk_index.json mapping
    idx_path = Path("exe4/united_parentSon_chunk_index.json")
    data = json.loads(idx_path.read_text(encoding="utf-8"))
    # normalize possible slashes
    norm = chunk_name.replace("\\", "/")
    # The json might keep keys with backslashes—try both
    if norm in data:
        return data[norm].get("parent_file", chunk_name)
    if chunk_name in data:
        return data[chunk_name].get("parent_file", chunk_name)

    # fallback
    return chunk_name


def _format_evidence(chunks: List[ChunkEvidence], *, title: str, chronological: bool) -> str:
    if chronological:
        ordered = sorted(chunks, key=lambda c: c.timestamp_iso)  # old -> new
        subtitle = "oldest ➜ newest"
    else:
        ordered = sorted(chunks, key=lambda c: c.timestamp_iso, reverse=True)  # new -> old
        subtitle = "newest ➜ oldest"

    lines = [f"## {title} ({subtitle})"]
    for i, c in enumerate(ordered, 1):
        lines.append(
            f"[{i}] ({c.corpus} | {c.timestamp_iso} | score={c.score:.4f})\n{c.text}\n"
        )
    return "\n".join(lines)


def build_stage4_prompt(
    *,
    query: str,
    nation: str,
    windows: TemporalWindows,
    early_chunks: List[ChunkEvidence],
    late_chunks: List[ChunkEvidence],
) -> str:
    # רק כרונולוגי פעם אחת (מוריד בלבול + מקצר)
    early_ordered = sorted(early_chunks, key=lambda c: c.timestamp_iso)  # old -> new
    late_ordered  = sorted(late_chunks,  key=lambda c: c.timestamp_iso)  # old -> new

    def fmt(chunks, title):
        lines = [f"## {title}"]
        for i, c in enumerate(chunks, 1):
            lines.append(f"[{i}] ({c.corpus} | {c.timestamp_iso} | score={c.score:.4f})\n{c.text}\n")
        return "\n".join(lines)

    early_min = early_ordered[0].timestamp_iso if early_ordered else "N/A"
    early_max = early_ordered[-1].timestamp_iso if early_ordered else "N/A"
    late_min  = late_ordered[0].timestamp_iso if late_ordered else "N/A"
    late_max  = late_ordered[-1].timestamp_iso if late_ordered else "N/A"

    return f"""
You are an analytical assistant specializing in Temporal RAG (time-aware comparison).

QUESTION:
"{query}"

CORPUS REGION: {nation.upper()}

TIME WINDOWS (do not mix them):
EARLY WINDOW: {windows.early[0].date().isoformat()} to {windows.early[1].date().isoformat()}
LATE  WINDOW: {windows.late[0].date().isoformat()} to {windows.late[1].date().isoformat()}

Quick sanity check of evidence dates:
- EARLY evidence spans: {early_min} .. {early_max}
- LATE  evidence spans: {late_min} .. {late_max}

EVIDENCE (EARLY first, then LATE):
{fmt(early_ordered, "EARLY evidence (use ONLY for early section)")}

{fmt(late_ordered, "LATE evidence (use ONLY for late section)")}

TASK (must follow strictly):
1) EARLY: Describe the main position/rhetoric in the EARLY window.
   - You MUST cite at least TWO different EARLY chunks by their [number] and date.
2) LATE: Describe the main position/rhetoric in the LATE window.
   - You MUST cite at least TWO different LATE chunks by their [number] and date.
3) CHANGE: Explain how it changed over time (or why it stayed stable).
   - You MUST reference at least one EARLY citation and one LATE citation.
4) Provide 2-4 bullet "change indicators" (themes/phrases/emphases).
5) If you cannot cite enough evidence for any part, write:
   "Insufficient evidence to determine change."

RULES:
- NEVER refer to a 2025 date in the EARLY section.
- NEVER refer to a 2023/2024 date in the LATE section.
- Do not invent facts.
""".strip()


# -----------------------------
# Main API
# -----------------------------
def run_stage4_temporal_evolution(
    *,
    query: str,
    nation: str,                 # "uk" or "us"
    chunking_method: str,        # "fixed" or "parent-son"
    vector_method: str,          # "bm25" or "dense"
    k: int = 5,
    window_months: int = 8,
    metadata_index_path: Path = Path("exe4/metadata_index.json"),
) -> Dict[str, Any]:
    """
    End-to-end Stage 4:
    - sets chunking method in exe3
    - loads vectors from exe3
    - uses exe4 metadata to build temporal windows and pick K evidence from early+late
    - builds explicit comparison prompt
    - calls ollama via your existing wrapper
    """

    nation = nation.lower().strip()
    if nation not in ("uk", "us"):
        raise ValueError("nation must be 'uk' or 'us'")

    if chunking_method not in ("fixed", "parent-son"):
        raise ValueError("chunking_method must be 'fixed' or 'parent-son'")

    if vector_method not in ("bm25", "dense"):
        raise ValueError("vector_method must be 'bm25' or 'dense'")

    # 1) align exe3 retrieval dirs
    change_chanking_method(chunking_method)

    # 2) load stores from exe3
    X_bm25, vocab, bm25_names = load_bm25_store()
    X_dense, dense_names = load_dense_store()

    # IMPORTANT: names order must match the scores array
    if vector_method == "bm25":
        names = bm25_names
    else:
        names = dense_names

    # 3) choose the right chunk_index in exe4 (united_* you already built)
    if chunking_method == "fixed":
        chunks_index_path = Path("exe4/united_fixed_chunk_index.json")
    else:
        chunks_index_path = Path("exe4/united_parentSon_chunk_index.json")

    # 4) build time cache + windows
    time_cache = _build_time_cache(
        names=names,
        chunking_method=chunking_method,
        chunks_index_path=chunks_index_path,
        metadata_index_path=metadata_index_path,
    )
    all_dates = [v["timestamp_dt"] for v in time_cache.values()]
    windows = _compute_windows(all_dates, window_months=window_months)

    # 5) score all docs for this query (semantic relevance)
    if vector_method == "bm25":
        scores = _bm25_scores_all(query, X_bm25, names)
    else:
        model = SentenceTransformer(MODEL_NAME)
        q_emb = model.encode("query: " + query)
        scores = _dense_scores_all(X_dense, q_emb)

    # 6) pick top-k in EARLY and LATE windows
    early_picks = _pick_topk_in_window(
        names=names,
        scores=scores,
        time_cache=time_cache,
        window=windows.early,
        nation=nation,
        k=k,
    )
    late_picks = _pick_topk_in_window(
        names=names,
        scores=scores,
        time_cache=time_cache,
        window=windows.late,
        nation=nation,
        k=k,
    )

    def materialize(picks: List[Tuple[int, str, float]]) -> List[ChunkEvidence]:
        out: List[ChunkEvidence] = []
        for row, chunk_name, sc in picks:
            meta = time_cache[chunk_name]
            text_path = _chunk_text_path_for_prompt(chunking_method, chunk_name)
            text = _read_text(text_path, max_chars=2200)
            out.append(
                ChunkEvidence(
                    row=int(row),
                    chunk=chunk_name,
                    score=float(sc),
                    timestamp_iso=meta["timestamp_iso"],
                    corpus=meta["corpus"],
                    text_path=text_path,
                    text=text,
                )
            )
        return out

    early_chunks = materialize(early_picks)
    late_chunks = materialize(late_picks)

    # 7) build prompt + call LLM (your existing ollama wrapper)
    prompt = build_stage4_prompt(
        query=query,
        nation=nation,
        windows=windows,
        early_chunks=early_chunks,
        late_chunks=late_chunks,
    )
    answer = call_ollama(prompt)

    return {
        "query": query,
        "nation": nation,
        "chunking_method": chunking_method,
        "vector_method": vector_method,
        "k": k,
        "window_months": window_months,
        "early_window": [windows.early[0].isoformat(), windows.early[1].isoformat()],
        "late_window": [windows.late[0].isoformat(), windows.late[1].isoformat()],
        "early_sources": [
            {
                "row": c.row,
                "chunk": c.chunk,
                "score": c.score,
                "timestamp": c.timestamp_iso,
                "corpus": c.corpus,
                "text_path": c.text_path,
            }
            for c in early_chunks
        ],
        "late_sources": [
            {
                "row": c.row,
                "chunk": c.chunk,
                "score": c.score,
                "timestamp": c.timestamp_iso,
                "corpus": c.corpus,
                "text_path": c.text_path,
            }
            for c in late_chunks
        ],
        "prompt": prompt,
        "answer": answer,
    }
