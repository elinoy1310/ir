# exe4/stage4_temporal_evolution_rag.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# EXE3
from exe3.stage3_retrieval import (
    load_bm25_store,
    load_dense_store,
    transform_query_to_bm25,
    MODEL_NAME,
    change_chanking_method,
)
from exe3.stage4_llm_rag import call_ollama

# EXE4
from exe4.utils import resolve_chunk_metadata


@dataclass
class ChunkEvidence:
    row: int
    chunk: str
    score: float
    timestamp_iso: str
    corpus: str
    text_path: str
    text: str


@dataclass
class TemporalWindows:
    early: Tuple[datetime, datetime]
    late: Tuple[datetime, datetime]


def _read_text(path: str, max_chars: int = 2200) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    txt = p.read_text(encoding="utf-8", errors="ignore")
    txt = " ".join(txt.split())
    return txt[:max_chars]


def _month_delta(months: int) -> timedelta:
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
    chunks_index_path: Path,
    metadata_index_path: Path,
) -> Dict[str, Dict[str, Any]]:
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
    nation: str,
    k: int,
) -> List[Tuple[int, str, float]]:
    start, end = window
    nation_u = nation.upper()

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

    sub_scores = scores[valid_idx]
    order = np.argsort(-sub_scores)[:k]
    picked = [(int(valid_idx[j]), names[int(valid_idx[j])], float(sub_scores[j])) for j in order]
    return picked


def _chunk_text_path_for_prompt(chunking_method: str, chunk_name: str) -> str:
    if chunking_method == "fixed":
        return chunk_name

    idx_path = Path("exe4/united_parentSon_chunk_index.json")
    data = json.loads(idx_path.read_text(encoding="utf-8"))
    norm = chunk_name.replace("\\", "/")
    if norm in data:
        return data[norm].get("parent_file", chunk_name)
    if chunk_name in data:
        return data[chunk_name].get("parent_file", chunk_name)
    return chunk_name


def _materialize(
    picks: List[Tuple[int, str, float]],
    time_cache: Dict[str, Dict[str, Any]],
    chunking_method: str,
) -> List[ChunkEvidence]:
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


# -----------------------------
# NEW: 3-step LLM enforcement
# -----------------------------
def _format_evidence_tagged(chunks: List[ChunkEvidence], prefix: str) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(f"{prefix}{i} ({c.corpus} | {c.timestamp_iso} | score={c.score:.4f})\n{c.text}\n")
    return "\n".join(lines).strip()


def _build_period_prompt(query: str, evidence_block: str, period_name: str) -> str:
    return f"""
You are given evidence from ONLY the {period_name} time period.

QUESTION:
"{query}"

EVIDENCE:
{evidence_block}

TASK:
1) Write a concise summary of the main position/rhetoric in this period (4-7 sentences).
2) Extract 2-4 key talking points as bullet points.
3) When referencing evidence, cite ONLY the ids shown above (e.g., E1, E2... or L1, L2...).

RULES:
- Do NOT mention the other time period.
- Do NOT invent facts.
- If evidence is insufficient, write: "Insufficient evidence."
""".strip()


def _build_change_prompt(query: str, early_summary: str, late_summary: str) -> str:
    return f"""
You are comparing two time periods based on summaries produced from evidence.

QUESTION:
"{query}"

EARLY SUMMARY:
{early_summary}

LATE SUMMARY:
{late_summary}

TASK:
1) Describe the change (or stability) between EARLY and LATE in 5-10 sentences.
2) Provide 2-4 bullet "change indicators" (themes/emphases that differ).
3) If either summary says "Insufficient evidence.", do NOT invent changes; explain what is missing.

RULES:
- Do not cite raw evidence ids here (you only see summaries).
- Be explicit about what changed and why (only if supported by the summaries).
""".strip()


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

    names = bm25_names if vector_method == "bm25" else dense_names

    # 3) choose the right chunk_index in exe4
    chunks_index_path = Path("exe4/united_fixed_chunk_index.json") if chunking_method == "fixed" else Path("exe4/united_parentSon_chunk_index.json")

    # 4) build time cache + windows
    time_cache = _build_time_cache(
        names=names,
        chunking_method=chunking_method,
        chunks_index_path=chunks_index_path,
        metadata_index_path=metadata_index_path,
    )
    all_dates = [v["timestamp_dt"] for v in time_cache.values()]
    windows = _compute_windows(all_dates, window_months=window_months)

    # 5) score all docs for this query
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

    early_chunks = _materialize(early_picks, time_cache, chunking_method)
    late_chunks = _materialize(late_picks, time_cache, chunking_method)

    # 7) NEW: three-pass LLM
    early_block = _format_evidence_tagged(early_chunks, prefix="E")
    late_block = _format_evidence_tagged(late_chunks, prefix="L")

    early_prompt = _build_period_prompt(query, early_block, period_name="EARLY")
    late_prompt = _build_period_prompt(query, late_block, period_name="LATE")

    early_summary = call_ollama(early_prompt)
    late_summary = call_ollama(late_prompt)

    change_prompt = _build_change_prompt(query, early_summary, late_summary)
    answer = call_ollama(change_prompt)

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
        "early_prompt": early_prompt,
        "late_prompt": late_prompt,
        "change_prompt": change_prompt,
        "early_summary": early_summary,
        "late_summary": late_summary,
        "answer": answer,
    }
