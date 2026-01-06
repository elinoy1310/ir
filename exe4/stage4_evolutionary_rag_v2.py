# exe4/stage4_evolutionary_rag_v2.py
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import requests

from exe3.stage3_retrieval import (
    load_bm25_store,
    load_dense_store,
    bm25_retrieve,
    dense_retrieve,
    change_chanking_method,
)

# Stage 3 (Soft-Decay)
from exe4.stage3_time_decay_scoring import compute_time_score

# Stage 2/metadata utils (cache + windows)
from exe4.utils import (
    load_time_cache,
    compute_early_late_windows,
    filter_names_by_window,
)

# Dense model
from sentence_transformers import SentenceTransformer


# ---------- Ollama ----------
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3:8b"


# ---------- Helpers ----------
def _slug(s: str, max_len: int = 60) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:max_len] if len(s) > max_len else s


def _read_text(path: str, max_chars: int = 1200) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    txt = p.read_text(encoding="utf-8", errors="ignore")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt[:max_chars]


def _subset_matrix(X, names: List[str], subset_names: List[str]):
    idx_map = {n: i for i, n in enumerate(names)}
    idxs = [idx_map[n] for n in subset_names if n in idx_map]
    if not idxs:
        return None, []
    return X[idxs], [names[i] for i in idxs]


def _normalize_scores_to_01(scores: List[float]) -> List[float]:
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]


def _chunk_index_path_for(chunking_method: str) -> str:
    # אצלך קיימים שני קבצים אלו (לפי מה שהעלית)
    if chunking_method == "fixed":
        return "exe4/united_fixed_chunk_index.json"
    if chunking_method in {"parent-son", "parent_son", "parent"}:
        return "exe4/united_parentSon_chunk_index.json"
    # fallback:
    return "exe4/united_fixed_chunk_index.json"


def call_ollama(
    prompt: str,
    *,
    model: str = OLLAMA_MODEL,
    temperature: float = 0.2,
    timeout_sec: int = 600,
    retries: int = 2,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": float(temperature)},
    }

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=timeout_sec)
            r.raise_for_status()
            return (r.json().get("response") or "").strip()
        except Exception as e:
            last_err = e
            # backoff קטן
            time.sleep(1.5 * (attempt + 1))

    # אם נכשל – נחזיר טקסט שמאפשר להמשיך ריצות בלי להפיל הכול
    return f"LLM_CALL_FAILED: {type(last_err).__name__}: {last_err}"


# ---------- Retrieval wrappers (exe3 signatures differ between bm25/dense) ----------
def _bm25_retrieve_topn(query: str, X_sub, vocab, names_sub: List[str], topn: int):
    # bm25_retrieve מחזיר [(idx, name, score), ...]
    hits = bm25_retrieve(query, X_sub, vocab, names_sub, top_k=topn)
    out: List[Tuple[str, float]] = []
    for h in hits:
        # h = (idx, name, score)
        name = h[1]
        score = float(h[2])
        out.append((name, score))
    return out


def _dense_retrieve_topn(query: str, X_sub, names_sub: List[str], model: SentenceTransformer, topn: int):
    # dense_retrieve מחזיר [(idx, name, score), ...] גם כן
    hits = dense_retrieve(query, X_sub, names_sub, model, top_k=topn)
    out: List[Tuple[str, float]] = []
    for h in hits:
        name = h[1]
        score = float(h[2])
        out.append((name, score))
    return out


def retrieve_candidates(
    *,
    query: str,
    vector_method: str,
    X_full,
    names_full: List[str],
    subset_names: List[str],
    candidate_n: int,
    vocab=None,
    dense_model: Optional[SentenceTransformer] = None,
) -> List[Tuple[str, float]]:
    X_sub, names_sub = _subset_matrix(X_full, names_full, subset_names)
    if X_sub is None:
        return []

    if vector_method == "bm25":
        if vocab is None:
            raise ValueError("vocab is required for bm25 retrieval")
        return _bm25_retrieve_topn(query, X_sub, vocab, names_sub, candidate_n)

    if vector_method == "dense":
        if dense_model is None:
            raise ValueError("dense_model is required for dense retrieval")
        return _dense_retrieve_topn(query, X_sub, names_sub, dense_model, candidate_n)

    raise ValueError("vector_method must be 'bm25' or 'dense'")


# ---------- Stage3 Soft-Decay rerank inside a window ----------
def rerank_with_stage3_soft_decay(
    *,
    candidates: List[Tuple[str, float]],  # (path, sim_raw)
    alpha: float,
    lambda_decay: float,
    query_date: datetime,
    chunking_method: str,
    metadata_index_path: str,
) -> List[Tuple[str, float, float, float]]:
    """
    final = (1-alpha)*SimNorm + alpha*TimeScore
    returns: (path, sim_raw, time_score, final_score)
    """
    if not candidates:
        return []

    sim_raw = [s for _, s in candidates]
    sim_norm = _normalize_scores_to_01(sim_raw)

    chunk_index_path = _chunk_index_path_for(chunking_method)

    reranked: List[Tuple[str, float, float, float]] = []
    for (path, s_raw), s_n in zip(candidates, sim_norm):
        time_score, _, _ = compute_time_score(
            chunk_path=path,
            chunk_index_path=chunk_index_path,
            metadata_index_path=metadata_index_path,
            chunking_method=chunking_method,
            query_date=query_date,
            lambda_decay=lambda_decay,
        )
        final = (1.0 - float(alpha)) * float(s_n) + float(alpha) * float(time_score)
        reranked.append((path, float(s_raw), float(time_score), float(final)))

    reranked.sort(key=lambda t: t[3], reverse=True)
    return reranked


def _pack_sources(
    *,
    ranked: List[Tuple[str, float, float, float]],
    cache: Dict[str, Any],
    max_chars: int = 1200,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path, sim_raw, time_score, final_score in ranked:
        ts = cache.get(path, {}).get("timestamp", "")
        out.append(
            {
                "chunk": path,
                "timestamp": ts,
                "sim_raw": sim_raw,
                "time_score": time_score,
                "final_score": final_score,
                "text": _read_text(path, max_chars=max_chars),
            }
        )
    return out


# ---------- Prompts (short, no duplicated text) ----------
def build_prompt_early_or_late(
    *,
    query: str,
    window_name: str,
    window: Tuple[str, str],
    items: List[Dict[str, Any]],
) -> str:
    # מספקים “מיון לשני כיוונים” בלי להכפיל טקסט: נותנים רשימת אינדקסים לפי זמן
    # והטקסט פעם אחת לכל מקור.
    order_old_new = sorted(range(len(items)), key=lambda i: items[i]["timestamp"])
    order_new_old = list(reversed(order_old_new))

    lines = []
    lines.append("You are a Temporal RAG assistant.")
    lines.append("RULES:")
    lines.append("1) Use ONLY the EVIDENCE provided.")
    lines.append("2) Every bullet MUST cite evidence like [2].")
    lines.append("3) If not enough evidence: write exactly: Insufficient evidence.")
    lines.append("")
    lines.append(f"QUESTION: {query}")
    lines.append(f"{window_name} WINDOW: {window[0]} to {window[1]}")
    lines.append("")
    lines.append("EVIDENCE ORDER (old->new): " + ", ".join(str(i + 1) for i in order_old_new))
    lines.append("EVIDENCE ORDER (new->old): " + ", ".join(str(i + 1) for i in order_new_old))
    lines.append("")
    lines.append("EVIDENCE:")
    for i, it in enumerate(items, 1):
        lines.append(f"[{i}] ({it['timestamp']} | score={it['final_score']:.4f}) {it['chunk']}")
        lines.append(it["text"])
        lines.append("")
    lines.append("TASK: Write 3-6 bullet points summarizing the position/rhetoric in this window. Cite at least 2 sources.")
    return "\n".join(lines)


def build_prompt_change(
    *,
    query: str,
    early_window: Tuple[str, str],
    late_window: Tuple[str, str],
    early_items: List[Dict[str, Any]],
    late_items: List[Dict[str, Any]],
) -> str:
    lines = []
    lines.append("You are a Temporal RAG assistant comparing EARLY vs LATE.")
    lines.append("RULES:")
    lines.append("1) Use ONLY the EVIDENCE below.")
    lines.append("2) EARLY statements cite EARLY[2], LATE statements cite LATE[1].")
    lines.append("3) If not enough evidence in either side: write exactly: Insufficient evidence to determine change.")
    lines.append("")
    lines.append(f"QUESTION: {query}")
    lines.append(f"EARLY WINDOW: {early_window[0]} to {early_window[1]}")
    lines.append(f"LATE  WINDOW: {late_window[0]} to {late_window[1]}")
    lines.append("")
    lines.append("EARLY EVIDENCE:")
    for i, it in enumerate(early_items, 1):
        lines.append(f"EARLY[{i}] ({it['timestamp']} | score={it['final_score']:.4f}) {it['chunk']}")
        lines.append(it["text"])
        lines.append("")
    lines.append("LATE EVIDENCE:")
    for i, it in enumerate(late_items, 1):
        lines.append(f"LATE[{i}] ({it['timestamp']} | score={it['final_score']:.4f}) {it['chunk']}")
        lines.append(it["text"])
        lines.append("")
    lines.append("TASK:")
    lines.append("A) Explain the CHANGE over time (must cite at least 1 EARLY and 1 LATE).")
    lines.append("B) Provide 2-4 short bullet 'change indicators'.")
    return "\n".join(lines)


def run_stage4_evolution_v2(
    *,
    query: str,
    nation: str,
    chunking_method: str,
    vector_method: str,
    k: int = 5,
    window_months: int = 8,
    candidate_n: int = 800,
    # Stage 3 params
    alpha: float = 0.3,
    lambda_decay: float = 0.6,
    # metadata index (זה מה-stage2)
    metadata_index_path: str = "exe4/metadata_index.json",
    # LLM options
    llm_timeout_sec: int = 600,
    evidence_max_chars: int = 1200,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Stage 4 Evolutionary RAG:
    - Hard temporal gating into EARLY/LATE pools
    - Retrieve Top-N inside each pool
    - Re-rank inside pool using Stage-3 Soft-Decay
    - 3 LLM calls: EARLY summary, LATE summary, CHANGE analysis
    """
    if verbose:
        print(f"[START] nation={nation} | chunking={chunking_method} | vec={vector_method} | k={k} | N={candidate_n}")
        print(f"        query: {query}")

    # STEP 0: chunking method affects exe3 retrieval paths
    if verbose:
        print("[STEP 0] Setting chunking method in exe3...")
    change_chanking_method(chunking_method)

    # STEP 1: time cache
    if verbose:
        print("[STEP 1] Loading time_cache...")
    cache = load_time_cache(chunking_method)
    if verbose:
        print(f"[CACHE] Loaded {len(cache)} cached chunk timestamps")

    # STEP 2: vector store + (vocab/model if needed)
    if verbose:
        print(f"[STEP 2] Loading vector store ({vector_method}) ...")
    vocab = None
    dense_model = None
    if vector_method == "bm25":
        X, vocab, names = load_bm25_store()
    else:
        X, names = load_dense_store()
        # Load model ONCE per run
        dense_model = SentenceTransformer("intfloat/e5-small-v2")

    # STEP 3: compute time windows
    if verbose:
        print("[STEP 3] Computing time windows...")
    early_w, late_w = compute_early_late_windows(cache, nation, window_months=window_months)
    early_window = (early_w.start.date().isoformat(), early_w.end.date().isoformat())
    late_window = (late_w.start.date().isoformat(), late_w.end.date().isoformat())
    if verbose:
        print(f"[WINDOWS] EARLY={early_window[0]} → {early_window[1]} | LATE={late_window[0]} → {late_window[1]}")

    # STEP 4: pools by corpus + window (hard gating)
    if verbose:
        print("[STEP 4] Building pools by corpus + window (hard temporal gating)...")
    early_names = filter_names_by_window(names, cache, nation, early_w)
    late_names = filter_names_by_window(names, cache, nation, late_w)
    if verbose:
        print(f"[POOLS] early_pool={len(early_names)} | late_pool={len(late_names)}")

    # STEP 5-6: retrieve inside each pool
    if verbose:
        print(f"[STEP 5] Retrieving candidates inside EARLY pool (Top-{candidate_n})...")
    early_candidates = retrieve_candidates(
        query=query,
        vector_method=vector_method,
        X_full=X,
        names_full=names,
        subset_names=early_names,
        candidate_n=candidate_n,
        vocab=vocab,
        dense_model=dense_model,
    )

    if verbose:
        print(f"[STEP 6] Retrieving candidates inside LATE pool (Top-{candidate_n})...")
    late_candidates = retrieve_candidates(
        query=query,
        vector_method=vector_method,
        X_full=X,
        names_full=names,
        subset_names=late_names,
        candidate_n=candidate_n,
        vocab=vocab,
        dense_model=dense_model,
    )

    if verbose:
        print(f"[CANDIDATES] early={len(early_candidates)} | late={len(late_candidates)}")

    # STEP 7-8: rerank using Stage-3 Soft Decay inside each window
    if verbose:
        print("[STEP 7] Reranking EARLY candidates using Stage-3 Soft-Decay...")
    early_ranked = rerank_with_stage3_soft_decay(
        candidates=early_candidates,
        alpha=alpha,
        lambda_decay=lambda_decay,
        query_date=early_w.end,  # anchor to end of early window
        chunking_method=chunking_method,
        metadata_index_path=metadata_index_path,
    )[:k]

    if verbose:
        print("[STEP 8] Reranking LATE candidates using Stage-3 Soft-Decay...")
    late_ranked = rerank_with_stage3_soft_decay(
        candidates=late_candidates,
        alpha=alpha,
        lambda_decay=lambda_decay,
        query_date=late_w.end,
        chunking_method=chunking_method,
        metadata_index_path=metadata_index_path,
    )[:k]

    if verbose:
        print(f"[TOP-K] EARLY picked {len(early_ranked)} | LATE picked {len(late_ranked)}")

    # STEP 9: pack sources with text
    early_items = _pack_sources(ranked=early_ranked, cache=cache, max_chars=evidence_max_chars)
    late_items = _pack_sources(ranked=late_ranked, cache=cache, max_chars=evidence_max_chars)

    # STEP 10-12: 3 LLM calls (short prompts)
    if verbose:
        print("[STEP 9] Calling LLM: EARLY summary...")
    early_prompt = build_prompt_early_or_late(
        query=query,
        window_name="EARLY",
        window=early_window,
        items=early_items,
    )
    early_answer = call_ollama(early_prompt, timeout_sec=llm_timeout_sec)

    if verbose:
        print("[STEP 10] Calling LLM: LATE summary...")
    late_prompt = build_prompt_early_or_late(
        query=query,
        window_name="LATE",
        window=late_window,
        items=late_items,
    )
    late_answer = call_ollama(late_prompt, timeout_sec=llm_timeout_sec)

    if verbose:
        print("[STEP 11] Calling LLM: CHANGE analysis...")
    change_prompt = build_prompt_change(
        query=query,
        early_window=early_window,
        late_window=late_window,
        early_items=early_items,
        late_items=late_items,
    )
    change_answer = call_ollama(change_prompt, timeout_sec=llm_timeout_sec)

    return {
        "query": query,
        "nation": nation,
        "chunking_method": chunking_method,
        "vector_method": vector_method,
        "k": k,
        "window_months": window_months,
        "candidate_n": candidate_n,
        "alpha": alpha,
        "lambda_decay": lambda_decay,
        "early_window": early_window,
        "late_window": late_window,
        "early_sources": early_items,
        "late_sources": late_items,
        "early_prompt": early_prompt,
        "late_prompt": late_prompt,
        "change_prompt": change_prompt,
        "early_answer": early_answer,
        "late_answer": late_answer,
        "change_answer": change_answer,
    }
