# # exe4/stage4_evolution_fast.py
# from __future__ import annotations

# import json
# from dataclasses import dataclass
# from datetime import datetime, timedelta
# from pathlib import Path
# from typing import Any, Dict, List, Tuple, Optional

# from sentence_transformers import SentenceTransformer

# from exe3.stage3_retrieval import (
#     load_bm25_store,
#     load_dense_store,
#     bm25_retrieve,
#     dense_retrieve,
#     change_chanking_method,
#     MODEL_NAME,
# )
# from exe3.stage4_llm_rag import call_ollama


# @dataclass
# class Evidence:
#     chunk_path: str
#     score: float
#     ts: str
#     corpus: str
#     text_preview: str


# # -------------------- helpers --------------------

# def _month_delta(m: int) -> timedelta:
#     return timedelta(days=30 * int(m))


# def _compute_windows(all_ts: List[str], window_months: int) -> Tuple[Tuple[datetime, datetime], Tuple[datetime, datetime]]:
#     dts = [datetime.fromisoformat(t) for t in all_ts]
#     mn, mx = min(dts), max(dts)
#     d = _month_delta(window_months)
#     early = (mn, min(mn + d, mx))
#     late = (max(mx - d, mn), mx)
#     return early, late


# def _load_time_cache(chunking_method: str) -> Dict[str, Dict[str, Any]]:
#     p = Path("exe4/cache/time_cache_fixed.json") if chunking_method == "fixed" else Path("exe4/cache/time_cache_parent-son.json")
#     if not p.exists():
#         raise FileNotFoundError(f"Time cache not found: {p}. Run once: python -m exe4.stage4_prepare_time_cache")
#     return json.loads(p.read_text(encoding="utf-8"))


# def _read_chunk_text(path: str, max_chars: int = 900) -> str:
#     p = Path(path)
#     if not p.exists():
#         return ""
#     txt = p.read_text(encoding="utf-8", errors="ignore")
#     txt = " ".join(txt.split())
#     return txt[:max_chars]


# def _make_preview(path: str, max_chars: int = 250) -> str:
#     txt = _read_chunk_text(path, max_chars=max_chars)
#     return txt


# # -------------------- topic filter --------------------
# # תיקון C: סינון טופיק פשוט כדי למנוע "Ukraine/China" וכו'
# TOPIC_KEYWORDS = [
#     "israel", "gaza", "hamas", "palestin", "hostage", "ceasefire",
#     "idf", "west bank", "rafah", "jerusalem", "two-state", "unrwa",
#     "october 7", "7 october", "netanyahu"
# ]

# def _topic_match(text: str) -> bool:
#     t = text.lower()
#     return any(k in t for k in TOPIC_KEYWORDS)


# # -------------------- prompts --------------------

# def _period_prompt(query: str, block: str, period: str, n_evidence: int) -> str:
#     return f"""
# You are given evidence from ONLY the {period} period.

# QUESTION:
# "{query}"

# EVIDENCE ITEMS COUNT: {n_evidence}

# EVIDENCE:
# {block}

# TASK:
# 1) Describe the dominant rhetoric/framing in this period (4-7 sentences).
# 2) Provide 2-4 bullet talking points.
# 3) Cite ONLY ids shown above (E1/E2... or L1/L2...) when referring to evidence.
# 4) Only write "Insufficient evidence." if EVIDENCE ITEMS COUNT is LESS THAN 2.

# RULES:
# - Do NOT mention the other time period.
# - Do NOT compare across periods.
# - Do NOT invent facts.
# """.strip()


# def _change_prompt(query: str, early_sum: str, late_sum: str) -> str:
#     return f"""
# QUESTION:
# "{query}"

# EARLY SUMMARY:
# {early_sum}

# LATE SUMMARY:
# {late_sum}

# TASK:
# 1) Explain the change (or stability) between EARLY and LATE in 5-10 sentences.
# 2) Provide 2-4 bullet "change indicators" (themes/emphases that differ).
# 3) If either summary contains "Insufficient evidence.", do NOT invent changes; explain what is missing.

# RULES:
# - Do not cite raw evidence ids here (you only see summaries).
# """.strip()


# def _format_block(evs: List[Evidence], prefix: str, max_chars_each: int = 700) -> str:
#     # יותר קטן כדי למנוע timeout (עדיין נותן מספיק)
#     lines = []
#     for i, e in enumerate(evs, 1):
#         text = _read_chunk_text(e.chunk_path, max_chars_each)
#         lines.append(f"{prefix}{i} ({e.corpus} | {e.ts} | score={e.score:.4f})\n{text}\n")
#     return "\n".join(lines).strip()


# # -------------------- main --------------------

# def run_stage4_fast(
#     *,
#     query: str,
#     nation: str,                 # "uk" / "us"
#     chunking_method: str,        # "fixed" / "parent-son"
#     vector_method: str,          # "bm25" / "dense"
#     k: int = 5,
#     window_months: int = 8,
#     candidate_n: int = 400,
#     min_evidence: int = 2,       # תיקון B
#     max_chars_each: int = 700,
#     verbose: bool = True,
# ) -> Dict[str, Any]:
#     nation = nation.lower().strip()
#     if nation not in ("uk", "us"):
#         raise ValueError("nation must be 'uk' or 'us'")
#     if chunking_method not in ("fixed", "parent-son"):
#         raise ValueError("chunking_method must be 'fixed' or 'parent-son'")
#     if vector_method not in ("bm25", "dense"):
#         raise ValueError("vector_method must be 'bm25' or 'dense'")

#     def log(msg: str):
#         if verbose:
#             print(msg)

#     log(f"[START] nation={nation} | chunking={chunking_method} | vec={vector_method} | k={k} | N={candidate_n}")

#     # 0) ensure retrieval names align with chunking
#     log("[STEP 0] Setting chunking method in exe3...")
#     change_chanking_method(chunking_method)

#     # 1) load time cache
#     log("[STEP 1] Loading time_cache...")
#     time_cache = _load_time_cache(chunking_method)
#     log(f"[CACHE] Loaded {len(time_cache)} cached chunk timestamps")

#     # 2) compute windows from nation subset
#     log("[STEP 2] Computing time windows...")
#     all_ts = [v["timestamp"] for v in time_cache.values() if v.get("corpus", "").lower() == nation]
#     if not all_ts:
#         raise ValueError(f"No timestamps in cache for nation='{nation}'")
#     early_w, late_w = _compute_windows(all_ts, window_months)
#     log(f"[WINDOWS] EARLY={early_w[0].date()} → {early_w[1].date()} | LATE={late_w[0].date()} → {late_w[1].date()}")

#     # 3) retrieve candidates
#     log(f"[STEP 3] Retrieving Top-{candidate_n} candidates using {vector_method}...")
#     if vector_method == "bm25":
#         X, vocab, names = load_bm25_store()
#         sim = bm25_retrieve(query, X, vocab, names, top_k=candidate_n)
#     else:
#         X, names = load_dense_store()
#         model = SentenceTransformer(MODEL_NAME)
#         sim = dense_retrieve(query, X, names, model, top_k=candidate_n)
#     log(f"[CANDIDATES] Retrieved {len(sim)} candidates")

#     # 4) corpus gate early (תיקון A)
#     log("[STEP 4] Corpus-gating candidates (keep only correct nation)...")
#     gated: List[Tuple[int, str, float]] = []
#     missing_meta = 0
#     other_nation = 0

#     for row, chunk_path, score in sim:
#         meta = time_cache.get(chunk_path)
#         if not meta:
#             missing_meta += 1
#             continue
#         if meta.get("corpus", "").lower() != nation:
#             other_nation += 1
#             continue
#         gated.append((row, chunk_path, float(score)))

#     log(f"[GATE] kept={len(gated)} | other_nation={other_nation} | missing_meta={missing_meta}")

#     # 5) temporal split + topic filter (תיקון C)
#     log("[STEP 5] Temporal split (EARLY/LATE) + topic filter...")
#     early_candidates: List[Evidence] = []
#     late_candidates: List[Evidence] = []
#     topic_rejects = 0

#     for _row, chunk_path, score in gated:
#         meta = time_cache[chunk_path]
#         ts = meta["timestamp"]
#         dt = datetime.fromisoformat(ts)

#         preview = _make_preview(chunk_path, max_chars=250)

#         # topic filter uses preview (fast) — can optionally read more, but preview is enough
#         if not _topic_match(preview):
#             topic_rejects += 1
#             continue

#         e = Evidence(chunk_path=chunk_path, score=score, ts=ts, corpus=meta["corpus"], text_preview=preview)

#         if early_w[0] <= dt <= early_w[1]:
#             early_candidates.append(e)
#         elif late_w[0] <= dt <= late_w[1]:
#             late_candidates.append(e)

#     log(f"[FILTER] early_hits={len(early_candidates)} | late_hits={len(late_candidates)} | topic_rejects={topic_rejects}")

#     early_candidates.sort(key=lambda e: e.ts)        # מהישן לחדש
#     late_candidates.sort(key=lambda e: e.ts, reverse=True)  # מהחדש לישן

#     early_top = early_candidates[:k]
#     late_top = late_candidates[:k]

#     log(f"[TOP-K] EARLY picked {len(early_top)} | LATE picked {len(late_top)}")

#     # 6) evidence gate (תיקון B)
#     log("[STEP 6] Evidence gate (min_evidence check) ...")
#     early_ok = len(early_top) >= min_evidence
#     late_ok = len(late_top) >= min_evidence

#     # build blocks only if needed
#     early_block = _format_block(early_top, "E", max_chars_each=max_chars_each) if early_top else ""
#     late_block = _format_block(late_top, "L", max_chars_each=max_chars_each) if late_top else ""

#     # EARLY summary
#     if early_ok:
#         log("[STEP 7] Calling LLM: EARLY summary...")
#         early_prompt = _period_prompt(query, early_block, "EARLY", n_evidence=len(early_top))
#         early_summary = call_ollama(early_prompt)
#     else:
#         early_prompt = _period_prompt(query, early_block, "EARLY", n_evidence=len(early_top))
#         early_summary = "Insufficient evidence."

#     # LATE summary
#     if late_ok:
#         log("[STEP 8] Calling LLM: LATE summary...")
#         late_prompt = _period_prompt(query, late_block, "LATE", n_evidence=len(late_top))
#         late_summary = call_ollama(late_prompt)
#     else:
#         late_prompt = _period_prompt(query, late_block, "LATE", n_evidence=len(late_top))
#         late_summary = "Insufficient evidence."

#     # CHANGE analysis
#     log("[STEP 9] Calling LLM: CHANGE analysis...")
#     change_prompt = _change_prompt(query, early_summary, late_summary)
#     answer = call_ollama(change_prompt)

#     log("[DONE] Finished one run.\n")

#     return {
#         "query": query,
#         "nation": nation,
#         "chunking_method": chunking_method,
#         "vector_method": vector_method,
#         "k": k,
#         "candidate_n": candidate_n,
#         "window_months": window_months,
#         "min_evidence": min_evidence,
#         "early_window": [early_w[0].isoformat(), early_w[1].isoformat()],
#         "late_window": [late_w[0].isoformat(), late_w[1].isoformat()],

#         # include previews for report/debug (תיקון D)
#         "early_sources": [e.__dict__ for e in early_top],
#         "late_sources": [e.__dict__ for e in late_top],

#         "stats": {
#             "missing_meta": missing_meta,
#             "other_nation": other_nation,
#             "gated_kept": len(gated),
#             "topic_rejects": topic_rejects,
#             "early_hits": len(early_candidates),
#             "late_hits": len(late_candidates),
#         },

#         "early_prompt": early_prompt,
#         "late_prompt": late_prompt,
#         "change_prompt": change_prompt,
#         "early_summary": early_summary,
#         "late_summary": late_summary,
#         "answer": answer,
#     }
