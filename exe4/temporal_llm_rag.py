# exe4/temporal_llm_rag.py
import re
import requests
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------
# Robust imports: allow running from repo root (ir/) while keeping exe3 intact
# ---------------------------------------------------------------------
try:
    # If you run as package: python -m exe4.temporal_llm_rag
    from exe3.stage3_retrieval import (
        load_chunkpath_to_source,
        load_bm25_store,
        load_dense_store,
        bm25_retrieve,
        dense_retrieve,
        enrich_results,
        uk_count,
        MODEL_NAME,
        change_chanking_method,
    )
except Exception:
    # If you run as a script from inside exe3/exe4 folders
    from exe3.stage3_retrieval import (
        load_chunkpath_to_source,
        load_bm25_store,
        load_dense_store,
        bm25_retrieve,
        dense_retrieve,
        enrich_results,
        uk_count,
        MODEL_NAME,
        change_chanking_method,
    )

# ---------------------------------------------------------------------
# Ollama settings
# ---------------------------------------------------------------------
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "gemma:2b"


# ------------------------- Stage 2: Temporal utils -------------------------
def extract_date_from_source_file(source_file: str):
    """
    Extract datetime from filenames like: debates2024-11-11.txt
    Returns datetime or None.
    """
    if not source_file:
        return None
    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", source_file)
    if not m:
        return None
    y, mth, d = map(int, m.groups())
    return datetime(y, mth, d)


def infer_time_constraint(query: str):
    """
    Stage 2 heuristic constraints:
      - explicit year: "... in 2024" -> {"type":"year","value":2024}
      - "last quarter of 2023" -> {"type":"q4","value":2023}
      - "current/latest/most recent" -> {"type":"recency"}
    """
    q = (query or "").lower()

    # last quarter of 2023
    m = re.search(r"last quarter of\s+(20\d{2})", q)
    if m:
        return {"type": "q4", "value": int(m.group(1))}

    # explicit year (2024, 2025...)
    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        return {"type": "year", "value": int(m.group(1))}

    if any(tok in q for tok in ["current", "latest", "most recent", "as of", "today"]):
        return {"type": "recency"}

    return None


def is_in_q4(dt: datetime, year: int) -> bool:
    return dt is not None and dt.year == year and dt.month in (10, 11, 12)


def temporal_filter_and_rerank(query: str, enriched: list[dict]):
    """
    Input: enriched results (must contain 'source_file')
    Output: filtered / reranked enriched results
    """
    c = infer_time_constraint(query)
    if not c:
        return enriched

    # attach parsed datetime for later sorting/printing
    for r in enriched:
        dt = extract_date_from_source_file(r.get("source_file"))
        r["doc_date"] = dt  # may be None
        r["doc_year"] = dt.year if dt else None

    if c["type"] == "year":
        year = c["value"]
        filtered = [r for r in enriched if r.get("doc_year") == year]
        # if nothing matched, keep empty (forces "I don't know") – good for analysis
        return filtered

    if c["type"] == "q4":
        year = c["value"]
        filtered = [r for r in enriched if is_in_q4(r.get("doc_date"), year)]
        return filtered

    if c["type"] == "recency":
        # recency rerank: newest first
        return sorted(enriched, key=lambda r: (r.get("doc_date").timestamp() if r.get("doc_date") else 0), reverse=True)

    return enriched


# ------------------------- LLM call -------------------------
def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 300
        }
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"Error calling Ollama: {e}"


def build_prompt(query: str, contexts: list[dict]) -> str:
    ctx_lines = []
    for i, c in enumerate(contexts, 1):
        ctx_lines.append(
            f"[{i}] SOURCE_FILE: {c.get('source_file')}\n"
            f"CHUNK: {c.get('chunk')}\n"
            f"TEXT: {c.get('text')}\n"
        )
    # fix: this is chars count, not "context chunks"
    print(f"\nBuilt prompt with {len(contexts)} retrieved chunks, total chars={len(''.join(ctx_lines))}.")
    return (
        "Answer ONLY using the CONTEXT below.\n"
        "If the answer is not in the context, say: \"I don't know based on the provided context.\" \n"
        "At the end, list the sources you used as: SOURCES: [1], [2], ...\n\n"
        f"QUESTION: {query}\n\n"
        "CONTEXT:\n"
        + "\n".join(ctx_lines)
    )


def run_rag(query: str, method: str = "hybrid", k: int = 5):
    # allow tuple (label, query)
    if isinstance(query, tuple):
        query = query[1]

    _ = uk_count()
    chunkpath_to_source = load_chunkpath_to_source()
    X_bm25, vocab, bm25_names = load_bm25_store()
    X_emb, dense_names = load_dense_store()
    st_model = SentenceTransformer(MODEL_NAME)

    # retrieval
    if method == "bm25":
        retrieved = bm25_retrieve(query, X_bm25, vocab, bm25_names, top_k=k)
    elif method == "dense":
        retrieved = dense_retrieve(query, X_emb, dense_names, st_model, top_k=k)
    else:
        # keep behavior explicit (your code doesn't define hybrid retrieval here)
        retrieved = dense_retrieve(query, X_emb, dense_names, st_model, top_k=k)

    # enrich (adds source_file, chunk, text, score)
    enriched = enrich_results(retrieved, chunkpath_to_source, max_chars=3500)

    # ---------------- Stage 2 temporal layer ----------------
    enriched = temporal_filter_and_rerank(query, enriched)

    # if filtering removed everything, still build prompt with empty context -> model should say "I don't know..."
    prompt = build_prompt(query, enriched[:k])
    answer = call_ollama(prompt)

    # output
    print("\n" + "=" * 90)
    print(f"METHOD={method}  K={k}")
    print("QUESTION:", query)
    print("\nANSWER:\n", answer)
    print("\nTOP CONTEXT SOURCES:")
    for i, c in enumerate(enriched[:k], 1):
        dt = c.get("doc_date")
        dt_str = dt.strftime("%Y-%m-%d") if isinstance(dt, datetime) else "NA"
        print(f"[{i}] {c.get('source_file')}  ({c.get('chunk')})  score={c.get('score'):.4f}  date={dt_str}")


def run_rag_with_multiple_configs(queries: list, chunk_method: str):
    # keep the same chunking switch behavior
    change_chanking_method(chunk_method)

    for method in ["dense", "bm25"]:
        for k in [3, 5, 8]:
            for i, query in enumerate(queries, 1):
                print(f"\nRunning RAG (Run {i}): Method={method}, k={k}, Chunking={chunk_method}")
                if isinstance(query, tuple):
                    label, q_text = query
                    print(f"\n[{label}] {q_text}\n")
                    run_rag(q_text, method=method, k=k)
                else:
                    run_rag(query, method=method, k=k)

                print(f"End of Run {i} for {method} with k={k}")
                print("=" * 90)


if __name__ == "__main__":
    # Example minimal temporal test set (feel free to edit)
    QUERIES = [
        ("HARD_FILTER", "What was the specific budget allocated to security in 2024?"),
        ("RECENCY", "What is the current official position regarding the State of Israel?"),
        ("RECENCY", "Has the official position in the last quarter of 2023 changed relative to the official position in the last quarter of 2025?"),
        ("EVOLUTION", "How did the Prime Minister/President's rhetoric regarding the war between Israel and Hamas/Gaza change between his first and last speech?"),
        ("AMBIGUITY", "Who is the Minister of Defense/Secretary of Defense?"),
    ]

    # run fixed chunking by default
    run_rag_with_multiple_configs(QUERIES, chunk_method="fixed")
