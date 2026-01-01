# exe4/ temporal_retrieval.py
from exe4.temporal_utils import (
    infer_time_constraint,
    extract_date_from_source_file,
    extract_year,
    is_in_q4,
)

def hard_filter_by_year(results: list[dict], year: int) -> list[dict]:
    out = []
    for r in results:
        y = extract_year(r.get("source_file"))
        if y == year:
            out.append(r)
    return out

def hard_filter_by_q4(results: list[dict], year: int) -> list[dict]:
    out = []
    for r in results:
        dt = extract_date_from_source_file(r.get("source_file"))
        if is_in_q4(dt, year):
            out.append(r)
    return out

def recency_rerank(results: list[dict]) -> list[dict]:
    # newest first
    def key(r):
        dt = extract_date_from_source_file(r.get("source_file"))
        return dt.timestamp() if dt else 0
    return sorted(results, key=key, reverse=True)

def apply_temporal_logic(query: str, enriched_results: list[dict]) -> list[dict]:
    """
    Input: results AFTER enrich_results (so each item has source_file)
    Output: filtered/reranked list
    """
    c = infer_time_constraint(query)
    if not c:
        return enriched_results

    if c["type"] == "year":
        return hard_filter_by_year(enriched_results, c["value"])

    if c["type"] == "q4":
        return hard_filter_by_q4(enriched_results, c["value"])

    if c["type"] == "recency":
        return recency_rerank(enriched_results)

    return enriched_results
