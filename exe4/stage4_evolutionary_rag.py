# exe4/stage4_evolutionary_rag.py
# Stage 4: Evolutionary RAG (Double Retrieval + Synthesis Prompt + LLM)

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from exe4.stage4_windows import load_metadata_index, compute_early_late_windows
from exe4.stage4_double_retrieval import prepare_backend, double_retrieve
from exe3.stage4_llm_rag import call_ollama


def sort_by_date(items: List[dict], asc: bool = True) -> List[dict]:
    """Sort contexts by doc_date (ISO string)."""
    return sorted(items, key=lambda x: x.get("doc_date", ""), reverse=not asc)


def _render_block(title: str, items: List[dict], tag: str) -> str:
    """Render a compact evidence block with stable citations [E1]/[L2]."""
    out = [f"=== {title} ==="]
    for i, c in enumerate(items, 1):
        text = (c.get("text") or "").strip()
        out.append(
            f"[{tag}{i}] DATE: {c.get('doc_date', '?')}\n"
            f"TEXT: {text}\n"
        )
    return "\n".join(out)


def build_prompt(query: str, early_ctx: List[dict], late_ctx: List[dict]) -> str:
    """
    Evidence-first evolutionary RAG prompt:
    1) Extract ONLY facts with short quotes.
    2) Synthesize change ONLY from extracted facts.
    """
    early = sort_by_date(early_ctx, asc=True)
    late  = sort_by_date(late_ctx, asc=True)

    return f"""
You are given evidence from TWO time periods: EARLY and LATE.
Your goal: describe how the position/policy evolved over time.

NON-NEGOTIABLE RULES:
1) Use ONLY the evidence below. Do NOT invent facts.
2) Every bullet MUST include:
   - a claim
   - a SHORT DIRECT QUOTE (5-20 words) from the evidence
   - a citation tag like [E2] or [L1]
3) If you cannot support a claim with a direct quote, write: "Not enough evidence".
4) No vague adjectives (e.g., proactive/reactive, traditional/modern) unless directly quoted.

QUESTION:
{query}

{_render_block("EARLY", early, "E")}

{_render_block("LATE", late, "L")}

OUTPUT (use EXACT headings):

A) EARLY facts (3-6 bullets, each with quote + [E*])
B) LATE facts  (3-6 bullets, each with quote + [L*])
C) What changed (1-3 bullets; each bullet MUST be:
   "Early: ... (quote) [E*]  ->  Late: ... (quote) [L*]")
D) Not enough evidence? (if any; otherwise "N/A")
""".strip()


def save_outputs(
    out_dir: Path,
    query_id: str,
    query: str,
    early_ctx: List[dict],
    late_ctx: List[dict],
    prompt: str,
    answer: str,
):
    """Save answer + contexts + prompt for reporting/debugging."""
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / f"{query_id}_answer.txt").write_text(answer, encoding="utf-8")
    (out_dir / f"{query_id}_prompt.txt").write_text(prompt, encoding="utf-8")

    payload = {
        "query_id": query_id,
        "query": query,
        "early": early_ctx,
        "late": late_ctx,
    }
    (out_dir / f"{query_id}_contexts.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_evolutionary_rag(
    query: str,
    query_id: str = "q1",
    metadata_index_path: str | None = None,
    method: str = "dense",
    chunk_method: str = "fixed",
    k: int = 5,
    window_months: int = 8,
    out_dir: str = "outputs/stage4_evolution",
) -> str:
    """
    Full Stage 4 pipeline:
    - load metadata
    - compute early/late windows
    - double retrieval
    - synthesis prompt
    - LLM answer
    - save outputs
    """
    if metadata_index_path is None:
        # Default: metadata_index.json located next to this file (exe4/)
        metadata_index_path = str(Path(__file__).resolve().parent / "metadata_index.json")

    meta = load_metadata_index(Path(metadata_index_path))
    early_w, late_w = compute_early_late_windows(meta, window_months)

    backend = prepare_backend(method, chunk_method)
    early_ctx, late_ctx = double_retrieve(backend, meta, early_w, late_w, query, k)

    prompt = build_prompt(query, early_ctx, late_ctx)
    answer = call_ollama(prompt)

    save_outputs(
        out_dir=Path(out_dir),
        query_id=query_id,
        query=query,
        early_ctx=early_ctx,
        late_ctx=late_ctx,
        prompt=prompt,
        answer=answer,
    )

    return answer


if __name__ == "__main__":
    q = "How did the Prime Minister/President's rhetoric regarding the war between Israel and Hamas/Gaza develop/change between his first and last speech?"
    ans = run_evolutionary_rag(
        query=q,
        query_id="q3",
        method="dense",
        chunk_method="fixed",
        k=5,
        window_months=8,
        out_dir="exe4/outputs/stage4_evolution",
    )
    print("Done. Saved outputs to exe4/outputs/stage4_evolution/")