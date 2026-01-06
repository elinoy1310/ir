# exe4/stage4_run_evolution_v2.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from exe4.stage4_evolutionary_rag_v2 import run_stage4_evolution_v2


def run_grid(
    *,
    queries: List[str],
    out_root: Path = Path("exe4/out_stage4_evolution_v2"),
    k: int = 5,
    window_months: int = 8,
    candidate_n: int = 800,
    alpha: float = 0.3,
    lambda_decay: float = 0.6,
    verbose_each_run: bool = True,
):
    run_dir = out_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    chunking_methods = ["fixed", "parent-son"]
    vector_methods = ["bm25", "dense"]
    nations = ["uk", "us"]

    total_runs = len(queries) * len(nations) * len(chunking_methods) * len(vector_methods)
    run_counter = 0

    all_results: List[Dict[str, Any]] = []

    for qi, query in enumerate(queries, 1):
        for nation in nations:
            for chunking in chunking_methods:
                for vec in vector_methods:
                    run_counter += 1
                    print("=" * 110)
                    print(
                        f"[RUN {run_counter}/{total_runs}] "
                        f"Q{qi} ({query[:40]}...) | nation={nation} | chunking={chunking} | vec={vec}"
                    )

                    res = run_stage4_evolution_v2(
                        query=query,
                        nation=nation,
                        chunking_method=chunking,
                        vector_method=vec,
                        k=k,
                        window_months=window_months,
                        candidate_n=candidate_n,
                        alpha=alpha,
                        lambda_decay=lambda_decay,
                        verbose=verbose_each_run,
                    )
                    res["query_index"] = qi
                    all_results.append(res)

                    out_path = run_dir / f"stage4_evo_v2_q{qi}_{nation}_{chunking}_{vec}_k{k}_N{candidate_n}.json"
                    print(f"[WRITE] Saving -> {out_path}")
                    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    combined = run_dir / "stage4_evo_v2_all_results.json"
    print(f"[WRITE] Saving combined -> {combined}")
    combined.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[ALL DONE] Saved {len(all_results)} runs into: {run_dir}")


if __name__ == "__main__":
    queries = [
        "How did the Prime Minister/President's rhetoric regarding the war between Israel and Hamas/Gaza develop/change between his first and last speech?",
        "How did the Prime Minister/President's rhetoric regarding immigration change between his first and last speech?",
    ]

    run_grid(
        queries=queries,
        k=5,
        window_months=8,
        candidate_n=800,
        alpha=0.3,
        lambda_decay=0.6,
        verbose_each_run=True,
    )
