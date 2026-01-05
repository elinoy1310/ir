# exe4/stage4_run_fast.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from exe4.stage4_evolution_fast import run_stage4_fast


def run_all_fast(
    *,
    queries: List[str],
    out_dir: Path = Path("exe4/out_stage4_fast"),
    k: int = 5,
    window_months: int = 8,
    candidate_n: int = 400,
    verbose_each_run: bool = True,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results: List[Dict[str, Any]] = []

    chunking_methods = ["fixed", "parent-son"]
    vector_methods = ["bm25", "dense"]
    nations = ["uk", "us"]

    total_runs = len(queries) * len(nations) * len(chunking_methods) * len(vector_methods)
    run_counter = 0

    for qi, query in enumerate(queries, 1):
        if query is None:
          continue

        for nation in nations:
            for chunking in chunking_methods:
                for vec in vector_methods:
                    run_counter += 1
                    print("=" * 110)
                    print(
                        f"[RUN {run_counter}/{total_runs}] "
                        f"Q{qi} | nation={nation} | chunking={chunking} | vec={vec} | "
                        f"k={k} | window_months={window_months} | candidate_n={candidate_n}"
                    )

                    res = run_stage4_fast(
                        query=query,
                        nation=nation,
                        chunking_method=chunking,
                        vector_method=vec,
                        k=k,
                        window_months=window_months,
                        candidate_n=candidate_n,
                        verbose=verbose_each_run,
                    )
                    res["query_index"] = qi
                    all_results.append(res)

                    out_path = out_dir / f"stage4_fast_q{qi}_{nation}_{chunking}_{vec}_k{k}_N{candidate_n}.json"
                    print(f"[WRITE] Saving -> {out_path}")
                    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    combined = out_dir / "stage4_fast_all_results.json"
    print(f"[WRITE] Saving combined -> {combined}")
    combined.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[ALL DONE] Saved {len(all_results)} runs into: {out_dir}")


if __name__ == "__main__":
    queries = [
    None,  # Q1 – כבר רץ, מדלגים
    "How did references to U.S. foreign policy and humanitarian concerns regarding Gaza change between the early and late periods of the corpus?",
    ]

    run_all_fast(
        queries=queries,
        k=5,
        window_months=8,
        candidate_n=400,   # אם חסר early/late לפעמים: תעלי ל-800
        verbose_each_run=True,
    )
