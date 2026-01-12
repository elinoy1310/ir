# # exe4/stage4_run_all_evolution_queries.py
# from __future__ import annotations

# import json
# from pathlib import Path
# from typing import List, Dict, Any

# from exe4.stage4_temporal_evolutionary_rag import run_stage4_temporal_evolution


# def run_all_stage4(
#     *,
#     queries: List[str],
#     k: int = 5,
#     window_months: int = 8,
#     out_dir: Path = Path("exe4/out_stage4"),
# ):
#     out_dir.mkdir(parents=True, exist_ok=True)

#     chunking_methods = ["fixed", "parent-son"]
#     vector_methods = ["bm25", "dense"]
#     nations = ["uk", "us"]

#     all_results: List[Dict[str, Any]] = []

#     for qi, query in enumerate(queries, 1):
#         for nation in nations:
#             for chunking in chunking_methods:
#                 for vec in vector_methods:
#                     print("=" * 110)
#                     print(f"Q{qi} | nation={nation} | chunking={chunking} | vec={vec} | k={k} | window_months={window_months}")
#                     res = run_stage4_temporal_evolution(
#                         query=query,
#                         nation=nation,
#                         chunking_method=chunking,
#                         vector_method=vec,
#                         k=k,
#                         window_months=window_months,
#                     )

#                     res["query_index"] = qi
#                     all_results.append(res)

#                     out_path = out_dir / f"stage4_q{qi}_{nation}_{chunking}_{vec}_k{k}.json"
#                     out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

#     combined = out_dir / "stage4_all_results.json"
#     combined.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
#     print(f"\nSaved {len(all_results)} runs into: {out_dir}")


# if __name__ == "__main__":
#     # שאילתת אבולוציה מההוראות + תוסיפי עוד שלך
#     queries = [
#         "How did the Prime Minister/President's rhetoric regarding the war between Israel and Hamas/Gaza develop/change between his first and last speech?",
#     ]
#     run_all_stage4(queries=queries, k=5, window_months=8)
