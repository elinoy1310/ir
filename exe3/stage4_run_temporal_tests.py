# exe3/stage4_run_temporal_tests.py

from pathlib import Path
from contextlib import redirect_stdout
from stage4_llm_rag import run_rag_with_multiple_configs




# =========================
# Main runner
# =========================

if __name__ == "__main__":
    # Output directory (kept separate from previous exercises)
    out_dir = Path("exe3") / "exe4_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = [
        ("HARD_FILTER", HARD_FILTER),
        ("RECENCY", RECENCY),
        ("EVOLUTION", EVOLUTION),
        ("AMBIGUITY", AMBIGUITY),
    ]

    for group_name, queries in groups:
        out_path = out_dir / f"{group_name}_fixed.txt"

        with out_path.open("w", encoding="utf-8") as f:
            with redirect_stdout(f):
                print("=" * 90)
                print(f"### TEMPORAL TEST GROUP: {group_name}")
                print("=" * 90)
                run_rag_with_multiple_configs(queries,k_list=[3],method_list=["dense","bm25"], chunk_method="fixed",answers_path_no_prefix="exe4/outputs/stage1/answers",
                                              sources_path_no_prefix="exe4/outputs/stage1/sources",nation="uk")
                # run_rag_with_multiple_configs(queries, chunk_method="parent-son",answers_path_no_prefix="exe4/outputs/stage1/answers",
                #                               sources_path_no_prefix="exe4/outputs/stage1/sources")

        print(f"✅ Saved output to: {out_path}")
