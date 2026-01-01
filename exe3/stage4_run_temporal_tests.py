# exe3/stage4_run_temporal_tests.py

from pathlib import Path
from contextlib import redirect_stdout
from stage4_llm_rag import run_rag_with_multiple_configs

# =========================
# Temporal query groups
# =========================

HARD_FILTER = [
    ("HARD_FILTER", "What was the specific budget allocated to security in 2024?"),
]

RECENCY = [
    ("RECENCY", "What is the current official position regarding the State of Israel?"),
    ("RECENCY", "What is the current official position regarding Hamas/Gaza?"),
    ("RECENCY", "Has the official position in the last quarter of 2023 changed relative to the official position in the last quarter of 2025?"),
]

EVOLUTION = [
    ("EVOLUTION", "How did the Prime Minister/President's rhetoric regarding the war between Israel and Hamas/Gaza change between his first and last speech?"),
]

AMBIGUITY = [
    ("AMBIGUITY", "Who is the Minister of Defense/Secretary of Defense?"),
]


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
                run_rag_with_multiple_configs(queries, chunk_method="fixed",answers_path_no_prefix="exe4/outputs/stage1/answers",
                                              sources_path_no_prefix="exe4/outputs/stage1/sources")
                # run_rag_with_multiple_configs(queries, chunk_method="parent-son",answers_path_no_prefix="exe4/outputs/stage1/answers",
                #                               sources_path_no_prefix="exe4/outputs/stage1/sources")

        print(f"✅ Saved output to: {out_path}")
