# exe4/run_temporal_tests
from pathlib import Path
from contextlib import redirect_stdout
from exe4.temporal_llm_rag import run_rag_with_multiple_configs_temporal

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

if __name__ == "__main__":
    out_dir = Path("exe4") / "outputs_stage2"
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = [
        ("HARD_FILTER", HARD_FILTER),
        ("RECENCY", RECENCY),
        ("EVOLUTION", EVOLUTION),
        ("AMBIGUITY", AMBIGUITY),
    ]

    for group_name, queries in groups:
        out_path = out_dir / f"{group_name}_temporal.txt"
        with out_path.open("w", encoding="utf-8") as f:
            with redirect_stdout(f):
                print(f"### STAGE 2 TEMPORAL GROUP: {group_name}")
                run_rag_with_multiple_configs_temporal(queries, chunk_method="fixed")
        print(f"✅ Saved: {out_path}")
