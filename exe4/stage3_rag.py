from pathlib import Path
import pandas as pd

from exe3.stage3_retrieval import enrich_results, load_chunkpath_to_source
from exe4.stage3_hard_filtering import save_results as run_hard_filter
from exe4.stage3_time_decay_scoring import save_soft_decay_results as run_soft_decay
from exe4.stage0_llm_rag import (
    build_prompt,
    call_ollama,
    save_answer_to_txt,
    save_sources_to_txt,
    save_answer_to_excel,
    save_sources_to_excel
)
from exe4.utils import get_queries

def stage3_csv_path(
    *,
    approach: str,          # "hard_filter" / "soft_decay"
    corpus: str,            # "uk" / "us"
    query_index: int,
    chunking: str,
    embedding: str,
    k: int,
):
    base = Path("exe4/outputs/stage3_tables") / approach / corpus

    if approach == "hard_filter":
        fname = (
            f"hard_filter_results_{query_index}_"
            f"{chunking}_{embedding}_k={k}.csv"
        )
    else:
        fname = (
            f"soft_decay_results_{query_index}_"
            f"{chunking}_{embedding}_k={k}_a=0.3_l=0.6.csv"
        )

    return base / fname

def load_retrieved_from_stage3(csv_path: Path,use_soft):
    df = pd.read_csv(csv_path)

    # הפורמט המקורי: (row_index, chunk_path, score)
    retrieved = [
        (int(row["row_index"]), row["chunk_path"], float(row["final_score" if use_soft else "score"]))
        for _, row in df.iterrows()
    ]
    return retrieved

def run_rag_on_stage3(
    *,
    query: str,
    retrieved: list,
    chunk_method
):
    chunkpath_to_source = load_chunkpath_to_source()

    enriched = enrich_results(
        retrieved,
        chunkpath_to_source,
        max_chars=3000
        # chunking_method=chunk_method
    )

    prompt = build_prompt(query, enriched)
    answer = call_ollama(prompt)

    sources = [
        f"{c['source_file']} ({c['chunk']}) score={c['score']:.4f}"
        for c in enriched
    ]

    return answer, sources

if __name__ == "__main__":

    queries = [get_queries()[0]]

    # corpora = ["uk"]
    # approaches = {
    #     #"hard_filter": run_hard_filter,
    #     "soft_decay": run_soft_decay
    # }
    # ks = [5]
    # embeddings = ["st"]
    # chunkings = ["parentSon","fixed"]
    corpora = ["uk", "us"]
    approaches = {
        "hard_filter": run_hard_filter,
        "soft_decay": run_soft_decay
    }
    ks = [3,5,8]
    embeddings = ["st", "bm25"]
    chunkings = ["fixed", "parentSon"]
    from exe3.stage3_retrieval import change_chanking_method

    for corpus in corpora:
        for approach, stage3_runner in approaches.items():
            for k in ks:
                for embedding in embeddings:
                    for chunking in chunkings:
                        change_chanking_method(chunking)
                        for q_idx, query in enumerate(queries):

                            print(
                                f"\nCORPUS={corpus} | "
                                f"APPROACH={approach} | "
                                f"K={k} | EMB={embedding} | "
                                f"CHUNK={chunking} | "
                                f"Q={q_idx}"
                            )

                            csv_path = stage3_csv_path(
                                approach=approach,
                                corpus=corpus,
                                query_index=q_idx,
                                chunking=chunking,
                                embedding=embedding,
                                k=k
                            )

                            # --- Stage 3 אם חסר ---
                            if not csv_path.exists():
                                print(f"Stage 3 missing → generating (K={k})")

                                chunks_index_path = Path(
                                    f"exe4/united_{chunking}_chunk_index.json"
                                )

                                stage3_runner(
                                    query=query,
                                    chunks_index_path=chunks_index_path,
                                    chunking_method=chunking,
                                    save_path=csv_path.parent,
                                    query_index=q_idx,
                                    top_k=k,
                                    use_dense=(embedding == "st"),
                                    nation=corpus
                                )

                            # --- טעינת תוצאות ---
                            retrieved = load_retrieved_from_stage3(csv_path, approach=="soft_decay")

                            # --- RAG ---
                            answer, sources = run_rag_on_stage3(
                                query=query,
                                retrieved=retrieved,
                                chunk_method=chunking
                            )

                            # --- שמירה ---
                            prefix = (
                                f"exe4/outputs/stage3_tables/{approach}/{corpus}/"
                            )

                            save_answer_to_txt(
                                Path(prefix + "answers.txt"),
                                query, k, embedding, chunking, answer
                            )
                            save_sources_to_txt(
                                Path(prefix + "sources.txt"),
                                query, k, embedding, chunking, sources
                            )

                            excel_method = "dense" if embedding == "st" else "bm25"
                            excel_chunking = "parent-son" if chunking == "parentSon" else chunking

                            save_answer_to_excel(
                                Path(prefix + "answers.xlsx"),
                                query, k, excel_method, excel_chunking, answer
                            )
                            save_sources_to_excel(
                                Path(prefix + "sources.xlsx"),
                                query, k, excel_method, excel_chunking, sources
                            )
