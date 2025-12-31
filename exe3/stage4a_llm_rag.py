# exe3/stage4_llm_rag.py
import requests
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ייבוא הפונקציות והקבועים מהקובץ המעודכן (stage3)
from stage3_retrieval import (
    load_chunkpath_to_source,
    load_bm25_store,
    load_dense_store,
    bm25_retrieve,
    dense_retrieve,
    enrich_results,
    uk_count,
    MODEL_NAME,
    change_chanking_method
)

# Ollama settings
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "gemma:2b"

# -------- LLM call --------
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
            f"[{i}] SOURCE_FILE: {c['source_file']}\n"
            f"CHUNK: {c['chunk']}\n"
            f"TEXT: {c['text']}\n"
        )
    print(f"\nBuilt prompt with {len(' '.join(ctx_lines))} context chunks.")
    return (
        "Answer ONLY using the CONTEXT below.\n"
        "If the answer is not in the context, say: \"I don't know based on the provided context.\" \n"
        "At the end, list the sources you used as: SOURCES: [1], [2], ...\n\n"
        f"QUESTION: {query}\n\n"
        "CONTEXT:\n"
        + "\n".join(ctx_lines)
    )

def run_rag(query: str, method: str = "hybrid", k: int = 5):
    # שימוש בפונקציות המיובאות
    if isinstance(query, tuple):
        query = query[1]
    uk_n = uk_count()
    chunkpath_to_source = load_chunkpath_to_source()
    X_bm25, vocab, bm25_names = load_bm25_store()
    X_emb, dense_names = load_dense_store()
    st_model = SentenceTransformer(MODEL_NAME)

    if method == "bm25":
        retrieved = bm25_retrieve(query, X_bm25, vocab, bm25_names, top_k=k)
    elif method == "dense":
        retrieved = dense_retrieve(query, X_emb, dense_names, st_model, top_k=k)
    # העשרת התוצאות באמצעות הפונקציה מ-stage3
    enriched = enrich_results(retrieved, chunkpath_to_source, max_chars=3500)

    # בניית הפרומפט וקריאה ל-LLM
    prompt = build_prompt(query, enriched)
    answer = call_ollama(prompt)

    # הצגת התוצאה
    print("\n" + "="*90)
    print(f"METHOD={method}  K={k}")
    print("QUESTION:", query)
    print("\nANSWER:\n", answer)
    print("\nTOP CONTEXT SOURCES:")
    for i, c in enumerate(enriched, 1):
        print(f"[{i}] {c['source_file']}  ({c['chunk']})  score={c['score']:.4f}")

def run_rag_with_multiple_configs(queries: list, chunk_method: str):
    # קודם תעדכן את שיטת הצ'אנקינג
    change_chanking_method(chunk_method)

    # ריצה עבור כל שיטה (DENSE ו-BM25) עם כל ערך של k
    for method in ["dense", "bm25"]:
        for k in [3, 5, 8]:
            for i, query in enumerate(queries, 1):
                print(f"\nRunning RAG (Run {i}): Method={method}, k={k}, Chunking={chunk_method}")
                if isinstance(query, tuple):
                    label, q_text = query
                    print(f"\n[{label}] {q_text}\n")
                else:
                    q_text = query
                run_rag(q_text, method=method, k=k)
                print(f"End of Run {i} for {method} with k={k}")
                print("="*90)

if __name__ == "__main__":

    queries=[ "On what dates did the British Prime Minister deliver his speech on the defense budget?",
	"what was the main argument regarding the immigration bill that was presented?",
	"What three industrial sectors were mentioned as the main victims of the new trade policy that was presented",
	"What organizations were mentioned by the speakers as supporting the proposed reform of the health system?",
	"How does the rhetoric on climate change vary between different speakers; is the emphasis on economic opportunity or existential crisis",
	"What is the central tension that emerges from the speeches between the need for national security and the protection of citizens’ privacy in the digital age?",
	"How is the state’s moral responsibility towards refugees and asylum seekers described, and what are the ethical (rather than economic) arguments given for and against their absorption?",
	"In what ways did speakers link investment in education to reducing future crime, and was there consensus on this issue?",

 "Which renewable energy technologies were explicitly mentioned in the debates, and in what national contexts were they discussed?",
    "What specific funding amounts or investment figures were mentioned in relation to energy, education, or decarbonisation programs?",
    "Which government departments or public bodies were explicitly mentioned as responsible for energy or climate-related policies?",
    "What geographic locations (regions or nations) were explicitly referenced in discussions about infrastructure or energy projects?",

    "How do speakers justify continued investment in both renewable energy and traditional energy sources, and what tensions arise between these approaches?",
    "How is energy security framed in the debates: primarily as an economic issue, a national security concern, or an environmental responsibility?",
    "How do speakers describe the role of government versus the private sector in driving technological innovation and infrastructure development?",
    "What arguments are used to link long-term public investment, such as in education or energy infrastructure, to future national resilience and prosperity?"
    ]
    # run_rag_with_multiple_configs(queries, chunk_method="fixed")  # שיטת צ'אנקינג ראשונית
    
    # # הרץ עם שינוי שיטת הצ'אנקינג ל"parent-son"
    # run_rag_with_multiple_configs(queries, chunk_method="parent-son")
    method="bm25"
    k=8
    chunk_method="fixed"
    for i, query in enumerate(queries, 1):
                print(f"\nRunning RAG (Run {i}): Method={method}, k={k}, Chunking={chunk_method}")
                run_rag(query, method=method, k=k)
                print(f"End of Run {i} for {method} with k={k}")
                print("="*90)
    run_rag_with_multiple_configs(queries, chunk_method="parent-son")