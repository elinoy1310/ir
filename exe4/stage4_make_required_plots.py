# stage4_make_required_plots.py
# Full analysis suite (1–10) for Stage 4 (Temporal RAG, evo_v2)
# Produces a set of plots directly aligned with the assignment requirements.

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
INPUT_JSON = Path(r"out_stage4_evolution_v2\run_20260105_172733\stage4_evo_v2_all_results.json")
OUT_DIR = Path("stage4_plots_full")

# Methods order for consistent axes
METHOD_ORDER = [
    ("bm25", "fixed"),
    ("dense", "fixed"),
    ("bm25", "parent-son"),
    ("dense", "parent-son"),
]

PHASES = ["early", "late", "change"]

# "No-answer" heuristic
NO_ANSWER_PATTERNS = [
    r"^\s*$",
    r"\bi don't\b",
    r"\bi cant\b",
    r"\bi can't\b",
    r"\bi cannot\b",
    r"\bcan't answer\b",
    r"\bcannot answer\b",
    r"\binsufficient evidence\b",
    r"\bwas not provided\b",
    r"\bwas not mentioned\b",
    r"\bnot mentioned\b",
    r"\bnot provided\b",
    r"\bno information\b",
    r"\bdoes not mention\b",
    r"\bthe context does not\b",
]

# Tokenization / overlap knobs
WORD_RE = re.compile(r"[A-Za-z0-9']+")
USEFUL_CHUNK_COVERAGE_THRESHOLD = 0.05  # for signal-to-noise (analysis #8)

# For failure modes (analysis #10)
LOW_FINAL_SCORE_THRESHOLD = 0.70     # median score below this => "low_score"
HIGH_LEAKAGE_THRESHOLD = 0.25        # leakage rate above this => "time_leak"


# =========================
# HELPERS
# =========================
def ensure_out():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def savefig(name: str):
    p = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.close()
    print("Saved:", p)

def method_label(vector_method: str, chunking_method: str, k: int | None = None) -> str:
    # We keep k in label only if needed. (Here k is constant, but we keep the function flexible.)
    base = f"{vector_method} | {chunking_method}"
    return base if k is None else f"{base} | k={k}"

def methods_in_order(k: int | None = None) -> list[str]:
    return [method_label(v, c, k=k) for (v, c) in METHOD_ORDER]

def load_runs(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))

def parse_dt(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def parse_year(ts: str) -> int | None:
    dt = parse_dt(ts)
    return dt.year if dt else None

def mid_date(window: list[str]) -> datetime | None:
    # window like ["2023-06-28","2024-02-23"]
    if not window or len(window) != 2:
        return None
    a = parse_dt(window[0])
    b = parse_dt(window[1])
    if not a or not b:
        return None
    return a + (b - a) / 2

def in_window(ts: str, window: list[str]) -> bool:
    dt = parse_dt(ts)
    if not dt or not window or len(window) != 2:
        return False
    a = parse_dt(window[0])
    b = parse_dt(window[1])
    if not a or not b:
        return False
    return a <= dt <= b

def is_no_answer(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(re.search(p, t) for p in NO_ANSWER_PATTERNS)

def tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(text or "")}

def coverage(answer: str, sources_text: str) -> float:
    a = tokenize(answer)
    if not a:
        return 0.0
    s = tokenize(sources_text)
    return len(a & s) / len(a)

def median(values: list[float]) -> float:
    if not values:
        return float("nan")
    vs = sorted(values)
    n = len(vs)
    mid = n // 2
    if n % 2 == 1:
        return vs[mid]
    return (vs[mid - 1] + vs[mid]) / 2.0

def quantile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    vs = sorted(values)
    n = len(vs)
    if n == 1:
        return vs[0]
    idx = p * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    w = idx - lo
    return vs[lo] * (1 - w) + vs[hi] * w

def iqr(values: list[float]) -> float:
    if not values:
        return float("nan")
    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)
    return q3 - q1

def safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)

def normalize_minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    if mx == mn:
        return [0.0 for _ in values]
    return [(v - mn) / (mx - mn) for v in values]


# =========================
# DATA EXTRACTION
# =========================
def phase_sources(run: dict, phase: str) -> list[dict]:
    if phase == "early":
        return run.get("early_sources", []) or []
    if phase == "late":
        return run.get("late_sources", []) or []
    # change: use both early+late sources as evidence pool
    return (run.get("early_sources", []) or []) + (run.get("late_sources", []) or [])

def phase_answer(run: dict, phase: str) -> str:
    if phase == "early":
        return run.get("early_answer", "") or ""
    if phase == "late":
        return run.get("late_answer", "") or ""
    return run.get("change_answer", "") or ""

def phase_window(run: dict, phase: str) -> list[str] | None:
    if phase == "early":
        return run.get("early_window", None)
    if phase == "late":
        return run.get("late_window", None)
    return None  # change is comparing, not a single window

def per_source_final_scores(run: dict, phase: str) -> list[float]:
    return [float(s.get("final_score")) for s in phase_sources(run, phase) if s.get("final_score") is not None]

def run_level_median_score(run: dict, phase: str) -> float:
    vals = per_source_final_scores(run, phase)
    return median(vals)

def run_level_iqr_score(run: dict, phase: str) -> float:
    vals = per_source_final_scores(run, phase)
    return iqr(vals)

def leakage_rate(run: dict, phase: str) -> float:
    window = phase_window(run, phase)
    if not window:
        return 0.0
    srcs = phase_sources(run, phase)
    if not srcs:
        return 0.0
    out = 0
    total = 0
    for s in srcs:
        ts = s.get("timestamp", "")
        if not ts:
            continue
        total += 1
        if not in_window(ts, window):
            out += 1
    return (out / total) if total else 0.0

def temporal_bias_years(run: dict, phase: str) -> float:
    window = phase_window(run, phase)
    if not window:
        return 0.0
    center = mid_date(window)
    if not center:
        return 0.0
    ys = []
    for s in phase_sources(run, phase):
        y = parse_year(s.get("timestamp", ""))
        if y is not None:
            ys.append(y)
    if not ys:
        return 0.0
    mean_y = sum(ys) / len(ys)
    # bias = absolute difference from window center year (in years)
    return abs(mean_y - center.year)

def rerank_delta(run: dict, phase: str) -> float:
    """
    Analysis #3: "reranking contribution"
    We compute: mean(final_score - normalized(sim_raw)) within a run+phase.
    This is NOT a universal metric across methods, but it shows how final_score
    differs from raw similarity once time weighting is applied.
    """
    srcs = phase_sources(run, phase)
    sim_raw = [float(s.get("sim_raw")) for s in srcs if s.get("sim_raw") is not None and s.get("final_score") is not None]
    fin = [float(s.get("final_score")) for s in srcs if s.get("sim_raw") is not None and s.get("final_score") is not None]
    if not sim_raw or not fin or len(sim_raw) != len(fin):
        return float("nan")
    sim_norm = normalize_minmax(sim_raw)
    deltas = [f - r for (f, r) in zip(fin, sim_norm)]
    return safe_mean(deltas) if deltas else float("nan")

def evidence_alignment(run: dict, phase: str) -> float:
    """
    Analysis #4: coverage of answer tokens by retrieved chunk texts.
    For change: evidence pool = early+late sources.
    """
    ans = phase_answer(run, phase)
    srcs = phase_sources(run, phase)
    pool = "\n".join([(s.get("text", "") or "") for s in srcs])
    return coverage(ans, pool)

def signal_to_noise(run: dict, phase: str) -> float:
    """
    Analysis #8: ratio of "useful" chunks among retrieved chunks.
    We mark a chunk useful if its text covers at least X% of answer tokens.
    """
    ans = phase_answer(run, phase)
    a = tokenize(ans)
    if not a:
        return 0.0
    srcs = phase_sources(run, phase)
    if not srcs:
        return 0.0

    useful = 0
    total = 0
    for s in srcs:
        txt = s.get("text", "") or ""
        cov = coverage(ans, txt)
        total += 1
        if cov >= USEFUL_CHUNK_COVERAGE_THRESHOLD:
            useful += 1
    return useful / total if total else 0.0

def top1_vs_topk_gain(run: dict, phase: str) -> float:
    """
    Analysis #5: k sensitivity even when k is fixed (k=5).
    gain = mean(topk final_score) - top1 final_score
    """
    srcs = phase_sources(run, phase)
    fin = [float(s.get("final_score")) for s in srcs if s.get("final_score") is not None]
    if not fin:
        return float("nan")
    top1 = fin[0]  # sources are already ranked
    topk = safe_mean(fin)
    return topk - top1


# =========================
# PLOTTING UTILITIES
# =========================
def barplot_simple(title: str, xlabel: str, ylabel: str, labels: list[str], values: list[float], fname: str, rotate: int = 20):
    plt.figure(figsize=(11, 5))
    x = list(range(len(labels)))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=rotate, ha="right")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    savefig(fname)

def barplot_grouped(title: str, ylabel: str, labels: list[str], series: dict[str, list[float]], fname: str, rotate: int = 20):
    """
    series: name -> values aligned with labels
    """
    plt.figure(figsize=(12, 5))
    x = list(range(len(labels)))
    width = 0.8 / max(1, len(series))
    names = list(series.keys())
    for i, name in enumerate(names):
        vals = series[name]
        offs = [xi - 0.4 + (i + 0.5) * width for xi in x]
        plt.bar(offs, vals, width=width, label=name)
    plt.xticks(x, labels, rotation=rotate, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    savefig(fname)

def heatmap(title: str, xlabels: list[str], ylabels: list[str], matrix: list[list[float]], fname: str):
    plt.figure(figsize=(9, 5))
    plt.imshow(matrix, aspect="auto")
    plt.xticks(range(len(xlabels)), xlabels, rotation=20, ha="right")
    plt.yticks(range(len(ylabels)), ylabels)
    plt.title(title)
    plt.colorbar()
    savefig(fname)


# =========================
# ANALYSES 1–10 (PLOTS)
# =========================
def analysis_1_stability_iqr(runs: list[dict]):
    # IQR of final_score across retrieved chunks (run-level), then aggregate by method/phase via median
    methods = methods_in_order()
    for phase in ["early", "late"]:
        by_method = defaultdict(list)
        for r in runs:
            m = method_label(r["vector_method"], r["chunking_method"])
            by_method[m].append(run_level_iqr_score(r, phase))
        vals = [median([v for v in by_method[m] if v == v]) for m in methods]
        barplot_simple(
            title=f"Stability by method (IQR of final_score) - {phase}",
            xlabel="method",
            ylabel="IQR(final_score)  (lower = more stable)",
            labels=methods,
            values=vals,
            fname=f"a1_stability_iqr_{phase}.png",
        )

def analysis_2_temporal_leakage(runs: list[dict]):
    # % chunks out of time window (run-level), aggregate by median per method/phase
    methods = methods_in_order()
    for phase in ["early", "late"]:
        by_method = defaultdict(list)
        for r in runs:
            m = method_label(r["vector_method"], r["chunking_method"])
            by_method[m].append(leakage_rate(r, phase))
        vals = [median(by_method[m]) for m in methods]
        barplot_simple(
            title=f"Temporal leakage by method - {phase}",
            xlabel="method",
            ylabel="% retrieved chunks OUT of window",
            labels=methods,
            values=vals,
            fname=f"a2_temporal_leakage_{phase}.png",
        )

def analysis_3_reranking_contribution(runs: list[dict]):
    # delta = mean(final_score - normalized(sim_raw)) per run; aggregate median by method/phase
    methods = methods_in_order()
    for phase in ["early", "late"]:
        by_method = defaultdict(list)
        for r in runs:
            m = method_label(r["vector_method"], r["chunking_method"])
            d = rerank_delta(r, phase)
            if d == d:  # not NaN
                by_method[m].append(d)
        vals = [median(by_method[m]) if by_method[m] else float("nan") for m in methods]
        barplot_simple(
            title=f"Reranking contribution proxy (final_score - norm(sim_raw)) - {phase}",
            xlabel="method",
            ylabel="delta (higher => more time-weight effect)",
            labels=methods,
            values=vals,
            fname=f"a3_rerank_delta_{phase}.png",
        )

def analysis_4_evidence_alignment(runs: list[dict]):
    # coverage(answer tokens by retrieved texts), median per method for each phase
    methods = methods_in_order()
    for phase in ["early", "late", "change"]:
        by_method = defaultdict(list)
        for r in runs:
            m = method_label(r["vector_method"], r["chunking_method"])
            by_method[m].append(evidence_alignment(r, phase))
        vals = [median(by_method[m]) for m in methods]
        barplot_simple(
            title=f"Evidence alignment (token coverage) - {phase}",
            xlabel="method",
            ylabel="coverage(answer in retrieved texts)",
            labels=methods,
            values=vals,
            fname=f"a4_evidence_alignment_{phase}.png",
        )

def analysis_5_k_sensitivity_top1_vs_topk(runs: list[dict]):
    # gain = mean(topk final_score) - top1 final_score (run-level), median per method/phase
    methods = methods_in_order()
    for phase in ["early", "late"]:
        by_method = defaultdict(list)
        for r in runs:
            m = method_label(r["vector_method"], r["chunking_method"])
            g = top1_vs_topk_gain(r, phase)
            if g == g:
                by_method[m].append(g)
        vals = [median(by_method[m]) if by_method[m] else float("nan") for m in methods]
        barplot_simple(
            title=f"Top-1 vs Top-k gain (k={runs[0].get('k', 'N/A')}) - {phase}",
            xlabel="method",
            ylabel="mean(topk) - top1 (final_score)",
            labels=methods,
            values=vals,
            fname=f"a5_top1_vs_topk_gain_{phase}.png",
        )

def analysis_6_conditional_superiority_heatmap(runs: list[dict]):
    """
    Dense advantage over BM25 across conditions:
    rows = nation + phase
    cols = chunking (fixed, parent-son)
    cell = median(run_median_score_dense - run_median_score_bm25)
    """
    rows = []
    nations = sorted({r["nation"] for r in runs})
    phases = ["early", "late"]
    chunkings = ["fixed", "parent-son"]

    # Build lookup: (nation, phase, chunking, vector_method) -> list of run-level median scores
    scores = defaultdict(list)
    for r in runs:
        for phase in phases:
            key = (r["nation"], phase, r["chunking_method"], r["vector_method"])
            scores[key].append(run_level_median_score(r, phase))

    matrix = []
    for nation in nations:
        for phase in phases:
            rows.append(f"{nation}_{phase}")
            row_vals = []
            for ch in chunkings:
                dense_list = scores.get((nation, phase, ch, "dense"), [])
                bm25_list = scores.get((nation, phase, ch, "bm25"), [])
                if not dense_list or not bm25_list:
                    row_vals.append(0.0)
                else:
                    row_vals.append(median(dense_list) - median(bm25_list))
            matrix.append(row_vals)

    heatmap(
        title="Dense advantage over BM25 (median final_score delta)",
        xlabels=chunkings,
        ylabels=rows,
        matrix=matrix,
        fname="a6_dense_advantage_heatmap.png",
    )

def analysis_7_temporal_bias(runs: list[dict]):
    # temporal bias = |mean(retrieved_years) - center_year|, median per method/phase
    methods = methods_in_order()
    for phase in ["early", "late"]:
        by_method = defaultdict(list)
        for r in runs:
            m = method_label(r["vector_method"], r["chunking_method"])
            by_method[m].append(temporal_bias_years(r, phase))
        vals = [median(by_method[m]) for m in methods]
        barplot_simple(
            title=f"Temporal bias by method - {phase}",
            xlabel="method",
            ylabel="|mean_year - window_center_year| (years)",
            labels=methods,
            values=vals,
            fname=f"a7_temporal_bias_{phase}.png",
        )

def analysis_8_signal_to_noise(runs: list[dict]):
    # ratio useful chunks among retrieved, median per method/phase
    methods = methods_in_order()
    for phase in ["early", "late", "change"]:
        by_method = defaultdict(list)
        for r in runs:
            m = method_label(r["vector_method"], r["chunking_method"])
            by_method[m].append(signal_to_noise(r, phase))
        vals = [median(by_method[m]) for m in methods]
        barplot_simple(
            title=f"Signal-to-noise (useful chunk ratio) - {phase}",
            xlabel="method",
            ylabel="useful_chunks / retrieved_chunks",
            labels=methods,
            values=vals,
            fname=f"a8_signal_to_noise_{phase}.png",
        )

def analysis_9_query_sensitivity(runs: list[dict]):
    """
    Sensitivity across query_index (q1 vs q2):
    For each method+phase: compute absolute difference between query_index medians (or std).
    """
    methods = methods_in_order()
    for phase in ["early", "late"]:
        # method -> query_index -> list of run median scores
        by = defaultdict(lambda: defaultdict(list))
        for r in runs:
            m = method_label(r["vector_method"], r["chunking_method"])
            qi = int(r.get("query_index", 0))
            by[m][qi].append(run_level_median_score(r, phase))

        # If you have exactly 2 queries: use abs(median(q1) - median(q2))
        vals = []
        for m in methods:
            qkeys = sorted(by[m].keys())
            if len(qkeys) >= 2:
                a = median(by[m][qkeys[0]])
                b = median(by[m][qkeys[1]])
                vals.append(abs(a - b))
            elif len(qkeys) == 1:
                vals.append(0.0)
            else:
                vals.append(float("nan"))

        barplot_simple(
            title=f"Query sensitivity (|median score q1 - q2|) - {phase}",
            xlabel="method",
            ylabel="absolute difference in median final_score",
            labels=methods,
            values=vals,
            fname=f"a9_query_sensitivity_{phase}.png",
        )

def analysis_10_failure_modes(runs: list[dict]):
    """
    Failure modes by method (counts), using simple heuristics:
    - no_answer: answer is empty/declines
    - time_leak: leakage_rate > HIGH_LEAKAGE_THRESHOLD
    - low_score: run-level median final_score < LOW_FINAL_SCORE_THRESHOLD
    We report counts for early+late+change (stacked bars).
    """
    methods = methods_in_order()
    categories = ["ok", "no_answer", "time_leak", "low_score"]

    # method -> category -> count
    counts = defaultdict(lambda: Counter())

    for r in runs:
        m = method_label(r["vector_method"], r["chunking_method"])

        # We score per phase; accumulate categories across phases (so you see patterns)
        for phase in ["early", "late", "change"]:
            ans = phase_answer(r, phase)
            if is_no_answer(ans):
                counts[m]["no_answer"] += 1
                continue

            if phase in ["early", "late"] and leakage_rate(r, phase) > HIGH_LEAKAGE_THRESHOLD:
                counts[m]["time_leak"] += 1
                continue

            if phase in ["early", "late"]:
                med = run_level_median_score(r, phase)
                if med == med and med < LOW_FINAL_SCORE_THRESHOLD:
                    counts[m]["low_score"] += 1
                    continue

            counts[m]["ok"] += 1

    # Plot stacked bars
    plt.figure(figsize=(12, 5))
    x = list(range(len(methods)))
    bottom = [0] * len(methods)

    for cat in categories:
        vals = [counts[m][cat] for m in methods]
        plt.bar(x, vals, bottom=bottom, label=cat)
        bottom = [b + v for (b, v) in zip(bottom, vals)]

    plt.xticks(x, methods, rotation=20, ha="right")
    plt.ylabel("# phase-results")
    plt.title("Failure modes by method (stacked across phases)")
    plt.legend()
    savefig("a10_failure_modes_stacked.png")


# =========================
# ALSO KEEP: assignment-core plots
# (final_score med/IQR, year distribution, no-answer counts)
# =========================
def core_final_score_median_iqr(runs: list[dict]):
    methods = methods_in_order()
    for phase in ["early", "late"]:
        med_vals = []
        iqr_low = []
        iqr_high = []

        # Collect per-method all source final_scores (global distribution)
        by_method = defaultdict(list)
        for r in runs:
            m = method_label(r["vector_method"], r["chunking_method"])
            by_method[m].extend(per_source_final_scores(r, phase))

        for m in methods:
            vals = by_method.get(m, [])
            med = median(vals)
            q1 = quantile(vals, 0.25) if vals else float("nan")
            q3 = quantile(vals, 0.75) if vals else float("nan")
            med_vals.append(med)
            iqr_low.append(max(0.0, med - q1) if q1 == q1 else 0.0)
            iqr_high.append(max(0.0, q3 - med) if q3 == q3 else 0.0)

        plt.figure(figsize=(11, 5))
        x = list(range(len(methods)))
        plt.bar(x, med_vals, yerr=[iqr_low, iqr_high], capsize=6)
        plt.xticks(x, methods, rotation=20, ha="right")
        plt.ylabel("final_score (median, IQR)")
        plt.title(f"Final_score comparison by method ({phase})")
        savefig(f"core_final_score_median_iqr_{phase}.png")

def core_retrieved_years_distribution(runs: list[dict]):
    for phase in ["early", "late"]:
        # Gather counts of years per method
        years = defaultdict(Counter)
        all_years = set()

        for r in runs:
            m = method_label(r["vector_method"], r["chunking_method"])
            for s in phase_sources(r, phase):
                y = parse_year(s.get("timestamp", ""))
                if y is not None:
                    years[m][y] += 1
                    all_years.add(y)

        if not all_years:
            continue

        ys = sorted(all_years)
        methods = methods_in_order()
        x = list(range(len(ys)))
        bar_w = 0.18

        plt.figure(figsize=(11, 5))
        for i, m in enumerate(methods):
            counts = [years[m].get(y, 0) for y in ys]
            x_off = [xi + (i - 1.5) * bar_w for xi in x]
            plt.bar(x_off, counts, width=bar_w, label=m)

        plt.xticks(x, [str(y) for y in ys])
        plt.ylabel("# retrieved chunks")
        plt.title(f"Retrieved chunk years distribution ({phase})")
        plt.legend()
        savefig(f"core_retrieved_years_{phase}.png")

def core_no_answer_counts(runs: list[dict]):
    counts = defaultdict(lambda: {"early": 0, "late": 0, "change": 0})
    for r in runs:
        m = method_label(r["vector_method"], r["chunking_method"])
        for ph in ["early", "late", "change"]:
            if is_no_answer(phase_answer(r, ph)):
                counts[m][ph] += 1

    methods = methods_in_order()
    series = {
        "early": [counts[m]["early"] for m in methods],
        "late": [counts[m]["late"] for m in methods],
        "change": [counts[m]["change"] for m in methods],
    }
    barplot_grouped(
        title="No-answer counts by method (heuristic)",
        ylabel="# no-answer runs",
        labels=methods,
        series=series,
        fname="core_no_answer_counts.png",
    )


# =========================
# MAIN
# =========================
def main():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Missing {INPUT_JSON} (edit INPUT_JSON in this script or put file next to it).")

    ensure_out()
    runs = load_runs(INPUT_JSON)
    print(f"Loaded {len(runs)} runs from {INPUT_JSON}")

    # Core plots (assignment-friendly)
    core_final_score_median_iqr(runs)
    core_retrieved_years_distribution(runs)
    core_no_answer_counts(runs)

    # Analyses 1–10
    analysis_1_stability_iqr(runs)
    analysis_2_temporal_leakage(runs)
    analysis_3_reranking_contribution(runs)
    analysis_4_evidence_alignment(runs)
    analysis_5_k_sensitivity_top1_vs_topk(runs)
    analysis_6_conditional_superiority_heatmap(runs)
    analysis_7_temporal_bias(runs)
    analysis_8_signal_to_noise(runs)
    analysis_9_query_sensitivity(runs)
    analysis_10_failure_modes(runs)

    print("\nDone. Plots are in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
