# exe4/stage1_plot.py
# ==========================================
# RAG Temporal & Failure Analysis Pipeline
# ==========================================

import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. Config
# ==========================================

FILES_CONFIG = [
    {
        'answers': r'exe4\outputs\stage1\uk\answers.txt',
        'sources': r'exe4\outputs\stage1\uk\sources.txt',
        'corpus': 'UK'
    },
    # {
    #     'answers': r'exe4\outputs\stage1\us\answers.txt',
    #     'sources': r'exe4\outputs\stage1\us\sources.txt',
    #     'corpus': 'US'
    # },
]

OUTPUT_DIR = r'exe4\outputs\stage1\uk\for_plot'
# OUTPUT_DIR = r'exe4\outputs\stage1\us\for_plot'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEPARATOR = '=' * 120


# ==========================================
# 2. Parsing Engine
# ==========================================

def parse_rag_data(config):
    answers, sources = [], []

    # ---------- Parse Answers ----------
    if os.path.exists(config['answers']):
        with open(config['answers'], encoding='utf-8') as f:
            blocks = f.read().split(SEPARATOR)

        for block in blocks:
            if not block.strip():
                continue

            q = re.search(r'QUERY:\s*(.*?)\n', block)
            k = re.search(r'K\s*=\s*(\d+)', block)
            m = re.search(r'(DENSE|BM25) with (fixed|parent-son) chunking', block)

            if q and k and m and 'SOURCES:' in block:
                ans_text = block[m.end():block.find('SOURCES:')].strip()
                answers.append({
                    'corpus': config['corpus'],
                    'query': q.group(1).strip(),
                    'k': int(k.group(1)),
                    'method': m.group(1),
                    'chunking': m.group(2),
                    'answer': ans_text
                })

    # ---------- Parse Sources ----------
    if os.path.exists(config['sources']):
        with open(config['sources'], encoding='utf-8') as f:
            blocks = f.read().split(SEPARATOR)

        for block in blocks:
            if not block.strip():
                continue

            q = re.search(r'QUERY:\s*(.*?)\n', block)
            k = re.search(r'K\s*=\s*(\d+)', block)
            m = re.search(r'(DENSE|BM25) with (fixed|parent-son) chunking', block)
            srcs = re.findall(r'-\s*(.*?)\s+score=([\d.]+)', block)

            if q and k and m:
                for rank, (path, score) in enumerate(srcs, start=1):
                    date_match = re.search(r'(\d{4})-\d{2}-\d{2}', path)
                    year = date_match.group(1) if date_match else None

                    sources.append({
                        'corpus': config['corpus'],
                        'query': q.group(1).strip(),
                        'k': int(k.group(1)),
                        'method': m.group(1),
                        'chunking': m.group(2),
                        'rank': rank,
                        'score': float(score),
                        'year': year,
                        'file': path
                    })

    return pd.DataFrame(answers), pd.DataFrame(sources)


# ==========================================
# 3. Failure Analysis
# ==========================================

def analyze_temporal_failures(df_ans, df_src):
    failures = []

    if df_src.empty or df_ans.empty:
        return pd.DataFrame()

    # --- A. Mixed Temporal Context ---
    for key, group in df_src.dropna(subset=['year']).groupby(
        ['corpus', 'query', 'k', 'method', 'chunking']
    ):
        years = sorted(group['year'].unique())
        if len(years) > 1:
            failures.append({
                'Type': 'Mixed Temporal Context',
                'Query': key[1],
                'Corpus': key[0],
                'Config': f'K={key[2]}, {key[3]}, {key[4]}',
                'Details': f'Years retrieved: {years}'
            })

    # --- B. Entity Drift ---
    for (corp, query), group in df_ans.groupby(['corpus', 'query']):
        entities = set(re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', ' '.join(group['answer'])))
        if len(entities) > 1:
            failures.append({
                'Type': 'Entity Drift',
                'Query': query,
                'Corpus': corp,
                'Config': 'Across settings',
                'Details': f'Entities found: {sorted(entities)}'
            })

    # --- C. Recency Bias ---
    for key, group in df_src.dropna(subset=['year']).groupby(['corpus', 'query', 'method']):
        group = group.sort_values('year', ascending=False)
        if len(group) > 1:
            newest = group.iloc[0]
            if any(group.iloc[1:]['score'] > newest['score']):
                failures.append({
                    'Type': 'Recency Failure',
                    'Query': key[1],
                    'Corpus': key[0],
                    'Config': f'Method={key[2]}',
                    'Details': 'Older documents scored higher than newer ones'
                })

    return pd.DataFrame(failures)


# ==========================================
# 4. Visualization
# ==========================================

def create_visualizations(df_src):
    df = df_src.dropna(subset=['year'])

    if df.empty:
        return

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='year', hue='corpus')
    plt.title('Retrieved Chunks by Year')
    plt.savefig(os.path.join(OUTPUT_DIR, 'retrieval_year_distribution.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    df = df.copy()
    df['score_norm'] = df.apply(
        lambda row: row['score'] * 100 if row['method'] == 'DENSE' else row['score'],
        axis=1
    )

    sns.lineplot(data=df, x='year', y='score_norm', hue='method', marker='o')
    plt.title('Retrieval Score vs Year')
    plt.savefig(os.path.join(OUTPUT_DIR, 'score_vs_year.png'))
    plt.close()


# ==========================================
# 5. Temporal Footprint Mapping
# ==========================================

def query_year_mapping(df_src):
    df = df_src.dropna(subset=['year'])
    q_summary = df.groupby('query')['year'].unique().reset_index()
    q_summary['years'] = q_summary['year'].apply(lambda x: sorted(map(str, x)))
    return q_summary[['query', 'years']]


def plot_temporal_footprint(df_map):
    plot_df = df_map.explode('years')
    plot_df['years'] = plot_df['years'].astype(int)
    plot_df['query_short'] = plot_df['query'].apply(lambda q: q[:50] + '...' if len(q) > 50 else q)

    plt.figure(figsize=(14, 8))
    sns.stripplot(
        data=plot_df,
        x='years',
        y='query_short',
        size=10,
        marker='X',
        color='darkred'
    )

    plt.title('Temporal Footprint per Query')
    plt.xlabel('Year')
    plt.ylabel('Query')
    plt.grid(axis='x', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'temporal_query_footprint.png'))
    plt.close()


# ==========================================
# 6. Runner
# ==========================================

if __name__ == "__main__":

    all_ans, all_src = [], []

    for cfg in FILES_CONFIG:
        a, s = parse_rag_data(cfg)
        all_ans.append(a)
        all_src.append(s)

    final_ans = pd.concat(all_ans, ignore_index=True)
    final_src = pd.concat(all_src, ignore_index=True)

    failures = analyze_temporal_failures(final_ans, final_src)

    create_visualizations(final_src)
    query_map = query_year_mapping(final_src)
    plot_temporal_footprint(query_map)

    final_ans.to_csv(os.path.join(OUTPUT_DIR, 'processed_answers.csv'), index=False)
    final_src.to_csv(os.path.join(OUTPUT_DIR, 'processed_sources.csv'), index=False)
    failures.to_csv(os.path.join(OUTPUT_DIR, 'failure_report.csv'), index=False)

    print("✅ Analysis complete")
    print(f"Failures detected: {len(failures)}")
