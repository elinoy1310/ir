import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from exe4.utils import resolve_chunk_metadata

# --------------------------------------------------
# Paths
# --------------------------------------------------
METADATA_INDEX = Path("exe4/metadata_index.json")

FIXED_CHUNKS_DIR = Path(r"exe3\chunked-text")
FIXED_CHUNK_INDEX_NAME = "reverse_chunk_index.json"

# PARENTSON_CHUNKS_DIR_UK = Path(r"exe3\parent-child-chunked-text\UK\children")
# PARENTSON_CHUNKS_DIR_US = Path(r"exe3\parent-child-chunked-text\US\children")
PARENTSON_CHUNKS_DIR = Path(r"exe3\parent-child-chunked-text")
PARENTSON_CHUNK_INDEX_NAME = "child_index.json"


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def plot_histogram(years_by_corpus: dict, title: str, ylabel: str):
    """
    years_by_corpus = {
        "UK": [2022, 2023, ...],
        "US": [2021, 2023, ...]
    }
    """
    counters = {k: Counter(v) for k, v in years_by_corpus.items()}
    all_years = sorted(set().union(*[c.keys() for c in counters.values()]))

    x = range(len(all_years))
    width = 0.35

    plt.figure(figsize=(10, 5))

    for i, (corpus, counter) in enumerate(counters.items()):
        values = [counter.get(y, 0) for y in all_years]
        plt.bar(
            [p + i * width for p in x],
            values,
            width=width,
            label=corpus
        )

    plt.xticks([p + width / 2 for p in x], all_years, rotation=45)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Plot 1: Source files
# --------------------------------------------------
def plot_source_files_distribution():
    with METADATA_INDEX.open(encoding="utf-8") as f:
        metadata = json.load(f)

    years_by_corpus = defaultdict(list)

    for meta in metadata.values():
        year = datetime.fromisoformat(meta["timestamp"]).year
        years_by_corpus[meta["corpus"]].append(year)

    plot_histogram(
        years_by_corpus,
        title="Temporal Distribution of Source Files",
        ylabel="Number of Source Files"
    )


# --------------------------------------------------
# Plot 2+3: Chunks
# --------------------------------------------------
def plot_chunks_distribution(chunks_dir: Path, chunk_index_name: Path, chunking_method: str):
    years_by_corpus = defaultdict(list)

    for corpus_dir in ["UK", "US"]:
        corpus_path = chunks_dir / corpus_dir
        if not corpus_path.exists():
            continue

                # אם קיימת תיקיית children פנימית, נעדכן את corpus_path
        children_path = corpus_path / "children"
        if children_path.exists() and children_path.is_dir():
            corpus_path = children_path
        
        chunk_index = corpus_path / chunk_index_name
        
        for chunk_path in corpus_path.rglob("*"):
            if not chunk_path.is_file() :
                continue

                    # רק קבצי טקסט
            if chunk_path.suffix.lower() != ".txt":
                continue

            corpus, timestamp = resolve_chunk_metadata(
                chunk_path=str(chunk_path),
                chunk_index_path=str(chunk_index),
                metadata_index_path=str(METADATA_INDEX),
                chunking_method=chunking_method
            )

            year = datetime.fromisoformat(timestamp).year
            years_by_corpus[corpus].append(year)


    plot_histogram(
        years_by_corpus,
        title=f"Temporal Distribution of Chunks ({chunking_method.upper()})",
        ylabel="Number of Chunks"
    )


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    print("📊 Plotting source files distribution...")
    plot_source_files_distribution()

    print("📊 Plotting FIXED chunks distribution...")
    plot_chunks_distribution(
        chunks_dir=FIXED_CHUNKS_DIR,
        chunk_index_name=FIXED_CHUNK_INDEX_NAME,
        chunking_method="fixed"
    )

    print("📊 Plotting PARENT-SON chunks distribution...")
    plot_chunks_distribution(
        chunks_dir= PARENTSON_CHUNKS_DIR,
        chunk_index_name=PARENTSON_CHUNK_INDEX_NAME,
        chunking_method="parent-son"
    )
