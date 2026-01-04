from typing import Tuple
from pathlib import Path
import json

def resolve_chunk_metadata(
    chunk_path: str,
    chunk_index_path: str,
    metadata_index_path: str,
    chunking_method: str
) -> Tuple[str, str]:
    """
    Returns:
        (corpus, timestamp)
    """
    # Load chunk index and metadata index
    with open(chunk_index_path, encoding="utf-8") as f:
        chunk_index = json.load(f)

    with open(metadata_index_path, encoding="utf-8") as f:
        metadata_index = json.load(f)

    # ---------- Normalize chunk_path ----------
    # תמיד forward slash
    normalized_chunk_path = Path(chunk_path).as_posix()

    if normalized_chunk_path not in chunk_index:
        # בדיקה נוספת אם במטא יש backslashes
        alt_chunk_path = str(chunk_path)
        if alt_chunk_path in chunk_index:
            normalized_chunk_path = alt_chunk_path
        else:
            raise KeyError(f"Chunk not found in index: {chunk_path}")

    # ---------- Resolve source file ----------
    if chunking_method == "fixed":
        source_file = chunk_index[normalized_chunk_path]
    elif chunking_method == "parent-son":
        source_file = chunk_index[normalized_chunk_path]["original_file"]
    else:
        raise ValueError("chunking_method must be 'fixed' or 'parent-son'")

    # ---------- Normalize source_file ----------
    normalized_source_file = Path(source_file).as_posix()

    if normalized_source_file not in metadata_index:
        # בדיקה נוספת אם הנתיב במטא נשמר עם backslashes
        alt_source_file = source_file.replace("/", "\\")
        if alt_source_file in metadata_index:
            normalized_source_file = alt_source_file
        else:
            raise KeyError(f"Source file not found in metadata index: {source_file}")

    meta = metadata_index[normalized_source_file]
    return meta["corpus"], meta["timestamp"]

# =========================
# Temporal query groups
# =========================

hard_filter = [
        "what was the specific budget allocated to security in 2024?",

        "What was the specific amount allocated to support the public sector for National Insurance costs in 2025?"
]

recency = [
        "What is the current official position regarding the State of Israel?",
"What is the current official position regarding Hamas/Gaza?",
"Was the official position in the last quarter of 2023 supportive of the State of Israel?",
"Was the official position in the last quarter of 2023 supportive of Hamas/Gaza?",
"Has the official position in the last quarter of 2023 changed relative to the official position in the last quarter of 2025?"
,

"What is the latest official position regarding legislation on the protection of veterans?"

]

evolution = [
    "how did the prime minister/president's rhetoric regarding the war between israel and hamas/gaza change between his first and last speech?",

"How has the approach to the restoration and maintenance of local and community infrastructure changed between late 2023 and late 2024?"

]

ambiguity = [
    "who is the minister of defense/secretary of defense?",

    "Who holds the position responsible for Rural Development?"

]
def get_queries():
    return hard_filter+ recency+evolution+ambiguity


def get_type(query):
    if query in hard_filter:
        return "hard_filter"
    if query in recency:
        return "recency"
    if query in evolution:
        return "evolution"
    if query in ambiguity:
        return "ambiguity"
