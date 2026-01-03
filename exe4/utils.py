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
