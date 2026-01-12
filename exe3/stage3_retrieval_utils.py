from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ChunkingConfig:
    method: str
    bm25_dir: Path
    dense_dir: Path
    uk_dir: Path
    us_dir: Path
    uk_index: Path
    us_index: Path


ROOT = Path("exe3")

def get_chunking_config(method: str) -> ChunkingConfig:
    method = method.lower().replace("-", "_")

    if method == "fixed":
        return ChunkingConfig(
            method="fixed",
            bm25_dir=ROOT / "bm25_vectors",
            dense_dir=ROOT / "st_vectors_fixed_chunks",
            uk_dir=ROOT / "chunked-text" / "UK",
            us_dir=ROOT / "chunked-text" / "US",
            uk_index=ROOT / "chunked-text" / "UK" / "reverse_chunk_index.json",
            us_index=ROOT / "chunked-text" / "US" / "reverse_chunk_index.json",
        )

    elif method == "parent_son":
        return ChunkingConfig(
            method="parent_son",
            bm25_dir=ROOT / "bm25_vectors_parentSon_chunks",
            dense_dir=ROOT / "st_vectors_parentSon_chunks",
            uk_dir=ROOT / "parent-child-chunked-text" / "UK" / "children",
            us_dir=ROOT / "parent-child-chunked-text" / "US" / "children",
            uk_index=ROOT / "parent-child-chunked-text" / "UK" / "children" / "child_index.json",
            us_index=ROOT / "parent-child-chunked-text" / "US" / "children" / "child_index.json",
        )

    else:
        raise ValueError(f"Unknown chunking method: {method}")
