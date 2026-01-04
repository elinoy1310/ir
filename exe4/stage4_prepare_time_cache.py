# exe4/stage4_prepare_time_cache.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from exe3.stage3_retrieval import load_bm25_store, load_dense_store, change_chanking_method
from exe4.utils import resolve_chunk_metadata

def build_time_cache(chunking_method: str, out_path: Path):
    change_chanking_method(chunking_method)

    # names from one store is enough; best use BM25 names
    X_bm25, vocab, names = load_bm25_store()

    chunk_index_path = Path("exe4/united_fixed_chunk_index.json") if chunking_method == "fixed" else Path("exe4/united_parentSon_chunk_index.json")
    metadata_index_path = Path("exe4/metadata_index.json")

    cache: Dict[str, Dict[str, Any]] = {}
    for name in names:
        corpus, ts = resolve_chunk_metadata(
            chunk_path=name,
            chunk_index_path=str(chunk_index_path),
            metadata_index_path=str(metadata_index_path),
            chunking_method=chunking_method
        )
        cache[name] = {"corpus": corpus.upper(), "timestamp": ts}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved cache with {len(cache)} entries -> {out_path}")

if __name__ == "__main__":
    build_time_cache("fixed", Path("exe4/cache/time_cache_fixed.json"))
    build_time_cache("parent-son", Path("exe4/cache/time_cache_parent-son.json"))
