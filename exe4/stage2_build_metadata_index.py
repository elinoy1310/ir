# exe4/stage2_build_metadata_index.py
import json
from pathlib import Path
from exe4.stage2_temporal_utils import extract_metadata_from_source_path

'''
python -m exe4.stage2_build_metadata_index


'''

OUTPUT_PATH = Path("exe4/metadata_index.json")

def build_metadata_index(source_root_dir: str):

    """
    Builds a metadata index:
    {
        source_file_path: {
            "corpus": "UK" | "US",
            "timestamp": "YYYY-MM-DDTHH:MM:SS"
        }
    }
    """


    print("🔄 Scanning source files directory...")
    source_root = Path(source_root_dir)

    metadata_index = {}
    skipped = 0

    for source_path in source_root.rglob("*.txt"):
        source_path = str(source_path)

        meta = extract_metadata_from_source_path(source_path)
        if meta is None:
            skipped += 1
            continue

        corpus, timestamp = meta
        metadata_index[source_path] = {
            "corpus": corpus,
            "timestamp": timestamp
        }


    # --- load existing metadata if exists ---
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as f:
            existing_metadata = json.load(f)
    else:
        existing_metadata = {}

    # --- merge (append logically) ---
    existing_metadata.update(metadata_index)

    # --- write back (overwrite file) ---
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(existing_metadata, f, indent=2)


    print("✅ Metadata index built")
    print(f"📦 Total source files indexed: {len(metadata_index)}")
    print(f"⚠️ Skipped files: {skipped}")
    print(f"💾 Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_metadata_index(r"exe3\clean-text\UK")
    build_metadata_index(r"exe3\clean-text\US")
