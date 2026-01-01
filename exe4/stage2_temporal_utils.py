# exe4/stage2_temporal_utils.py
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

UK_PATTERN = re.compile(r"debates(20\d{2})-(\d{2})-(\d{2})")
US_PATTERN = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")

def extract_metadata_from_source_path(source_path: str) -> Optional[Tuple[str, str]]:
    """
    Extract corpus type (UK / US) and ISO timestamp from source file path.

    Returns:
        (corpus_type, iso_timestamp) or None if extraction fails
    """
    if not source_path:
        return None

    path = Path(source_path)
    filename = path.name.lower()
    full_path = str(path).lower()

    # Determine corpus
    if "uk" in full_path:
        corpus = "UK"
        match = UK_PATTERN.search(filename)
    elif "us" in full_path:
        corpus = "US"
        match = US_PATTERN.search(filename)
    else:
        return None

    if not match:
        return None

    year, month, day = map(int, match.groups())

    try:
        dt = datetime(year, month, day)
        return corpus, dt.isoformat()
    except ValueError:
        return None
