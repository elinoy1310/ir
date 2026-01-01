#exe4/ temporal_utils.py
import re
from datetime import datetime

def extract_date_from_source_file(source_file: str):
    """
    Extract datetime from filenames like: debates2024-11-11.txt
    Returns datetime or None.
    """
    if not source_file:
        return None
    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", source_file)
    if not m:
        return None
    y, mth, d = map(int, m.groups())
    return datetime(y, mth, d)

def extract_year(source_file: str):
    dt = extract_date_from_source_file(source_file)
    return dt.year if dt else None

def infer_time_constraint(query: str):
    """
    Returns:
      {"type": "year", "value": 2024}
      {"type": "recency"}
      {"type": "q4", "value": 2023}
      or None
    """
    q = (query or "").lower()

    # explicit year like 2024
    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        return {"type": "year", "value": int(m.group(1))}

    # last quarter of 2023
    m = re.search(r"last quarter of\s+(20\d{2})", q)
    if m:
        return {"type": "q4", "value": int(m.group(1))}

    # "current" / "latest" / "recent" / recency-type
    if any(tok in q for tok in ["current", "latest", "most recent", "as of", "today"]):
        return {"type": "recency"}

    return None

def is_in_q4(dt: datetime, year: int) -> bool:
    return dt is not None and dt.year == year and dt.month in (10, 11, 12)
