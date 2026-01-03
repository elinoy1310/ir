# stage4_windows.py
# Utilities for computing EARLY / LATE time windows in the corpus

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from dateutil.relativedelta import relativedelta



def norm_path(p: str) -> str:
    # Normalize paths so metadata keys match source_file paths
    return p.replace("\\", "/").strip()


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


@dataclass(frozen=True)
class TimeWindow:
    start: datetime
    end: datetime

    def contains(self, dt: datetime) -> bool:
        return self.start <= dt <= self.end


def load_metadata_index(path: Path) -> dict:
    # Load metadata_index.json and normalize keys
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {norm_path(k): v for k, v in raw.items()}


def corpus_min_max_dates(meta: dict):
    # Find earliest and latest document dates in the corpus
    dates = [parse_iso(v["timestamp"]) for v in meta.values() if v.get("timestamp")]
    return min(dates), max(dates)


def compute_early_late_windows(meta: dict, window_months: int = 8):
    # Define early and late windows of equal duration
    mn, mx = corpus_min_max_dates(meta)
    early = TimeWindow(mn, mn + relativedelta(months=window_months))
    late = TimeWindow(mx - relativedelta(months=window_months), mx)
    return early, late


def get_doc_date(meta: dict, source_file: str):
    # Get document date for a given source_file
    rec = meta.get(norm_path(source_file))
    return parse_iso(rec["timestamp"]) if rec else None
