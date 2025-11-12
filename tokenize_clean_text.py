#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path
"""python tokenize_clean_text.py --input clean_text --output tokens"""

# תבנית טוקן "מילה" הכוללת:
# - אותיות עברית/לטינית ומספרים
# - גרש עברי (׳ U+05F3), גרשיים עבריים (״ U+05F4), גרש רגיל (' או ’) ומקף באמצע מילה
#   כדי לשמור על מחרוזות כמו צה״ל/צה"ל ו-United-Kingdom
WORD = r"[A-Za-z\u0590-\u05FF0-9]+(?:[’'\u05F3\u05F4\-][A-Za-z\u0590-\u05FF0-9]+)*"

# אליפסיס ("…" או "...") כטוקן אחד
ELLIPSIS = r"(?:\.\.\.|…)"  # … = U+2026

# סימני פיסוק שיופרדו כמילים בפני עצמן
# נכללים: נקודה, פסיק, נקודתיים, נקודה-פסיק, סימני שאלה/קריאה,
# סוגריים [], (), {}, גרשיים כפולים/בודדים, מקפים ארוכים/קצרים
PUNCT = r"[\.,;:\?!\(\)\[\]\{\}\"״'׳\-–—]"

# סדר עדיפויות: קודם מילה, אח"כ אליפסיס, ואז פיסוק בודד
TOKEN_RE = re.compile(f"{WORD}|{ELLIPSIS}|{PUNCT}", flags=re.UNICODE)

def tokenize_text(text: str) -> str:
    tokens = [m.group(0) for m in TOKEN_RE.finditer(text)]
    # איחוד רווחים ופסילת רווחים כפולים מהקלט המקורי
    return " ".join(tokens).strip()

def process_folder(input_folder: Path, output_folder: Path, suffix_in=".txt", suffix_out=".txt"):
    output_folder.mkdir(parents=True, exist_ok=True)
    count_in, count_out = 0, 0

    for p in sorted(input_folder.iterdir()):
        if not p.is_file() or not p.name.lower().endswith(suffix_in):
            continue
        count_in += 1
        out_path = output_folder / p.name.replace(suffix_in, suffix_out)

        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            tokenized = tokenize_text(text)
            out_path.write_text(tokenized, encoding="utf-8")
            count_out += 1
            print(f"✔ {p.name} → {out_path.name} ({len(tokenized)} chars)")
        except Exception as e:
            print(f"⚠ שגיאה בעיבוד {p.name}: {e}")

    print(f"\nסיכום: נקראו {count_in} קבצים, נוצרו {count_out} קבצים בתיקייה: {output_folder.resolve()}")

def main():
    ap = argparse.ArgumentParser(description="Tokenize TXT files: separate words and punctuation into standalone tokens.")
    ap.add_argument("--input", "-i", required=True, help="תיקיית קלט עם קבצי TXT הנקיים (למשל clean_text)")
    ap.add_argument("--output", "-o", default="tokens", help="תיקיית פלט לקבצים המטוקננים (ברירת מחדל: tokens)")
    args = ap.parse_args()

    process_folder(Path(args.input), Path(args.output))

if __name__ == "__main__":
    main()
