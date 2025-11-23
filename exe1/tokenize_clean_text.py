import argparse
import os
import string
from pathlib import Path

# יצירת טבלת תרגום שמוסיפה רווח לפני ואחרי כל סימן פיסוק
PUNCT_TRANSLATION = str.maketrans({
    p: f" {p} " for p in string.punctuation
})

# פונקציית טוקניזציה ללא רג׳קס
def tokenize_text(text: str) -> str:
    # מפריד סימני פיסוק ע"י רווחים
    separated = text.translate(PUNCT_TRANSLATION)
    # מנקה רווחים כפולים/מיותרים
    tokens = separated.split()
    return " ".join(tokens)

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
    input_folder = Path("clean_text")
    output_folder = Path("tokens")
    process_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()
