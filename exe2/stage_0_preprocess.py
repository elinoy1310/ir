from pathlib import Path
import string
import spacy


def separate_punctuation(text: str) -> str:
    """
    מפריד סימני פיסוק בעזרת רווחים לפני ואחרי.
    כמו בתרגיל 1 – משתמש ב-string.punctuation
    """
    trans_table = str.maketrans({
        ch: f" {ch} " for ch in string.punctuation
    })
    return text.translate(trans_table)


def lemmatize_text(nlp, text: str) -> str:
    """
    מקבל טקסט אחרי הפרדת פיסוק ומחזיר טקסט של למות.
    משאיר רק טוקנים שהם לא רווח.
    """
    doc = nlp(text)
    lemmas = []
    for token in doc:
        if token.is_space:
            continue
        # אפשר לבחור להוריד פיסוק כאן:
        # if token.is_punct: continue
        lemmas.append(token.lemma_)
    return " ".join(lemmas)


def process_parliament_dir(nlp, in_dir: Path, tokens_dir: Path, lemmas_dir: Path):
    """
    מעבד תיקייה אחת (UK או US):
    - קורא את כל קבצי ה-txt
    - מפריד סימני פיסוק -> שומר ב-tokens_dir
    - עושה lemmatization -> שומר ב-lemmas_dir
    """
    tokens_dir.mkdir(parents=True, exist_ok=True)
    lemmas_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in sorted(in_dir.glob("*.txt")):
        with txt_file.open("r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        # שלב 1 – הפרדת פיסוק
        with_punct = separate_punctuation(raw)

        # שמירת גרסת tokens
        tokens_path = tokens_dir / txt_file.name
        with tokens_path.open("w", encoding="utf-8") as f_out:
            f_out.write(with_punct)

        # שלב 2 – Lemmatization
        lemm_text = lemmatize_text(nlp, with_punct)

        # שמירת גרסת lemmas
        lemmas_path = lemmas_dir / txt_file.name
        with lemmas_path.open("w", encoding="utf-8") as f_out:
            f_out.write(lemm_text)

        print(f"Processed: {txt_file.name}")


if __name__ == "__main__":
    # לעדכן אם הנתיב שונה
    base_dir = Path(r"C:\Users\user\Desktop\שנה ד\איחזור מידע\ir\exe2")

    uk_in = base_dir / "UK"
    us_in = base_dir / "US"

    # תיקיות פלט
    uk_tokens = base_dir / "tokens" / "UK"
    us_tokens = base_dir / "tokens" / "US"

    uk_lemmas = base_dir / "lemmas" / "UK"
    us_lemmas = base_dir / "lemmas" / "US"

    print("Loading spaCy model en_core_web_sm ...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    print("\n=== Processing UK documents ===")
    process_parliament_dir(nlp, uk_in, uk_tokens, uk_lemmas)

    print("\n=== Processing US documents ===")
    process_parliament_dir(nlp, us_in, us_tokens, us_lemmas)

    print("\nDone. Cleaned files saved under:")
    print(f"  Tokens: {base_dir / 'tokens'}")
    print(f"  Lemmas: {base_dir / 'lemmas'}")
