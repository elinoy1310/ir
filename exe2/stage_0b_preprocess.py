from pathlib import Path
import string
import spacy
from exe1.tokenize_clean_text import process_folder
from exe1.lemma import lemmatize_folder

'''
python -m exe2.stage_0b_preprocess
'''


# def separate_punctuation(text: str) -> str:
#     """
#     מפריד סימני פיסוק בעזרת רווחים לפני ואחרי.
#     כמו בתרגיל 1 – משתמש ב-string.punctuation
#     """
#     trans_table = str.maketrans({
#         ch: f" {ch} " for ch in string.punctuation
#     })
#     return text.translate(trans_table)


# def lemmatize_text(nlp, text: str) -> str:
#     """
#     מקבל טקסט אחרי הפרדת פיסוק ומחזיר טקסט של למות.
#     משאיר רק טוקנים שהם לא רווח.
#     """
#     doc = nlp(text)
#     lemmas = []
#     for token in doc:
#         if token.is_space:
#             continue
#         # אפשר לבחור להוריד פיסוק כאן:
#         # if token.is_punct: continue
#         lemmas.append(token.lemma_)
#     return " ".join(lemmas)

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import os

def remove_stopwords(input_dir: Path, output_dir: Path):
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".txt"):
            continue
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        words = text.split()
        no_stopwords= ' '.join([word for word in words if word not in ENGLISH_STOP_WORDS])
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(" ".join(no_stopwords))



def process_parliament_dir( in_dir: Path, tokens_dir: Path, lemmas_dir: Path,cleaned_dir:Path):
    """
    מעבד תיקייה אחת (UK או US):
    - קורא את כל קבצי ה-txt
    - מפריד סימני פיסוק -> שומר ב-tokens_dir
    - עושה lemmatization -> שומר ב-lemmas_dir
    """
    tokens_dir.mkdir(parents=True, exist_ok=True)
    lemmas_dir.mkdir(parents=True, exist_ok=True)
    

    # # שלב 1 – הפרדת פיסוק
    # with_punct = separate_punctuation(raw)
    print(f"remove punc")
    process_folder(in_dir,tokens_dir)

    # # שמירת גרסת tokens
    # tokens_path = tokens_dir / txt_file.name
    # with tokens_path.open("w", encoding="utf-8") as f_out:
    #     f_out.write(with_punct)

    # שלב 2 – Lemmatization
    print(f"lemmatize")
    lemmatize_folder(tokens_dir,lemmas_dir)

    # # שמירת גרסת lemmas
    # lemmas_path = lemmas_dir / txt_file.name
    # with lemmas_path.open("w", encoding="utf-8") as f_out:
    #     f_out.write(lemm_text)

    print(f"remove stopwords")
    #remove_stopwords(lemmas_dir,cleaned_dir)




if __name__ == "__main__":
    # לעדכן אם הנתיב שונה
    # base_dir = Path(r"C:\Users\user\Desktop\שנה ד\איחזור מידע\ir\exe2")
    base_dir = Path(r"exe2")

    uk_in = base_dir / "UK"
    us_in = base_dir / "US"

    # תיקיות פלט
    uk_tokens = base_dir / "tokens" / "UK"
    us_tokens = base_dir / "tokens" / "US"

    uk_lemmas = base_dir / "lemmas" / "UK"
    us_lemmas = base_dir / "lemmas" / "US"

    uk_cleaned = base_dir / "cleaned" / "UK"
    us_cleaned = base_dir / "cleaned" / "US"

    #print("Loading spaCy model en_core_web_sm ...")
    #nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    print("\n=== Processing UK documents ===")
    process_parliament_dir( uk_in, uk_tokens, uk_lemmas, uk_cleaned)

    print("\n=== Processing US documents ===")
    process_parliament_dir( us_in, us_tokens, us_lemmas, us_cleaned)

    print("\nDone. Cleaned files saved under:")
    print(f"  Tokens: {base_dir / 'tokens'}")
    print(f"  Lemmas: {base_dir / 'lemmas'}")
