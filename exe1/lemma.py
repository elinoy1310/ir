import os
import spacy
'''
pip install spacy
python -m spacy download en_core_web_sm

'''
def lemmatize_folder(input_folder, output_folder="lemmatized_text"):
    """
    יוצר גרסה עם למות לכל קובץ טקסט בתיקייה.
    משתמש במודל של spaCy לעיבוד שפה טבעית.
    """
    nlp = spacy.load("en_core_web_sm")
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".txt"):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        doc = nlp(text)
        lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(" ".join(lemmas))

        print(f"✔ נוצר קובץ למות: {filename}")


# --- דוגמת הרצה ---
if __name__ == "__main__":
    lemmatize_folder(input_folder="tokens", output_folder="lemmatized_text")
