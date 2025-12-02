from pathlib import Path


def load_corpus(base_dir: str):
    """
    טוען מסמכים מהלמות:
    base_dir/lemmas/UK
    base_dir/lemmas/US

    מחזיר:
    texts  – רשימת טקסטים (למות)
    labels – 0 עבור UK, 1 עבור US
    filenames – שם הקובץ
    """
    base = Path(base_dir)

    uk_dir = base / "lemmas" / "UK"
    us_dir = base / "lemmas" / "US"

    texts = []
    labels = []
    filenames = []

    # קודם UK – label 0
    for path in sorted(uk_dir.glob("*.txt")):
        txt = path.read_text(encoding="utf-8", errors="ignore")
        texts.append(txt)
        labels.append(0)
        filenames.append(path.name)

    # אחר כך US – label 1
    for path in sorted(us_dir.glob("*.txt")):
        txt = path.read_text(encoding="utf-8", errors="ignore")
        texts.append(txt)
        labels.append(1)
        filenames.append(path.name)

    return texts, labels, filenames


if __name__ == "__main__":
    base_dir = r"C:\Users\user\Desktop\שנה ד\איחזור מידע\ir\exe2"
    texts, labels, filenames = load_corpus(base_dir)
    print("Total documents:", len(texts))
    print("First file:", filenames[0] if filenames else "NONE")
    print("Preview:", texts[0][:200] if texts else "")
