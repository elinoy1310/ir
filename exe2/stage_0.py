import os
from glob import glob
from sklearn.feature_extraction.text import TfidfVectorizer


def load_corpus(base_dir):
    uk_dir = os.path.join(base_dir, "UK")
    us_dir = os.path.join(base_dir, "US")

    texts = []
    labels = []   # 0 = UK, 1 = US
    filenames = []

    # UK
    for path in sorted(glob(os.path.join(uk_dir, "*.txt"))):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
        labels.append(0)
        filenames.append(os.path.basename(path))

    # US
    for path in sorted(glob(os.path.join(us_dir, "*.txt"))):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
        labels.append(1)
        filenames.append(os.path.basename(path))

    return texts, labels, filenames

def build_tfidf_vectors(texts, max_features=5000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english"
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


if __name__ == "__main__":
    base_dir = r"C:\Users\user\Desktop\שנה ד\איחזור מידע\ir\exe2"

    texts, labels, filenames = load_corpus(base_dir)
    X_tfidf, tfidf_vectorizer = build_tfidf_vectors(texts)

    print("TF-IDF shape:", X_tfidf.shape)
    print("Labels example:", labels[:10])
    print("First vector preview:", X_tfidf[0][:20])






