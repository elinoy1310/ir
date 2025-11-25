from sklearn.feature_extraction.text import TfidfVectorizer

from stage_0_load import load_corpus


def build_tfidf_vectors(texts, max_features: int = 5000):
    """
    בונה מטריצת TF-IDF מתוך רשימת טקסטים.
    מחזיר:
      X          – מטריצת TF-IDF (sparse)
      vectorizer – אובייקט ה-TfidfVectorizer (כולל vocabulary)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


if __name__ == "__main__":
    base_dir = r"C:\Users\user\Desktop\שנה ד\איחזור מידע\ir\exe2"

    # שלב 1 – טעינת המסמכים
    texts, labels, filenames = load_corpus(base_dir)
    print("Loaded documents:", len(texts))

    # שלב 2 – בניית TF-IDF
    X_tfidf, tfidf_vectorizer = build_tfidf_vectors(texts)
    print("TF-IDF shape:", X_tfidf.shape)

    # הצצה קטנה לווקטור הראשון (פורמט sparse)
    print("First document TF-IDF vector (sparse preview):")
    print(X_tfidf[0])
