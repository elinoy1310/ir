import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from stage_0_load import load_corpus
from stage_1_tfidf import build_tfidf_vectors


def run_kmeans(X, y, n_clusters: int = 2):
    """
    מריץ KMeans על מטריצת ה-TF-IDF ומחשב מדדי איכות מול התוויות האמיתיות (y).
    X – מטריצת TF-IDF (numpy array או sparse שהומר ל-dense)
    y – labels אמיתיים: 0 = UK, 1 = US
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    # התאמה לנתונים
    cluster_labels = kmeans.fit_predict(X)

    # KMeans לא יודע מי זה UK ומי זה US – הוא רק נותן 0/1
    # נבדוק שתי אפשרויות:
    # 1. כמו שזה
    # 2. הפוך (1 - cluster_labels)
    acc_direct = accuracy_score(y, cluster_labels)
    acc_flipped = accuracy_score(y, 1 - cluster_labels)

    if acc_flipped > acc_direct:
        mapped = 1 - cluster_labels
    else:
        mapped = cluster_labels

    # חישוב מדדים אחרי המיפוי ל-0/1
    accuracy = accuracy_score(y, mapped)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        mapped,
        average="binary"  # המחלקה החיובית היא 1 (US)
    )

    cm = confusion_matrix(y, mapped)

    return {
        "raw_clusters": cluster_labels,   # לפני מיפוי
        "mapped_clusters": mapped,       # אחרי מיפוי ל-0/1
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    # !!! לעדכן לנתיב האמיתי אצלך !!!
    base_dir = r"C:\Users\user\Desktop\שנה ד\איחזור מידע\ir\exe2"

    # שלב 1 – טעינת המסמכים
    texts, labels, filenames = load_corpus(base_dir)
    print("Loaded documents:", len(texts))

    # שלב 2 – בניית TF-IDF
    X_tfidf, tfidf_vectorizer = build_tfidf_vectors(texts)
    print("TF-IDF shape:", X_tfidf.shape)

    # חלק מגרסאות KMeans ב-sklearn לא אוהבות sparse → נעשה toarray()
    X_dense = X_tfidf.toarray()

    # שלב 3 – KMeans
    results = run_kmeans(X_dense, labels)

    print("\n=== KMeans clustering evaluation (2 clusters: UK/US) ===")
    print("Accuracy :", round(results["accuracy"], 4))
    print("Precision:", round(results["precision"], 4))
    print("Recall   :", round(results["recall"], 4))
    print("F1       :", round(results["f1"], 4))

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(results["confusion_matrix"])

    # כמה מסמכים בכל תווית אחרי המיפוי
    unique, counts = np.unique(results["mapped_clusters"], return_counts=True)
    print("\nCluster sizes (after mapping to 0/1):")
    for u, c in zip(unique, counts):
        print(f"Label {u}: {c} docs")
