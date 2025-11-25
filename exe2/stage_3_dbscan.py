import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics.pairwise import cosine_distances

from stage_0_load import load_corpus
from stage_1_tfidf import build_tfidf_vectors


def run_dbscan(X, y, eps=0.4, min_samples=5):
    """
    מעביר DBSCAN על ייצוג TF-IDF.
    משתמשים ב-cosine distance כי זה הסטנדרט בטקסט.
    """
    # חישוב מטריצת מרחקים מבוססת Cosine
    dist_matrix = cosine_distances(X)

    # DBSCAN מקבל מטריצת מרחקים כשהפרמטר metric='precomputed'
    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="precomputed"
    )

    cluster_labels = db.fit_predict(dist_matrix)

    # -1 = רעש (noise), לא ניתן למפות ל-UK/US
    mask = cluster_labels != -1

    if mask.sum() == 0:
        return None

    mapped_input = cluster_labels[mask]
    true_labels = y[mask]

    acc_direct = accuracy_score(true_labels, mapped_input)
    acc_flip = accuracy_score(true_labels, 1 - mapped_input)

    if acc_flip > acc_direct:
        mapped = 1 - mapped_input
    else:
        mapped = mapped_input

    accuracy = accuracy_score(true_labels, mapped)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        mapped,
        average="binary"
    )
    cm = confusion_matrix(true_labels, mapped)

    return {
        "total_points": len(cluster_labels),
        "noise_points": np.sum(cluster_labels == -1),
        "cluster_labels": cluster_labels,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    base_dir = r"C:\Users\user\Desktop\שנה ד\איחזור מידע\ir\exe2"

    texts, labels, filenames = load_corpus(base_dir)
    print("Loaded:", len(texts))

    # TF-IDF
    X_tfidf, tfidf_vectorizer = build_tfidf_vectors(texts)
    X_dense = X_tfidf.toarray()

    # DBSCAN – אפשר לשנות eps ו-min_samples ולבדוק
    results = run_dbscan(X_dense, labels, eps=0.45, min_samples=5)

    if results is None:
        print("DBSCAN found no clusters (all noise). Try lowering eps.")
    else:
        print("\n=== DBSCAN results ===")
        print("Total points:", results["total_points"])
        print("Noise points:", results["noise_points"])
        print("Accuracy :", round(results["accuracy"], 4))
        print("Precision:", round(results["precision"], 4))
        print("Recall   :", round(results["recall"], 4))
        print("F1       :", round(results["f1"], 4))
        print("\nConfusion matrix:")
        print(results["confusion_matrix"])
