import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics.pairwise import cosine_distances
import hdbscan

from stage_0_load import load_corpus
from stage_1_tfidf import build_tfidf_vectors


def run_hdbscan(X, y, min_cluster_size=30, min_samples=5):
    """
    מריץ HDBSCAN על מטריצת מרחקים מבוססת Cosine.
    בגלל בעיות תמיכה ב-metric='cosine' בגרסאות חדשות של sklearn,
    אנחנו מחשבים מראש מטריצת מרחקים ומשתמשים ב-metric='precomputed'.
    X – מטריצת TF-IDF (dense)
    y – תוויות אמת: 0 = UK, 1 = US
    """

    # מחשבים מטריצת מרחקים קוסינוס (כמו ב-DBSCAN)
    dist_matrix = cosine_distances(X)

    # HDBSCAN עם מטריצת מרחקים מוכנה
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )

    cluster_labels = clusterer.fit_predict(dist_matrix)

    # מזהים רעש (-1)
    noise_mask = (cluster_labels == -1)
    n_noise = int(noise_mask.sum())

    # משתמשים רק בנקודות שאינן רעש
    mask = ~noise_mask
    if mask.sum() == 0:
        # אם הכול רעש אין מה להעריך
        return None

    used_clusters = cluster_labels[mask]   # תוויות אשכול
    true_labels = y[mask]                  # תוויות אמת (0/1)

    # --- מיפוי אשכולות ל-0/1 לפי רוב בכתובות האמת ---
    unique_clusters = np.unique(used_clusters)
    cluster_to_label = {}

    for c in unique_clusters:
        c_mask = (used_clusters == c)
        mean_label = true_labels[c_mask].mean()  # אחוז ה-1 באשכול
        cluster_to_label[c] = 1 if mean_label >= 0.5 else 0

    mapped = np.array([cluster_to_label[c] for c in used_clusters])

    # --- חישוב מדדים ---
    accuracy = accuracy_score(true_labels, mapped)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        mapped,
        average="binary"   # 1 = US
    )
    cm = confusion_matrix(true_labels, mapped)

    # גודל כל אשכול
    cluster_sizes = {
        int(c): int((used_clusters == c).sum())
        for c in unique_clusters
    }

    return {
        "total_points": len(cluster_labels),
        "noise_points": n_noise,
        "n_clusters": len(unique_clusters),
        "cluster_labels": cluster_labels,
        "cluster_sizes": cluster_sizes,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    base_dir = r"C:\Users\user\Desktop\שנה ד\איחזור מידע\ir\exe2"

    # שלב 1 – טעינת המסמכים
    texts, labels, filenames = load_corpus(base_dir)
    print("Loaded documents:", len(texts))

    # שלב 2 – בניית TF-IDF
    X_tfidf, tfidf_vectorizer = build_tfidf_vectors(texts)
    print("TF-IDF shape:", X_tfidf.shape)

    # המרה ל-dense לשימוש ב-cosine_distances
    X_dense = X_tfidf.toarray()

    # שלב 3 – HDBSCAN
    results = run_hdbscan(
        X_dense,
        labels,
        min_cluster_size=30,
        min_samples=5
    )

    if results is None:
        print("HDBSCAN: all points were labeled as noise – try smaller min_cluster_size or min_samples.")
    else:
        print("\n=== HDBSCAN results ===")
        print("Total points :", results["total_points"])
        print("Noise points :", results["noise_points"])
        print("Num clusters :", results["n_clusters"])
        print("Cluster sizes (cluster_id: size):", results["cluster_sizes"])

        print("\nAccuracy :", round(results["accuracy"], 4))
        print("Precision:", round(results["precision"], 4))
        print("Recall   :", round(results["recall"], 4))
        print("F1       :", round(results["f1"], 4))

        print("\nConfusion matrix (rows=true, cols=pred):")
        print(results["confusion_matrix"])
