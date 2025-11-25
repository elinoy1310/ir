import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def run_gmm(X, y, n_components=2, n_dim=100):
    """
    מריץ GMM על נתוני TF-IDF לאחר הורדת מימד באמצעות TruncatedSVD.
    """

    print(f"Running SVD reduction → {n_dim} dimensions ...")
    svd = TruncatedSVD(n_components=n_dim, random_state=42)
    X_reduced = svd.fit_transform(X)

    print("Running Gaussian Mixture Model ...")
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=42
    )

    cluster_labels = gmm.fit_predict(X_reduced)

    # מיפוי אשכול → תגית 0/1 (לפי רוב)
    cluster_to_label = {}
    for c in np.unique(cluster_labels):
        cluster_to_label[c] = 1 if y[cluster_labels == c].mean() >= 0.5 else 0

    mapped = np.array([cluster_to_label[c] for c in cluster_labels])

    # חישובי איכות
    accuracy = accuracy_score(y, mapped)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, mapped, average="binary"
    )
    cm = confusion_matrix(y, mapped)

    return {
        "cluster_sizes": {int(c): int((cluster_labels == c).sum()) for c in np.unique(cluster_labels)},
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    from stage_0_load import load_corpus  
    from stage_1_tfidf import build_tfidf_vectors 
    base_dir = r"C:\Users\user\Desktop\שנה ד\איחזור מידע\ir\exe2"
  
    docs, labels, _ = load_corpus(base_dir)
    X, _ = build_tfidf_vectors(docs)
    print(f"Loaded documents: {len(docs)}")
    print(f"TF-IDF shape: {X.shape}")

    results = run_gmm(X, labels, n_components=2, n_dim=100)

    print("\n=== GMM Results ===")
    print("Cluster sizes:", results["cluster_sizes"])
    print("Accuracy :", round(results["accuracy"], 4))
    print("Precision:", round(results["precision"], 4))
    print("Recall   :", round(results["recall"], 4))
    print("F1       :", round(results["f1"], 4))
    print("\nConfusion matrix:")
    print(results["confusion_matrix"])
