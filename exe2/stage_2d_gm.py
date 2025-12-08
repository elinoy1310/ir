# exe2/ stage_2d_gm.py
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix



def run_gmm(X, y, n_components=2, n_dim=100):
    """
    מריץ GMM על נתוני TF-IDF לאחר הורדת מימד באמצעות TruncatedSVD.
    X – מטריצת TF-IDF (sparse או dense)
    y – תוויות אמת (0 = UK, 1 = US)
    """
    # לוודא ש-y הוא numpy array
    y = np.asarray(y)

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

    # מיפוי אשכול → תגית 0/1 (לפי רוב התוויות האמיתיות באשכול)
    cluster_to_label = {}
    for c in np.unique(cluster_labels):
        mask_c = (cluster_labels == c)
        # y הוא עכשיו numpy array, אז אינדוקס בוליאני עובד
        cluster_to_label[c] = 1 if y[mask_c].mean() >= 0.5 else 0

    mapped = np.array([cluster_to_label[c] for c in cluster_labels])

    # חישובי איכות
    accuracy = accuracy_score(y, mapped)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, mapped, average="binary"
    )
    cm = confusion_matrix(y, mapped)

    return {
        "cluster_sizes": {
            int(c): int((mapped == c).sum())
            for c in np.unique(mapped)
        },
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "cluster_labels": cluster_labels
    }


if __name__ == "__main__":
    base_dir = r"exe2"

    # טוענים קורפוס מהלמות (אחרי ה-preprocess)
    X_tfidf = sparse.load_npz(Path(base_dir) / r"vectors_tfidf\TFIDF-Documents.npz")
    labels = np.array(json.loads(Path(base_dir).joinpath(r"vectors_tfidf\TFIDF-Documents_labels.json").read_text())['labels'])
    filenames = json.loads(Path(base_dir).joinpath(r"vectors_tfidf\TFIDF-Documents_files.json").read_text())['files']
    #features_names = json.loads(Path(base_dir).joinpath(r"vectors_tfidf\TFIDF-Documents_vocabulary.json").read_text())
    # חשוב: להפוך ל-numpy array
    labels = np.array(labels)

    results = run_gmm(X_tfidf, labels, n_components=2, n_dim=100)

    print("\n=== GMM Results ===")
    print("Cluster sizes:", results["cluster_sizes"])
    print("Accuracy :", round(results["accuracy"], 4))
    print("Precision:", round(results["precision"], 4))
    print("Recall   :", round(results["recall"], 4))
    print("F1       :", round(results["f1"], 4))
    print("\nConfusion matrix:")
    print(results["confusion_matrix"])

    from eval_and_plot import visualize_clusters, plot_umap_results

    plot_umap_results(X_tfidf, results["cluster_labels"], "GMM Clusters", "gmm_umap.png")
