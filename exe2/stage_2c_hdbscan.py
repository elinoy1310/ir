import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics.pairwise import cosine_distances
import hdbscan
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from eval_and_plot import visualize_clusters
from stage_0_load import load_corpus
from stage_1_tfidf import build_tfidf_vectors

'''
pip install hdbscan
'''
import hdbscan
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
def determine_min_samples(tfidf_matrix, k=30):
    nn = NearestNeighbors(n_neighbors=k, metric='cosine')
    nn.fit(tfidf_matrix)
    distances, _ = nn.kneighbors(tfidf_matrix)
    k_distances = distances[:, -1]
    avg_distance = np.mean(k_distances)
    print(f"Average k-distance for min_samples: {avg_distance}")
    return avg_distance

def determine_min_cluster_size(tfidf_matrix):
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import minimum_spanning_tree

    graph = kneighbors_graph(tfidf_matrix, n_neighbors=10, metric='cosine')
    mst = minimum_spanning_tree(graph)
    mst_distances = mst.data
    mst_distances.sort()
    median_distance = mst_distances[int(len(mst_distances) * 0.5)] if len(mst_distances) > 0 else 1.0
    print(f"Median MST distance for min_cluster_size: {median_distance}")
    return median_distance

# פונקציה לבחירת min_samples בעזרת k-distances
# def determine_min_samples(tfidf_matrix, k=10):
#     """
#     מחשבת את min_samples בעזרת k-distances
#     tfidf_matrix: מטריצת ה-TF-IDF (כמטריצה דלילה)
#     k: מספר השכנים הקרובים שברצוננו לבדוק
#     """
#     nn = NearestNeighbors(n_neighbors=k, metric='cosine')
#     nn.fit(tfidf_matrix)
#     distances, _ = nn.kneighbors(tfidf_matrix)
#     k_distances = distances[:, -1]  # המרחק של השכן ה-k
#     return np.mean(k_distances)  # שימוש בממוצע המרחקים

# # פונקציה לבחירת min_cluster_size בעזרת MST (Minimum Spanning Tree)
# def determine_min_cluster_size(tfidf_matrix):
#     """
#     מחשבת את min_cluster_size בעזרת MST
#     tfidf_matrix: מטריצת ה-TF-IDF (כמטריצה דלילה)
#     """
#     from sklearn.neighbors import kneighbors_graph
#     from scipy.sparse.csgraph import minimum_spanning_tree

#     # יצירת גרף של שכנים קרובים
#     graph = kneighbors_graph(tfidf_matrix, n_neighbors=10, metric='cosine')
    
#     # חישוב MST (Minimum Spanning Tree)
#     mst = minimum_spanning_tree(graph)
#     mst_distances = mst.data
    
#     # מיון המרחקים של ה-MST
#     mst_distances.sort()
#     return mst_distances[int(len(mst_distances) * 0.5)]  # ערך החצי של המרחקים


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
    # המרה למטריצה מסוג np.float64
    dist_matrix = np.array(dist_matrix, dtype=np.float64)

    import math
    # אם min_samples לא הוזן, חישוב עם היוריסטיקה
    if min_samples is None:
        min_samples =math.ceil(determine_min_samples(X))
        print(f"Determined min_samples: {min_samples}")

    # חישוב min_cluster_size עם היוריסטיקה
    if min_cluster_size is None:
        min_cluster_size = int(determine_min_cluster_size(X))
        print(f"Determined min_cluster_size: {min_cluster_size}")

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
    base_dir = r"exe2"
    X_tfidf = sparse.load_npz(Path(base_dir) / r"vectors_tfidf\TFIDF-Documents.npz")
    labels = np.array(json.loads(Path(base_dir).joinpath(r"vectors_tfidf\TFIDF-Documents_labels.json").read_text())['labels'])
    filenames = json.loads(Path(base_dir).joinpath(r"vectors_tfidf\TFIDF-Documents_files.json").read_text())['files']
    # print("Loaded documents:", len(texts))

    # # שלב 2 – בניית TF-IDF
    # X_tfidf, tfidf_vectorizer = build_tfidf_vectors(texts)
    # print("TF-IDF shape:", X_tfidf.shape)

    # # המרה ל-dense לשימוש ב-cosine_distances
    # X_dense = X_tfidf.toarray()

    # שלב 3 – HDBSCAN
    results = run_hdbscan(
        X_tfidf,
        labels,
        30,3
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
        visualize_clusters(
            X=X_tfidf,
            cluster_labels=results["cluster_labels"],
            true_labels=labels,
            title_prefix="HDBSCAN"
        )
