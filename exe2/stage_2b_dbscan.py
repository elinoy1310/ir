import json
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, calinski_harabasz_score, precision_recall_fscore_support, confusion_matrix, silhouette_score
from sklearn.metrics.pairwise import cosine_distances
import umap
import plotly.express as px
import pandas as pd


def run_dbscan(X, y, min_samples=5, eps=None):
    """
    הפעלת DBSCAN על ייצוג TF-IDF.
    משתמשים ב-cosine distance כי זה הסטנדרט בטקסט.
    """
    # חישוב מטריצת מרחקים מבוססת Cosine
    dist_matrix = cosine_distances(X)

    # אם eps לא הוזן, מחשבים אותו
    if eps is None:
        eps = compute_k_distance_eps(dist_matrix,k=2)
    
    # DBSCAN מקבל מטריצת מרחקים כשהפרמטר metric='precomputed'
    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="precomputed"
    )

    # הפעלת DBSCAN וקבלת תוויות האשכולות
    cluster_labels = db.fit_predict(dist_matrix)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    # -1 = רעש (noise), לא ניתן למפות ל-UK/US
    mask = cluster_labels != -1

    if mask.sum() == 0:
        return None
    
    # חישוב מדדים פנימיים אם יש יותר מאשכול אחד
    if n_clusters >= 2:
        silhouette = silhouette_score(X[mask], cluster_labels[mask], metric='cosine')
        calinski_harabasz = calinski_harabasz_score(X[mask].toarray(), cluster_labels[mask])
    else:
        silhouette = None
        calinski_harabasz = None

    mapped_input = cluster_labels[mask]
    true_labels = y[mask]

    # חישוב דיוק ע"י השוואת תוויות אמת (true labels) עם התוויות המתקבלות
    acc_direct = accuracy_score(true_labels, mapped_input)
    acc_flip = accuracy_score(true_labels, 1 - mapped_input)

    if acc_flip > acc_direct:
        mapped = 1 - mapped_input
    else:
        mapped = mapped_input

    accuracy = accuracy_score(true_labels, mapped)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, mapped, average="binary")
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
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette": silhouette,
        "calinski_harabasz": calinski_harabasz,
    }


def compute_k_distance_eps(distance_matrix, k=5):
    """
    פונקציה לחישוב eps עבור DBSCAN באמצעות גרף k-מרחק.
    """
    # חישוב מרחקים ל-k-השכנים הקרובים ביותר לכל נקודה
    k_distance = np.sort(distance_matrix, axis=1)[:, k-1]  # המרחק ל-k השכנים הקרובים ביותר
    eps = np.max(k_distance)  # eps הוא המרחק שבו אנחנו רואים שינוי חד במרחקים
    print(f"  k-distance eps computed: {eps:.4f}")
    return eps


if __name__ == "__main__":
    base_dir = r"exe2"
    X_tfidf = sparse.load_npz(Path(base_dir) / r"vectors_tfidf\TFIDF-Documents.npz")
    labels = np.array(json.loads(Path(base_dir).joinpath(r"vectors_tfidf\TFIDF-Documents_labels.json").read_text())['labels'])
    filenames = json.loads(Path(base_dir).joinpath(r"vectors_tfidf\TFIDF-Documents_files.json").read_text())['files']
    
    print(f"Loaded matrix shape: {X_tfidf.shape}")
    print(f"Loaded labels shape: {labels.shape}")

    # חישוב eps באמצעות k-distance
    # eps_k_distance = compute_k_distance_eps(cosine_distances(X_tfidf))
    # print(f"EPS שנבחר באמצעות k-distance: {eps_k_distance}")

    # הפעלת DBSCAN עם eps שהתקבל
    results = run_dbscan(X_tfidf, labels, min_samples=5)

    if results is None:
        print("DBSCAN found no clusters (all noise). Try lowering eps.")
    else:
        print("\n=== DBSCAN results ===")
        print("Total points:", results["total_points"])
        print("Noise points:", results["noise_points"])
        print("Accuracy:", round(results["accuracy"], 4))
        print("Precision:", round(results["precision"], 4))
        print("Recall:", round(results["recall"], 4))
        print("F1:", round(results["f1"], 4))
        print("\nConfusion matrix:")
        print(results["confusion_matrix"])

    # --- חלק ב: UMAP והצגה ויזואלית ---
    # הפחתת ממדים עם UMAP
    print("\n--- Running UMAP to visualize (Cosine Metric) ---")
    reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
    X_umap = reducer.fit_transform(X_tfidf)

    # הכנת DataFrame ל-Plotly
    df = pd.DataFrame(X_umap, columns=['UMAP-1', 'UMAP-2'])
    df['DBSCAN Cluster'] = [f"Cluster {c}" if c != -1 else "Noise (-1)" for c in results["cluster_labels"]]
    df['True Label'] = np.where(labels == 0, 'UK (True)', 'US (True)')
    df['Filename'] = filenames

    # יצירת גרף אינטראקטיבי עם Plotly
    print("Generating Plotly interactive visualization...")
    fig = px.scatter(
        df,
        x='UMAP-1',
        y='UMAP-2',
        color='DBSCAN Cluster',
        symbol='True Label',
        hover_data=['Filename', 'True Label', 'DBSCAN Cluster'],
        title=f'DBSCAN Clustering (Clusters: {results["n_clusters"]}, Noise: {results["n_noise"]}) on TF-IDF (UMAP)',
    )
    fig.show()  # פתיחת הגרף בדפדפן

