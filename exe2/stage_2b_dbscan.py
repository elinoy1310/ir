# exe2/stage_2b_dbscan.py
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from eval_and_plot import plot_umap_results
import umap
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from kneed import KneeLocator

'''
pip install umap-learn matplotlib plotly
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import pairwise_distances

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import pairwise_distances

def calculate_mst_cosine_edges(X):
    """
    מחשבת את משקלי קשתות ה-MST לפי מרחק קוסינוס ומחזירה אותם ממוינים.
    
    Parameters:
    -----------
    X : array-like
        הדאטה (Feature Matrix).
        
    Returns:
    --------
    sorted_edges : np.array
        מערך של משקלי הקשתות (Cosine Distances) ממוין מהקטן לגדול.
    """
    
    # 1. חישוב מטריצת מרחקים לפי מרחק קוסינוס (1 - דמיון קוסינוס)
    # הערכים יהיו בטווח [0, 2]
    dist_matrix = pairwise_distances(X, metric='cosine')
        
    # 2. חישוב עץ פורש מינימלי (MST)
    mst = minimum_spanning_tree(dist_matrix)
    
    # 3. חילוץ המשקלים של הקשתות ב-MST
    edges = mst.data
    
    # 4. מיון הקשתות מהקטן לגדול
    sorted_edges = np.sort(edges)
    
    # 5. החזרת המערך
    return sorted_edges

def run_dbscan(X, y, eps: float = 0.09, min_samples: int = 5):
    """
    מריץ DBSCAN על מטריצת ה-TF-IDF ומחשב מדדי איכות מול התוויות האמיתיות (y).
    X – מטריצת TF-IDF (numpy array או sparse שהומר ל-dense)
    y – labels אמיתיים: 0 = UK, 1 = US
    eps – רדיוס של הקבוצה
    min_samples – מספר הנקודות המינימלי לכל קבוצה
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')

    # התאמה לנתונים
    cluster_labels = dbscan.fit_predict(X)

    # DBSCAN לא תמיד יוצר קבוצות נפרדות – ייתכן שהיו noise points (label == -1)
    # נבדוק שתי אפשרויות:
    # 1. כמו שזה
    # 2. הפוך (1 - cluster_labels) – אם יש יותר קבוצות אחרי ההפיכה
    acc_direct = accuracy_score(y, cluster_labels)
    print("acc_direct: ",acc_direct)
    acc_flipped = accuracy_score(y, 1 - cluster_labels)
    print("acc_flipped: ",acc_flipped)

    if acc_flipped > acc_direct:
        mapped = 1 - cluster_labels
        print("Mapping clusters: flipped 0<->1")
    else:
        mapped = cluster_labels
        print("Mapping clusters: direct 0->0, 1->1")

    print("y: ",y[328:340])
    print("mapped: ",mapped[328:340])

    # חישוב מדדים אחרי המיפוי ל-0/1
    accuracy = accuracy_score(y, mapped)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        mapped,
        average="macro"  # המחלקה החיובית היא 1 (US)
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

import numpy as np
from sklearn.neighbors import NearestNeighbors

def calculate_k_distance_cosine_edges(X, k):
    """
    מחשבת את המרחק לשכן ה-k-י הקרוב ביותר עבור כל נקודה, 
    באמצעות מרחק קוסינוס, ומחזירה את התוצאות ממוינות.

    Parameters:
    -----------
    X : array-like
        הדאטה (Feature Matrix).
    k : int
        מספר השכנים הקרובים ביותר שיש לבחון (min_samples).
        
    Returns:
    --------
    sorted_k_distances : np.array
        מערך של מרחקי ה-k-NN ממוין מהקטן לגדול.
    """
    
    # 1. הגדרת מודל השכנים
    # k+1: אנחנו מבקשים את k השכנים הקרובים ביותר, כולל הנקודה עצמה (שהיא השכן ב-0).
    # metric='cosine': שימוש במרחק קוסינוס (1 - דמיון קוסינוס).
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
    
    # 2. אימון המודל וחישוב מרחקים
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # 3. חילוץ המרחק לשכן ה-k-י
    # מכיוון ש-distances כולל את המרחק לנקודה עצמה (0), 
    # אנחנו צריכים את העמודה במקום k (שהיא האינדקס ה-k-י)
    # לדוגמה: אם k=4, אנחנו לוקחים את האינדקס 4.
    k_distances = distances[:, k]
    
    # 4. מיון המרחקים
    sorted_k_distances = np.sort(k_distances)
    
    return sorted_k_distances

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def plot_eps(X, min_samples):
    # חיפוש עבור מרחקים בין הנקודות
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    
    # סידור המרחקים
    distances = np.sort(distances[:, -1], axis=0)
    
    # הצגת גרף המרחקים
    plt.plot(distances)
    plt.title("k-distance graph")
    plt.xlabel("Points sorted by distance")
    plt.ylabel("Distance to k-th nearest neighbor")
    plt.show()



if __name__ == "__main__":

    X_tfidf = sparse.load_npz(Path(r"exe2\vectors_tfidf\TFIDF-Documents.npz"))
    print("TF-IDF shape:", X_tfidf.shape)

    # חלק מגרסאות DBSCAN ב-sklearn לא אוהבות sparse → נעשה toarray()
    #X_dense = X_tfidf.toarray()

    # --- טעינת שמות הקבצים והמילים ---
    with open(r"exe2\vectors_tfidf\TFIDF-Documents_labels.json", "r", encoding="utf-8") as f:
        files_data = json.load(f)
    labels = files_data["labels"]
        # 2. חישוב הקשתות
    #mst_edges = calculate_mst_cosine_edges(X_dense)
    min_samples_k=20
    plot_eps(X_tfidf, min_samples=min_samples_k)
    #optimal_eps = kneedle.knee_y
    optimal_eps = 0.84
    print(f"Optimal eps found at: {optimal_eps}")

    # שלב 3 – DBSCAN
    results = run_dbscan(X_tfidf, labels,eps=optimal_eps,min_samples=min_samples_k)

    print("\n=== DBSCAN clustering evaluation ===")
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

    # ----------------------------------------------------
    # 🌟 שלב 4 – הפחתת ממדים והצגה ויזואלית באמצעות UMAP
    # ----------------------------------------------------
    print("\n--- Running UMAP to visualize ---")
    plot_umap_results(X_tfidf, results["mapped_clusters"], title="DBSCAN Clustering on TF-IDF Vectors", filename="dbscan_umap.png")
