import json
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import umap
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

'''
pip install umap-learn matplotlib plotly
'''

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
        print("Mapping clusters: flipped 0<->1")
    else:
        mapped = cluster_labels
        print("Mapping clusters: direct 0->0, 1->1")

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
    base_dir = r"C:\Users\user\Desktop\שנה ד\איחזור מידע\ir\exe2"

    # # שלב 1 – טעינת המסמכים
    # texts, labels, filenames = load_corpus(base_dir)
    # print("Loaded documents:", len(texts))

    # שלב 2 – בניית TF-IDF
    X_tfidf=sparse.load_npz(Path(r"exe2\vectors_tfidf\TFIDF-Documents.npz"))
    print("TF-IDF shape:", X_tfidf.shape)

    # חלק מגרסאות KMeans ב-sklearn לא אוהבות sparse → נעשה toarray()
    X_dense = X_tfidf.toarray()
    # --- טעינת שמות הקבצים והמילים ---
    with open(r"exe2\vectors_tfidf\TFIDF-Documents_labels.json", "r", encoding="utf-8") as f:
        files_data = json.load(f)
    labels = files_data["labels"]
    

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

    # ----------------------------------------------------
    # 🌟 שלב 4 – הפחתת ממדים והצגה ויזואלית באמצעות UMAP
    # ----------------------------------------------------
    print("\n--- Running UMAP to visualize ---")
    
    # הפחתת ממדים ל-2D באמצעות UMAP
    # ניתן להשתמש ב-X_tfidf הדלילה ישירות ללא toarray()!
    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        metric='cosine' # קוסינוס עדיפה למטריצות TF-IDF
    )
    
    # הפחתת הנתונים
    X_umap = reducer.fit_transform(X_tfidf)

    # יצירת DataFrame לטובת ציור
    df = pd.DataFrame(X_umap, columns=['UMAP-1', 'UMAP-2'])
    df['True Label'] = np.where(labels == 0, 'UK (True)', 'US (True)') # יצירת תוויות טקסט
    df['Cluster Result'] = np.where(results["mapped_clusters"] == 0, 'UK (Predicted)', 'US (Predicted)')
    with open(r"exe2\vectors_tfidf\TFIDF-Documents_files.json", "r", encoding="utf-8") as f:
        files_data = json.load(f)
    file_names = files_data["files"]
    df['Filename'] = file_names 
        
    
    ## אפשרות 1: גרף אינטראקטיבי (מומלץ)
    # ניתן להשתמש ב-Plotly כדי לקבל גרף שמאפשר זום וריחוף מעל נקודות
    print("Generating Plotly interactive visualization...")
    fig = px.scatter(
        df,
        x='UMAP-1',
        y='UMAP-2',
        color='Cluster Result', # צבע לפי התוצאה
        symbol='True Label', # צורה לפי התווית האמיתית
        hover_data=['Filename', 'True Label', 'Cluster Result'],
        title='K-Means Clustering on TF-IDF Vectors (UMAP Reduction)',
        labels={'color': 'Predicted Cluster', 'symbol': 'True Country'}
    )
    fig.show() # פתיחת הגרף בדפדפן
    
    
    ## אפשרות 2: גרף סטטי (Matplotlib) - לצרכי שמירה
    plt.figure(figsize=(10, 8))
    
    # ציור לפי התווית האמיתית
    scatter = plt.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=results["mapped_clusters"], # צבע לפי תוצאת האשכול
        cmap='viridis',
        s=50,
        alpha=0.6
    )
    plt.title('K-Means Clusters (UMAP Reduction)')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    
    # הוספת מקרא
    legend1 = plt.legend(*scatter.legend_elements(), 
                         loc="lower left", title="Cluster")
    plt.gca().add_artist(legend1)
    
    plt.show()
