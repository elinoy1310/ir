import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics.pairwise import cosine_distances
import hdbscan
import json
from pathlib import Path
from scipy import sparse
import umap
import matplotlib.pyplot as plt
import seaborn as sns

def plot_umap_results(x, predicted_labels, title,filename, save_path=Path("exe2")):
    """
    יוצר גרף UMAP של תוצאות הסיווג.
    """
    print(f"🎨 Generating UMAP plot: {title}...")
    
    # הפחתת מימדים ל-2 בעזרת UMAP
    reducer = umap.UMAP(random_state=42 ,metric='cosine')
    embedding = reducer.fit_transform(x)
    
    plt.figure(figsize=(10, 7))
    
    # יצירת סקאטר פלוט
    # אנו צובעים את הנקודות לפי הסיווג שהמודל חזה (y_pred)
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=predicted_labels, cmap='coolwarm', s=10, alpha=0.7)
    
    handles, labels = scatter.legend_elements()
    plt.legend(handles, labels, title="Predicted Class")

    plt.title(title)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    # שמירת הגרף
    save_path = save_path / filename
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Plot saved to {save_path}")

# --- פונקציה גנרית לוויזואליזציה ---
def visualize_clusters(X, cluster_labels, true_labels, n_neighbors=15, min_dist=0.1, title_prefix=""):
    """
    מבצע הפחתת מימד באמצעות UMAP ומשרטט את האשכולות.
    """
    print(f"\nStarting UMAP dimensionality reduction for {title_prefix}...")
    
    # הפחתת מימד באמצעות UMAP
    reducer = umap.UMAP(
        n_components=2,
        # n_neighbors=n_neighbors,
        # min_dist=min_dist,
        metric='cosine', # עקביות עם מרחק
        random_state=42
    )
    
    # המרה לצפוף אם דליל
   # X_dense = X.toarray() if sparse.issparse(X) else X
        
    embedding = reducer.fit_transform(X)
    print("UMAP reduction complete.")

    plt.figure(figsize=(14, 6))
    
    # -------------------
    # ציור 1: תוצאות האשכולות
    # -------------------
    plt.subplot(1, 2, 1) 
    
    # מספר ייחודי של תוויות (כולל רעש -1)
    n_clusters = len(np.unique(cluster_labels))
    sns.scatterplot(
        x=embedding[:, 0], 
        y=embedding[:, 1], 
        hue=cluster_labels, 
        palette=sns.color_palette("hls", n_clusters),
        legend="full",
        s=50,
        alpha=0.6
    )
    plt.title(f'{title_prefix} Clusters (Colored by Cluster ID)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    
    # -------------------
    # ציור 2: תוויות האמת
    # -------------------
    plt.subplot(1, 2, 2) 
    
    # צובעים לפי תוויות האמת (0 ו-1)
    sns.scatterplot(
        x=embedding[:, 0], 
        y=embedding[:, 1], 
        hue=true_labels, 
        palette=['skyblue', 'salmon'],
        legend='full',
        s=50,
        alpha=0.6
    )
    plt.title('True Labels (0=UK, 1=US)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')

    plt.tight_layout() 
    plt.show()


# --- פונקציה גנרית להערכת אשכולות ---
def evaluate_clustering(clusterer, X, y, title="Clustering"):
    """
    מבצעת אישכול, מיפוי תוצאות, הערכת מדדים והצגת תוצאות.
    
    clusterer: אובייקט מאשכל מאומן (או כזה שטרם אומן) כמו HDBSCAN, DBSCAN, KMeans וכו'.
    X: מטריצת TF-IDF (צפופה או דלילה)
    y: תוויות אמת (0 = UK, 1 = US)
    title: כותרת להצגת התוצאות
    """
    print(f"\n--- Running {title} Evaluation ---")
    
    # 1. חיזוי תוויות
    # אם זה HDBSCAN/DBSCAN עם precomputed, אנו צריכים מטריצת מרחק
    if isinstance(clusterer, hdbscan.HDBSCAN) and clusterer.metric == "precomputed":
        dist_matrix = cosine_distances(X)
        cluster_labels = clusterer.fit_predict(dist_matrix)
    else:
        # עבור מאשכלים אחרים שפועלים ישירות על X (כמו K-Means)
        # מכיוון ש-X הוא sparse, נשתמש ב-fit_predict ישירות (HDBSCAN/DBSCAN לא תמיד תומכים בזה)
        cluster_labels = clusterer.fit_predict(X)
    
    # 2. טיפול בנקודות רעש (בדרך כלל -1)
    noise_mask = (cluster_labels == -1)
    n_noise = int(noise_mask.sum())

    # אם כל הנקודות רעש, אין מה להעריך
    if (~noise_mask).sum() == 0:
        print(f"{title}: All points were labeled as noise.")
        return None

    # 3. מיפוי אשכולות לתוויות אמת (0/1)
    mask = ~noise_mask
    used_clusters = cluster_labels[mask]
    true_labels = y[mask]

    unique_clusters = np.unique(used_clusters)
    cluster_to_label = {}

    for c in unique_clusters:
        c_mask = (used_clusters == c)
        mean_label = true_labels[c_mask].mean()
        # מיפוי לפי רוב קולות: אם ממוצע התוויות האמיתיות >= 0.5, נסמן כ-1 (US), אחרת 0 (UK)
        cluster_to_label[c] = 1 if mean_label >= 0.5 else 0

    mapped_labels = np.array([cluster_to_label[c] for c in used_clusters])
    
    # 4. חישוב והצגת מדדים
    accuracy = accuracy_score(true_labels, mapped_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, mapped_labels, average="binary"
    )
    cm = confusion_matrix(true_labels, mapped_labels)

    # גודל כל אשכול
    cluster_sizes = {
        int(c): int((used_clusters == c).sum())
        for c in unique_clusters
    }

    # הדפסת התוצאות
    print("Total points :", len(cluster_labels))
    print("Noise points :", n_noise)
    print("Num clusters :", len(unique_clusters))
    print("Cluster sizes (cluster_id: size):", cluster_sizes)

    print("\nAccuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1       :", round(f1, 4))

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)
    
    # 5. ויזואליזציה
    visualize_clusters(
        X=X, 
        cluster_labels=cluster_labels, 
        true_labels=y,
        title_prefix=title
    )
    
    return {
        "accuracy": accuracy,
        "f1": f1
    }
