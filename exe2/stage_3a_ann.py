# exe2/stage_3a_ann.py
'''
pip install tensorflow umap-learn matplotlib seaborn
'''

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.activations import relu, softmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from pathlib import Path

# ייבוא פונקציית הטעינה מקובץ העזר
from important_features import load_data, OUTPUT_DIR

# הגדרת Random Seed לתוצאות עקביות
tf.random.set_seed(42)
np.random.seed(42)

def get_activation_function(name):
    """מחזיר את פונקציית האקטיבציה המתאימה."""
    if name.lower() == 'relu':
        return 'relu'
    elif name.lower() == 'gelu':
        # GELU נתמך בגרסאות חדשות של TF. אם יש שגיאה, ניתן להשתמש בקירוב.
        return tf.keras.activations.gelu
    return 'relu'

def build_ann_model(input_dim, topology_type='relu'):
    """
    בונה את המודל בהתאם לטופולוגיה המבוקשת (א' או ב').
    ההבדל הוא בפונקציית האקטיבציה.
    """
    activation = get_activation_function(topology_type)
    
    model = Sequential()
    
    # שכבת קלט - מקבלת את וקטור ה-TF-IDF
    # הערה: מכיוון שהקלט הוא TF-IDF (ערכים רציפים), אנו משתמשים ב-Dense כשכבה ראשונה
    # במקום Embedding שמיועדת לקבלת אינדקסים של מילים.
    model.add(Input(shape=(input_dim,)))
    
    # שכבה שנייה Hidden layer - 10 קודקודים
    model.add(Dense(10, activation=activation, name=f"Hidden_1_{topology_type}"))
    
    # שכבה שלישית Hidden layer - 10 קודקודים
    model.add(Dense(10, activation=activation, name=f"Hidden_2_{topology_type}"))
    
    # שכבה רביעית Hidden layer - 7 קודקודים
    model.add(Dense(7, activation=activation, name=f"Hidden_3_{topology_type}"))
    
    # שכבה אחרונה Activation layer עם softmax
    # גודל 2 כיוון שיש לנו 2 מחלקות (בריטי/אמריקאי) וביקשו Softmax
    model.add(Dense(2, activation='softmax', name="Output"))
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def plot_umap_results(X_test, y_pred, title, filename):
    """
    יוצר גרף UMAP של תוצאות הסיווג.
    """
    print(f"🎨 Generating UMAP plot: {title}...")
    
    # הפחתת מימדים ל-2 בעזרת UMAP
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(X_test)
    
    plt.figure(figsize=(10, 7))
    
    # יצירת סקאטר פלוט
    # אנו צובעים את הנקודות לפי הסיווג שהמודל חזה (y_pred)
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y_pred, cmap='coolwarm', s=10, alpha=0.7)
    
    plt.colorbar(scatter, label='Predicted Class')
    plt.title(title)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    # שמירת הגרף
    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Plot saved to {save_path}")

def run_ann():
    print("==========================================")
    print("      Starting ANN Training Sequence      ")
    print("==========================================")
    
    # 1. טעינת הנתונים
    X, y, features_names = load_data()
    
    # המרה ל-Dense (Keras עובד מהר יותר עם Dense arrays בקנה מידה הזה)
    if hasattr(X, "toarray"):
        X = X.toarray()

    input_dim = X.shape[1]
    
    # 2. חלוקת הנתונים לפי הדרישה:
    # 80% למידה (שמתוכם 10% ולידציה), 20% בחינה.
    
    # שלב א: הפרדת ה-Test (20%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # שלב ב: מתוך ה-80% שנשארו, נפריד 10% ל-Validation
    # חישוב: 10% מתוך ה-Training Full Set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.10, random_state=42, stratify=y_train_full
    )
    
    print(f"Data Split Summary:")
    print(f"Total: {X.shape[0]}")
    print(f"Test (20%): {X_test.shape[0]}")
    print(f"Train Full (80%): {X_train_full.shape[0]} -> Split into: Train={X_train.shape[0]}, Val={X_val.shape[0]}")

    # הגדרת רשימת הטופולוגיות להרצה
    topologies = ['ReLU', 'GELU']
    
    for topo in topologies:
        print(f"\nTraining ANN with Topology: {topo}")
        print("-" * 30)
        
        # בניית המודל
        model = build_ann_model(input_dim, topology_type=topo)
        
        # הגדרת Callbacks
        checkpoint_path = OUTPUT_DIR / f"best_model_{topo}.keras"
        callbacks = [
            # עצירה אם אין שיפור ב-val_accuracy במשך 3 איטרציות
            EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, restore_best_weights=True),
            # שמירת המודל הטוב ביותר
            ModelCheckpoint(filepath=str(checkpoint_path), monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
        
        # אימון המודל
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=15,
            batch_size=16,
            callbacks=callbacks,
            verbose=1
        )
        
        # ביצוע תחזית על ה-Test Set
        # המודל מחזיר הסתברויות (Softmax), נבחר את האינדקס הגבוה ביותר
        y_prob = model.predict(X_test)
        y_pred = np.argmax(y_prob, axis=1)
        
        # הצגת תוצאות מספריות
        print(f"\n--- Results for ANN ({topo}) ---")
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
        
        # הצגת תוצאות ויזואליות (UMAP)
        plot_umap_results(
            X_test, 
            y_pred, 
            title=f"ANN ({topo}) Classification Results (UMAP)", 
            filename=f"umap_ann_{topo}.png"
        )
        
        # שמירת מטריצת הבלבול (אופציונלי)
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    run_ann()