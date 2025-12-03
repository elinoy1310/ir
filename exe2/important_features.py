# utils.py
import numpy as np
import pandas as pd
from scipy import sparse
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import ClassifierMixin
from typing import List, Tuple, Dict

# הגדרת תיקיית הבסיס - הנחתי שהסקריפט מורץ מתיקיית ה-project_root
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output_features"
OUTPUT_DIR.mkdir(exist_ok=True) # יצירת תיקיית פלט

def load_data(base_dir: Path = BASE_DIR) -> Tuple[sparse.csr_matrix, np.ndarray, List[str]]:
    """טוען את מטריצת ה-TF-IDF, התוויות ושמות המאפיינים כפי שצוין."""
    try:
        # 1. טעינת מטריצת TFIDF
        X_tfidf = sparse.load_npz(base_dir / r"vectors_tfidf\TFIDF-Documents.npz")
        
        # 2. טעינת התוויות
        labels = np.array(json.loads((base_dir / r"vectors_tfidf\TFIDF-Documents_labels.json").read_text())['labels'])
        
        # 3. טעינת שמות המאפיינים
        # המפתח features_names הוא ה-keys של מילון האוצר המילים
        features_dict = json.loads((base_dir / r"vectors_tfidf\TFIDF-Documents_vocabulary.json").read_text())
        features_names = list(features_dict.keys())
        
        print(f"✅ נתונים נטענו בהצלחה: {X_tfidf.shape[0]} מסמכים, {X_tfidf.shape[1]} מאפיינים.")
        # ודא כי מספר המסמכים תואם למספר התוויות
        if X_tfidf.shape[0] != len(labels):
             print("⚠️ אזהרה: חוסר התאמה בין מספר המסמכים במטריצה למספר התוויות.")
             
        return X_tfidf, labels, features_names

    except FileNotFoundError as e:
        print(f"❌ שגיאה בטעינת הקבצים. אנא ודא שהנתיב הבסיסי נכון ושהקבצים קיימים: {e}")
        # יצירת נתוני דמה להרצת הדוגמה אם הקבצים חסרים
        print("יצירת נתוני דמה לצורך הרצה ובדיקה של הקוד...")
        return create_dummy_data()
    except Exception as e:
         print(f"❌ שגיאה לא צפויה: {e}")
         return create_dummy_data()

def create_dummy_data():
    """יוצר נתוני דמה לצורך הפעלת הסקריפט כאשר אין קבצים אמיתיים."""
    X_tfidf = sparse.csr_matrix(np.random.rand(100, 1000))
    labels = np.random.randint(0, 2, 100) # סיווג בינארי (0 ו-1)
    features_names = [f"word_{i}" for i in range(1000)]
    return X_tfidf, labels, features_names


def run_cross_validation(model: ClassifierMixin, X: sparse.csr_matrix, y: np.ndarray, model_name: str):
    """מריץ 10-fold Stratified Cross-Validation ומציג תוצאות."""
    
    # StratifiedKFold מבטיח שיחס התוויות נשמר בכל Fold.
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    
    print(f"\n--- תוצאות 10-Fold Cross-Validation עבור {model_name} ---")
    
    # n_jobs=-1 משתמש בכל הליבות הזמינות להאצת התהליך
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1, verbose=0)
    
    # הדפסת התוצאות הממוצעות
    print(f"Accuracy ממוצע: {results['test_accuracy'].mean():.4f} (סטיית תקן: {results['test_accuracy'].std():.4f})")
    print(f"F1-Score ממוצע: {results['test_f1_macro'].mean():.4f} (סטיית תקן: {results['test_f1_macro'].std():.4f})")
    print("-" * 50)
    
    return results

# def extract_and_save_top_features(
#     model: ClassifierMixin, 
#     features_names: List[str], 
#     model_name: str, 
#     num_features: int = 20, 
#     class_labels: Tuple[int, int] = (0, 1) # הנחה של סיווג בינארי
# ):
#     """
#     מחפש את 20 המאפיינים החשובים ביותר לכל קבוצה (כיתה) ושומר לקובץ אקסל.
    
#     מנגנון החילוץ מותאם לסוג המודל:
#     - Logistic Regression, SVM (Linear): מקדמים (Coefficients).
#     - Naive Bayes: הפרש של feature_log_prob_.
#     - Random Forest: חשיבות מאפיין כוללת (Feature Importance).
#     """
    
#     print(f"\n--- חילוץ ושמירת מאפיינים חשובים עבור {model_name} ---")
    
#     # אימון המודל על כל הנתונים כדי לקבל את המקדמים/חשיבויות הסופיות
#     # יש לוודא שהמודל אומן לפני הקריאה לפונקציה, אך נעשה זאת כאן לשם ודאות.
#     # המודל כבר מאומן אם הוא הועבר מפונקציית cross_validation, אך נבצע אימון נוסף (fit) קצרצר
#     # כדי לוודא שיש לו את כל המאפיינים הנדרשים (coef_, feature_importances_ וכו').
#     # בפועל, בפונקציות הנפרדות, נבצע fit לפני קריאה ל-extract_and_save_top_features.
    
#     weights = None
#     df = pd.DataFrame(index=features_names)
    
#     # --- 1. חילוץ המשקולות (Weights) בהתאם למודל ---
    
#     if hasattr(model, 'coef_'):
#         # מודלים לינאריים: Logistic Regression, Linear SVM
#         # coef_.shape הוא (1, num_features) עבור סיווג בינארי.
#         weights = model.coef_[0]
#         # נשמור את המשקולות כחלק מה-DataFrame
#         df['Weight'] = weights
#         df['Abs_Weight'] = np.abs(weights)
#         df['Feature'] = features_names

#         # Class 1 (מקדם חיובי):
#         top_c1 = df.sort_values(by='Weight', ascending=False).head(num_features)
#         # Class 0 (מקדם שלילי):
#         top_c0 = df.sort_values(by='Weight', ascending=True).head(num_features)
#         # נהפוך את המשקולת לערך מוחלט לצורך התצוגה, אך נזכור את המשמעות ב-README
#         top_c0['Weight'] = top_c0['Abs_Weight']
        
        
#     elif hasattr(model, 'feature_importances_'):
#         # מודלים מבוססי עצים: Random Forest
#         importances = model.feature_importances_
#         df['Weight'] = importances
#         df['Abs_Weight'] = importances # ב-RF אין ערך שלילי
#         df['Feature'] = features_names
        
#         # מכיוון ש-RF נותן חשיבות כוללת (לא ספציפית לקבוצה), נשתמש בדירוג הכללי:
#         top_features = df.sort_values(by='Weight', ascending=False).head(num_features)
        
#         # כדי לעמוד בדרישת המשתמש למאפיינים לכל קבוצה, נחלק את ה-20 המובילים לשתי הקבוצות
#         # עם הערה ברורה ב-README שמדובר בחשיבות כוללת.
#         top_c1 = top_features.iloc[:num_features // 2] if num_features > 1 else top_features.copy()
#         top_c0 = top_features.iloc[num_features // 2:num_features] if num_features > 1 else top_features.copy()
        
#         print(f"⚠️ הערה: עבור {model_name}, נשמרה חשיבות מאפיין כוללת (לא ספציפית לקבוצה).")

#     elif hasattr(model, 'feature_log_prob_'):
#         # מודלים הסתברותיים: Naive Bayes (MultinomialNB)
#         # feature_log_prob_.shape הוא (num_classes, num_features)
        
#         # ההפרש log(P(f|C1)) - log(P(f|C0))
#         diff = model.feature_log_prob_[class_labels[1]] - model.feature_log_prob_[class_labels[0]]
        
#         df['Weight'] = diff
#         df['Abs_Weight'] = np.abs(diff)
#         df['Feature'] = features_names
        
#         # Class 1 (הפרש חיובי גדול)
#         top_c1 = df.sort_values(by='Weight', ascending=False).head(num_features)
#         # Class 0 (הפרש שלילי גדול)
#         top_c0 = df.sort_values(by='Weight', ascending=True).head(num_features)
#         # נהפוך את המשקולת לערך מוחלט לצורך התצוגה
#         top_c0['Weight'] = top_c0['Abs_Weight']


#     else:
#         print(f"❌ לא ניתן לחלץ מאפיינים חשובים מהמודל {model_name}.")
#         return

#     # --- 2. שמירה לקובץ Excel ---
    
#     # הכנת הפלט לקובץ Excel
#     writer = pd.ExcelWriter(OUTPUT_DIR / f'top_features_{model_name}.xlsx', engine='xlsxwriter')
    
#     # טבלת סיכום לקבוצה 1 (למשל: American)
#     top_c1[['Feature', 'Weight']].sort_values(by='Weight', ascending=False).to_excel(
#         writer, 
#         sheet_name=f'Class_{class_labels[1]}_Top_{num_features}', 
#         index=False,
#         header=['מאפיין', f'משקולת (ערך מוחלט)'])
        
#     # טבלת סיכום לקבוצה 0 (למשל: British)
#     top_c0[['Feature', 'Weight']].sort_values(by='Weight', ascending=False).to_excel(
#         writer, 
#         sheet_name=f'Class_{class_labels[0]}_Top_{num_features}', 
#         index=False,
#         header=['מאפיין', f'משקולת (ערך מוחלט)'])
        
#     writer.close()
#     print(f"✅ מאפיינים חשובים נשמרו בהצלחה בקובץ: top_features_{model_name}.xlsx")

def extract_top_features(
    model: ClassifierMixin, 
    features_names: List[str], 
    num_features: int = 20,
    class_labels: Tuple[int, int] = (0, 1)
) -> Dict[str, pd.DataFrame]:
    """
    מחזיר מילון עם שני DataFrames: אחד לקבוצה 0 ואחד לקבוצה 1.
    """
    df = pd.DataFrame(index=features_names)
    top_c0 = None
    top_c1 = None
    
    # 1. Linear Models (LoR, SVM)
    if hasattr(model, 'coef_'):
        weights = model.coef_[0]
        df['Weight'] = weights
        df['Abs_Weight'] = np.abs(weights)
        df['Feature'] = features_names
        
        # Class 1 (Positive weights)
        top_c1 = df.sort_values(by='Weight', ascending=False).head(num_features)
        # Class 0 (Negative weights)
        top_c0 = df.sort_values(by='Weight', ascending=True).head(num_features)
        
        # Formatting
        top_c0 = top_c0[['Feature', 'Abs_Weight']].rename(columns={'Abs_Weight': 'Weight'})
        top_c1 = top_c1[['Feature', 'Abs_Weight']].rename(columns={'Abs_Weight': 'Weight'})

    # 2. Random Forest (Feature Importances)
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        df['Weight'] = importances
        df['Feature'] = features_names
        
        # RF gives global importance
        top_features = df.sort_values(by='Weight', ascending=False).head(num_features)
        
        # תיקון: בחירה מפורשת של סדר העמודות ['Feature', 'Weight']
        top_c1 = top_features[['Feature', 'Weight']].copy()
        top_c0 = top_features[['Feature', 'Weight']].copy()

    # 3. Naive Bayes (Log Probabilities)
    elif hasattr(model, 'feature_log_prob_'):
        diff = model.feature_log_prob_[class_labels[1]] - model.feature_log_prob_[class_labels[0]]
        df['Weight'] = diff
        df['Abs_Weight'] = np.abs(diff)
        df['Feature'] = features_names
        
        top_c1 = df.sort_values(by='Weight', ascending=False).head(num_features)[['Feature', 'Abs_Weight']].rename(columns={'Abs_Weight': 'Weight'})
        top_c0 = df.sort_values(by='Weight', ascending=True).head(num_features)[['Feature', 'Abs_Weight']].rename(columns={'Abs_Weight': 'Weight'})

    return {'class_0': top_c0, 'class_1': top_c1}

def create_consolidated_excel(all_models_data: Dict[str, Dict[str, pd.DataFrame]]):
    """
    מקבלת את כל הנתונים מכל המודלים ויוצרת קובץ אקסל אחד מרוכז.
    """
    print("\n--- Generating Consolidated Excel File ---")
    
    # רשימות לאגירת הנתונים לפי קבוצות
    dfs_class_0 = []
    dfs_class_1 = []
    
    for model_name, data in all_models_data.items():
        if data is None: continue # Skip models with no features (like ANN)
        
        # הכנת DataFrame לקבוצה 0
        c0 = data['class_0'].reset_index(drop=True)
        c0.columns = [f'{model_name}_Feature', f'{model_name}_Weight']
        dfs_class_0.append(c0)
        
        # הכנת DataFrame לקבוצה 1
        c1 = data['class_1'].reset_index(drop=True)
        c1.columns = [f'{model_name}_Feature', f'{model_name}_Weight']
        dfs_class_1.append(c1)
    
    # חיבור אופקי של כל הטבלאות
    final_df_0 = pd.concat(dfs_class_0, axis=1)
    final_df_1 = pd.concat(dfs_class_1, axis=1)
    
    # שמירה לאקסל
    output_path = OUTPUT_DIR / 'Unified_Top_Features.xlsx'
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        final_df_0.to_excel(writer, sheet_name='Class_0_Features', index=False)
        final_df_1.to_excel(writer, sheet_name='Class_1_Features', index=False)
        
    print(f"✅ Excel file created successfully: {output_path}")