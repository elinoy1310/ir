#exe2/stage_3c_svm.py
from important_features import load_data, run_cross_validation, extract_top_features
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline

def run_svm():
    # טעינת הנתונים
    X, y, features_names = load_data()
    
    # הגדרת מודל Linear SVM
    # LinearSVC הוא יעיל ומתאים לבעיות עם הרבה מאפיינים.
    # מכיוון ש-SVM רגיש לסקאלת הנתונים, נשתמש ב-Pipeline עם Scaler מתאים למטריצות דלילות.
    model = Pipeline([
        ('scaler', MaxAbsScaler()), # שומר על דלילות המטריצה ומבצע סקייל (מומלץ ל-TF-IDF)
        ('svc', LinearSVC(random_state=42, max_iter=1000, C=1.0))
    ])
    
    # הרצת Cross-Validation
    run_cross_validation(model, X, y, "Support Vector Machine (SVM)")
    
    # אימון על כל הנתונים לצורך חילוץ מאפיינים
    model.fit(X, y)
    
    # חילוץ ושמירת מאפיינים
    return extract_top_features(model['svc'], features_names)

if __name__ == "__main__":
    run_svm()