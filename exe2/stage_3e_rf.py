# exe2/stage_3e_rf.py
from important_features import load_data, run_cross_validation, extract_top_features
from sklearn.ensemble import RandomForestClassifier

def run_random_forest():
    # טעינת הנתונים
    X, y, features_names = load_data()
    
    # הגדרת מודל Random Forest
    # n_estimators הוא מספר העצים. ככל שיותר, יותר מדויק אך איטי יותר.
    # max_depth מונע התאמת יתר (Overfitting).
    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    
    # הרצת Cross-Validation
    run_cross_validation(model, X, y, "Random Forest (RF)")
    
    # אימון על כל הנתונים לצורך חילוץ מאפיינים
    model.fit(X, y)
    
    # חילוץ ושמירת מאפיינים
    return extract_top_features(model, features_names)

if __name__ == "__main__":
    run_random_forest()