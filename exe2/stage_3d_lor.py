# logistic_regression_classifier.py
from important_features import load_data, run_cross_validation, extract_top_features
from sklearn.linear_model import LogisticRegression

def run_logistic_regression():
    # טעינת הנתונים
    X, y, features_names = load_data()
    
    # הגדרת מודל Logistic Regression
    # solver='liblinear' טוב לסיווג בינארי ו-L1/L2, תומך במטריצות דלילות.
    model = LogisticRegression(solver='liblinear', random_state=42, C=1.0, penalty='l2', max_iter=1000)
    
    # הרצת Cross-Validation
    run_cross_validation(model, X, y, "Logistic Regression (LoR)")
    
    # אימון על כל הנתונים לצורך חילוץ מאפיינים
    model.fit(X, y)
    
    # חילוץ ושמירת מאפיינים
    return extract_top_features(model, features_names)

if __name__ == "__main__":
    run_logistic_regression()