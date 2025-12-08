# exe2/stage_3b_nb.py
from important_features import load_data, run_cross_validation, extract_top_features
from sklearn.naive_bayes import MultinomialNB

def run_naive_bayes():
    # טעינת הנתונים
    X, y, features_names = load_data()
    
    # הגדרת מודל Naive Bayes (MultinomialNB מתאים לנתוני ספירת מילים/TF-IDF)
    # alpha=1.0 הוא ה-Laplace smoothing, ערך ברירת מחדל טוב.
    model = MultinomialNB(alpha=1.0)
    
    # הרצת Cross-Validation
    run_cross_validation(model, X, y, "Naive Bayes (NB)")
    
    # אימון על כל הנתונים לצורך חילוץ מאפיינים
    model.fit(X, y)
    
    # חילוץ ושמירת מאפיינים
    return extract_top_features(model, features_names)

if __name__ == "__main__":
    run_naive_bayes()