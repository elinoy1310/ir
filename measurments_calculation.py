import json
import numpy as np
from scipy.sparse import load_npz
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

def calculate_information_gain(input_folder):
    """
    מחשב את ה-Information Gain (Mutual Information) עבור תכונות מבוססות TF-IDF.
    קורא את המטריצה השמורה ואת המילון, ומחשב את ה-Information Gain לכל תכונה.
    התוצאות נשמרות בקובץ CSV.
    """
    # קריאת המטריצה ששמרת (TFIDF)
    tfidf_matrix = load_npz(f"{input_folder}/BM25_Word.npz")

    # קריאת המילים מתוך קובץ ה-vocabulary
    with open(f"{input_folder}/BM25_Word_vocabulary.json", "r", encoding="utf-8") as f:
        vocabulary = json.load(f)

    # הוצאת המילים מתוך המילון (keys) כדי להפיק רשימה של הפיצ'רים
    feature_names = list(vocabulary.keys())

    # קריאת שמות הקבצים (למשל, ייצוג של קטגוריות)
    file_names = pd.read_json(f"{input_folder}/BM25_Word_files.json")["files"].values

    # המרת שמות הקבצים לקטגוריות (למשל, כל קובץ הוא קטגוריה)
    le = LabelEncoder()
    labels = le.fit_transform(file_names)

    # חישוב ה-Information Gain (Mutual Information)
    mi = mutual_info_classif(tfidf_matrix, labels)

    # יצירת DataFrame להציג את התוצאות
    mi_df = pd.DataFrame({"Feature": feature_names, "Information Gain": mi})
    mi_df = mi_df.sort_values(by="Information Gain", ascending=False)

    # הצגת התוצאה ושמירה לקובץ CSV
    mi_df.to_csv("tfidf_information_gain.csv", index=False)
    print(mi_df.head())

def calculate_chi2(input_folder): 
    import json
    import numpy as np
    from scipy.sparse import load_npz
    import pandas as pd
    from sklearn.feature_selection import chi2
    from sklearn.preprocessing import LabelEncoder

    # קריאת המטריצה ששמרת (TFIDF)
    tfidf_matrix = load_npz(f"{input_folder}/BM25_Word.npz")

    # קריאת המילים מתוך קובץ ה-vocabulary
    with open(f"{input_folder}/BM25_Word_vocabulary.json", "r", encoding="utf-8") as f:
        vocabulary = json.load(f)

    # הוצאת המילים מתוך המילון (keys) כדי להפיק רשימה של הפיצ'רים
    feature_names = list(vocabulary.keys())

    # קריאת שמות הקבצים (למשל, ייצוג של קטגוריות)
    file_names = pd.read_json(f"{input_folder}/BM25_Word_files.json")["files"].values

    # המרת שמות הקבצים לקטגוריות (למשל, כל קובץ הוא קטגוריה)
    le = LabelEncoder()
    labels = le.fit_transform(file_names)

    # חישוב ה-Chi-Square Statistic עבור כל פיצ'ר במטריצה
    chi2_values, p_values = chi2(tfidf_matrix, labels)

    # יצירת DataFrame להציג את התוצאות
    chi2_df = pd.DataFrame({
        "Feature": feature_names,
        "Chi-Square": chi2_values,
        "P-Value": p_values
    })

    # מיון התוצאות לפי ערך ה-chi-square בסדר יורד
    chi2_df = chi2_df.sort_values(by="Chi-Square", ascending=False)

    # הצגת התוצאה ושמירה לקובץ CSV
    chi2_df.to_csv("tfidf_chi_square.csv", index=False)
    print(chi2_df.head())


if __name__ == "__main__":
    print("info gain calculation")
    print("calc for not lemmatized text")
    calculate_information_gain(input_folder="vectors_word")
    print("calc for lemmatized text")
    calculate_information_gain(input_folder="vectors_lemm")
    print("chi2 calculation")
    print("calc for not lemmatized text")
    calculate_chi2(input_folder="vectors_word")
    print("calc for lemmatized text")
    calculate_chi2(input_folder="vectors_lemm")