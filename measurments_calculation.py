from collections import Counter
import json
import numpy as np
from scipy.sparse import load_npz
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.preprocessing import LabelEncoder
'''
pip install openpyxl

'''

def calculate_mutual_information(input_folder):
    """
    מחשב את ה-Information Gain (Mutual Information) עבור תכונות מבוססות TF-IDF.
    קורא את המטריצה השמורה ואת המילון, ומחשב את ה-Information Gain לכל תכונה.
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
    mi_df = pd.DataFrame({"Feature": feature_names, "Mutual Information": mi})
    mi_df = mi_df.sort_values(by="Mutual Information", ascending=False)

    return mi_df


def calculate_information_gain(input_folder):
    # פונקציה לחישוב Entropy
    def entropy(labels):
        count = Counter(labels)
        total = len(labels)
        probs = [count[label] / total for label in count]
        return -sum(p * np.log2(p) for p in probs)

    # פונקציה לחישוב Information Gain
    def information_gain(data, target, feature_column):
        # חישוב ה-Entropy הכללי של היעד (המטרה)
        total_entropy = entropy(target)
        
        # חישוב ה-Entropy לכל ערך במאפיין (feature)
        values = np.unique(data[:, feature_column])  # ערכים במאפיין
        weighted_entropy = 0
        for value in values:
            subset = target[data[:, feature_column] == value]
            weighted_entropy += (len(subset) / len(target)) * entropy(subset)
        
        # חישוב ה-Information Gain
        return total_entropy - weighted_entropy

    # קריאת המילים מתוך קובץ ה-vocabulary # שים כאן את הנתיב הנכון
    with open(f"{input_folder}/BM25_Word_vocabulary.json", "r", encoding="utf-8") as f:
        vocabulary = json.load(f)

    # הוצאת המילים מתוך המילון (keys) כדי להפיק רשימה של הפיצ'רים
    feature_names = list(vocabulary.keys())

    # טוען את מטריצת ה-TFIDF
    tfidf_matrix = load_npz(f"{input_folder}/BM25_Word.npz")

    # קריאת שמות הקבצים (כמו קטגוריות)
    file_names = pd.read_json(f"{input_folder}/BM25_Word_files.json")["files"].values

    # המרת שמות הקבצים לקטגוריות (תוויות)
    le = LabelEncoder()
    labels = le.fit_transform(file_names)

    # המרת המטריצה לפורמט numpy למחשוב נוח יותר
    tfidf_array = tfidf_matrix.toarray()

    # חישוב ה-Information Gain עבור כל מאפיין במטריצה
    ig_word = {}
    for col in range(tfidf_array.shape[1]):
        ig_word[feature_names[col]] = information_gain(tfidf_array, labels, col)

    # יצירת DataFrame עם התוצאות
    ig_word_df = pd.DataFrame(list(ig_word.items()), columns=['Feature', 'Information Gain'])
    return ig_word_df

def save_to_excel(input_folder, ig_df, mi_df):
    """
    שומר את התוצאות לקובץ Excel באותו דף עם שתי עמודות לכל מדד.
    """
    merged_df = pd.merge(ig_df, mi_df, on="Feature", how="outer")
    
    # שמירת התוצאות ב-Excel
    merged_df.to_excel(f"{input_folder}/tfidf_resultsV3.xlsx", index=False)
    print(f"✅ התוצאות נשמרו ב-{input_folder}/tfidf_resultsV3.xlsx")


if __name__ == "__main__":
    # חישוב ה-Information Gain ו- Chi-Square עבור כל מטריצה
    for tfidf_folder in ["vectors_word", "vectors_lemm"]:
        print(f"\nחישוב Information Gain ו-Chi-Square עבור: {tfidf_folder}")
        
        # חישוב ה-Information Gain
        ig_df = calculate_information_gain(tfidf_folder)

        # חישוב ה-Chi-Square
        mi_df = calculate_mutual_information(tfidf_folder)

        # שמירת התוצאות בקובץ Excel
        save_to_excel(tfidf_folder, ig_df, mi_df)
