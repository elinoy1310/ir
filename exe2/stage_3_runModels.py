import sys
from important_features import create_consolidated_excel

# ייבוא הפונקציות מהקבצים הנפרדים
# הערה: וודא שכל הקבצים נמצאים באותה תיקייה
try:
    from stage_3a_ann import run_ann
    from stage_3d_lor import run_logistic_regression
    from stage_3c_svm import run_svm
    from stage_3b_nb import run_naive_bayes
    from stage_3e_rf import run_random_forest
except ImportError as e:
    print("❌ שגיאה בייבוא אחד הקבצים. וודא ששמות הקבצים תואמים ושהם באותה תיקייה.")
    print(f"Details: {e}")
    sys.exit(1)

def main():
    print("🚀 Starting Classification & Feature Extraction Pipeline...\n")
    
    # מילון לאגירת התוצאות מכל מודל
    all_results = {}

    # 1. ANN (No features to extract)
    print(">>> Processing ANN...(already exist)")
    # run_ann() # רק מדפיס תוצאות CV
    
    # 2. Naive Bayes
    print("\n>>> Processing Naive Bayes...")
    all_results['NB'] = run_naive_bayes()

    # 3. SVM
    print("\n>>> Processing SVM...")
    all_results['SVM'] = run_svm()

    # 4. Logistic Regression
    print("\n>>> Processing Logistic Regression...")
    all_results['LoR'] = run_logistic_regression()

    # 5. Random Forest
    print("\n>>> Processing Random Forest...")
    all_results['RF'] = run_random_forest()

    # יצירת קובץ אקסל מרוכז
    create_consolidated_excel(all_results)
    
    print("\n🏁 Process Finished Successfully!")

if __name__ == "__main__":
    main()