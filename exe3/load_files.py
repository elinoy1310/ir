import os
from tqdm import tqdm  

def load_text(folder_path):
    # יצירת רשימה של צ'אנקים (כל צ'אנק הוא תוכן של קובץ טקסט)
    corpus = []
    filenames = [filename for filename in os.listdir(folder_path) if filename.endswith(".txt")]
    
    # הוספת פס התקדמות
    for filename in tqdm(filenames, desc="Loading text files", unit="file"):  # פס התקדמות עם שם ותיאור
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            corpus.append(text)
    
    return corpus, filenames
