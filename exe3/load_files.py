import os
from tqdm import tqdm  

def load_text(folder_path):
    # יצירת רשימה של צ'אנקים (כל צ'אנק הוא תוכן של קובץ טקסט)
    corpus = []
    filenames = []
    
    # הוספת פס התקדמות
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".txt"):
                filenames.append(os.path.join(root, filename))
    
    for filepath in tqdm(filenames, desc="Loading text files", unit="file"):
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
            corpus.append(text)
    
    return corpus, filenames
