import os
from pathlib import Path
import scipy.sparse
import numpy as np
from load_files import load_text
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # יבוא של tqdm
from transformers import AutoTokenizer

MODEL='embaas/sentence-transformers-multilingual-e5-base'
# יצירת טוקניזר עבור המודל
tokenizer = AutoTokenizer.from_pretrained(MODEL)



# פונקציה לחיתוך טקסטים אם הם ארוכים מדי (לפי 512 טוקנים)
def chunk_text(text, max_tokens=512):
    # טוקניזציה של הטקסט
    tokens = tokenizer.encode(text)
    
    # חיתוך הטקסט לצ'אנקים לפי אורך הטוקנים
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    
    # המרת כל צ'אנק חזרה לטקסט
    return [tokenizer.decode(chunk) for chunk in chunks]

def generate_embeddings(texts, filenames, output_folder):
    # יצירת מודל Sentence-Transformer
    model = SentenceTransformer(MODEL)  # המודל החדש
    
    # חישוב האימבדינגים לכל הצ'אנקים עם פס התקדמות
    embeddings = []
    for text in tqdm(texts, desc="Generating embeddings", unit="chunk"):
        # חיתוך הטקסטים אם הם ארוכים מדי
        chunked_texts = chunk_text(text)
        
        chunk_embeddings = []
        # חישוב האימבדינגים עבור כל צ'אנק
        for chunk in chunked_texts:
            embedding = model.encode(chunk, convert_to_tensor=True)
            chunk_embeddings.append(embedding)
        
        # חישוב ממוצע של האימבדינגים של הצ'אנקים עבור הטקסט
        avg_embedding = np.mean(chunk_embeddings, axis=0)
        embeddings.append(avg_embedding)
    
    # המרת האימבדינגים לפורמט Sparse
    embeddings = np.array(embeddings)
    sparse_embeddings = scipy.sparse.csr_matrix(embeddings)

    # יצירת תיקיית פלט אם לא קיימת
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # שמירת האימבדינגים בפורמט Sparse (.npz)
    embeddings_filename = os.path.join(output_folder, 'embeddings_sparse.npz')
    scipy.sparse.save_npz(embeddings_filename, sparse_embeddings)

    # שמירה של שמות הקבצים כקובץ טקסט נפרד
    filenames_filename = os.path.join(output_folder, 'filenames.txt')
    with open(filenames_filename, 'w') as f:
        for filename in filenames:
            f.write(f"{filename}\n")

    print(f"Sparse embeddings saved to {embeddings_filename}")
    print(f"Filenames saved to {filenames_filename}")


# def generate_embeddings(texts, filenames, output_folder):
#     # יצירת מודל Sentence-Transformer
#     model = SentenceTransformer('embaas/sentence-transformers-multilingual-e5-base')
#   # ניתן לשנות את המודל אם צריך

#     # חישוב האימבדינגים לכל הצ'אנקים עם פס התקדמות
#     embeddings = []
#     for text in tqdm(texts, desc="Generating embeddings", unit="chunk"):
#         # חיתוך הטקסטים אם הם ארוכים מדי
#         chunked_texts = chunk_text(text)
        
#         # חישוב האימבדינגים עבור כל צ'אנק
#         for chunk in chunked_texts:
#             embedding = model.encode(chunk, convert_to_tensor=True)
#             embeddings.append(embedding)
    
#     # המרת האימבדינגים לפורמט Sparse
#     embeddings = np.array(embeddings)
#     sparse_embeddings = scipy.sparse.csr_matrix(embeddings)

#     # יצירת תיקיית פלט אם לא קיימת
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # שמירת האימבדינגים בפורמט Sparse (.npz)
#     embeddings_filename = os.path.join(output_folder, 'embeddings_sparse.npz')
#     scipy.sparse.save_npz(embeddings_filename, sparse_embeddings)

#     # שמירה של שמות הקבצים כקובץ טקסט נפרד
#     filenames_filename = os.path.join(output_folder, 'filenames.txt')
#     with open(filenames_filename, 'w') as f:
#         for filename in filenames:
#             f.write(f"{filename}\n")

#     print(f"Sparse embeddings saved to {embeddings_filename}")
#     print(f"Filenames saved to {filenames_filename}")

if __name__ == "__main__":
    uk_chanks_texts, uk_filenames = load_text(r"exe3\fixed-chunked-text\UK")
    us_chanks_texts, us_filenames = load_text(r"exe3\fixed-chunked-text\US")
    print(f"Loaded {len(uk_chanks_texts)} UK chunks and {len(us_chanks_texts)} US chunks.")
    all_texts = uk_chanks_texts + us_chanks_texts
    all_filenames = uk_filenames + us_filenames
    print(f"Total chunks: {len(all_texts)}")
    out_directory = Path(r"exe3\st_vectors")
    generate_embeddings(all_texts, all_filenames, out_directory)
