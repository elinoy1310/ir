import os
from pathlib import Path
import scipy.sparse
import numpy as np
from load_files import load_text
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # יבוא של tqdm
from transformers import AutoTokenizer

#MODEL='embaas/sentence-transformers-multilingual-e5-base'
# יצירת טוקניזר עבור המודל


MODEL_NAME =MODEL= 'intfloat/multilingual-e5-small'


# פונקציה לחיתוך טקסטים אם הם ארוכים מדי (לפי 512 טוקנים)
def chunk_text(text, max_tokens=512):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # טוקניזציה של הטקסט
    tokens = tokenizer.encode(text)
    final_max_tokens = max_tokens - 2 -len(tokenizer.encode("passage: ")) # להוריד 2 לטוקנים מיוחדים [CLS] ו-[SEP]
    # חיתוך הטקסט לצ'אנקים לפי אורך הטוקנים
    chunks = [tokens[i:i + final_max_tokens] for i in range(0, len(tokens), final_max_tokens)]
    
    # המרת כל צ'אנק חזרה לטקסט
    return ["passage: " + tokenizer.decode(chunk) for chunk in chunks]


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

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# שינוי המודל לגרסה הקטנה והמהירה

def generate_embeddings_optimized(texts, filenames, output_folder):
    # זיהוי אם יש כרטיס גרפי (GPU) - קריטי למהירות!
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    all_embeddings = []
    
    # עבודה ב-Batches במקום צ'אנק-צ'אנק
    # SentenceTransformer יודע לעשות batching פנימי אם נותנים לו רשימה
    for text in tqdm(texts, desc="Processing documents"):
        chunks = chunk_text(text)
        
        # חישוב כל הצ'אנקים של המסמך במכה אחת
        # המודל יריץ אותם במקביל על ה-GPU
        chunk_embeddings = model.encode(chunks, convert_to_numpy=True, batch_size=32)
        
        # ממוצע צ'אנקים
        avg_embedding = np.mean(chunk_embeddings, axis=0)
        all_embeddings.append(avg_embedding)
    
    # המרה למערך NumPy אחד גדול
    final_matrix = np.vstack(all_embeddings).astype('float32')

    # שמירה כקובץ NumPy רגיל (לא Sparse!)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'embeddings.npy')
    np.save(output_path, final_matrix)

    # שמירה של שמות הקבצים כקובץ טקסט נפרד
    filenames_filename = os.path.join(output_folder, 'filenames.txt')
    with open(filenames_filename, 'w') as f:
        for filename in filenames:
            f.write(f"{filename}\n")
    
    print(f"Saved {final_matrix.shape} matrix to {output_path}")


if __name__ == "__main__":
    # uk_chanks_texts, uk_filenames = load_text(r"exe3\fixed-chunked-text\UK")
    # us_chanks_texts, us_filenames = load_text(r"exe3\fixed-chunked-text\US")
    # print(f"Loaded {len(uk_chanks_texts)} UK chunks and {len(us_chanks_texts)} US chunks.")
    # all_texts = uk_chanks_texts + us_chanks_texts
    # all_filenames = uk_filenames + us_filenames
    # print(f"Total chunks: {len(all_texts)}")
    # out_directory = Path(r"exe3\st_vectors")
    # generate_embeddings(all_texts, all_filenames, out_directory)
    uk_chanks_texts, uk_filenames = load_text(r"exe3\chunked-text\UK")
    us_chanks_texts, us_filenames = load_text(r"exe3\chunked-text\US")
    print(f"Loaded {len(uk_chanks_texts)} UK chunks and {len(us_chanks_texts)} US chunks.")
    all_texts = uk_chanks_texts + us_chanks_texts
    all_filenames = uk_filenames + us_filenames
    print(f"Total chunks: {len(all_texts)}")
    out_directory = Path(r"exe3\st_vectors_fixed_chunks")
    # במקום לשלוח את הטקסט כמו שהוא:
    #texts_with_prefix = ["passage: " + t for t in all_texts]
    generate_embeddings_optimized(all_texts, all_filenames, out_directory)
    # uk_chanks_texts, uk_filenames = load_text(r"exe3\parent-child-chunked-text\UK\children")
    # us_chanks_texts, us_filenames = load_text(r"exe3\parent-child-chunked-text\US\children")
    # print(f"Loaded {len(uk_chanks_texts)} UK chunks and {len(us_chanks_texts)} US chunks.")
    # all_texts = uk_chanks_texts + us_chanks_texts
    # all_filenames = uk_filenames + us_filenames
    # print(f"Total chunks: {len(all_texts)}")
    # out_directory = Path(r"exe3\st_vectors_parentSon_chunks")
    # # במקום לשלוח את הטקסט כמו שהוא:
    # #texts_with_prefix = ["passage: " + t for t in all_texts]
    # generate_embeddings_optimized(all_texts, all_filenames, out_directory)