import os
from glob import glob
import numpy as np


def load_corpus(base_dir: str):
    """
    טוען את כל המסמכים משתי תיקיות: UK ו-US
    מחזיר:
      texts    – רשימת טקסטים (str) של המסמכים
      labels   – numpy array של תוויות: 0 = UK, 1 = US
      filenames – רשימת שמות קבצים (str)
    """
    uk_dir = os.path.join(base_dir, "UK")
    us_dir = os.path.join(base_dir, "US")

    texts = []
    labels = []
    filenames = []

    # --- UK = 0 ---
    for path in sorted(glob(os.path.join(uk_dir, "*.txt"))):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
        labels.append(0)
        filenames.append(os.path.basename(path))

    # --- US = 1 ---
    for path in sorted(glob(os.path.join(us_dir, "*.txt"))):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
        labels.append(1)
        filenames.append(os.path.basename(path))

    return texts, np.array(labels), filenames


if __name__ == "__main__":
    texts, labels, filenames = load_corpus("")

    print("Total documents:", len(texts))
    print("Num UK (0):", int((labels == 0).sum()))
    print("Num US (1):", int((labels == 1).sum()))
    print("First filename:", filenames[0] if filenames else "N/A")
    print("First label:", labels[0] if len(labels) else "N/A")
    print("First document preview:")
    if texts:
        print(texts[0][:200], "...")
