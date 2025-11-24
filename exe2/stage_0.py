import os
from glob import glob

def load_corpus(base_dir):
    uk_dir = os.path.join(base_dir, "UK")
    us_dir = os.path.join(base_dir, "US")

    texts = []
    labels = []   # 0 = UK, 1 = US
    filenames = []

    # UK
    for path in sorted(glob(os.path.join(uk_dir, "*.txt"))):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
        labels.append(0)
        filenames.append(os.path.basename(path))

    # US
    for path in sorted(glob(os.path.join(us_dir, "*.txt"))):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
        labels.append(1)
        filenames.append(os.path.basename(path))

    return texts, labels, filenames

# שימוש:
# base_dir = r"C:\Users\user\Desktop\שנה ד\איחזור מידע\ir\parliament_small"
# texts, labels, filenames = load_corpus(base_dir)
