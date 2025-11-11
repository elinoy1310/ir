import os

folder = r"C:\Users\user\Desktop\שנה ד\איחזור מידע\ir\clean_text" 
empty_files = []

for filename in os.listdir(folder):
    path = os.path.join(folder, filename)
    if os.path.isfile(path):
        size = os.path.getsize(path)
        if size == 0:
            empty_files.append(filename)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                if len(content) < 10: 
                    empty_files.append(filename)

print(f"נמצאו {len(empty_files)} קבצים ריקים או כמעט ריקים:")
for f in empty_files:
    print("-", f)
