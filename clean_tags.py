import os
from lxml import etree

def clean_xml_folder(input_folder, output_folder="clean_text"):
    """
    קורא את כל קבצי ה-XML בתיקייה, מנקה את התגים ומשאיר רק את הטקסט ביניהם.
    משתמש ב-lxml לעיבוד, ומדלג על קבצים ריקים או פגומים.
    
    :param input_folder: נתיב לתיקייה עם קבצי XML
    :param output_folder: נתיב לתיקייה לשמירת הקבצים הנקיים (נוצרה אוטומטית אם אינה קיימת)
    """

    # יצירת תיקייה ליצוא אם לא קיימת
    os.makedirs(output_folder, exist_ok=True)

    # מעבר על כל הקבצים בתיקייה
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".xml"):
            continue  # דילוג על קבצים לא XML

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".xml", ".txt"))

        try:
            # קריאה ופרסינג עם lxml
            tree = etree.parse(input_path)
            root = tree.getroot()
            text = "".join(root.itertext())  # חילוץ כל הטקסט שבין התגים

            # ניקוי רווחים מיותרים
            text = " ".join(text.split())

            # שמירה לקובץ חדש
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text.strip())

            print(f"✔ נוקה בהצלחה: {filename}")

        except Exception as e:
            # טיפול בשגיאות פרסינג
            print(f"⚠ שגיאה בעיבוד הקובץ {filename}: {e}")
            with open(os.path.join(output_folder, "error_log.txt"), "a", encoding="utf-8") as log:
                log.write(f"{filename}: {e}\n")


# --- דוגמת הרצה ---
if __name__ == "__main__":
    # לדוגמה, אם הקבצים נמצאים בתיקייה "raw_xml"
    clean_xml_folder(input_folder="debates_xml", output_folder="clean_text")
    import os

    folder = r"clean_text" 
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
