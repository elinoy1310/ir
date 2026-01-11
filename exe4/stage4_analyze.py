import json
import pandas as pd
from pathlib import Path
import re

def sanitize_filename(name):
    # ניקוי תווים לא חוקיים לשמות קבצים
    return re.sub(r'[\\/*?:"<>|]', "", name)[:50]

def process_evolution_json_to_csv(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # קיבוץ התוצאות לפי (שאילתה, מדינה, שיטת צ'אנקינג)
    groups = {}
    for entry in data:
        key = (entry['query'], entry['nation'], entry['chunking_method'])
        if key not in groups:
            groups[key] = []
        groups[key].append(entry)
    
    output_files = []
    
    for (query, nation, chunking), entries in groups.items():
        # יצירת שורה אחת המייצגת את השאילתה עם עמודות דינמיות לכל קונפיגורציה
        row_data = {
            "Query": query,
            "Nation": nation,
            "Chunking_Method": chunking
        }
        
        for entry in entries:
            method = entry['vector_method'].upper()
            k = entry['k']
            prefix = f"{method}_K{k}"
            
            # הוספת התשובות כעמודות חדשות
            row_data[f"{prefix}_Early_Answer"] = entry.get('early_answer', '')
            row_data[f"{prefix}_Late_Answer"] = entry.get('late_answer', '')
            row_data[f"{prefix}_Final_Analysis"] = entry.get('change_answer', '')
            
        df = pd.DataFrame([row_data])
        
        # יצירת שם קובץ תקין
        safe_q = sanitize_filename(query)
        filename = f"evolution_results_{nation}_{chunking}_{safe_q}.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        output_files.append(filename)
        
    return output_files

# הרצה על הקובץ שלך
csv_files = process_evolution_json_to_csv(r'exe4\outputs\run_20260105_172733\stage4_evo_v2_all_results.json')
print(f"נוצרו הקבצים הבאים: {csv_files}")