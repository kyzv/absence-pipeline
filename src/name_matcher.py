# src/name_matcher.py
import os
import json
import sys

def main(doc_id: str):
    attendance_file = os.path.join("data", "metadata", doc_id, "attendance_results.json")
    if not os.path.exists(attendance_file):
        print(f"Error: Missing matrix data. Run checkbox_detection.py first.")
        sys.exit(1)
        
    with open(attendance_file, 'r', encoding='utf-8') as f:
        records = json.load(f)

    print(f"\n=======================================================")
    print(f" COMPREHENSIVE STATUS LOG FOR {doc_id.upper()}")
    print(f"=======================================================")

    for s in range(1, 11):
        s_key = str(s)
        column_states = [r['seances'][s_key] for r in records]
        if "Not Conducted" in column_states:
            print(f"Séance {s:02d}: [Not Conducted Yet]")
            continue
            
        absentees = [r for r in records if r['seances'][s_key] == "Absent"]
        print(f"Séance {s:02d}: {len(absentees)} Absences found")
        for r in absentees:
            print(f"  -> {r['id']:<10} | {r['name']}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/name_matcher.py <doc_id>")
        sys.exit(1)
    main(sys.argv[1])