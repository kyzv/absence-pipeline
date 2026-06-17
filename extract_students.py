import os
import re
import csv
import cv2
import numpy as np
from paddleocr import PaddleOCR
from preprocessing import load_image, deskew

def main():
    raw_dir = os.path.join("data", "raw")
    out_dir = os.path.join("data", "config", "groups")
    os.makedirs(out_dir, exist_ok=True)
    
    ocr = PaddleOCR(lang='fr', use_textline_orientation=False)
    
    doc_dirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    for doc in doc_dirs:
        doc_path = os.path.join(raw_dir, doc)
        images = [f for f in os.listdir(doc_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        images.sort()
        
        students = []
        n_apo_counter = 1
        
        for img_name in images:
            img_path = os.path.join(doc_path, img_name)
            print(f"Processing {img_path}...")
            
            img = load_image(img_path)
            img = deskew(img)
            
            raw = list(ocr.predict(img))
            if not raw or not raw[0]:
                continue
                
            page = raw[0]
            if 'rec_texts' not in page or 'rec_polys' not in page:
                continue
                
            texts = page['rec_texts']
            boxes = page['rec_polys']
            
            # Find the "Nom" and "N°" headers to define columns
            nom_x_min, nom_x_max = -1, -1
            apo_x_min, apo_x_max = -1, -1
            header_y = -1
            
            for box, text in zip(boxes, texts):
                text = text.lower().strip()
                y_center = sum(p[1] for p in box) / 4
                x_center = sum(p[0] for p in box) / 4
                
                if 'nom' in text or 'prenom' in text or 'prénom' in text:
                    nom_x_min = min(p[0] for p in box) - 50
                    nom_x_max = max(p[0] for p in box) + 300 # Names can be long
                    header_y = max(header_y, y_center)
                elif text == 'n°' or 'apo' in text or text == 'n':
                    apo_x_min = min(p[0] for p in box) - 20
                    apo_x_max = max(p[0] for p in box) + 50
                    header_y = max(header_y, y_center)
            
            if header_y == -1:
                # If we didn't find headers clearly, we just assume any long text below top 20% is a name
                header_y = img.shape[0] * 0.2
                nom_x_min, nom_x_max = 50, img.shape[1] // 2
            
            # Group by row
            row_elements = []
            for box, text in zip(boxes, texts):
                text = text.strip()
                if not text or len(text) < 2:
                    continue
                y_center = sum(p[1] for p in box) / 4
                x_center = sum(p[0] for p in box) / 4
                
                if y_center > header_y + 20: # Only below header
                    row_elements.append({
                        'text': text,
                        'x': x_center,
                        'y': y_center
                    })
            
            # Sort by Y
            row_elements.sort(key=lambda e: e['y'])
            
            # Form rows
            current_row = []
            rows = []
            for e in row_elements:
                if not current_row:
                    current_row.append(e)
                else:
                    if abs(e['y'] - current_row[-1]['y']) < 15: # Same row
                        current_row.append(e)
                    else:
                        rows.append(current_row)
                        current_row = [e]
            if current_row:
                rows.append(current_row)
            
            # Extract names and N° Apo
            for row in rows:
                apo = None
                nom = []
                for e in row:
                    text = e['text']
                    x = e['x']
                    # Check if it's apo
                    if apo_x_min <= x <= apo_x_max and text.isdigit():
                        apo = text
                    # Check if it's name
                    elif nom_x_min <= x <= nom_x_max:
                        # Ignore common header texts or single characters
                        if text.lower() not in ['nom', 'prenom', 'prénom', 'absent', 'présent'] and len(text) > 3:
                            nom.append(text)
                
                if nom:
                    full_name = " ".join(nom)
                    if not apo:
                        apo = str(n_apo_counter)
                    students.append({'n_apo': apo, 'nom': full_name})
                    n_apo_counter += 1
        
        if students:
            csv_path = os.path.join(out_dir, f"{doc}.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['n_apo', 'nom'])
                writer.writeheader()
                writer.writerows(students)
            print(f"Saved {len(students)} students to {csv_path}")

if __name__ == "__main__":
    main()
