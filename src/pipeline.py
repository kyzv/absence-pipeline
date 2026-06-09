import os
import cv2
import json
from paddleocr import PaddleOCR
from .preprocessing import load_image, deskew, enhance, binarize, _natural_sorted
from .ocr_header import extract_document_metadata
from .table_analyzer import analyze_table

def run_pipeline(doc_id: str, raw_root: str = "data/raw", config_root: str = "data/config/groups"):
    """
    Main pipeline orchestrator.
    Returns (metadata, table_results)
    """
    raw_dir = os.path.join(raw_root, doc_id)
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Directory not found: {raw_dir}")
        
    csv_path = os.path.join(config_root, f"{doc_id}.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Student CSV not found: {csv_path}. Please run extract_students.py first.")

    exts = ('.jpg', '.jpeg', '.png')
    image_files = _natural_sorted([f for f in os.listdir(raw_dir) if f.lower().endswith(exts)])
    
    ocr = PaddleOCR(lang='fr', use_textline_orientation=False)
    
    all_results = []
    final_metadata = {}
    
    # Process pages
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(raw_dir, img_file)
        print(f"Processing {img_file}...")
        
        # 1. Preprocessing
        img = load_image(img_path)
        img_deskewed = deskew(img)
        img_enhanced = enhance(img_deskewed)
        img_binary = binarize(img_enhanced)
        
        # 2. OCR
        raw_ocr = list(ocr.predict(img_deskewed))
        if not raw_ocr or not raw_ocr[0] or 'rec_texts' not in raw_ocr[0]:
            continue
            
        texts = raw_ocr[0]['rec_texts']
        boxes = raw_ocr[0]['rec_polys']
        
        ocr_blocks = []
        for box, text in zip(boxes, texts):
            ocr_blocks.append({
                'text': text.strip(),
                'box': box,
                'center': (sum(p[0] for p in box) / 4, sum(p[1] for p in box) / 4)
            })
            
        # 3. Smart Split
        header_y = -1
        for b in ocr_blocks:
            t = b['text'].lower()
            if 'nom' in t or 'prénom' in t or 'séance' in t or 'seance' in t:
                header_y = max(header_y, b['center'][1])
                
        if header_y == -1:
            header_y = img_deskewed.shape[0] * 0.25 # Fallback
            
        # 4. Header Extraction (Only from first page usually)
        if idx == 0:
            header_blocks = [b for b in ocr_blocks if b['center'][1] < header_y + 20]
            final_metadata = extract_document_metadata(header_blocks)
            
        # 5. Table Analysis
        page_results = analyze_table(
            image=img_deskewed,
            binary=img_binary,
            ocr_blocks=ocr_blocks,
            student_csv_path=csv_path,
            header_y=header_y,
            doc_id=doc_id + f"_page_{idx}"
        )
        all_results.extend(page_results)
        
    # Aggregate duplicate students across pages if any (unlikely unless table wraps)
    # The analyze_table already filters empty sessions, but we should do a global merge
    merged_results = {}
    for res in all_results:
        n_apo = res['n_apo']
        if n_apo not in merged_results:
            merged_results[n_apo] = res
        else:
            # Merge sessions
            existing = {s['seance']: s for s in merged_results[n_apo]['sessions']}
            for s in res['sessions']:
                if s['seance'] not in existing or s['status'] == 'Present':
                    existing[s['seance']] = s
            merged_results[n_apo]['sessions'] = list(existing.values())
            
    final_list = list(merged_results.values())
    
    # Save results
    out_dir = os.path.join("data", "output", doc_id)
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(final_metadata, f, indent=2, ensure_ascii=False)
        
    with open(os.path.join(out_dir, "absences.json"), "w", encoding="utf-8") as f:
        json.dump(final_list, f, indent=2, ensure_ascii=False)
        
    return final_metadata, final_list

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/pipeline.py <doc_id>")
        sys.exit(1)
    doc = sys.argv[1]
    meta, tbl = run_pipeline(doc)
    print("Pipeline finished.")
