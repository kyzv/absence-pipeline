"""
Calibration script: measure dark pixel density for all seance cells in doc_8 debug dir.
Outputs a sorted CSV to help set thresholds.
"""
import cv2
import os
import csv
import glob
import re

DEBUG_DIR = r"c:\Users\abde\Desktop\absence-pipeline\debug\doc_8"
OUT_CSV = os.path.join(DEBUG_DIR, "cell_densities.csv")

results = []

for f in glob.glob(os.path.join(DEBUG_DIR, "row*_seance*.jpg")):
    img = cv2.imread(f)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use threshold 200 (matching ocr_students.py)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    total_pixels = img.shape[0] * img.shape[1]
    dark_pixels = cv2.countNonZero(binary)
    density = dark_pixels / total_pixels if total_pixels > 0 else 0
    size_bytes = os.path.getsize(f)
    name = os.path.basename(f)
    m = re.match(r'row(\d+)_seance(\d+)', name)
    row_num = int(m.group(1)) if m else 0
    seance_num = int(m.group(2)) if m else 0
    results.append({
        "file": name, "row": row_num, "seance": seance_num,
        "size_bytes": size_bytes, "dark_pixels": dark_pixels,
        "total_pixels": total_pixels, "density": round(density, 4)
    })

# Sort by density to understand distribution
results.sort(key=lambda x: x["density"])

with open(OUT_CSV, "w", newline="") as csvf:
    writer = csv.DictWriter(csvf, fieldnames=["file","row","seance","size_bytes","dark_pixels","total_pixels","density"])
    writer.writeheader()
    writer.writerows(results)

print(f"Written {len(results)} entries to {OUT_CSV}")

# Print distribution overview
densities = [r["density"] for r in results]
thresholds = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
print("\nDensity distribution:")
for t in thresholds:
    count_below = sum(1 for d in densities if d < t)
    print(f"  < {t:.2f}: {count_below} cells ({count_below/len(densities)*100:.1f}%)")

print(f"\nMin density: {min(densities):.4f}")
print(f"Max density: {max(densities):.4f}")

# Show first 20 (lowest density = likely blank)
print("\nLOWEST 20 densities (likely blank/absent):")
for r in results[:20]:
    print(f"  {r['file']}: density={r['density']}, size={r['size_bytes']}B")

# Show 20 around median
mid = len(results) // 2
print(f"\nMIDDLE 20 densities (ambiguous zone):")
for r in results[mid-10:mid+10]:
    print(f"  {r['file']}: density={r['density']}, size={r['size_bytes']}B")
