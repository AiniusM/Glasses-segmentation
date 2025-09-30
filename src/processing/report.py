# src/processing/report.py
from pathlib import Path
import csv
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASKS_DIR    = PROJECT_ROOT / "results" / "masks"
REPORT_CSV   = PROJECT_ROOT / "results" / "report.csv"

def analyze_mask(mask_path: Path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    # Binarizuojam (jei kaukė 0/255 – tiesiog >0)
    binm = (mask > 0).astype(np.uint8)

    # Komponentų skaičius
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    components = max(0, num - 1)  # be fono

    # Plotas (pikseliais)
    area = int(binm.sum())

    # Perimetras: sumuojam visų kontūrų ilgius
    contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = 0.0
    for c in contours:
        perimeter += cv2.arcLength(c, True)

    return area, perimeter, components

def make_report():
    rows = [("filename", "area_px", "perimeter_px", "components")]
    for p in sorted(MASKS_DIR.glob("*_mask.png")):
        res = analyze_mask(p)
        if res is None:
            continue
        area, perim, comps = res
        rows.append((p.name, area, round(float(perim), 2), int(comps)))

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)
    print(f"OK: ataskaita -> {REPORT_CSV}")

if __name__ == "__main__":
    make_report()