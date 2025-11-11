"""
12-01 – Konstrukciniai komponentai (YOLO versija be Detectron2)
Ainius ir Iveta projektas: akinių komponentų segmentacija.
"""

from pathlib import Path
import cv2
import numpy as np

# YOLO (Ultralytics)
from ultralytics import YOLO

# ---- Katalogai ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "results" / "components_yolo_rcnn"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---- 1. YOLO komponentų atpažinimas ----
def detect_with_yolo(image_path, model):
    img = cv2.imread(str(image_path))
    results = model.predict(source=str(image_path), conf=0.5, verbose=False)
    boxes = results[0].boxes
    names = model.names
    overlay = img.copy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls)
        label = names[cls] if cls in names else f"cls_{cls}"
        conf = float(box.conf)
        color = (0, 255, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return overlay


# ---- 3. Pagrindinė funkcija ----
def process_all():
    # Įkeliame modelius
    print("🔹 Keliame YOLOv8 modelį...")
    yolo_model = YOLO("yolov8n.pt")  # bazinis modelis (galima pakeisti į custom)

    # Pereiname per visas nuotraukas
    img_paths = sorted(list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.png")))

    for p in img_paths:
        print(f"🖼  Apdorojama: {p.name}")
        try:
            yolo_out = detect_with_yolo(p, yolo_model)

            # Išsaugom YOLO variantą
            cv2.imwrite(str(OUT_DIR / f"{p.stem}_yolo.png"), yolo_out)
            print(f"✅ {p.name} apdorota.")
        except Exception as e:
            print(f"⚠️ Klaida su {p.name}: {e}")


if __name__ == "__main__":
    process_all()