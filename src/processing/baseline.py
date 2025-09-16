# src/processing/baseline.py
from pathlib import Path
import cv2
import numpy as np
from typing import List

# ---- Keliai (nepriklausomai nuo working directory) ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MASKS_DIR = PROJECT_ROOT / "results" / "masks"
OVERLAYS_DIR = PROJECT_ROOT / "results" / "overlays"
MASKS_DIR.mkdir(parents=True, exist_ok=True)
OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ---- Pagalbinės ----
def list_images(folder: Path, exts=IMG_EXTS) -> List[Path]:
    folder = Path(folder)
    files = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files

def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    mask: 0/255 (uint8) arba bool. Uždedam raudoną sluoksnį ant mask'ės vietų.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8) * 255
    mask_bool = mask > 0
    color = np.zeros_like(img_bgr)
    color[..., 2] = 255  # raudonas BGR kanalas
    out = img_bgr.copy()
    out[mask_bool] = cv2.addWeighted(img_bgr[mask_bool], 1 - alpha, color[mask_bool], alpha, 0)
    return out

# ---- LABAI paprastas baseline segmentavimas rėmeliui ----
def segment_frame_baseline(img_bgr: np.ndarray) -> np.ndarray:
    """
    Idėja: kraštai (Canny) -> pastorinam -> užpildom kontūrų skyles.
    Tai nėra tikra semantinė segmentacija, bet duoda „pirmą rezultatą“.
    Grąžina 0/255 uint8 maskę to paties dydžio kaip img.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Lengvas išlyginimas prieš kraštus
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny kraštai – ribas dažniausiai pagaus
    edges = cv2.Canny(blur, 50, 150)

    # Pastorinam kraštus, kad susijungtų plyšiai
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dil = cv2.dilate(edges, kernel, iterations=2)

    # Užpildom vidines skyles (kontūrai -> užpildyti)
    mask = np.zeros((h, w), dtype=np.uint8)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)

    # Šiek tiek „sutvarkom“: atmetam labai mažas dėmes
    mask = remove_small_objects(mask, min_area=max(200, (h * w) // 5000))

    # Uždengiame nedideles skyles rėmelyje
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask

def remove_small_objects(mask_u8: np.ndarray, min_area: int = 200) -> np.ndarray:
    """
    Pašalina mažus plotus iš binarinės maskės.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask_u8)
    for i in range(1, num_labels):  # 0 = fonas
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out

# ---- Vykdymas visiems failams ----
def process_all():
    img_paths = list_images(RAW_DIR)
    print(f"Rasta nuotraukų: {len(img_paths)}")
    for p in img_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[ĮSPĖJIMAS] Nepavyko nuskaityti: {p}")
            continue

        mask = segment_frame_baseline(img)

        # Išsaugom maskę
        mask_path = MASKS_DIR / f"{p.stem}_frame_mask.png"
        cv2.imwrite(str(mask_path), mask)

        # Išsaugom overlay
        over = overlay_mask(img, mask)
        over_path = OVERLAYS_DIR / f"{p.stem}_overlay.png"
        cv2.imwrite(str(over_path), over)

        print(f"OK: {p.name} -> {mask_path.name}, {over_path.name}")

if __name__ == "__main__":
    process_all()
