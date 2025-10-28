# src/processing/shadow_remove.py
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
import csv
from tqdm import tqdm

# ---- Keliai ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR           = PROJECT_ROOT / "data" / "raw"
FRAME_MASKS_DIR   = PROJECT_ROOT / "results" / "masks"          # jūsų rėmelio maskės (*_mask.png)
OUT_DIR           = PROJECT_ROOT / "results"
OUT_SHADOW_DIR    = OUT_DIR / "shadow_masks"
OUT_CLEAN_DIR     = OUT_DIR / "frame_no_shadow_masks"
OUT_OVERLAY_DIR   = OUT_DIR / "frame_no_shadow_overlays"
REPORT_CSV        = OUT_DIR / "shadow_report.csv"
for d in [OUT_SHADOW_DIR, OUT_CLEAN_DIR, OUT_OVERLAY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ---- Helperiai ----
def list_images(folder: Path, exts=IMG_EXTS) -> List[Path]:
    files = [p for p in Path(folder).rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files

def read_bgr_with_alpha(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        b,g,r,a = cv2.split(img)
        alpha = (a.astype(np.float32)/255.0)[...,None]
        rgb = (alpha*img[...,:3] + (1-alpha)*255).astype(np.uint8)
        return rgb
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def load_frame_mask_for(stem: str, target_hw: Tuple[int,int]) -> Optional[np.ndarray]:
    cand = list(FRAME_MASKS_DIR.glob(f"{stem}*_mask.png"))
    if not cand:
        return None
    m = cv2.imread(str(cand[0]), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = (m > 0).astype(np.uint8)
    h,w = target_hw
    if m.shape != (h,w):
        m = cv2.resize(m, (w,h), interpolation=cv2.INTER_NEAREST)
        m = (m > 0).astype(np.uint8)
    return m

def overlay_red(img_bgr, mask01, alpha=0.45):
    mask01 = (mask01 > 0).astype(np.uint8)
    if mask01.sum() == 0:
        return img_bgr.copy()
    # sukonstruojam spalvotą sluoksnį tik maskės vietose
    overlay = np.zeros_like(img_bgr, dtype=np.uint8)
    overlay[..., 2] = 255  # raudonas BGR kanalas
    # nulinam už maskės ribų, kad neįtakotų
    overlay = (overlay * mask01[..., None]).astype(np.uint8)
    out = cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)
    return out

def overlay_green(img_bgr, mask01, alpha=0.45):
    mask01 = (mask01 > 0).astype(np.uint8)
    if mask01.sum() == 0:
        return img_bgr.copy()
    overlay = np.zeros_like(img_bgr, dtype=np.uint8)
    overlay[..., 1] = 255  # žalias
    overlay = (overlay * mask01[..., None]).astype(np.uint8)
    out = cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)
    return out

def band_near_boundary(mask01: np.ndarray, r_out: int = 6, r_in: int = 3) -> np.ndarray:
    """Žiedas aplink rėmelio ribą: (dilate(mask,r_out) - erode(mask,r_in))"""
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r_out+1, 2*r_out+1))
    k_in  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r_in +1, 2*r_in +1))
    dil = cv2.dilate(mask01, k_out, iterations=1)
    ero = cv2.erode(mask01, k_in,  iterations=1)
    band = (dil & (1-ero)).astype(np.uint8)
    return band

# ---- Šešėlio išskyrimas ----
def compute_shadow_candidate(img_bgr: np.ndarray, frame_mask01: np.ndarray) -> np.ndarray:
    """Šešėlis: tamsu + mažai kraštų + ribos zonoje (prie rėmelio)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[...,2].astype(np.uint8)

    # kraštų energija (Sobel magnitude -> 0..255)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # tikrinam ribos žiede aplink rėmelį
    band = band_near_boundary(frame_mask01, r_out=8, r_in=2)
    # praplėskim šiek tiek į išorę, kad „pagautų“ šešėlį
    ext = cv2.dilate(frame_mask01, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), 1) & (1-frame_mask01)
    band = ((band | ext) & (1-frame_mask01)).astype(np.uint8)

    # adapt. slenksčiai iš band zonos
    vals_V   = V[band>0]
    vals_mag = mag_u8[band>0]
    if len(vals_V)==0:
        return np.zeros_like(frame_mask01)
    v_thr  = np.percentile(vals_V, 35)      # 35-asis percentilis – „tamsu“
    g_thr  = np.percentile(vals_mag, 40)    # mažai kraštų

    shadow = ((V <= v_thr) & (mag_u8 <= g_thr)).astype(np.uint8)
    shadow = shadow & band

    # sutvarkom:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, k, iterations=1)
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, k, iterations=2)

    # pašalinam trupinius
    num, labels, stats, _ = cv2.connectedComponentsWithStats(shadow, 8)
    out = np.zeros_like(shadow)
    h,w = shadow.shape
    min_area = max(80, (h*w)//12000)
    for i in range(1,num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels==i] = 1
    return out

def remove_shadow_from_frame(frame_mask01: np.ndarray, shadow01: np.ndarray) -> np.ndarray:
    # Jei rėmelio maskė „prilipusi“ prie šešėlio – atimame jų sankirtą
    # (ir šiek tiek buferio aplink šešėlį, kad neliktų tamsaus hal’o)
    buf = cv2.dilate(shadow01, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    clean = (frame_mask01 & (1-buf)).astype(np.uint8)

    # pasaugom rėmelio vientisumą
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=1)
    return clean

# ---- Evaluacija + išsaugojimai ----
def process_all():
    rows = [("filename","orig_area","shadow_cand_area","removed_area","removed_ratio","success")]

    for p in tqdm(list_images(RAW_DIR), desc="Šešėlio atskyrimas"):
        img = read_bgr_with_alpha(p)
        if img is None:
            continue
        h,w,_ = img.shape

        frame_mask01 = load_frame_mask_for(p.stem, (h,w))
        if frame_mask01 is None:
            # peršokam, jei neturit rėmelio maskės šitam kadrui
            continue

        # Šešėlio kandidatas
        shadow01 = compute_shadow_candidate(img, frame_mask01)

        # Kiek šešėlio buvo pačioje maskėje
        orig_area = int(frame_mask01.sum())
        inter = int((frame_mask01 & shadow01).sum())

        # Išimam
        clean = remove_shadow_from_frame(frame_mask01, shadow01)
        clean_area = int(clean.sum())

        removed_area = orig_area - clean_area
        removed_ratio = (removed_area / max(1, shadow01.sum())) if shadow01.sum()>0 else 1.0

        # paprastas „success“ – jei pavyko iš rėmelio išmesti ≥70% to, kas atitiko šešėlio požymius
        success = 1 if shadow01.sum()==0 or removed_ratio >= 0.7 else 0

        # įrašai
        cv2.imwrite(str(OUT_SHADOW_DIR / f"{p.stem}_shadow.png"), (shadow01*255).astype(np.uint8))
        cv2.imwrite(str(OUT_CLEAN_DIR  / f"{p.stem}_frame_no_shadow.png"), (clean*255).astype(np.uint8))

        # overlay: žalia – rėmelis be šešėlio, raudona – šešėlis kandidatas
        over = overlay_green(img, clean, 0.45)
        over = overlay_red(over, shadow01, 0.40)
        cv2.imwrite(str(OUT_OVERLAY_DIR / f"{p.stem}_overlay.png"), over)

        rows.append((p.name, orig_area, int(shadow01.sum()), removed_area,
                     round(float(removed_ratio),3), success))

    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerows(rows)
    print(f"OK: ataskaita -> {REPORT_CSV}")

if __name__ == "__main__":
    process_all()