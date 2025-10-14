# src/processing/lens_extract.py
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm

# ---- Keliai (nepriklausomai nuo working directory) ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR            = PROJECT_ROOT / "data" / "raw"
MASKS_FRAME_DIR    = PROJECT_ROOT / "results" / "masks"         # rėmelio maskės (iš baseline/boundaries)
LENS_MASKS_DIR     = PROJECT_ROOT / "results" / "lens_masks"
LENS_OVERLAYS_DIR  = PROJECT_ROOT / "results" / "lens_overlays"
for d in [LENS_MASKS_DIR, LENS_OVERLAYS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ---- Pagalbinės ----
def list_images(folder: Path, exts=IMG_EXTS) -> List[Path]:
    files = [p for p in Path(folder).rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files

def read_gray(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def read_color(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img

def find_frame_mask_for(stem: str) -> Optional[Path]:
    # ieškome bet kokios rėmelio maskės pagal failo vardą (pvz., *_canny_mask.png ar *_mask.png)
    candidates = list(MASKS_FRAME_DIR.glob(f"{stem}*_mask.png"))
    if candidates:
        return candidates[0]
    return None

def load_binary_mask(path: Path) -> Optional[np.ndarray]:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 0).astype(np.uint8)

def erode_mask(mask: np.ndarray, k: int = 3, it: int = 1) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask, kernel, iterations=it)

def remove_small_components(mask01: np.ndarray, min_area: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask01.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask01, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out

def keep_largest_components(mask01: np.ndarray, max_k: int = 3) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask01.astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask01
    areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num)]
    areas.sort(reverse=True)
    keep = [i for _, i in areas[:max_k]]
    out = np.zeros_like(mask01, dtype=np.uint8)
    for i in keep:
        out[labels == i] = 1
    return out

def overlay_green(img_bgr: np.ndarray, mask01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = np.zeros_like(img_bgr)
    color[..., 1] = 255  # žalias kanalas
    out = img_bgr.copy()
    sel = mask01.astype(bool)
    out[sel] = cv2.addWeighted(img_bgr[sel], 1 - alpha, color[sel], alpha, 0)
    return out

# ---- Smoothing metodai ----
def smooth_gaussian(gray: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(gray, (5, 5), 0)

def smooth_bilateral(gray: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(gray, d=9, sigmaColor=50, sigmaSpace=50)

def smooth_guided(gray: np.ndarray) -> Optional[np.ndarray]:
    # Reikia opencv-contrib-python
    if not hasattr(cv2, "ximgproc"):
        return None
    try:
        gf = cv2.ximgproc.guidedFilter(guide=gray, src=gray, radius=8, eps=1e-2)
        return gf
    except Exception:
        return None

# ---- Threshold metodai (ROI viduje) ----
def threshold_otsu(src_gray: np.ndarray, roi01: np.ndarray) -> np.ndarray:
    _, th = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ((th > 0).astype(np.uint8) & roi01).astype(np.uint8)

def threshold_adaptive(src_gray: np.ndarray, roi01: np.ndarray) -> np.ndarray:
    th = cv2.adaptiveThreshold(src_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, blockSize=21, C=2)
    return ((th > 0).astype(np.uint8) & roi01).astype(np.uint8)

# ---- Pagrindinė lęšio išskyrimo logika ----
def extract_lens_mask(img_gray: np.ndarray, frame_mask01: np.ndarray,
                      smooth_method: str, thr_method: str) -> np.ndarray:
    # 1) ROI – rėmelio vidus (šiek tiek eroduojam, kad nepatektų rėmelio kraštai)
    roi = erode_mask(frame_mask01, k=5, it=1)
    if roi.sum() == 0:
        # jei per agresyvu – bandome švelniau
        roi = erode_mask(frame_mask01, k=3, it=1)
        if roi.sum() == 0:
            # kraštutinis atvejis – dirbam su visa rėmelio kauke
            roi = frame_mask01.copy()

    # 2) smoothing
    if smooth_method == "gaussian":
        sm = smooth_gaussian(img_gray)
    elif smooth_method == "bilateral":
        sm = smooth_bilateral(img_gray)
    elif smooth_method == "guided":
        g = smooth_guided(img_gray)
        sm = g if g is not None else smooth_gaussian(img_gray)  # fallback
    else:
        sm = img_gray

    # 3) threshold
    if thr_method == "otsu":
        m = threshold_otsu(sm, roi)
    elif thr_method == "adaptive":
        m = threshold_adaptive(sm, roi)
    else:
        m = (sm > 0).astype(np.uint8) & roi

    # 4) post-proc: mažos sritys, skylės, paliekam kelis didžiausius (dažniausiai 2 lęšiai)
    h, w = img_gray.shape
    min_area = max(100, (h * w) // 8000)
    m = remove_small_components(m, min_area=min_area)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
    m = keep_largest_components(m, max_k=3)

    return m  # 0/1

def process_all():
    img_paths = list_images(RAW_DIR)
    if not img_paths:
        print("Nerasta nuotraukų data/raw/")
        return

    combos: List[Tuple[str, str]] = [
        ("gaussian", "otsu"),
        ("gaussian", "adaptive"),
        ("bilateral", "otsu"),
        ("bilateral", "adaptive"),
        ("guided", "otsu"),
        ("guided", "adaptive"),
    ]

    for p in tqdm(img_paths, desc="Lęšio išskyrimas"):
        gray = read_gray(p)
        color = read_color(p)
        if gray is None or color is None:
            print(f"[Įspėjimas] nepavyko nuskaityti: {p}")
            continue

        # Rėmelio maskė
        frame_mask_path = find_frame_mask_for(p.stem)
        if frame_mask_path is None:
            print(f"[Praleidžiama] Nerasta rėmelio maskė: {p.stem}_*mask.png")
            continue

        frame_mask01 = load_binary_mask(frame_mask_path)
        if frame_mask01 is None:
            print(f"[Praleidžiama] Nepavyko perskaityti rėmelio maskės: {frame_mask_path}")
            continue

        # >>> Svarbu: priderinam kaukės dydį prie vaizdo dydžio
        h, w = gray.shape
        if frame_mask01.shape != (h, w):
            frame_mask01 = (frame_mask01 > 0).astype(np.uint8)
            frame_mask01 = cv2.resize(frame_mask01, (w, h), interpolation=cv2.INTER_NEAREST)
            frame_mask01 = (frame_mask01 > 0).astype(np.uint8)

        for sm, th in combos:
            lens01 = extract_lens_mask(gray, frame_mask01, sm, th)

            # Išsaugom lęšio maskę
            mask_out = LENS_MASKS_DIR / f"{p.stem}_{sm}_{th}_lens.png"
            cv2.imwrite(str(mask_out), (lens01 * 255).astype(np.uint8))

            # Išsaugom overlay (žalias)
            over = overlay_green(color, lens01, alpha=0.45)
            over_out = LENS_OVERLAYS_DIR / f"{p.stem}_{sm}_{th}_overlay.png"
            cv2.imwrite(str(over_out), over)

if __name__ == "__main__":
    process_all()
