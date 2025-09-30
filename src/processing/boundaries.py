# src/processing/boundaries.py
from pathlib import Path
from typing import List, Literal, Tuple
import cv2
import numpy as np

# Keliai (nepriklausomai nuo working dir)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
OUT_DIR       = PROJECT_ROOT / "results"
MASKS_DIR     = OUT_DIR / "masks"
OVERLAYS_DIR  = OUT_DIR / "overlays"
BOUNDARY_DIR  = OUT_DIR / "boundaries"   # čia bus tik ribos (linijos)
for d in [MASKS_DIR, OVERLAYS_DIR, BOUNDARY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ---------- Pagalbinės ----------
def list_images(folder: Path, exts=IMG_EXTS) -> List[Path]:
    files = [p for p in Path(folder).rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files

def remove_small_objects(mask_u8: np.ndarray, min_area: int = 200) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask_u8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out

def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8) * 255
    color = np.zeros_like(img_bgr)
    color[..., 2] = 255
    out = img_bgr.copy()
    sel = mask > 0
    out[sel] = cv2.addWeighted(img_bgr[sel], 1 - alpha, color[sel], alpha, 0)
    return out

def draw_boundaries(img_bgr: np.ndarray, contours: List[np.ndarray],
                    thickness: int = 2) -> np.ndarray:
    out = img_bgr.copy()
    # žymim ryškiai raudonai ribas
    cv2.drawContours(out, contours, -1, color=(0, 0, 255), thickness=thickness)
    return out


# ---------- Kraštai (edges) ----------
def edges_canny(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)

def edges_sobel(gray: np.ndarray) -> np.ndarray:
    # Sobel X ir Y, tada magnitude
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Otsu ar adaptacinis slenkstis, kad gautume binarinį kraštų žemėlapį
    _, edges = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return edges


# ---------- Iš kraštų -> maskė (region-based) ----------
def mask_from_edges(edges: np.ndarray, min_area: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    # Pastorinam, kad sujungti spragas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dil = cv2.dilate(edges, kernel, iterations=2)

    # Kontūrai iš „kraštų“
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Užpildom kontūrus į maskę
    h, w = edges.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)

    # Pašalinam smulkmenas
    mask = remove_small_objects(mask, min_area=min_area)

    # Truputį uždarom skyles
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Po filtravimo – atnaujinti kontūrus (tiksliai ribai užpiešti)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return mask, contours


# ---------- Vykdymas ----------
def process_all(method: Literal["canny", "sobel"] = "canny"):
    paths = list_images(RAW_DIR)
    print(f"Rasta nuotraukų: {len(paths)} | metodas: {method}")
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[Įspėjimas] Nepavyko nuskaityti: {p}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = edges_canny(gray) if method == "canny" else edges_sobel(gray)

        # min_area pagal vaizdo dydį, kad adaptyvu
        h, w = gray.shape
        min_area = max(200, (h * w) // 5000)

        mask, contours = mask_from_edges(edges, min_area=min_area)

        # Išsaugom pilną maskę
        cv2.imwrite(str(MASKS_DIR / f"{p.stem}_{method}_mask.png"), mask)

        # Overlay (uždažytas regionas)
        over = overlay_mask(img, mask, alpha=0.45)
        cv2.imwrite(str(OVERLAYS_DIR / f"{p.stem}_{method}_overlay.png"), over)

        # Tik ribos (linijos)
        boundary = draw_boundaries(img, contours, thickness=2)
        cv2.imwrite(str(BOUNDARY_DIR / f"{p.stem}_{method}_boundary.png"), boundary)

        print(f"OK: {p.name} -> mask/overlay/boundary")

if __name__ == "__main__":
    # pakeisk į "sobel", jei norit pabandyti kitą metodą
    process_all(method="canny")