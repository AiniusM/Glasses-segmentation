"""
12-01 – Konstrukciniai komponentai

Užduotys:
- Nustatyti pagrindinius akinių rėmelio komponentus (tiltelis, abi kojelės, lęšio apvadas).
- Sugeneruoti vaizdą su spalvotais pažymėjimais.
- Remiasi paprasta geometrija ir simetrija, be neuroninio tinklo.
Rezultatas: katalogas su pavyzdžiais, kur pažymėti bent 2 komponentai.
"""

from pathlib import Path
import cv2
import numpy as np

# ----- Katalogų struktūra -----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MASKS_DIR = PROJECT_ROOT / "results" / "masks"
OUT_DIR = PROJECT_ROOT / "results" / "components"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----- Pagalbinės funkcijos -----
def load_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def load_mask_for(stem: str):
    candidates = list(MASKS_DIR.glob(f"{stem}*_mask.png"))
    if not candidates:
        return None
    m = cv2.imread(str(candidates[0]), cv2.IMREAD_GRAYSCALE)
    return (m > 0).astype(np.uint8)


def draw_component_label(img, contour, label, color):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


# ----- Komponentų paieška -----
def detect_components(frame_mask):
    """
    Heuristinis metodas:
    - Suranda visus rėmelio kontūrus.
    - Pagal padėtį skiria:
        * Tiltelis – viršutinė centrinė sritis
        * Kojelės – toliausiai į kairę/dešinę nutolusios dalys
        * Apvadai – likusios dvi centrinės sritys aplink lęšius
    """
    contours, _ = cv2.findContours(frame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Apskaičiuojam masės centrus
    comps = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        area = cv2.contourArea(c)
        comps.append((c, cx, cy, area))

    # Rūšiuojam pagal x (horizontalią padėtį)
    comps.sort(key=lambda x: x[1])

    result = []
    if len(comps) >= 3:
        left = comps[0]
        right = comps[-1]
        center = comps[len(comps)//2]
        result.append(("Left temple", left[0], (255, 0, 0)))   # mėlyna
        result.append(("Bridge", center[0], (0, 255, 255)))    # geltona
        result.append(("Right temple", right[0], (0, 0, 255))) # raudona
    elif len(comps) == 2:
        # paprastas atvejis: du lęšiai
        result.append(("Left lens rim", comps[0][0], (0, 255, 0)))
        result.append(("Right lens rim", comps[1][0], (0, 255, 0)))
    else:
        # vienas komponentas – laikom visą rėmelį
        result.append(("Frame", comps[0][0], (255, 255, 255)))

    return result


# ----- Pagrindinė procedūra -----
def process_all():
    img_paths = sorted(list(RAW_DIR.glob("*.png")) + list(RAW_DIR.glob("*.jpg")))

    for img_path in img_paths:
        try:
            img, gray = load_gray(img_path)
        except Exception:
            continue

        mask = load_mask_for(img_path.stem)
        if mask is None:
            print(f"[WARN] Mask not found for {img_path.name}")
            continue

        components = detect_components(mask)
        overlay = img.copy()

        for label, contour, color in components:
            draw_component_label(overlay, contour, label, color)

        out_path = OUT_DIR / f"{img_path.stem}_components.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"[OK] {img_path.name} -> {out_path.name}")


if __name__ == "__main__":
    process_all()