from pathlib import Path
import cv2
import numpy as np
from typing import List

def list_images(folder: Path, exts=(".jpg", ".jpeg", ".png")) -> List[Path]:
    folder = Path(folder)
    files = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files

def read_image_cv2(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise FileNotFoundError(f"Nepavyko nuskaityti: {path}")
    return img

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    raw_dir = Path("data/raw")
    paths = list_images(raw_dir)
    print(f"Rasta nuotraukų: {len(paths)}")
    if paths:
        img = read_image_cv2(paths[0])
        print("Pirmo vaizdo dydis:", img.shape)
