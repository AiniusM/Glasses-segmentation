import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from skimage.util import img_as_ubyte
from math import log2
import sys

# ---- Auto paths (run without CLI) ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_MASKS_DIR_PRIMARY = PROJECT_ROOT / "results" / "masks"         # filled frame masks (preferred)
DEFAULT_MASKS_DIR_FALLBACK = PROJECT_ROOT / "results" / "boundaries"   # boundary lines (fallback)
DEFAULT_OUT_DIR = PROJECT_ROOT / "results"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img

def load_mask(path: Path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError(f"Failed to read mask: {path}")
    # Binarize (in case it's an edge map, we’ll dilate/close later)
    _, m = cv2.threshold(m, 0, 255, cv2.THRESH_OTSU)
    return m

def dilate_and_fill(mask: np.ndarray, dilate_iter=2, kernel_size=5, close_iter=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    m = cv2.dilate(mask, kernel, iterations=max(0, dilate_iter))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=max(0, close_iter))
    return m

def apply_mask(img: np.ndarray, mask: np.ndarray):
    return cv2.bitwise_and(img, img, mask=mask)

def entropy_from_hist(hist):
    # hist is counts; normalize
    hist = hist.astype(np.float64)
    s = hist.sum()
    if s == 0:
        return 0.0
    p = hist / s
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def lbp_features(gray: np.ndarray, P=8, R=1):
    # uniform LBP with 59 bins (for P=8)
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    # bins: 0..P*(P-1)+2 (59 for P=8)
    n_bins = int(P * (P - 1) + 3)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=False)
    hist = hist.astype(np.float64)
    hist /= (hist.sum() + 1e-8)
    return hist

def patch_features(patch_bgr: np.ndarray, mask_patch: np.ndarray):
    # focus only on masked pixels
    m = mask_patch > 0
    if m.sum() == 0:
        return None

    # grayscale
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    gray_m = gray[m]

    # basic stats
    mean_val = float(np.mean(gray_m))
    std_val = float(np.std(gray_m))

    # intensity histogram (32 bins)
    hist = cv2.calcHist([gray], [0], mask_patch, [32], [0, 256]).flatten()
    hist_entropy = entropy_from_hist(hist)

    # LBP (use only masked pixels by zeroing outside)
    gray_masked = gray.copy()
    gray_masked[~m] = 0
    lbp_hist = lbp_features(gray_masked)

    feats = [mean_val, std_val, hist_entropy]
    feats.extend(lbp_hist.tolist())
    return np.array(feats, dtype=np.float32)

def sliding_windows(h, w, patch_size, stride):
    for y in range(0, max(1, h - patch_size + 1), stride):
        for x in range(0, max(1, w - patch_size + 1), stride):
            yield x, y

def stem_of(path: Path):
    return path.stem

def find_matching_mask(image_path: Path, masks_dirs):
    """
    Match by the longest common stem prefix: e.g.
    image: pic10.jpg -> look for files starting with 'pic10' in masks dirs.
    If multiple candidates, prefer those containing 'boundary' or 'mask'.
    `masks_dirs` can be a Path or a list/tuple of Paths (searched in order).
    """
    if isinstance(masks_dirs, (list, tuple)):
        dirs = [Path(d) for d in masks_dirs]
    else:
        dirs = [Path(masks_dirs)]

    stem = image_path.stem

    def candidates_in_dir(d: Path):
        if not d.exists():
            return []
        c = list(d.glob(f"{stem}*.*"))
        if not c:
            prefix = stem.split('_')[0]
            c = list(d.glob(f"{prefix}*.*"))
        return c

    candidates = []
    for d in dirs:
        candidates.extend(candidates_in_dir(d))

    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda p: (
            0 if any(k in p.name.lower() for k in ["boundary", "mask"]) else 1,
            len(p.name)
        )
    )
    return candidates[0]

def main():
    # If run without CLI args, fall back to defaults
    use_defaults = (len(sys.argv) == 1)

    ap = argparse.ArgumentParser(description="Extract frame texture patches from masked frame regions.")
    ap.add_argument("--images_dir", type=str, required=False,
                    default=str(DEFAULT_IMAGES_DIR),
                    help="Directory with frame images (default: data/raw)")
    ap.add_argument("--masks_dir", type=str, required=False,
                    default=str(DEFAULT_MASKS_DIR_PRIMARY),
                    help="Directory with corresponding masks (default: results/masks; falls back to results/boundaries)")
    ap.add_argument("--out_dir", type=str, required=False,
                    default=str(DEFAULT_OUT_DIR),
                    help="Output base directory (default: results/)")
    ap.add_argument("--patch_size", type=int, default=200, help="Patch size (square). Default: 200")
    ap.add_argument("--stride", type=int, default=200, help="Stride for sliding window. Default: 200")
    ap.add_argument("--mask_dilate", type=int, default=2, help="Iterations to dilate mask (expands coverage).")
    ap.add_argument("--mask_close", type=int, default=1, help="Iterations to close small holes in mask.")
    ap.add_argument("--min_mask_coverage", type=float, default=0.35, help="Min fraction of patch covered by mask to keep (0..1).")
    ap.add_argument("--min_mean_intensity", type=float, default=10.0, help="Min mean intensity (0..255) to keep patch.")
    ap.add_argument("--examples_per_image", type=int, default=6, help="How many representative example patches to keep per image.")

    # Parse args (even if none provided, defaults will be used)
    args = ap.parse_args([] if use_defaults else None)

    images_dir = Path(args.images_dir)
    masks_dir_primary = Path(args.masks_dir)
    masks_dir_fallback = DEFAULT_MASKS_DIR_FALLBACK
    out_dir = Path(args.out_dir)

    # Choose masks directory: prefer primary if exists, else fallback to boundaries
    masks_dirs = []
    if masks_dir_primary.exists():
        masks_dirs.append(masks_dir_primary)
    if masks_dir_fallback.exists():
        masks_dirs.append(masks_dir_fallback)
    if not masks_dirs:
        print(f"[WARN] No masks directories found. Expected {masks_dir_primary} or {masks_dir_fallback}.")
        return

    # Output dirs
    frame_areas_dir = out_dir / "frame_areas"
    textures_dir = out_dir / "textures"
    examples_dir = out_dir / "texture_examples"
    meta_csv = textures_dir / "patch_metadata.csv"

    ensure_dir(frame_areas_dir)
    ensure_dir(textures_dir)
    ensure_dir(examples_dir)

    rows = []
    # find images (recursively)
    img_paths = [p for p in images_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
    img_paths = sorted(img_paths)

    for img_path in img_paths:
        try:
            img = load_image(img_path)
        except Exception as e:
            print(e)
            continue

        mask_path = find_matching_mask(img_path, masks_dirs)
        if mask_path is None:
            print(f"[WARN] No mask found for {img_path.name}")
            continue

        try:
            mask_raw = load_mask(mask_path)
        except Exception as e:
            print(e)
            continue

        mask = dilate_and_fill(mask_raw, dilate_iter=args.mask_dilate, close_iter=args.mask_close)
        frame_only = apply_mask(img, mask)
        frame_save_path = frame_areas_dir / f"{img_path.stem}_frame_only.png"
        cv2.imwrite(str(frame_save_path), frame_only)

        h, w = frame_only.shape[:2]
        patch_size = args.patch_size
        stride = args.stride
        min_cov = float(args.min_mask_coverage)

        per_image_patch_paths = []
        per_image_features = []

        # slide over image (x is column, y is row)
        for y in range(0, max(1, h - patch_size + 1), stride):
            for x in range(0, max(1, w - patch_size + 1), stride):
                patch = frame_only[y:y+patch_size, x:x+patch_size]
                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    continue

                mask_patch = mask[y:y+patch_size, x:x+patch_size]
                coverage = (mask_patch > 0).mean()

                if coverage < min_cov:
                    continue

                mean_intensity = float(np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)[mask_patch > 0]))
                if mean_intensity < args.min_mean_intensity:
                    continue

                feats = patch_features(patch, mask_patch)
                if feats is None:
                    continue

                patch_name = f"{img_path.stem}_x{x}_y{y}.png"
                patch_out = textures_dir / patch_name
                cv2.imwrite(str(patch_out), patch)

                rows.append({
                    "image": str(img_path.relative_to(images_dir)),
                    "mask": mask_path.name,
                    "patch_file": patch_out.name,
                    "x": x,
                    "y": y,
                    "patch_size": patch_size,
                    "mask_coverage": coverage,
                    "mean_intensity": mean_intensity,
                    "feat_mean": float(feats[0]),
                    "feat_std": float(feats[1]),
                    "feat_entropy": float(feats[2]),
                    **{f"lbp_{i}": float(feats[3+i]) for i in range(len(feats)-3)}
                })

                per_image_patch_paths.append(patch_out)
                per_image_features.append(feats)

        # pick representative examples via KMeans on features
        if per_image_features:
            F = np.vstack(per_image_features)
            k = min(args.examples_per_image, len(per_image_patch_paths))
            try:
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = km.fit_predict(F)
                centers = km.cluster_centers_
                for c in range(k):
                    idxs = np.where(labels == c)[0]
                    if len(idxs) == 0:
                        continue
                    cluster_feats = F[idxs]
                    dists = np.linalg.norm(cluster_feats - centers[c], axis=1)
                    best_local = idxs[np.argmin(dists)]
                    src = per_image_patch_paths[best_local]
                    dst = examples_dir / f"{src.stem}_example.png"
                    img_patch = cv2.imread(str(src), cv2.IMREAD_COLOR)
                    cv2.imwrite(str(dst), img_patch)
            except Exception as e:
                print(f"[WARN] KMeans failed for {img_path.name}: {e}")
                for src in per_image_patch_paths[:k]:
                    dst = examples_dir / f"{src.stem}_example.png"
                    img_patch = cv2.imread(str(src), cv2.IMREAD_COLOR)
                    cv2.imwrite(str(dst), img_patch)

    # write metadata CSV
    if rows:
        ensure_dir(textures_dir)
        df = pd.DataFrame(rows)
        df.to_csv(meta_csv, index=False, encoding="utf-8")
        print(f"[OK] Saved patches metadata: {meta_csv}")
    else:
        print("[INFO] No patches extracted. Check mask coverage thresholds or paths.")

if __name__ == "__main__":
    # Run without needing any terminal arguments (uses defaults)
    main()