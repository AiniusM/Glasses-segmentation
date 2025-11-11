# src/main.py
"""
Vieno mygtuko pipeline'as visam projektui.
Paleidžia:
1) Rėmelio ribas/maskes (Canny/Sobel) -> results/masks, results/boundaries, results/overlays
2) CSV ataskaitą apie maskes -> results/report.csv
3) Lęšio išskyrimą (gaussian/bilateral/guided + otsu/adaptive) -> results/lens_masks, results/lens_overlays
4) Šešėlio atskyrimą -> results/frame_no_shadow_masks, results/frame_no_shadow_overlays, results/shadow_report.csv
5) Heuristiniai konstrukciniai komponentai -> results/components
6) YOLO komponentų aptikimas -> results/components_yolo_rcnn
7) Rėmelio tekstūros iškarpos -> results/frame_areas, results/textures, results/texture_examples, results/textures/patch_metadata.csv
"""

from pathlib import Path
import sys
import shutil
import traceback

# ---- Projekto keliai ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # src/ -> projekto šaknis
SRC_DIR = PROJECT_ROOT / "src"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
RESULTS = PROJECT_ROOT / "results"

# Leisk importuoti `processing` modulius kaip paketą
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# ---- Importai iš jūsų modulių ----
# 1) Rėmelio ribos/maskės ir ataskaita
from processing.boundaries import process_all as run_boundaries
from processing.report import make_report as run_masks_report

# 2) Lęšio išskyrimas
from processing.lens_extract import process_all as run_lens

# 3) Šešėlio atskyrimas
from processing.shadow_remove import process_all as run_shadow

# 4) Konstrukciniai komponentai: heuristinis
from processing.frame_components import process_all as run_components_heur

# 5) YOLO-only komponentai
from processing.frame_componenrs_yolo_rcnn import process_all as run_components_yolo

# 6) Tekstūros iškarpos
# (failas pritaikytas veikti be CLI – turi `main()` funkciją)
try:
    from processing.extract_frame_textures import main as run_textures
except Exception:
    run_textures = None


# ---- Konfigūracija (įjunk/išjunk žingsnius) ----
class Config:
    RUN_BOUNDARIES_CANNY = True
    RUN_BOUNDARIES_SOBEL = False      # jei norit – įjunkite
    RUN_MASKS_REPORT     = True

    RUN_LENS             = True
    RUN_SHADOW           = True

    RUN_COMPONENTS_HEUR  = True
    RUN_COMPONENTS_YOLO  = True       # reikalauja ultralytics (YOLO)

    RUN_TEXTURES         = True       # reikalingas jūsų `extract_frame_textures.py`

    CLEAR_PREV_RESULTS   = False      # jei True, prieš startą išvalys kai kurias results šakas


def safe_step(name, fn, *args, **kwargs):
    print(f"\n=== ▶ {name} ===")
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        traceback.print_exc()
        return None


def ensure_dirs():
    (PROJECT_ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "results").mkdir(parents=True, exist_ok=True)


def clear_previous():
    # Norint švaraus starto – saugiai trinam tik automatiškai generuojamus katalogus
    to_clear = [
        RESULTS / "masks",
        RESULTS / "overlays",
        RESULTS / "boundaries",
        RESULTS / "lens_masks",
        RESULTS / "lens_overlays",
        RESULTS / "shadow_masks",
        RESULTS / "frame_no_shadow_masks",
        RESULTS / "frame_no_shadow_overlays",
        RESULTS / "components",
        RESULTS / "components_yolo_rcnn",
        RESULTS / "frame_areas",
        RESULTS / "textures",
        RESULTS / "texture_examples",
    ]
    for d in to_clear:
        if d.exists():
            print(f" - removing {d}")
            shutil.rmtree(d, ignore_errors=True)


def main():
    ensure_dirs()

    if Config.CLEAR_PREV_RESULTS:
        print("⚠ Clearing previous results...")
        clear_previous()

    # Greitas patikrinimas: ar yra naujų nuotraukų
    imgs = list(DATA_RAW.rglob("*"))
    if not any(p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff") for p in imgs):
        print("ℹ Nerasta vaizdų kataloge data/raw/. Įkelkite naujas nuotraukas ir paleiskite dar kartą.")
        return

    # 1) Rėmelio ribos/maskės
    if Config.RUN_BOUNDARIES_CANNY:
        safe_step("Rėmelio ribos (Canny)", run_boundaries, method="canny")
    if Config.RUN_BOUNDARIES_SOBEL:
        safe_step("Rėmelio ribos (Sobel)", run_boundaries, method="sobel")

    # 2) CSV ataskaita apie maskes
    if Config.RUN_MASKS_REPORT:
        safe_step("Maskių ataskaita (CSV)", run_masks_report)

    # 3) Lęšio išskyrimas
    if Config.RUN_LENS:
        safe_step("Lęšio išskyrimas", run_lens)

    # 4) Šešėlio atskyrimas
    if Config.RUN_SHADOW:
        safe_step("Šešėlio atskyrimas", run_shadow)

    # 5) Konstrukciniai komponentai: heuristika
    if Config.RUN_COMPONENTS_HEUR:
        safe_step("Konstrukciniai komponentai (heuristiniai)", run_components_heur)

    # 6) YOLO-only komponentai
    if Config.RUN_COMPONENTS_YOLO:
        safe_step("Konstrukciniai komponentai (YOLO)", run_components_yolo)

    # 7) Rėmelio tekstūros iškarpos
    if Config.RUN_TEXTURES:
        if run_textures is not None:
            safe_step("Rėmelio tekstūros iškarpos", run_textures)
        else:
            print("⚠ `extract_frame_textures.py` neimportuojamas. Patikrinkite, ar failas egzistuoja ir neturi klaidų.")

    print("\n✅ Viskas. Rezultatus rasite kataloge `results/`.")


if __name__ == "__main__":
    main()