"""
Experiment 02 - YOLO + OpenCV, no LLM. Batch run over the CI2N validation set.

    python run.py

Writes the same layout as 01-vllm-only, so experiments/evaluate.py scores it directly:
    results/<VARIANT>/preds/<id>.json   predicted netlists
    results/<VARIANT>/golden/<id>.json  golden copies for the images used
    results/<VARIANT>/log.csv           per-image status and seconds (no API cost)
    results/<VARIANT>/debug/<id>.png    overlays: wires coloured by net, pins labelled
"""

# ============================ EDITABLE CONSTANTS ============================

VARIANT = "v2"  # v0 | v1g | v1 | v2 | v2g - see README.md "Variants"
MODELS_DIR = "models"  # trained .pt files (see TRAINING.md)
DATASET_DIR = "../../datasets/ci2n_datasets-main/validation"  # has images/ and golden/
NUM_FILES = None  # how many images (sorted by name); None = all 122
RESULTS_DIR = "results"
SAVE_DEBUG = True

# ============================================================================

import csv
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from netlist_cv import VARIANTS, Pipeline  # noqa: E402
from netlist_cv.debug import overlay  # noqa: E402


def main():
    dataset = (HERE / DATASET_DIR).resolve()
    images = sorted((dataset / "images").glob("*.png"))[:NUM_FILES]
    out = HERE / RESULTS_DIR / VARIANT
    for sub in ("preds", "golden") + (("debug",) if SAVE_DEBUG else ()):
        (out / sub).mkdir(parents=True, exist_ok=True)

    pipe = Pipeline(HERE / MODELS_DIR, VARIANTS[VARIANT])
    log_path = out / "log.csv"
    with log_path.open("w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["id", "model", "status", "seconds", "prompt_tokens", "completion_tokens",
                      "cost_usd", "n_components", "problems"])
        print(f"Variant {VARIANT}: {len(images)} images -> {out}")
        for n, img in enumerate(images, 1):
            golden = dataset / "golden" / f"{img.stem}.json"
            if golden.exists():
                shutil.copy(golden, out / "golden" / golden.name)
            t0 = time.time()
            try:
                netlist, trace = pipe.run(img)
                status, problem = "ok", ""
                (out / "preds" / f"{img.stem}.json").write_text(json.dumps(netlist, indent=4))
                if SAVE_DEBUG:
                    overlay(img, trace, out / "debug" / f"{img.stem}.png")
                n_comp = len(netlist["ckt_netlist"])
            except Exception as e:  # keep going; a missing prediction is scored as empty
                status, problem, n_comp = "error", f"{type(e).__name__}: {e}", 0
                traceback.print_exc()
            secs = time.time() - t0
            log.writerow([img.stem, f"yolo-cv-{VARIANT}", status, f"{secs:.2f}", "", "", "", n_comp, problem])
            f.flush()
            print(f"[{n}/{len(images)}] {img.stem}: {status} {secs:.2f}s {n_comp} comps")

    print(f"\nScore with:\n  python3 {HERE.parent / 'evaluate.py'} {out}")


if __name__ == "__main__":
    main()
