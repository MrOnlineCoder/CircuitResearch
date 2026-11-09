"""
Score the wire + pin stages alone, with ground-truth component boxes ("oracle detection").

    python oracle_boxes.py        # after prepare_data.py; no trained models needed for v0/v1g

46 device_identification images are duplicates of validation circuits (found by
prepare_data.py, data/leakage.json). They have hand-drawn boxes AND a golden netlist, so
running the pipeline on them with those boxes instead of the YOLO detector measures how
good the OpenCV wiring and pin rules are on their own: an upper bound for experiment 02
and a separate row in the results table.

Output: results/oracle-<VARIANT>/ (preds, golden, debug) -> score with experiments/evaluate.py
"""

# ============================ EDITABLE CONSTANTS ============================

VARIANT = "v1g"  # v0 or v1g need no models; v1/v2 also use the trained crossing/orientation models
CI2N_DIR = "../../datasets/ci2n_datasets-main"
RESULTS_DIR = "results"

# ============================================================================

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from netlist_cv import VARIANTS, Pipeline  # noqa: E402
from netlist_cv.debug import overlay  # noqa: E402
from netlist_cv.detect import Box, Models  # noqa: E402


class OracleModels(Models):
    """Device boxes from the Labelme annotation; everything else from the trained models."""

    def __init__(self, models_dir, settings, annotation):
        super().__init__(models_dir, settings)
        self.annotation = annotation

    def devices(self, image):
        boxes = []
        for s in self.annotation["shapes"]:
            (xa, ya), (xb, yb) = s["points"][:2]
            boxes.append(Box(s["label"], 1.0, min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)))
        return boxes


def main():
    ci2n = (HERE / CI2N_DIR).resolve()
    leak_file = HERE / "data" / "leakage.json"
    if not leak_file.exists():
        sys.exit("Run prepare_data.py first (it finds the duplicated circuits).")
    pairs, seen = json.loads(leak_file.read_text()), set()
    out = HERE / RESULTS_DIR / f"oracle-{VARIANT}"
    for sub in ("preds", "golden", "debug"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    pipe = Pipeline(HERE / "models", VARIANTS[VARIANT])
    for p in pairs:
        img = Path(p["train_image"])
        vid = Path(p["validation_image"]).stem
        if "device_identification" not in str(img) or vid in seen:
            continue
        seen.add(vid)
        pipe.models = OracleModels(HERE / "models", VARIANTS[VARIANT],
                                   json.loads(img.with_suffix(".json").read_text()))
        netlist, trace = pipe.run(img)
        (out / "preds" / f"{vid}.json").write_text(json.dumps(netlist, indent=4))
        (out / "golden" / f"{vid}.json").write_text((ci2n / "validation" / "golden" / f"{vid}.json").read_text())
        overlay(img, trace, out / "debug" / f"{vid}.png")
    print(f"{len(seen)} circuits -> {out}\nScore with:\n  python3 {HERE.parent / 'evaluate.py'} {out}")


if __name__ == "__main__":
    main()
