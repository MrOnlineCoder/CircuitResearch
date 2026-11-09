"""
Train all experiment-02 models with Ultralytics and collect the weights in models/.

    python train.py                      # everything (run prepare_data.py first)
    python train.py devices jumpers      # only some jobs
    python train.py --smoke              # 1 epoch on 5 % of the data, to test the setup

Jobs:
    devices              YOLO11s detector, 31 classes (components, gnd/vdd, ports, hop arcs)
    jumpers              YOLO11s-OBB, 3 crossing classes
    orient_*  (7 jobs)   YOLO11n-cls orientation / mirror classifiers

Each finished job copies its best checkpoint to models/<job>.pt. These .pt files run
unchanged on macOS (CPU or Apple GPU via MPS) with the same ultralytics version.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODELS = HERE / "models"

ORIENT_JOBS = ["orient_mos", "orient_bjt_d", "orient_bjt_m", "orient_amp_d", "orient_amp_m",
               "orient_diode", "orient_voltagelines"]

JOBS = {
    "devices": dict(weights="yolo11s.pt", data=DATA / "devices.yaml", imgsz=1024, epochs=150, batch=16),
    "jumpers": dict(weights="yolo11s-obb.pt", data=DATA / "jumpers.yaml", imgsz=1024, epochs=150, batch=16),
    **{name: dict(weights="yolo11n-cls.pt", data=DATA / name, imgsz=128, epochs=60, batch=128,
                  # orientation/mirror labels change under flips: disable flip augmentation
                  fliplr=0.0, flipud=0.0)
       for name in ORIENT_JOBS},
}
COMMON = dict(patience=30, seed=0, workers=8, cache=True, plots=True, exist_ok=True)


def train_job(name: str, cfg: dict, device: str, smoke: bool) -> dict:
    from ultralytics import YOLO

    cfg = dict(cfg)
    weights = cfg.pop("weights")
    args = {**COMMON, **cfg, "data": str(cfg["data"]), "device": device,
            "project": str(HERE / "runs"), "name": name}
    if smoke:
        args.update(epochs=1, fraction=0.05, imgsz=min(args["imgsz"], 320), batch=4, cache=False, plots=False)
    t0 = time.time()
    model = YOLO(weights)
    model.train(**args)
    best = Path(model.trainer.best)
    MODELS.mkdir(exist_ok=True)
    shutil.copy(best, MODELS / f"{name}.pt")
    metrics = {k: round(float(v), 4) for k, v in (model.trainer.metrics or {}).items()
               if isinstance(v, (int, float))}
    return {"weights": weights, "epochs": args["epochs"], "imgsz": args["imgsz"],
            "minutes": round((time.time() - t0) / 60, 1), "metrics": metrics}


def main():
    import torch
    import ultralytics

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="*", help=f"subset of: {', '.join(JOBS)}")
    ap.add_argument("--device", default=None, help="'0' (GPU), 'cpu', 'mps'; default: auto")
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 5%% of the data")
    args = ap.parse_args()

    jobs = args.jobs or list(JOBS)
    unknown = set(jobs) - set(JOBS)
    if unknown:
        ap.error(f"unknown jobs: {sorted(unknown)}")
    device = args.device or ("0" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"ultralytics {ultralytics.__version__}, torch {torch.__version__}, device {device}")

    manifest_path = MODELS / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["ultralytics"] = ultralytics.__version__
    for name in jobs:
        print(f"\n=== {name} ===")
        manifest[name] = train_job(name, JOBS[name], device, args.smoke)
        MODELS.mkdir(exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))  # save progress after each job
        print(f"-> {MODELS / (name + '.pt')}  {manifest[name]['metrics']}")
    print(f"\nDone. Copy {MODELS} to your Mac (see TRAINING.md).")


if __name__ == "__main__":
    main()
