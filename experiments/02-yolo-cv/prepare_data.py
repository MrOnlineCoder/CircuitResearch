"""
Convert the CI2N training folders into Ultralytics YOLO datasets.

    python prepare_data.py [--ci2n PATH] [--out data]

Creates:
    data/devices/       detection (Labelme rectangles -> YOLO txt), devices.yaml
    data/jumpers/       oriented boxes (already YOLO-OBB), jumpers.yaml
    data/orient_*/      one classification dataset per orientation task (train/ val/ per class)
    data/leakage.json   training images removed because they match a validation image

Leakage filter: any training image whose perceptual hash (dHash) is within LEAK_MAX_BITS
of a validation/ image is dropped, so the pipeline is never scored on circuits it trained on.
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from netlist_cv.config import DEVICE_CLASSES, JUMPER_CLASSES, ORIENTATION_TASKS  # noqa: E402

DEFAULT_CI2N = HERE / "../../datasets/ci2n_datasets-main"
VAL_FRACTION = 0.10  # devices: share of images held out for YOLO validation
ORIENT_VAL_FRACTION = 0.15
LEAK_MAX_BITS = 5  # dHash Hamming distance (of 64) that counts as the same circuit
SEED = 0
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def dhash(path: Path) -> int:
    a = np.asarray(Image.open(path).convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


class LeakFilter:
    def __init__(self, validation_images: list[Path]):
        self.val = {p.name: dhash(p) for p in validation_images}
        self.hits = []

    def leaked(self, path: Path) -> bool:
        h = dhash(path)
        for name, vh in self.val.items():
            if bin(h ^ vh).count("1") <= LEAK_MAX_BITS:
                self.hits.append({"train_image": str(path), "validation_image": name})
                return True
        return False


def reset(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def write_yaml(path: Path, root: Path, names: list[str], extra: str = "") -> None:
    lines = [f"path: {root.resolve()}", "train: images/train", "val: images/val", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    path.write_text("\n".join(lines) + "\n" + extra)


def prepare_devices(ci2n: Path, out: Path, leaks: LeakFilter) -> dict:
    src = ci2n / "device_identification"
    root = reset(out / "devices")
    cls_id = {n: i for i, n in enumerate(DEVICE_CLASSES)}
    items, skipped_labels = [], {}
    for j in sorted(src.glob("*.json")):
        img = next((p for p in (src / f"{j.stem}{e}" for e in IMAGE_EXTS) if p.exists()), None)
        if img is None or leaks.leaked(img):
            continue
        items.append((j, img))
    random.Random(SEED).shuffle(items)
    n_val = int(len(items) * VAL_FRACTION)
    counts = {"train": 0, "val": 0}
    for k, (j, img) in enumerate(items):
        split = "val" if k < n_val else "train"
        ann = json.loads(j.read_text())
        W, H = Image.open(img).size
        lines = []
        for shp in ann.get("shapes", []):
            label = shp["label"]
            if label not in cls_id:
                skipped_labels[label] = skipped_labels.get(label, 0) + 1
                continue
            (xa, ya), (xb, yb) = shp["points"][:2]
            x0, x1 = max(0.0, min(xa, xb)), min(float(W), max(xa, xb))
            y0, y1 = max(0.0, min(ya, yb)), min(float(H), max(ya, yb))
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            lines.append(f"{cls_id[label]} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                         f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
        for sub in ("images", "labels"):
            (root / sub / split).mkdir(parents=True, exist_ok=True)
        shutil.copy(img, root / "images" / split / img.name)
        (root / "labels" / split / f"{img.stem}.txt").write_text("\n".join(lines) + "\n")
        counts[split] += 1
    write_yaml(out / "devices.yaml", root, DEVICE_CLASSES)
    return {"images": counts, "unknown_labels_skipped": skipped_labels}


def prepare_jumpers(ci2n: Path, out: Path, leaks: LeakFilter) -> dict:
    src = ci2n / "jumper_identification"
    root = reset(out / "jumpers")
    counts = {}
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        n = 0
        for img in sorted((src / "images" / split).iterdir()):
            lab = src / "labels" / split / f"{img.stem}.txt"
            if img.suffix.lower() not in IMAGE_EXTS or not lab.exists() or leaks.leaked(img):
                continue
            shutil.copy(img, root / "images" / split / img.name)
            shutil.copy(lab, root / "labels" / split / lab.name)
            n += 1
        counts[split] = n
    write_yaml(out / "jumpers.yaml", root, JUMPER_CLASSES)
    return {"images": counts}


def prepare_orientation(ci2n: Path, out: Path) -> dict:
    stats = {}
    for name, folder in ORIENTATION_TASKS.items():
        src = ci2n / "device_orientation" / folder
        root = reset(out / name)
        stats[name] = {}
        for cls_dir in sorted(p for p in src.iterdir() if p.is_dir() and not p.name.startswith(".")):
            files = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            random.Random(SEED).shuffle(files)
            n_val = max(1, int(len(files) * ORIENT_VAL_FRACTION))
            for k, f in enumerate(files):
                dst = root / ("val" if k < n_val else "train") / cls_dir.name
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dst / f.name)
            stats[name][cls_dir.name] = len(files)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ci2n", default=str(DEFAULT_CI2N), help="path to ci2n_datasets-main")
    ap.add_argument("--out", default=str(HERE / "data"))
    args = ap.parse_args()
    ci2n, out = Path(args.ci2n).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    leaks = LeakFilter(sorted((ci2n / "validation" / "images").iterdir()))
    print("devices ...")
    report = {"devices": prepare_devices(ci2n, out, leaks)}
    print("jumpers ...")
    report["jumpers"] = prepare_jumpers(ci2n, out, leaks)
    print("orientation ...")
    report["orientation"] = prepare_orientation(ci2n, out)
    report["leakage"] = {"removed": len(leaks.hits), "max_hamming_bits": LEAK_MAX_BITS, "pairs": leaks.hits}
    (out / "leakage.json").write_text(json.dumps(leaks.hits, indent=2))
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "leakage"}, indent=2))
    print(f"Removed {len(leaks.hits)} training images matching validation images (see {out / 'leakage.json'})")


if __name__ == "__main__":
    main()
