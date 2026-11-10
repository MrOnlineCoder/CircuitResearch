"""
Experiment 04 - step 1: run the experiment-02 YOLO + OpenCV pipeline and save what the LLM gets.

    python3 prepare.py        # re-launches itself in ../02-yolo-cv/.venv (needs ultralytics)

Same as 03-hybrid/prepare.py, but WITHOUT the annotated overlay image. Instead of drawing
component IDs and pins on an image, the context gives their positions as pixel coordinates
in the original image, so the LLM can still find "M3" and its drain lead.

For every validation image, writes:
    data/context/<id>.json   the pipeline's netlist with component IDs, image size, component
                             boxes, pin lead positions, crossing positions, uncertain spots,
                             pipeline seconds
    data/manifest.json       variant, models, image count

No API calls. run.py then sends original image + this context to the LLM.
"""

# ============================ EDITABLE CONSTANTS ============================

VARIANT = "v2"  # experiment-02 variant whose output the LLM corrects (README of 02)
MODELS_DIR = "../02-yolo-cv/models"
DATASET_DIR = "../../datasets/ci2n_datasets-main/validation"
NUM_FILES = None  # None = all 122
OUT_DIR = "data"
LOW_CONF = 0.5  # detections below this confidence are listed as uncertain

# ============================================================================

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CV_DIR = (HERE / "../02-yolo-cv").resolve()

try:
    import ultralytics  # noqa: F401
except ImportError:  # use experiment 02's environment
    venv_py = CV_DIR / ".venv" / "bin" / "python"
    if not venv_py.exists() or Path(sys.prefix).resolve() == (CV_DIR / ".venv").resolve():
        sys.exit(f"Needs ultralytics: create {CV_DIR / '.venv'} (see 02-yolo-cv/README.md).")
    os.execv(str(venv_py), [str(venv_py), str(Path(__file__).resolve()), *sys.argv[1:]])

sys.path.insert(0, str(CV_DIR))
from netlist_cv import VARIANTS, Pipeline  # noqa: E402
from netlist_cv import wires as W  # noqa: E402

# Component ID prefixes (as on a schematic: M1, Q2, R3, ...)
PREFIX = {"NMOS": "M", "PMOS": "M", "NPN": "Q", "PNP": "Q", "Res": "R", "Cap": "C", "Ind": "L",
          "Switch": "S", "Diode": "D", "Voltage": "V", "Current": "I",
          "Diso_amp": "A", "Siso_amp": "A", "Dido_amp": "A"}


def build(image_path: Path, pipe: Pipeline, settings):
    t0 = time.time()
    netlist, tr = pipe.run(image_path)
    seconds = time.time() - t0

    # netlist entries are in the same order as components sorted by (y0, x0)
    comps = sorted(tr.components, key=lambda x: (x[0].y0, x[0].x0))
    counters, components = {}, []
    for (box, contacts, pins), entry in zip(comps, netlist["ckt_netlist"]):
        dtype = entry["component_type"]
        p = PREFIX.get(dtype, "X")
        counters[p] = counters.get(p, 0) + 1
        components.append({"id": f"{p}{counters[p]}", "type": dtype, "box": box, "contacts": contacts,
                           "pin_contacts": pins, "pins": dict(entry["port_connection"])})

    # --- uncertain spots (same as experiment 03, without references to overlay marks) ---
    notes = []
    if tr.ink_mode != "otsu":
        notes.append("The lines are faint (light grey), so wire tracing used a lenient threshold; "
                     "nearby text or arrows may have been merged into wires.")
    pin_count = {}
    for c in components:
        for net in c["pins"].values():
            pin_count[net] = pin_count.get(net, 0) + 1
    terminal_nets = set()
    for b in tr.terminals:
        for r in W.regions_in_box(b, tr.labels, settings):
            if r in tr.region_net:
                terminal_nets.add(tr.region_net[r])
    for c in components:
        box = c["box"]
        if box.conf < LOW_CONF:
            notes.append(f"{c['id']}: detected as {c['type']} with low confidence ({box.conf:.2f}).")
        for pin, net in c["pins"].items():
            idx = c["pin_contacts"].get(pin, [])
            if net.startswith("nc"):
                notes.append(f"{c['id']}.{pin}: no wire found at this lead (left unconnected as {net}).")
            elif idx and c["contacts"][idx[0]].weak:
                notes.append(f"{c['id']}.{pin}: attached through a very small blob, maybe text "
                             f"rather than a wire ({net}).")
        used = {i for ids in c["pin_contacts"].values() for i in ids}
        extra = [c["contacts"][i] for i, k in enumerate(c["contacts"])
                 if i not in used and k.region and not k.weak]
        if extra:
            where = ", ".join(f"({round(k.x)}, {round(k.y)})" for k in extra)
            notes.append(f"{c['id']}: {len(extra)} extra line(s) touch the box but were not assigned "
                         f"to any pin, at {where}.")
        if box in tr.guessed_inputs:
            notes.append(f"{c['id']}: +/- inputs were guessed (orientation classifier unsure).")
    for net, n in sorted(pin_count.items()):
        if n == 1 and not net.startswith("nc") and net not in ("GND", "VDD") and net not in terminal_nets:
            owner = next(f"{c['id']}.{p}" for c in components for p, v in c["pins"].items() if v == net)
            notes.append(f"{net}: only one pin ({owner}) and no input/output port - a dangling wire, "
                         f"or it should join another net.")

    # crossings
    crossings = []
    for k, b in enumerate(tr.jumpers, 1):
        kind = "hop arc" if b.label == "cross-line-curved" else "flat crossing without a dot"
        crossings.append({"id": f"J{k}", "decision": "NOT connected", "kind": kind, "box": b})
    for k, b in enumerate(tr.connected_crossings, 1):
        crossings.append({"id": f"X{k}", "decision": "connected", "kind": "flat crossing without a dot",
                          "box": b})
    return netlist, tr, components, crossings, notes, seconds


def box_json(b):
    return [round(b.x0), round(b.y0), round(b.x1), round(b.y1)]


def pin_positions(c) -> dict:
    """Pin -> [x, y] of the lead where the pipeline attached it (original image pixels)."""
    out = {}
    for pin, idx in c["pin_contacts"].items():
        if idx and not pin.startswith("_"):
            k = c["contacts"][idx[0]]
            out[pin] = [round(k.x), round(k.y)]
    return out


def main():
    dataset = (HERE / DATASET_DIR).resolve()
    images = sorted((dataset / "images").glob("*.png"))[:NUM_FILES]
    out = HERE / OUT_DIR
    (out / "context").mkdir(parents=True, exist_ok=True)
    settings = VARIANTS[VARIANT]
    pipe = Pipeline((HERE / MODELS_DIR).resolve(), settings)

    print(f"Variant {VARIANT}: {len(images)} images -> {out}")
    for n, img in enumerate(images, 1):
        netlist, tr, components, crossings, notes, seconds = build(img, pipe, settings)
        h, w = tr.labels.shape[:2]
        ctx = {
            "id": img.stem,
            "variant": VARIANT,
            "pipeline_seconds": round(seconds, 3),
            "image_size": [w, h],
            "ink_mode": tr.ink_mode,
            "components": [{"id": c["id"], "type": c["type"], "detector_label": c["box"].label,
                            "confidence": round(c["box"].conf, 3), "box": box_json(c["box"]),
                            "pins": c["pins"], "pin_xy": pin_positions(c)} for c in components],
            "crossings": [{"id": c["id"], "decision": c["decision"], "kind": c["kind"],
                           "box": box_json(c["box"])} for c in crossings],
            "uncertain": notes,
            "netlist": netlist,  # plain pipeline output (fallback if the LLM call fails)
        }
        (out / "context" / f"{img.stem}.json").write_text(json.dumps(ctx, indent=2))
        print(f"[{n}/{len(images)}] {img.stem}: {len(components)} comps, {len(crossings)} crossings, "
              f"{len(notes)} notes, {seconds:.2f}s")

    (out / "manifest.json").write_text(json.dumps({
        "variant": VARIANT, "models_dir": str((HERE / MODELS_DIR).resolve()),
        "dataset": str(dataset), "images": len(images),
    }, indent=2))
    print(f"\nDone. Next: python3 run.py  (see README.md)")


if __name__ == "__main__":
    main()
