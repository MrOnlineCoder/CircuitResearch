"""
Experiment 03 - step 1: run the experiment-02 YOLO + OpenCV pipeline and save what the LLM gets.

    python3 prepare.py        # re-launches itself in ../02-yolo-cv/.venv (needs ultralytics)

For every validation image, writes:
    data/overlays/<id>.png   annotated image for the LLM: component IDs, wires coloured by net
                             with net-name tags, pin letters, crossing decisions (J / X)
    data/context/<id>.json   the pipeline's netlist with component IDs, box positions,
                             uncertain spots, pipeline seconds
    data/manifest.json       variant, models, image count

No API calls. run.py then sends original image + overlay + context to the LLM.
"""

# ============================ EDITABLE CONSTANTS ============================

VARIANT = "v2"  # experiment-02 variant whose output the LLM corrects (README of 02)
MODELS_DIR = "../02-yolo-cv/models"
DATASET_DIR = "../../datasets/ci2n_datasets-main/validation"
NUM_FILES = None  # None = all 122
OUT_DIR = "data"
OVERLAY_LONG_SIDE = 1400  # upscale small images (max x2) so the labels stay readable
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
    import cv2
    import numpy as np
    import ultralytics  # noqa: F401
except ImportError:  # use experiment 02's environment
    venv_py = CV_DIR / ".venv" / "bin" / "python"
    if not venv_py.exists() or Path(sys.prefix).resolve() == (CV_DIR / ".venv").resolve():
        sys.exit(f"Needs ultralytics: create {CV_DIR / '.venv'} (see 02-yolo-cv/README.md).")
    os.execv(str(venv_py), [str(venv_py), str(Path(__file__).resolve()), *sys.argv[1:]])

sys.path.insert(0, str(CV_DIR))
from netlist_cv import VARIANTS, Pipeline  # noqa: E402
from netlist_cv import wires as W  # noqa: E402
from netlist_cv.config import DEVICE_LABELS  # noqa: E402

# Component ID prefixes (as on a schematic: M1, Q2, R3, ...)
PREFIX = {"NMOS": "M", "PMOS": "M", "NPN": "Q", "PNP": "Q", "Res": "R", "Cap": "C", "Ind": "L",
          "Switch": "S", "Diode": "D", "Voltage": "V", "Current": "I",
          "Diso_amp": "A", "Siso_amp": "A", "Dido_amp": "A"}
PIN_ABBR = {"Drain": "D", "Gate": "G", "Source": "S", "Body": "B", "Base": "B", "Emitter": "E",
            "Collector": "C", "Positive": "+", "Negative": "-", "InP": "+", "InN": "-",
            "OutP": "o+", "OutN": "o-", "Pos": "1", "Neg": "2", "In": "in", "Out": "out"}


def net_colors(names):
    import colorsys
    out = {"GND": (90, 90, 90), "VDD": (0, 0, 200)}  # BGR
    i = 0
    for n in sorted(names):
        if n not in out:
            r, g, b = colorsys.hsv_to_rgb((i * 0.618033988) % 1.0, 0.9, 0.75)
            out[n] = (int(b * 255), int(g * 255), int(r * 255))
            i += 1
    return out


def tag(img, text, x, y, color, scale):
    """Text on a white box with a coloured border, centred on (x, y), kept inside the image."""
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, max(1, round(scale))
    (tw, tht), base = cv2.getTextSize(text, font, fs, th)
    pad = max(2, round(2 * scale))
    x0 = int(min(max(0, x - tw / 2 - pad), img.shape[1] - tw - 2 * pad - 1))
    y0 = int(min(max(0, y - tht / 2 - pad), img.shape[0] - tht - base - 2 * pad - 1))
    cv2.rectangle(img, (x0, y0), (x0 + tw + 2 * pad, y0 + tht + base + 2 * pad), (255, 255, 255), -1)
    cv2.rectangle(img, (x0, y0), (x0 + tw + 2 * pad, y0 + tht + base + 2 * pad), color, 1)
    cv2.putText(img, text, (x0 + pad, y0 + pad + tht), font, fs, color, th, cv2.LINE_AA)


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

    # --- uncertain spots ---
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
        extra = [i for i, k in enumerate(c["contacts"]) if i not in used and k.region and not k.weak]
        if extra:
            notes.append(f"{c['id']}: {len(extra)} extra line(s) touch the box but were not assigned "
                         f"to any pin (red '?' marks).")
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


def draw_overlay(image_path: Path, tr, components, crossings, out_path: Path):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    k = min(2.0, max(1.0, OVERLAY_LONG_SIDE / max(h, w)))
    canvas = (img * 0.55 + 255 * 0.45).astype(np.uint8)  # keep the original lines visible
    colors = net_colors(set(tr.region_net.values()))
    for region, net in tr.region_net.items():  # tint wire pixels with their net colour
        mask = tr.labels == region
        canvas[mask] = (0.25 * canvas[mask] + 0.75 * np.array(colors[net])).astype(np.uint8)
    if k != 1.0:
        canvas = cv2.resize(canvas, (round(w * k), round(h * k)), interpolation=cv2.INTER_CUBIC)

    def S(v):
        return int(round(v * k))

    # crossings
    for c in crossings:
        b = c["box"]
        col = (0, 0, 220) if c["id"].startswith("J") else (0, 150, 0)
        cv2.rectangle(canvas, (S(b.x0), S(b.y0)), (S(b.x1), S(b.y1)), col, max(1, S(1)))
        tag(canvas, c["id"], S(b.x1) + S(8), S(b.y0) - S(6), col, k)
    # components: box, ID, pin letters, unused contacts
    for c in components:
        b = c["box"]
        cv2.rectangle(canvas, (S(b.x0), S(b.y0)), (S(b.x1), S(b.y1)), (40, 40, 40), max(1, S(1)))
        tag(canvas, c["id"], S(b.x0) + S(10), S(b.y0) - S(7), (0, 0, 0), k)
        pin_of = {i: p for p, ids in c["pin_contacts"].items() for i in ids}
        for i, ct in enumerate(c["contacts"]):
            x, y = S(ct.x), S(ct.y)
            if i in pin_of:
                cv2.circle(canvas, (x, y), max(2, S(2.5)), (0, 0, 0), -1)
                cv2.putText(canvas, PIN_ABBR.get(pin_of[i], pin_of[i]), (x + S(3), y - S(3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4 * k, (0, 0, 0), max(1, round(k)), cv2.LINE_AA)
            elif ct.region and not ct.weak:
                cv2.circle(canvas, (x, y), max(2, S(3)), (0, 0, 255), -1)
                cv2.putText(canvas, "?", (x + S(3), y - S(3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * k,
                            (0, 0, 255), max(1, round(k)), cv2.LINE_AA)
    # one name tag per net, on the largest wire region of that net
    best = {}
    for region, net in tr.region_net.items():
        ys, xs = np.nonzero(tr.labels == region)
        if len(xs) and len(xs) > best.get(net, (0,))[0]:
            j = len(xs) // 2  # a pixel on the wire itself (not the bbox centre)
            best[net] = (len(xs), xs[j], ys[j])
    for net, (_, x, y) in best.items():
        tag(canvas, net, S(x), S(y), colors[net], k)
    cv2.imwrite(str(out_path), canvas)
    return k


def box_json(b):
    return [round(b.x0), round(b.y0), round(b.x1), round(b.y1)]


def main():
    dataset = (HERE / DATASET_DIR).resolve()
    images = sorted((dataset / "images").glob("*.png"))[:NUM_FILES]
    out = HERE / OUT_DIR
    for sub in ("overlays", "context"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    settings = VARIANTS[VARIANT]
    pipe = Pipeline((HERE / MODELS_DIR).resolve(), settings)

    print(f"Variant {VARIANT}: {len(images)} images -> {out}")
    for n, img in enumerate(images, 1):
        netlist, tr, components, crossings, notes, seconds = build(img, pipe, settings)
        scale = draw_overlay(img, tr, components, crossings, out / "overlays" / f"{img.stem}.png")
        ctx = {
            "id": img.stem,
            "variant": VARIANT,
            "pipeline_seconds": round(seconds, 3),
            "overlay_scale": round(scale, 3),
            "ink_mode": tr.ink_mode,
            "components": [{"id": c["id"], "type": c["type"], "detector_label": c["box"].label,
                            "confidence": round(c["box"].conf, 3), "box": box_json(c["box"]),
                            "pins": c["pins"]} for c in components],
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
        "dataset": str(dataset), "images": len(images), "overlay_long_side": OVERLAY_LONG_SIDE,
    }, indent=2))
    print(f"\nDone. Next: python3 run.py  (see README.md)")


if __name__ == "__main__":
    main()
