# Experiment 02: YOLO + OpenCV, no LLM

Converts a circuit-diagram image into a CI2N netlist JSON with trained detectors and classical image processing, without any language model. Output format and folder layout match experiment 01, so `experiments/evaluate.py` scores both the same way.

## Pipeline

| # | Stage | How | Code |
|---|---|---|---|
| 1 | Find components, gnd/vdd symbols, ports, hop arcs | YOLO11s detector (`models/devices.pt`) | `netlist_cv/detect.py` |
| 2 | Find crossings; decide which are jumpers | YOLO11s-OBB (`models/jumpers.pt`), **or** model-free: intersect horizontal/vertical strokes and count arms (`wires.find_crossings`); then a drawing-style rule | `pipeline.py: _choose_jumpers` |
| 3 | Wire regions | Otsu threshold (redone with a background-relative threshold if most pins end up unconnected, i.e. light-grey wires) → erase component and jumper boxes → connected components: **one region = one net** | `netlist_cv/wires.py` |
| 4 | Join nets | Jumpers: lead *i* ↔ opposite lead *i + N/2*. All regions under `gnd` symbols → `GND`, under `vdd` symbols → `VDD`. | `pipeline.py` |
| 5 | Pins | Wire blobs in a thin ring around each box = contacts. Pin names from orientation classifiers (v2) or geometry rules (v0/v1). | `netlist_cv/pins.py` |
| 6 | Output | Net names (`GND`, `VDD`, `net1…`; an unconnected pin gets its own `ncK` net), CI2N JSON | `pipeline.py` |

**Crossing rule** (class `cross` = 4-way crossing without a dot):
- **Hop arcs** are never connected.
- **A flat crossing:**
  - is **connected** if the drawing uses hop arcs elsewhere (hops mark the non-connections);
  - is a **jumper** if the drawing uses dot junctions (dots mark the connections);
  - otherwise falls back to `flat_crossing_default` (jumper).

**Pin rules** (`pins.py`), checked against the CI2N orientation datasets and golden netlists:
- **With orientation classifiers (v2):** they give the gate/base side, the amplifier output direction, the diode cathode and the battery's positive plate. The mirror classifiers give emitter vs collector and + vs − input.
- **Geometry fallback:**
  - **Gate/base:** the only lead on a left/right side of the box while the other two use top/bottom (or rotated by 90°).
  - **C/E and D/S:** by position when clearly one above the other (NPN emitter and NMOS source lower, PNP emitter and PMOS source upper). Otherwise by net: `GND` / `VDD`.
  - **Op-amp inputs:** "+" counter-clockwise from the output tip, unless exactly one input is fed back from the output: that one is "−".
  - **Current / voltage sources:** In and Positive on the supply side, Out and Negative on ground.
- **Text next to a box** (e.g. "M7") creates contacts too. Small regions that touch only one component and no terminal are marked *weak* and used only when nothing better is there.

## Variants (for the results table)

| Variant | Crossings | Orientation classifiers | Needs trained models |
|---|---|---|---|
| `v0` | none: every crossing connects, except hop arcs from the detector | – (geometry rules) | `devices.pt` |
| `v1g` | geometric detector (no model) | – | `devices.pt` |
| `v1` | YOLO-OBB model | – | `devices.pt`, `jumpers.pt` |
| `v2` | YOLO-OBB model | ✓ | all |
| `v2g` | geometric detector | ✓ | `devices.pt`, `orient_*.pt` |

The variants are defined in `netlist_cv/config.py` (`VARIANTS`), together with all thresholds. Comparing `v1` and `v1g` shows whether the crossing model is worth training.

## Wiring and pins alone: ground-truth boxes

46 training images duplicate validation circuits (see below). They come with hand-drawn component boxes, so `oracle_boxes.py` runs stages 2–6 with **perfect detection** and scores them against the golden netlists. This gives an upper bound for the pipeline and a separate row for the paper. The rules above were developed with it; see the results under [Development notes](#development-notes).

```bash
.venv/bin/python oracle_boxes.py          # VARIANT constant at the top; v0/v1g need no models
python3 ../evaluate.py results/oracle-v1g
```

## Setup (Mac)

1. Train the models on a GPU pod: see **[TRAINING.md](TRAINING.md)**. Copy `models/` here.
2. Install:

```bash
cd experiments/02-yolo-cv
python3 -m venv .venv
.venv/bin/pip install -r requirements-infer.txt
```

## Usage

One image → JSON:

```bash
.venv/bin/python infer.py path/to/schematic.png                       # JSON to stdout
.venv/bin/python infer.py img.png -o netlist.json --debug overlay.png --variant v1
```

The debug overlay colours each net and labels each pin (`G/D/S`, `B/C/E`, `+/-`, …). Jumpers are red `J`, connected crossings green `X`, and contacts that were ignored are red dots. It's the fastest way to see what went wrong on an image.

Whole validation set → scored results:

```bash
# edit the constants at the top of run.py (VARIANT, NUM_FILES), then
.venv/bin/python run.py
python3 ../evaluate.py results/v2
```

`run.py` writes `results/<variant>/preds/`, `golden/`, `log.csv` (seconds per image) and `debug/` overlays.

## Data preparation and leakage

`prepare_data.py` converts the CI2N folders into YOLO datasets. It also **drops every training image that duplicates a validation circuit**, compared by perceptual hash. 65 training images matched 49 of the 122 validation circuits; they are listed in `data/leakage.json`. Without this filter, experiment-02 scores would be inflated compared with experiment 01, whose LLM never trained on these images.

## Development notes

The rules were developed with `oracle_boxes.py` (ground-truth boxes, 46 circuits), scored with a short 3-second GED search. Each row adds one change:

| Change | Exact netlists | NED (upper bound) |
|---|---|---|
| First version (boxes erased with 4 % padding, v0) | 8.7 % | 0.41 |
| No padding: padding cut wires running along boxes | 15.2 % | 0.34 |
| Op-amp input rule + text contacts marked weak + position before net for C/E, D/S | 17.4 % | 0.27 |
| Geometric crossing detector (v1g) | 17.4 % | 0.22 |
| Gate/base from box sides + text threshold relative to image size | **26.1 %** | **0.15** |

The remaining errors are mostly MOSFET drain/source leads attached to the wrong net in dense circuits: about 20 % of D/S pins, against 7–14 % of gate pins.

## Known limitations

- **No OCR.** Net labels with the same text in two places (e.g. two `VOUT` tags without a wire between them) are not merged.
- **All `vdd` symbols are merged into one `VDD` net,** because without OCR `VDD`, `VCC` and `VSS` drawn with the same symbol can't be told apart.
- **Contacts come from the detected boxes.** A box that cuts across a wire or swallows a junction changes the topology; boxes are padded slightly (`box_pad_frac`) to reduce this.
- **The amplifier mirror classifier was trained on only 369 crops.** Fully differential output polarity (OutP/OutN) follows a convention, not a classifier.
- **Switches are output as `Switch`.** The golden netlists contain none.
