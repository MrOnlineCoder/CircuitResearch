# Experiment 02: YOLO + OpenCV netlist extraction, no LLM (variant v2)

**Date:** 2026-11-09 · **Data:** CI2N `validation/` (122 image–netlist pairs) · **Raw metrics:** [`results/v2/metrics.json`](results/v2/metrics.json)

## Setup

| | |
|---|---|
| Pipeline | YOLO11s finds components → YOLO11s-OBB finds crossings → OpenCV connected components trace wires → 7 YOLO11n-cls orientation classifiers name the pins (see [README.md](README.md)) |
| Variant | `v2`: crossing model and orientation classifiers both on |
| Training | Runpod GPU, CI2N training folders. The 65 training images that duplicate validation circuits were removed first (`data/leakage.json`), so no validation circuit was seen in training. |
| Detector scores (held-out split) | Components: mAP50 0.893, mAP50-95 0.731 (early stop at epoch 54, best 24). Crossings: mAP50 0.975, mAP50-95 0.824 (stop 116, best 86). Orientation classifiers: 98.6–100 % top-1. |
| Inference | MacBook (Ultralytics 8.3.253, torch 2.14), single process. Wire threshold `ink_threshold = "auto"` (see obs. 4) |
| Scoring | `experiments/evaluate.py`, same settings as experiment 01 (60 s GED search plus bounds) |

## Results, with experiment 01 for comparison

| Metric | **YOLO + CV (v2)** | LLM (Sonnet 5.5, exp. 01) |
|---|---|---|
| **Success rate** (GED = 0, exact netlist) | **29.5 %** (36 / 122) | 41.8 % (51 / 122) |
| **NED, mean / median** (best known GED*) | **0.218 / 0.199** | 0.137 / 0.110 |
| NED lower bound, mean / median | 0.119 / 0.078 | 0.079 / 0.055 |
| Images with exact GED | 55 / 122 | 65 / 122 |
| **Component-count F1** (micro) | **0.984** (P 0.982, R 0.986) | 0.954 |
| Component-count F1 (macro over images) | 0.981 | 0.959 |
| Valid JSON output | 122 / 122 | 122 / 122 |
| Seconds per image, mean / median | **0.34 s / 0.25 s** (max 2.8 s, the first image, which includes model loading) | 9.2 s / 4.7 s |
| Cost per image | $0 (local); training was a one-off GPU rental | $0.016 |

\* As in experiment 01, GED is exact only for some images (55 here: 36 identical, 14 search completed, 5 bounds met). For the other 67 it is a proven range, and NED is reported from the upper bound, so **the true mean NED is between 0.119 and 0.218**. The ranges overlap with experiment 01's (0.079–0.137), but the upper bound, the lower bound and the success rate all favour the LLM.

### By circuit size (golden device count)

| Size | Images | Success: YOLO / LLM | NED mean [lower bound]: YOLO | NED mean [lower bound]: LLM | Component F1: YOLO / LLM |
|---|---|---|---|---|---|
| Small (≤ 8) | 31 | 54.8 % / 71.0 % | 0.115 [0.105] | 0.059 [0.055] | 0.967 / 0.971 |
| Medium (9–15) | 61 | 31.1 % / 39.3 % | 0.164 [0.083] | 0.129 [0.075] | 0.986 / 0.961 |
| Large (> 15) | 30 | **0.0 %** / 16.7 % | 0.434 [0.205] | 0.233 [0.111] | 0.987 / 0.942 |

### Per image, against experiment 01

| | Images |
|---|---|
| Exact in both | 23 |
| Exact only with YOLO + CV | 13 |
| Exact only with the LLM | 28 |
| Exact in neither | 58 |
| Lower NED: YOLO + CV / LLM / tie | 35 / 62 / 25 |

### Component types with notable errors

| Type | Precision | Recall | F1 | Golden count |
|---|---|---|---|---|
| Dido_amp | 1.00 | 0.14 | 0.25 | 7 |
| Voltage | 0.82 | 0.82 | 0.82 | 11 |
| Siso_amp | 0.75 | 1.00 | 0.86 | 3 |
| Diso_amp | 0.82 | 1.00 | 0.90 | 28 |
| PNP | 1.00 | 0.94 | 0.97 | 47 |

All other types (NMOS, PMOS, NPN, Res, Cap, Current, Diode) have F1 ≥ 0.97. The LLM's main error, PNP read as NPN (F1 0.80), is mostly gone here.

## Observations

1. **Components are recognised better than by the LLM, but wiring is worse.** Component F1 is 0.984 against 0.954, yet the success rate is 12 points lower. 99 images have a perfect component count, and 63 of those still have a wrong netlist (LLM: 34 of 85). The errors are in connectivity and pin naming, not detection.
2. **Large circuits fail completely.** No circuit with more than 15 devices is exact, while component F1 on them stays 0.987. Each wiring or pin rule has a small error rate, and a large circuit has many more chances for one to go wrong. The development notes put the main remaining error in MOSFET drain/source leads attached to the wrong net, about 20 % of D/S pins.
3. **Fully differential amplifiers are detected as single-ended** (6 of 7 Dido_amp, in images 092, 104 and 105). The training data has few of them, and they look almost the same as single-ended ones. This also explains most of the extra Diso_amp predictions.
4. **Light-grey drawings: the wire threshold is a trade-off.** In image 050 the wires are light grey (pixel values 150–195) and the text is darker. A global Otsu threshold (183 there) keeps the text and drops the wires, so almost every pin came out unconnected (NED 1.27). A threshold relative to the background colour finds those wires, but used on every image it also picks up light-grey annotations (current arrows, "50 µA", maths labels). Where they touch wires they merge nets: success fell from 29.5 % to 23.0 % (9 circuits lost, 1 gained; `results/v2-bg`). The pipeline therefore uses Otsu and redoes an image with the background threshold only when more than half of its pins end up unconnected. That only happens for 050 (93 % unconnected; next highest 12.5 %). Its NED improves to 0.77; the other 121 predictions are unchanged (`results/v2-otsu` has the run without the fallback, NED 0.222 [0.122]). 050 is still wrong because the detector reads its two BJTs as `nmos-bulk` and the crossing model misses a flat crossing.
5. **The two methods fail on different images.** 13 circuits are exact only with YOLO + CV and 28 only with the LLM. Together they would get 64 exact (52 %), which supports a hybrid (experiments 03/04): YOLO's component list and positions, with the LLM fixing the wiring.
6. **It is about 20× faster and free to run.** Median 0.24 s per image on a laptop against 4.7 s and $0.01 per image through the API.

### Duplicated circuits and the ground-truth-box test

49 validation circuits have copies in the CI2N training folders. Those copies were removed before training, so they are not leaked into this result. Both methods simply find them easier: YOLO + CV 38.8 % success on them against 23.3 % on the other 73; the LLM 44.9 % against 39.7 %.

46 of them have hand-drawn component boxes, which `oracle_boxes.py` uses to test wiring alone. On those 46 circuits:

| Run | Success | NED [lower bound] |
|---|---|---|
| Ground-truth boxes, `v1g` (geometric crossings, no classifiers) | 26.1 % | 0.150 [0.091] |
| Trained detector, `v2` (this report) | 39.1 % | 0.217 [0.123] |
| LLM (experiment 01) | 43.5 % | 0.117 [0.076] |

`v2` gets more exact netlists than the ground-truth-box run even with detected boxes, so the orientation classifiers and the crossing model clearly help pin naming. Its higher NED shows that, when it does go wrong, it goes wrong by more. A `v2` run with ground-truth boxes would separate detection errors from wiring errors.

## Reference

Image2Net (hybrid detection + rules pipeline) reports 80.77 % success and 0.116 mean NED on **104** validation circuits. This pipeline reimplements that kind of approach from the paper's description (no code was published) and reaches 29.5 %. Image2Net's own rules and post-processing are evidently much more refined; the numbers are indicative only, since this repository's validation set has 122 circuits.

## Caveats and next steps

- 67 GED values are still ranges (most of them large circuits). Small NED differences between methods should not be over-read.
- Single run, but the pipeline is deterministic, so repeats would give the same result.
- Not yet run: variants `v0`, `v1g`, `v1` and `v2g` (for the effect of each model), and `oracle-v2`.
- Most promising fixes: the D/S pin rule for MOSFETs (obs. 2), and more Dido_amp training examples (obs. 3). Removing light-grey annotation layers (arrows, labels) before thresholding would let the background threshold be used everywhere (obs. 4).
