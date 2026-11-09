# Experiment 01: VLM-only netlist extraction: Claude Sonnet 5.5

**Date:** 2026-11-09 · **Data:** CI2N `validation/` (122 image–netlist pairs) · **Raw metrics:** [`results/claude-sonnet-5.5/metrics.json`](results/claude-sonnet-5.5/metrics.json)

## Setup

| | |
|---|---|
| Model | `anthropic/claude-sonnet-5.5` via OpenRouter, default reasoning, `max_tokens` 16000 |
| Method | Zero-shot, one request per image: image + prompt listing the CI2N component types and port names → CI2N JSON netlist |
| Parsing | Last JSON object in the reply containing `ckt_netlist` (handles model self-corrections) |
| Scoring | `experiments/evaluate.py` (dataset's own GED code), GED timeout 60 s per image |
| Avg. tokens per image | 1,860 in / 1,226 out |

## Results

| Metric | Value |
|---|---|
| **Success rate** (GED = 0, exact netlist) | **41.8 %** (51 / 122) |
| NED, mean / median (all images, upper bound*) | 0.490 / 0.598 |
| NED, mean / median (56 images with exact GED) | 0.018 / 0.000 |
| **Component-count F1** (micro) | **0.954** (P 0.949, R 0.959) |
| Component-count F1 (macro over images) | 0.959 |
| Valid JSON output | 122 / 122 (2 recovered from self-corrections, see obs. 4) |
| Seconds per image, mean / median | 9.2 s / 4.7 s (min 1.6, max 45.7) |
| USD per image, mean / median | $0.016 / $0.010 |
| Total cost / cost per correct netlist | $1.95 / $0.038 |

\* **66 of the 71 imperfect netlists hit the GED timeout**, so their NED is only an upper bound, and often a loose one. Image 095 has 14 predicted vs 12 golden devices with near-identical net structure but scores NED 1.21 (worse than an empty prediction); re-running with a 300 s timeout did not improve it. **The all-images NED overstates the error and should not be reported as-is.** The exact-subset NED is biased the other way (it includes mostly easy, small circuits). The success rate is unaffected because the exact-match (isomorphism) check runs without a timeout.

### By circuit size (golden device count)

| Size | Images | Success rate | Component F1 (macro) | GED timeouts |
|---|---|---|---|---|
| Small (≤ 8) | 31 | 71.0 % | 0.971 | 4 |
| Medium (9–15) | 61 | 39.3 % | 0.961 | 37 |
| Large (> 15) | 30 | 16.7 % | 0.942 | 25 |

### Component types with notable errors

| Type | Precision | Recall | F1 | Golden count |
|---|---|---|---|---|
| NPN | 0.68 | 0.97 | 0.80 | 33 |
| PNP | 0.97 | 0.68 | 0.80 | 47 |
| Voltage | 0.63 | 0.91 | 0.74 | 11 |
| PMOS | 0.98 | 0.93 | 0.95 | 582 |
| NMOS | 0.92 | 0.99 | 0.96 | 548 |

All other types (Res, Cap, Current, Diode, all amplifier types) have F1 ≥ 0.97.

## Observations

1. **Recognition is strong, connectivity is the bottleneck.** Component-count F1 is 0.95, yet only 42 % of netlists are exactly right. 34 images have a perfect component count but a wrong netlist, i.e. the errors are in the wiring.
2. **Accuracy drops sharply with circuit size**, from 71 % success on small circuits to 17 % on large ones, while component F1 stays around 0.94–0.97.
3. **PNP is often labelled NPN.** The mirrored precision/recall (≈ 15 extra NPN, ≈ 15 missing PNP) points to an emitter-arrow direction confusion. Some PMOS are also labelled NMOS. **Voltage sources are over-predicted** (16 predicted vs 11), likely from input/supply symbols.
4. **The model sometimes self-corrects mid-answer** (2 / 122: images 031, 049). It emits a netlist, then "Wait, I need to correct the connections…", then a second netlist. The parser now takes the last netlist. 049's correction is exactly right; 031 still has wiring errors.
5. **Speed and cost are low**: median 4.7 s and $0.01 per image, with the full set for under $2.

## Reference

Image2Net (hybrid detection + rules pipeline) reports 80.77 % success and 0.116 mean NED on **104** validation circuits. This repo version has 122, so the numbers are indicative, not directly comparable.

## Caveats and next steps

- **Fix NED reliability before comparing methods.** Options: an exact MIP-based GED or a better initial matching for the search. A longer timeout alone did not help.
- One run only. VLM output varies, so run 2–3 repeats to get a spread.
- `log.csv` has 14 extra rows from earlier failed attempts (the JPEG-as-PNG 400 errors, since fixed). The evaluation uses the last row per image.
