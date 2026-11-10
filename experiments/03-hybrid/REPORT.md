# Experiment 03: hybrid, YOLO + OpenCV corrected by Claude Sonnet 5.5

**Date:** 2026-11-10 · **Data:** CI2N `validation/` (122 image–netlist pairs) · **Raw metrics:** [`results/claude-sonnet-5.5-hybrid/metrics.json`](results/claude-sonnet-5.5-hybrid/metrics.json)

## Setup

| | |
|---|---|
| Stage 1 | Experiment-02 pipeline, variant `v2` (YOLO11 detectors + OpenCV wire tracing + orientation classifiers), run by `prepare.py` |
| Stage 2 | `anthropic/claude-sonnet-5.5` via OpenRouter (same model and settings as experiment 01: default reasoning, `max_tokens` 16000), one request per image |
| LLM input | Original image + annotated overlay (component IDs, wires coloured by net with net-name tags, pin letters, crossing decisions) + the pipeline netlist as text + the pipeline's uncertain spots (see [README.md](README.md)) |
| LLM output | A list of **edits** (`set_pin`, `merge_nets`, `change_type`, …), applied in code (`edits.py`). No edits = pipeline netlist unchanged. |
| Scoring | `experiments/evaluate.py`, same settings as experiments 01 and 02 |
| Avg. tokens per image | 4,701 in / 2,167 out |

All 122 requests were answered; no edit was rejected as invalid, and no fallback to the pipeline netlist was needed. (A first run hit OpenRouter's credit limit on 45 images; those were deleted and re-run. Image 069 was re-applied after fixing a bug in `edits.py`, see Caveats.)

## Results

| Metric | **Hybrid (03)** | YOLO + CV (02, v2) | LLM only (01) |
|---|---|---|---|
| **Success rate** (GED = 0, exact netlist) | **62.3 %** (76 / 122) | 29.5 % (36) | 41.8 % (51) |
| **NED, mean / median** (best known GED*) | **0.065 / 0.000** | 0.218 / 0.199 | 0.137 / 0.110 |
| NED lower bound, mean / median | 0.036 / 0.000 | 0.119 / 0.078 | 0.079 / 0.055 |
| Images with exact GED | 87 / 122 | 55 | 65 |
| **Component-count F1** (micro) | **0.984** | 0.984 | 0.954 |
| Component-count F1 (macro over images) | 0.986 | 0.981 | 0.959 |
| Seconds per image, mean / median | 19.6 s / 11.0 s (max 158 s) | 0.34 s / 0.25 s | 9.2 s / 4.7 s |
| USD per image, mean / median | $0.031 / $0.021 | $0 | $0.016 / $0.010 |
| Total cost / cost per correct netlist | $3.79 / $0.050 | – | $1.95 / $0.038 |

\* GED is exact for 87 images (76 identical, 5 search completed, 6 bounds met). For the other 35 it is a proven range, so **the true mean NED is between 0.036 and 0.065**. Unlike the 01-vs-02 comparison, the ranges no longer overlap: the hybrid's upper bound (0.065) is below the LLM-only lower bound (0.079).

### By circuit size (golden device count)

| Size | Images | Success: 03 / 02 / 01 | NED mean [lower bound]: 03 | 02 | 01 |
|---|---|---|---|---|---|
| Small (≤ 8) | 31 | **87.1 %** / 54.8 % / 71.0 % | 0.016 [0.016] | 0.115 [0.105] | 0.059 [0.055] |
| Medium (9–15) | 61 | **65.6 %** / 31.1 % / 39.3 % | 0.057 [0.031] | 0.164 [0.083] | 0.129 [0.075] |
| Large (> 15) | 30 | **30.0 %** / 0.0 % / 16.7 % | 0.134 [0.065] | 0.434 [0.205] | 0.233 [0.111] |

### Per image

| Against experiment 02 (the netlist it corrects) | Images |
|---|---|
| Wrong in 02, **fixed** (exact in 03) | 45 |
| Exact in 02, **broken** by an edit | 5 |
| Lower / higher NED than 02 | 76 / 11 (35 equal) |

| Against experiment 01 (LLM alone) | Images |
|---|---|
| Exact in both | 47 |
| Exact only in the hybrid | 29 |
| Exact only with the LLM alone | 4 |
| Lower / higher NED than 01 | 56 / 11 |

21 circuits are exact **only** in the hybrid, wrong in both 01 and 02. The hybrid (76) also beats the best-of-both upper bound of 01 and 02 together (64).

### Edits

| | |
|---|---|
| Edits requested | 825 (mean 6.8, median 3, max 40 per image); 0 rejected as invalid |
| Images with no edits | 34, of which 31 were already exact in 02 and stayed exact |
| By operation | `set_pin` 727, `merge_nets` 51, `change_type` 20, `remove_pin` 18, `remove_component` 5, `swap_pins` 3, `add_component` 1 |
| Type changes | to NMOS 8, PMOS 5, PNP 4, Voltage 2, NPN 1 |

## Observations

1. **The hybrid is clearly best on every measure except speed and cost.** Exact netlists go from 29.5 % (02) and 41.8 % (01) to 62.3 %, and mean NED falls to a third of 02's and half of 01's. The gain holds at every circuit size, and large circuits go from 0 % (02) to 30 %.
2. **The LLM is much better at fixing a netlist than at writing one.** Given the same image, the same model writes 51 exact netlists from scratch but reaches 76 by correcting the pipeline's. Component counting stays at the pipeline's level (F1 0.984 against 0.954 for the LLM alone): the LLM inherits the detector's component list and spends its effort on connections. 97 % of all edits are wiring edits (`set_pin`, `merge_nets`, `swap_pins`).
3. **It rarely breaks what is right.** On the 36 netlists already exact in 02 it made no edits in 31 cases and broke 5, each with one or two edits:
   - 2 by merging nets that share a text label ("latch" in 046, a VSS arrow in 085); the golden netlist keeps them separate;
   - 3 by changing a transistor type that was already right (078, 093 to NMOS; 110 to PNP).
4. **Some systematic pipeline errors stay.** Fully differential amplifiers are still reported as single-ended (Dido_amp recall 0.14, as in 02): the LLM did not change any of them, although the prompt names this error. Detection mistakes that need a different pin structure are harder to fix with edits than wiring mistakes.
5. **Faint drawings are fixed too.** Image 050 (light-grey wires, worst in 02 at NED 1.27 before the threshold fallback) is now exact: the LLM separated the wrongly merged N and M nodes ("the vertical N wire crosses the M wire with no junction dot") and moved the op-amp input.
6. **Cost:** twice experiment 01 per image ($0.031 vs $0.016) and slower (median 11 s vs 4.7 s) because of two images, a longer prompt and longer replies. Per correct netlist the difference is smaller ($0.050 vs $0.038). The YOLO stage adds only 0.3 s.

## Reference

Image2Net reports 80.77 % success and 0.116 mean NED on 104 validation circuits. Our 122-circuit set differs, so the numbers are only indicative. The hybrid's mean NED (0.036–0.065) is below Image2Net's reported 0.116, while its success rate is lower.

## Caveats and next steps

- **Tuned on the test set.** The experiment-02 rules and the light-grey threshold fallback were developed on the validation images (partly via the 46 duplicated circuits). The prompt lists generic pipeline errors only, but the pipeline under it was not developed on held-out data. A separate test split would settle this.
- **One run.** LLM output varies; 2–3 repeats would give a spread, particularly for the 5 broken netlists.
- **Bug fixed during the run:** `edits.py` dropped all pins when a type changed within the BJT or MOSFET family (NPN → PNP). It hit one image (069); its saved edits were re-applied with the fixed code (no new API call) before scoring.
- **35 GED values are ranges,** mostly on large circuits.
- **Next:** experiment 04 (same text context without the overlay image) shows how much the annotated image contributes; a text-only ablation shows how much comes from circuit knowledge alone.
