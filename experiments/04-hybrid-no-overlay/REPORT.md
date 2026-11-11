# Experiment 04: hybrid without the overlay image

**Date:** 2026-11-10 · **Data:** CI2N `validation/` (122 image–netlist pairs) · **Raw metrics:** [`results/claude-sonnet-5.5-hybrid-no-overlay/metrics.json`](results/claude-sonnet-5.5-hybrid-no-overlay/metrics.json)

## Question

Experiment 03 gave the LLM the original image, an annotated overlay and the pipeline's data as text. This experiment drops the overlay and sends the **same information as text only**, to measure what the annotated image contributes.

## Setup

| | |
|---|---|
| Stage 1 | Experiment-02 pipeline (`v2`). Its netlists are identical to experiment 03's for all 122 images. |
| Stage 2 | `anthropic/claude-sonnet-5.5` via OpenRouter, same settings as 01 and 03 (default reasoning, `max_tokens` 16000) |
| LLM input | Original image only + text: image size, pipeline netlist with each component's box and each pin's lead position in pixels, crossings with their boxes, the same uncertain spots as 03 |
| LLM output | Edits, applied by the same `edits.py` as 03 |
| Avg. tokens per image | 3,162 in / 2,397 out (03: 4,701 / 2,167) |

**Run notes.** 5 requests first failed for lack of OpenRouter credit and 1 reply came back without an edit list; all 6 succeeded on retry. **Image 039 failed for good:** the model used its whole 16,000-token output limit on reasoning (14,810 reasoning tokens) and was cut off before writing any edits, so the pipeline netlist was kept (status `fallback (invalid_json)`). It would not have been exact anyway: 03 got NED 0.198 on it.

## Results

| Metric | **04: no overlay** | 03: with overlay | 02: YOLO + CV | 01: LLM only |
|---|---|---|---|---|
| **Success rate** (exact netlist) | **57.4 %** (70) | 62.3 % (76) | 29.5 % (36) | 41.8 % (51) |
| **NED, mean / median** (best known GED*) | **0.080 / 0.000** | 0.065 / 0.000 | 0.218 / 0.199 | 0.137 / 0.110 |
| NED lower bound, mean | 0.039 | 0.036 | 0.119 | 0.079 |
| Images with exact GED | 85 | 87 | 55 | 65 |
| **Component-count F1** (micro) | **0.989** | 0.984 | 0.984 | 0.954 |
| Seconds per image, mean / median | 19.8 s / 13.3 s | 19.6 s / 11.0 s | 0.34 s / 0.25 s | 9.2 s / 4.7 s |
| USD per image, mean / median | $0.030 / $0.020 | $0.031 / $0.021 | $0 | $0.016 / $0.010 |
| Total cost / per correct netlist | $3.70 / $0.053 | $3.79 / $0.050 | – | $1.95 / $0.038 |

\* As in the other experiments, NED is exact for some images and a proven range for the rest; **the true mean NED for 04 is between 0.039 and 0.080** (03: 0.036–0.065). Total cost counts the final request per image; including failed and retried requests, the run cost $3.86.

### By circuit size (golden device count)

| Size | Images | Success: 04 / 03 / 02 / 01 | NED mean [lower bound]: 04 | 03 |
|---|---|---|---|---|
| Small (≤ 8) | 31 | 83.9 % / 87.1 % / 54.8 % / 71.0 % | 0.021 [0.021] | 0.016 [0.016] |
| Medium (9–15) | 61 | 60.7 % / 65.6 % / 31.1 % / 39.3 % | 0.065 [0.034] | 0.057 [0.031] |
| Large (> 15) | 30 | 23.3 % / 30.0 % / 0.0 % / 16.7 % | 0.169 [0.069] | 0.134 [0.065] |

### Per image

| Against 02 (the netlist it corrects) | 04 | 03 |
|---|---|---|
| Wrong in 02, fixed | 39 | 45 |
| Exact in 02, broken | 5 | 5 |
| Lower / higher NED than 02 | 75 / 11 | 76 / 11 |

| 04 against 03 | Images |
|---|---|
| Exact in both | 65 |
| Exact only in 04 | 5 (036, 078, 094, 110, 127) |
| Exact only in 03 | 11 |
| Lower / higher NED in 04 | 14 / 26 |

### Edits

| | 04 | 03 |
|---|---|---|
| Edits requested | 795 (mean 6.5, median 3, max 42) | 825 |
| Rejected as invalid | 0 | 0 |
| Images with no edits | 32 (29 already exact in 02, all still exact) | 34 |
| By operation | `set_pin` 702, `merge_nets` 46, `change_type` 24, `remove_pin` 12, `swap_pins` 5, `remove_component` 5, `add_component` 1 | `set_pin` 727, `merge_nets` 51, `change_type` 20, … |

## Observations

1. **The overlay helps, but only a little.** Without it the hybrid loses 6 exact netlists (62.3 % → 57.4 %) and mean NED rises from 0.065 to 0.080. Large circuits lose most (30.0 % → 23.3 %). Images where only one of the two is exact go 11 to 5 in favour of the overlay. That difference is **not statistically significant** for one run (exact two-sided binomial test on the 16 differing images, p ≈ 0.21). The direction is consistent, though: higher NED in 26 images, lower in 14.
2. **Most of the hybrid's gain does not need the overlay.** Text with coordinates still takes the pipeline from 29.5 % to 57.4 % exact, against 62.3 % with the overlay. The LLM can match `M3 … Drain=net2@(272, 370)` to the drawing from coordinates alone.
3. **Without the overlay the LLM checks component types more carefully.** Component F1 is the best of all four experiments (0.989). It fixed 4 of the 6 fully differential amplifiers the pipeline reports as single-ended (Dido_amp recall 0.14 → 0.71, images 104 and 105), which 03 never did. One explanation: without an overlay that already labels every box, the model looks at the symbols itself more.
4. **Mistakes on correct netlists are the same kind as in 03.** Of the 5 broken netlists, 3 are the same images as in 03 (046, 085, 093) and 2 are new (056, 115):
   - merging nets that share a text label ("latch", V_SS, VDD arrows): 046, 085, 115. The golden netlists keep them separate, so this looks like a labelling convention the LLM does not know;
   - one wrong type change (093, as in 03) and one wrong net split (056).
5. **Reasoning can run out of budget.** Image 039 used the full 16,000 output tokens on reasoning and returned no edits. A larger `max_tokens`, or a lower `REASONING_EFFORT`, would avoid this; it did not change the success rate here.
6. **Cost and time are about the same as 03.** One image fewer cuts input tokens by a third (4,701 → 3,162), but the replies are longer (2,167 → 2,397 output tokens; output costs more per token). Median time is 2.3 s higher.

## Conclusion for the comparison

| | Exact | Mean NED (range) | Cost / image |
|---|---|---|---|
| 01 LLM only | 41.8 % | 0.079–0.137 | $0.016 |
| 02 YOLO + OpenCV | 29.5 % | 0.119–0.218 | $0 |
| 04 hybrid, text context | 57.4 % | 0.039–0.080 | $0.030 |
| 03 hybrid, text + overlay | **62.3 %** | **0.036–0.065** | $0.031 |

Giving the LLM the pipeline's result is what matters most (+15.6 points over the LLM alone). Adding the annotated image on top gives a further, smaller gain (+4.9 points) at no extra cost, mainly on large circuits, while the text-only version is better at correcting component types.

## Caveats

- **One run per experiment.** The 03-vs-04 difference is within what run-to-run variation could produce; 2–3 repeats of both would settle it.
- **Same test-set caveat as 03:** the experiment-02 pipeline was developed on these validation images.
- **37 GED values are ranges,** mostly for large circuits.
- **The prompts differ slightly** beyond the overlay itself (coordinates instead of overlay labels), as described in [README.md](README.md).
