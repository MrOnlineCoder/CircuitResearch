# Circuit diagram images to netlists

University (PhD) research on converting images of analog circuit schematics into a machine-readable, structured format: a netlist that lists every component and the net each of its pins is connected to.

## Sources

| Source | What it is | Used for |
|---|---|---|
| **Image2Net** (Xu et al., 2025, [arXiv:2508.13157](https://arxiv.org/abs/2508.13157)) | CI2N dataset, GED/NED benchmark, and a hybrid framework: detector + wire tracing + rules | Dataset, metrics, and the design of experiment 02. **No source code for the framework is published**, so experiment 02 is our own reimplementation. |
| **OmniSch** (Lu et al., 2026, [arXiv:2604.00270](https://arxiv.org/abs/2604.00270)) | Multimodal benchmark for PCB schematics and visual reasoning | Background: shows that large multimodal models are weak at schematic connectivity. Different task (PCB) and metrics, so its numbers are not compared directly. |

Two literature surveys (2024–2026) are in `sources/`, with text extracts in `sources/txt/`.

## Dataset

**CI2N**: [github.com/LAD021/ci2n_datasets](https://github.com/LAD021/ci2n_datasets), unpacked in `datasets/ci2n_datasets-main/`.

- `validation/`: 122 schematic images with golden JSON netlists, plus the official GED scorer. All experiments are scored on this set.
- `device_identification/`, `jumper_identification/`, `device_orientation/`: training data for the YOLO models in experiment 02.
- Data issues found:
  - 21 validation "PNG" files are really JPEGs.
  - 49 validation circuits have copies in the training folders. These copies are removed before training (`02-yolo-cv/prepare_data.py`, `data/leakage.json`).

## Methodology

Four independent experiments. Each writes predictions in the CI2N JSON format and is scored by the same script, `experiments/evaluate.py`.

| # | Experiment | Method |
|---|---|---|
| 01 | [`01-vllm-only`](experiments/01-vllm-only) | **LLM only.** Claude Sonnet 5.5 (via OpenRouter) gets the image and returns the netlist zero-shot. |
| 02 | [`02-yolo-cv`](experiments/02-yolo-cv) | **YOLO + OpenCV, no LLM.** Several models and image processing steps: <br>• A YOLO11s detector finds components, ground/supply symbols and ports. <br>• A YOLO11s-OBB model finds wire crossings and junctions. <br>• OpenCV connected components trace the wires (one region = one net). <br>• 7 YOLO11n classifiers detect component orientation, which assigns the pin names. |
| 03 | [`03-hybrid`](experiments/03-hybrid) | **Hybrid.** The 02 pipeline extracts a netlist. Claude Sonnet 5.5 gets the original image, an annotated overlay (component IDs, nets coloured, pins labelled) and the netlist as text. It returns **edits** (`set_pin`, `merge_nets`, `change_type`, …), which are applied in code. |
| 04 | [`04-hybrid-no-overlay`](experiments/04-hybrid-no-overlay) | **Hybrid without overlay.** Same as 03, but without the overlay. Component boxes and pin positions are given as pixel coordinates instead. |

**Metrics:**
- **Success rate:** share of images whose netlist is exactly right (graph edit distance, GED = 0).
- **NED:** GED divided by the size of the golden graph; lower is better. Exact GED is too slow on large circuits, so some images get a proven range instead, and the mean is reported as `[lower bound] – upper bound`.
- **Component-count F1:** how well component types and counts match, ignoring wiring.
- **Speed and cost** per image.

## Results

CI2N validation set, 122 circuits, one run per experiment.

| Metric | 01 LLM only | 02 YOLO + CV | 03 Hybrid | 04 Hybrid, no overlay |
|---|---|---|---|---|
| **Success rate** (exact netlist) | 41.8 % (51) | 29.5 % (36) | **62.3 %** (76) | 57.4 % (70) |
| **NED mean**, range [lower – upper] | 0.079 – 0.137 | 0.119 – 0.218 | **0.036 – 0.065** | 0.039 – 0.080 |
| NED median (upper bound) | 0.110 | 0.199 | **0.000** | **0.000** |
| **Component-count F1** (micro) | 0.954 | 0.984 | 0.984 | **0.989** |
| Seconds per image, mean / median | 9.2 / 4.7 | **0.34 / 0.25** | 19.6 / 11.0 | 19.8 / 13.3 |
| USD per image, mean | $0.016 | **$0** | $0.031 | $0.030 |
| Total cost (122 images) | $1.95 | $0 | $3.79 | $3.70 |

**Success rate by circuit size** (number of devices in the golden netlist):

| Size | Images | 01 | 02 | 03 | 04 |
|---|---|---|---|---|---|
| Small (≤ 8) | 31 | 71.0 % | 54.8 % | **87.1 %** | 83.9 % |
| Medium (9–15) | 61 | 39.3 % | 31.1 % | **65.6 %** | 60.7 % |
| Large (> 15) | 30 | 16.7 % | 0.0 % | **30.0 %** | 23.3 % |

**02 detector training** (held-out split): components mAP50 0.893 / mAP50-95 0.731; crossings mAP50 0.975 / mAP50-95 0.824; orientation classifiers 98.6–100 % top-1.

Full reports, with per-image analysis and error types: [01](experiments/01-vllm-only/REPORT.md) · [02](experiments/02-yolo-cv/REPORT.md) · [03](experiments/03-hybrid/REPORT.md) · [04](experiments/04-hybrid-no-overlay/REPORT.md).

**For reference**, Image2Net reports 80.77 % success and 0.116 mean NED on an earlier 104-circuit version of the validation set. That is indicative only, not directly comparable.

## Summary

- **Components are easy and connections are hard.** Every method reaches component F1 ≥ 0.95, but exact netlists stay well below that. Most errors are in the wiring.
- **The LLM alone** gets the connections right more often than YOLO + CV (41.8 % vs 29.5 %) but is slower and paid.
- **YOLO + CV** is the best detector, about 20× faster, and free to run. But its rule-based wiring fails on large circuits (0 % exact).
- **The hybrid is best by a wide margin** (62.3 % exact; mean NED 0.036–0.065). This agrees with Image2Net and the surveys: detection plus explicit wire tracing, with an LLM to correct the result.
  - The LLM is much better at correcting a netlist than at writing one from scratch.
  - The annotated overlay adds a little (+4.9 points, not statistically significant in one run).
- **Caveats:**
  - Single runs.
  - The 02 rules were developed on the same validation images.
  - Some NED values are ranges.

**Further research:**
- **Optimisation:** a cheaper or local VLM, fewer tokens, faster runs.
- **Accuracy:**
  - Repeated runs and a held-out test split.
  - OCR for net labels.
  - Better detection of fully differential amplifiers and other rare classes.
- **Better wiring algorithms:** learned connectivity instead of rules, e.g. graph models or transformers that predict pin-to-net links (Netlistify, AMSnet 2.0-style wire segmentation, GNNs).
- **Semantic reasoning:**
  - Use circuit knowledge (circuit type, typical topologies, device roles) to check and repair netlists.
  - Export to SPICE (`experiments/ci2n_to_spice.py`) for simulation-based validation.

## How to run

**Prerequisites:**
- Python 3.12.
- [uv](https://docs.astral.sh/uv/) (for the dataset's GED scorer).
- An [OpenRouter](https://openrouter.ai) API key (experiments 01, 03, 04).
- A CUDA GPU for training only. We used a Runpod RTX 4090 for about 2–3 hours. Inference runs on a Mac CPU or MPS.

**1. Dataset and scorer**

```bash
cd datasets
wget -O ci2n.zip https://github.com/LAD021/ci2n_datasets/archive/refs/heads/main.zip && unzip -q ci2n.zip
cd ci2n_datasets-main/validation && uv sync     # environment used by experiments/evaluate.py
```

**2. Train the YOLO models (experiment 02)**

The full guide for a Runpod pod is in [TRAINING.md](experiments/02-yolo-cv/TRAINING.md). In short:

```bash
cd experiments/02-yolo-cv
pip install -r requirements-train.txt
python prepare_data.py --ci2n <path>/ci2n_datasets-main   # YOLO datasets; removes duplicated validation circuits
python train.py                                             # -> models/devices.pt, jumpers.pt, orient_*.pt
```

Trained models are already included in `experiments/02-yolo-cv/models/`.

**3. Run and evaluate each experiment**

```bash
export OPENROUTER_API_KEY=...

# 01 - LLM only
cd experiments/01-vllm-only && pip install requests && python3 run.py
python3 ../evaluate.py results/claude-sonnet-5.5

# 02 - YOLO + OpenCV (set VARIANT at the top of run.py; default v2)
cd ../02-yolo-cv && python3 -m venv .venv && .venv/bin/pip install -r requirements-infer.txt
.venv/bin/python run.py
python3 ../evaluate.py results/v2
.venv/bin/python infer.py image.png -o netlist.json --debug overlay.png   # single image

# 03 - hybrid with overlay (prepare.py uses the 02 .venv automatically)
cd ../03-hybrid && python3 prepare.py && python3 run.py
python3 ../evaluate.py results/claude-sonnet-5.5-hybrid

# 04 - hybrid without overlay
cd ../04-hybrid-no-overlay && python3 prepare.py && python3 run.py
python3 ../evaluate.py results/claude-sonnet-5.5-hybrid-no-overlay
```

Each script has its settings (model, number of images, variant, …) as constants at the top. `evaluate.py` writes `metrics.json` into the results folder. To convert any netlist to SPICE, run `python3 experiments/ci2n_to_spice.py <preds dir> -o <out dir>`.

## Credits

- **Image2Net / CI2N:** H. Xu et al., *Image2Net: Datasets, Benchmark and Hybrid Framework to Convert Analog Circuit Diagrams into Netlists*, [arXiv:2508.13157](https://arxiv.org/abs/2508.13157). Dataset and GED scorer: [LAD021/ci2n_datasets](https://github.com/LAD021/ci2n_datasets).
- **OmniSch:** T. Lu et al., *OmniSch: A Multimodal PCB Schematic Benchmark For Structured Diagram Visual Reasoning*, [arXiv:2604.00270](https://arxiv.org/abs/2604.00270).
- **Models and libraries:**
  - [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics).
  - [OpenCV](https://opencv.org).
  - [NetworkX](https://networkx.org) (GED).
  - Claude Sonnet 5.5 (Anthropic) via [OpenRouter](https://openrouter.ai).
- **AI assistance:** [Claude Code](https://claude.com/claude-code) (Anthropic) was used to generate the experiment code, to analyse the source papers, and to draft the reports. All results were produced by running that code and are reported as measured.
