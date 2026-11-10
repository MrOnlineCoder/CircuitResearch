# Experiment 03: hybrid, YOLO + OpenCV corrected by an LLM

The experiment-02 pipeline (variant `v2`) extracts a netlist. Claude Sonnet 5.5 (the same model as in experiment 01) then checks it against the image and returns **edits**, not a new netlist. The edits are applied in code, and the result is scored with `experiments/evaluate.py`, like experiments 01 and 02.

## What the LLM gets (one request per image)

| Input | Content | Made by |
|---|---|---|
| Image 1 | Original schematic | dataset |
| Image 2 | Annotated overlay: component IDs (`M1`, `R2`, …), wires tinted by net with a name tag per net, pin letters at each lead, red `?` = lead not assigned to any pin, `J` / `X` = crossing treated as not connected / connected. Small images are enlarged (up to ×2, long side 1400 px) so the labels stay readable. | `prepare.py` |
| Text | The pipeline netlist (`M3 NMOS Drain=net2, Gate=net5, Source=GND`), crossing decisions, and the spots where the pipeline was unsure (list below) | `prepare.py` → `run.py` |

**Uncertain spots** (`data/context/<id>.json` → `uncertain`):
- pins left unconnected (`ncK`);
- pins attached only through a tiny blob that may be text;
- lines touching a component that were not assigned to any pin;
- detections with confidence < 0.5;
- op-amp inputs guessed (orientation classifier unsure);
- nets with a single pin and no port;
- faint drawings where the lenient wire threshold was used.

**Edit operations** (`edits.py`): `set_pin`, `swap_pins`, `merge_nets`, `change_type`, `add_component`, `remove_component`, `remove_pin`. Invalid edits are rejected one by one and logged; the rest are applied. With no edits the output is exactly the pipeline netlist.

## Run

```bash
cd experiments/03-hybrid
python3 prepare.py                 # once, ~1 min: runs the 02 pipeline (uses ../02-yolo-cv/.venv), writes data/
export OPENROUTER_API_KEY=...      # or set it at the top of run.py
python3 run.py                     # needs: pip install requests
python3 ../evaluate.py results/claude-sonnet-5.5-hybrid
```

Constants at the top of `run.py`: `MODEL`, `NUM_FILES`, `REASONING_EFFORT`, `SKIP_EXISTING` (resume after a stop), `FALLBACK_TO_PIPELINE`.

`data/` is already prepared for all 122 validation images. Rerun `prepare.py` only after changing the experiment-02 pipeline or models.

## Output

`results/claude-sonnet-5.5-hybrid/`:

| Path | Content |
|---|---|
| `preds/<id>.json` | final netlist (pipeline + accepted edits) |
| `edits/<id>.json` | every requested edit with `ok` / `error`; the `why` field holds the model's reason |
| `raw/<id>.txt` | raw reply |
| `debug/<id>/request.json` | full API request body; images replaced by a placeholder (type, size, path), no API key |
| `debug/<id>/response.json` | full API response; on an HTTP error, `http_status` plus the error body |
| `golden/`, `log.csv` | as in experiment 01. `seconds` = pipeline + LLM; extra columns `n_edits`, `n_rejected`, `pipeline_seconds`, `llm_seconds` |

If a request fails or the reply has no parsable edits, the pipeline netlist is kept and the status is `fallback (…)`. Set `FALLBACK_TO_PIPELINE = False` to score such images as empty, as experiment 01 does.

## What to report

Besides the usual metrics (`evaluate.py`), compare per image with `../02-yolo-cv/results/v2`:
- **fixed**: wrong in 02, exact here;
- **broken**: exact in 02 (36 images), not exact here;
- edits per image, share rejected, and whether NED went up or down per image.

Baselines: 02 alone 29.5 % exact; 01 (LLM alone) 41.8 %; best of both per image 52 %.

## Variations (not implemented)

- **Experiment 04, rich context without the overlay:** drop image 2 and keep the text context. Answers whether the annotated image helps.
- **Text-only ablation:** neither image, only the context. Shows how much comes from circuit knowledge alone.
