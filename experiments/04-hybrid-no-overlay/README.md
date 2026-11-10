# Experiment 04: hybrid without the overlay image

Same as [experiment 03](../03-hybrid/README.md), with one difference: the LLM gets **only the original image and text**, not the annotated overlay. Comparing 03 and 04 shows whether the annotated image helps the LLM beyond the same information given as text.

The experiment-02 pipeline (variant `v2`) extracts a netlist, and Claude Sonnet 5.5 returns **edits** to it, applied in code (`edits.py`, identical to 03). The pipeline netlists are identical to experiment 03's for all 122 images.

## What the LLM gets (one request per image)

| Input | Content |
|---|---|
| Image | Original schematic only |
| Text | Image size, and the pipeline netlist with each component's bounding box and each pin's lead position in pixels, e.g. `M1 NMOS [218, 373, 278, 440]  Drain=net2@(272, 370), Gate=net3@(216, 406), …`. Crossings with their box and decision. The same uncertain spots as in 03. |

**Differences from 03:**
- No image 2 (overlay). The prompt's description of the overlay is replaced by an explanation of the coordinates.
- Component boxes and pin lead positions are given as pixel coordinates. They do the overlay's job of showing which component is `M3` and which lead is its drain.
- Crossings include their box, so `J1` can be found in the image.
- "Extra line" notes give the position of the unassigned line instead of referring to red `?` marks.

Everything else is unchanged: model, settings, edit operations, fallback, logging, `SKIP_EXISTING`.

## Run

```bash
cd experiments/04-hybrid-no-overlay
python3 prepare.py                 # once, ~1 min: runs the 02 pipeline, writes data/context (no API calls)
export OPENROUTER_API_KEY=...      # or set it at the top of run.py
python3 run.py
python3 ../evaluate.py results/claude-sonnet-5.5-hybrid-no-overlay
```

`data/` is already prepared for all 122 validation images (1.3 MB, no images).

## Output

`results/claude-sonnet-5.5-hybrid-no-overlay/`:

| Path | Content |
|---|---|
| `preds/<id>.json` | final netlist (pipeline + accepted edits) |
| `edits/<id>.json` | every requested edit with `ok` / `error`; `why` holds the model's reason |
| `raw/<id>.txt` | raw reply |
| `debug/<id>/request.json` | full API request body; image replaced by a placeholder (type, size, path), no API key |
| `debug/<id>/response.json` | full API response; on an HTTP error, `http_status` plus the error body |
| `golden/`, `log.csv` | as in experiment 03 |

## What to compare with 03

- Success rate, NED, component F1 (`evaluate.py`).
- Per image: fixed / broken relative to 02 (03 fixed 45, broke 5).
- Edits per image and the edit types used.
- Cost and time: one image instead of two, so it should be cheaper and faster.
