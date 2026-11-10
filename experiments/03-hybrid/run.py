"""
Experiment 03 - hybrid: YOLO + OpenCV pipeline (experiment 02), corrected by a vision LLM.

For each validation image the LLM gets
    1. the original schematic,
    2. the pipeline's annotated overlay (component IDs, wires coloured by net, net names,
       pin letters, crossing decisions),
    3. the pipeline's netlist as text, its crossing decisions and a list of uncertain spots,
and returns EDITS (not a new netlist). The edits are applied in code (edits.py) and the result
is written in CI2N format, so experiments/evaluate.py scores it like experiments 01 and 02.

    python3 prepare.py   # once: runs the pipeline, writes data/ (no API calls)
    python3 run.py       # this script: one OpenRouter request per image

Output (under RESULTS_DIR/<model-slug>/):
    preds/<id>.json     final netlist (pipeline netlist + accepted edits)
    edits/<id>.json     requested edits, each marked applied / rejected (with reason)
    raw/<id>.txt        raw model reply
    golden/<id>.json    golden copies for the images used
    log.csv             per image: status, seconds (pipeline + LLM), tokens, cost, edit counts
    debug/<id>/request.json   full API request body (images replaced by placeholders, no key)
    debug/<id>/response.json  full API response (or HTTP status + error body)
"""

# ============================ EDITABLE CONSTANTS ============================

OPENROUTER_API_KEY = ""  # leave empty to read the OPENROUTER_API_KEY env var
MODEL = "anthropic/claude-sonnet-5.5"  # same model as experiment 01
DATASET_DIR = "../../datasets/ci2n_datasets-main/validation"  # has images/ and golden/
DATA_DIR = "data"  # written by prepare.py
NUM_FILES = None  # how many images to process (sorted by name); None = all prepared

# Less commonly changed
RESULTS_DIR = "results"
MAX_TOKENS = 16000
REASONING_EFFORT = None  # e.g. "low" | "medium" | "high"; None = model default (as in exp. 01)
REQUEST_TIMEOUT_S = 600
SKIP_EXISTING = True  # don't re-query images that already have a model reply (failed ones are retried)
# If the request fails or the reply has no parsable edits, keep the pipeline's netlist
# (status "fallback"). False = no prediction, scored as an empty netlist like in exp. 01.
FALLBACK_TO_PIPELINE = True

# ============================================================================

import base64
import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import requests

from edits import DEVICE_PORTS, OPS_HELP, apply_edits, validate

HERE = Path(__file__).resolve().parent
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

PROMPT = """You are checking a netlist that an automatic computer-vision pipeline extracted from an \
analog circuit schematic. Your job is to find and fix its mistakes.

You are given two images:
  IMAGE 1: the original schematic. This is the ground truth - trust it over everything else.
  IMAGE 2: the same schematic annotated by the pipeline{scale_note}:
    - grey boxes with IDs (M1, R2, ...) = detected components;
    - letters next to black dots = the pin the pipeline assigned to that lead
      (MOSFET D/G/S/B, BJT C/B/E, op-amp +/-/out, o+/o- for differential outputs,
      sources +/- or in/out, two-terminal parts 1 = Pos, 2 = Neg);
    - wires are coloured by net, and each net has a name tag in its colour;
      the SAME colour means the pipeline thinks the wires are connected;
    - red "?" dot = a line touching a component that was not assigned to any pin;
    - red box "J" = crossing treated as NOT connected; green box "X" = crossing treated as connected.
  The annotation can be wrong. Check every suspicious spot against IMAGE 1.

Pipeline netlist (component ID, type, pin -> net). GND and VDD come from ground / supply
symbols; netK are the other nets; ncK means the pin was left unconnected:
{netlist}

Crossings:
{crossings}

Spots where the pipeline itself was unsure:
{uncertain}

Allowed component types and their pins:
{types}

Typical pipeline errors to look for:
- a wire attached to the wrong pin, or a MOSFET drain and source swapped;
- two different nets merged (text, arrows or labels touching wires join them) or one net split;
- a crossing decided wrongly (dot = connected; hop arc = not connected);
- a wrong component type (NMOS/PMOS/NPN/PNP, single-ended vs fully differential amplifier),
  a missed component, or a detection that is not a component;
- op-amp + and - inputs swapped.

Rules:
- Most of the netlist is probably correct. Only edit what IMAGE 1 shows to be wrong.
- Net names do not matter, only which pins share a net. Do not rename nets.
- Ground and supply symbols are not components.
- If everything is correct, return an empty list.

Return ONLY a JSON object (no markdown fences) of the form
{{"edits": [ ... ]}}
where each edit is one of the following (you may add a short "why" field to any edit):
{ops}
"""


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.split("/")[-1]) + "-hybrid"


def media_type(data: bytes) -> str:
    """Real image type from magic bytes (some CI2N '.png' files are JPEGs)."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError(f"unsupported image format (first bytes: {data[:8]!r})")


def image_part(path: Path) -> dict:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{media_type(data)};base64,{b64}"}}


def loggable(body: dict, images: list[Path]) -> dict:
    """Request body for request.json: each base64 image replaced by a short placeholder."""
    body = json.loads(json.dumps(body))
    paths = iter(images)
    for msg in body["messages"]:
        for part in msg["content"]:
            if part.get("type") == "image_url":
                url, path = part["image_url"]["url"], next(paths)
                part["image_url"]["url"] = (f"<{url.split(';', 1)[0][5:]} image, {path.stat().st_size} bytes: "
                                            f"{path}>")
    return body


def build_prompt(ctx: dict) -> str:
    lines = []
    for c in ctx["components"]:
        pins = ", ".join(f"{p}={n}" for p, n in c["pins"].items())
        lines.append(f"  {c['id']:<4} {c['type']:<9} {pins}")
    crossings = [f"  {x['id']}: {x['kind']} -> {x['decision']}" for x in ctx["crossings"]]
    scale = ctx.get("overlay_scale", 1.0)
    return PROMPT.format(
        scale_note=f" (enlarged x{scale:.2f})" if abs(scale - 1.0) > 0.01 else "",
        netlist="\n".join(lines) or "  (no components detected)",
        crossings="\n".join(crossings) or "  (none detected)",
        uncertain="\n".join(f"  - {u}" for u in ctx["uncertain"]) or "  (none)",
        types="\n".join(f"  {t}: {', '.join(p)}" + ("  (Body optional)" if t in ("NMOS", "PMOS") else "")
                        for t, p in DEVICE_PORTS.items()),
        ops=OPS_HELP,
    )


def extract_edits(text: str):
    """The LAST top-level JSON object with an "edits" list (handles self-corrections and prose)."""
    decoder, found, i = json.JSONDecoder(), [], text.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict) and isinstance(obj.get("edits"), list):
            found.append(obj)
        i = text.find("{", end)
    if not found:
        raise json.JSONDecodeError('no JSON object with an "edits" list in reply', text, 0)
    return found[-1]["edits"]


def query(image: Path, overlay: Path, prompt: str, api_key: str, debug_dir: Path) -> dict:
    """One OpenRouter request. Writes debug_dir/request.json (images as placeholders, no API
    key) and debug_dir/response.json (the parsed reply, or status + raw text on an error)."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "IMAGE 1 (original schematic):"},
            image_part(image),
            {"type": "text", "text": "IMAGE 2 (pipeline annotation):"},
            image_part(overlay),
            {"type": "text", "text": prompt},
        ]}],
        "usage": {"include": True},  # ask OpenRouter to report cost
    }
    if REASONING_EFFORT:
        body["reasoning"] = {"effort": REASONING_EFFORT}
    (debug_dir / "request.json").write_text(json.dumps(loggable(body, [image, overlay]), indent=2))
    (debug_dir / "response.json").unlink(missing_ok=True)  # no stale reply from an earlier run
    resp = requests.post(OPENROUTER_URL, json=body, timeout=REQUEST_TIMEOUT_S,
                         headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        logged = resp.json()
    except ValueError:  # not JSON (e.g. an HTML error page)
        logged = {"http_status": resp.status_code, "raw_text": resp.text}
    if not resp.ok and "http_status" not in logged:
        logged = {"http_status": resp.status_code, "body": logged}
    (debug_dir / "response.json").write_text(json.dumps(logged, indent=2, ensure_ascii=False))
    if not resp.ok:
        raise requests.HTTPError(f"{resp.status_code}: {resp.text[:500]}", response=resp)
    return resp.json()


def main():
    api_key = OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit("Set OPENROUTER_API_KEY at the top of run.py or in the environment.")
    dataset, data = (HERE / DATASET_DIR).resolve(), HERE / DATA_DIR
    contexts = sorted((data / "context").glob("*.json"))[:NUM_FILES]
    if not contexts:
        sys.exit(f"No prepared data in {data}. Run prepare.py first.")

    out = HERE / RESULTS_DIR / slug(MODEL)
    for sub in ("preds", "edits", "raw", "golden"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    log_path = out / "log.csv"
    new_log = not log_path.exists()
    log_file = log_path.open("a", newline="")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["id", "model", "status", "seconds", "prompt_tokens", "completion_tokens", "cost_usd",
                      "n_components", "n_edits", "n_rejected", "pipeline_seconds", "llm_seconds", "problems"])

    print(f"Model: {MODEL}\nImages: {len(contexts)}\nOutput: {out}\n")
    for n, ctx_path in enumerate(contexts, 1):
        ctx = json.loads(ctx_path.read_text())
        stem = ctx["id"]
        golden = dataset / "golden" / f"{stem}.json"
        if golden.exists():
            shutil.copy(golden, out / "golden" / golden.name)
        pred_path = out / "preds" / f"{stem}.json"
        raw_path = out / "raw" / f"{stem}.txt"
        # skip only images with a real reply; fallbacks (failed requests) are retried
        if SKIP_EXISTING and pred_path.exists() and raw_path.exists() and raw_path.read_text().strip():
            print(f"[{n}/{len(contexts)}] {stem}: exists, skipped")
            continue

        status, problems, usage, text, edit_log, netlist = "ok", [], {}, "", [], None
        t0 = time.time()
        try:
            reply = query(dataset / "images" / f"{stem}.png", data / "overlays" / f"{stem}.png",
                          build_prompt(ctx), api_key, out / "debug" / stem)
            usage = reply.get("usage") or {}
            text = reply["choices"][0]["message"].get("content") or ""
            netlist, edit_log = apply_edits(ctx["components"], extract_edits(text))
            problems = validate(netlist)
            if problems:
                status = "schema_warning"
        except json.JSONDecodeError as e:
            status, problems = "invalid_json", [str(e)]
        except (requests.RequestException, KeyError, IndexError, ValueError) as e:
            status, problems = "request_error", [str(e)]
        llm_seconds = time.time() - t0
        if netlist is None and FALLBACK_TO_PIPELINE:
            netlist, status = ctx["netlist"], f"fallback ({status})"
        if netlist is not None:
            pred_path.write_text(json.dumps(netlist, indent=4))

        raw_path.write_text(text)
        (out / "edits" / f"{stem}.json").write_text(json.dumps(edit_log, indent=2))
        n_rej = sum(not e["ok"] for e in edit_log)
        seconds = ctx["pipeline_seconds"] + llm_seconds
        n_comp = len(netlist["ckt_netlist"]) if netlist else 0
        log.writerow([stem, MODEL, status, f"{seconds:.2f}", usage.get("prompt_tokens"),
                      usage.get("completion_tokens"), usage.get("cost"), n_comp, len(edit_log), n_rej,
                      ctx["pipeline_seconds"], f"{llm_seconds:.2f}", " | ".join(problems)])
        log_file.flush()
        print(f"[{n}/{len(contexts)}] {stem}: {status} {seconds:.1f}s, {len(edit_log)} edits"
              + (f" ({n_rej} rejected)" if n_rej else "") + (f" - {problems[0]}" if problems else ""))

    log_file.close()
    print(f"\nDone. Score with:\n  python3 {HERE.parent / 'evaluate.py'} {out}")


if __name__ == "__main__":
    main()
