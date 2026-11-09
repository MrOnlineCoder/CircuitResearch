"""
Experiment 01 - VLM only.

Sends each schematic image from the CI2N validation set to a vision LLM via
OpenRouter and asks for a netlist in the CI2N golden JSON format. Predictions
are written so they can be scored directly with the dataset's calc_ged.py.

Usage:
    python run.py

Output (under RESULTS_DIR/<model-slug>/):
    preds/<id>.json     parsed netlist, same name as golden/<id>.json
    raw/<id>.txt        raw model reply (for debugging failures)
    golden/<id>.json    copy of the golden files used (pass this as --golden_dir)
    log.csv             per-image status, latency, token usage, cost
"""

# ============================ EDITABLE CONSTANTS ============================

OPENROUTER_API_KEY = "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"  # leave empty to read the OPENROUTER_API_KEY env var
MODEL = "anthropic/claude-sonnet-5.5"  # any OpenRouter model id with image input
DATASET_DIR = "../../datasets/ci2n_datasets-main/validation"  # has images/ and golden/
NUM_FILES = None  # how many images to process (sorted by name); None = all 122

# Less commonly changed
RESULTS_DIR = "results"
MAX_TOKENS = 16000
REASONING_EFFORT = None  # e.g. "low" | "medium" | "high"; None = model default
REQUEST_TIMEOUT_S = 600
SKIP_EXISTING = True  # don't re-query images that already have a prediction
REPARSE_ONLY = False  # True = no API calls; rebuild preds/ from saved raw/ replies that failed to parse

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

HERE = Path(__file__).resolve().parent
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Mirrors [device_ports] in validation/device_def.toml (the GED tool's schema).
DEVICE_PORTS = {
    "NMOS": ["Drain", "Source", "Gate", "Body"],
    "PMOS": ["Drain", "Source", "Gate", "Body"],
    "NPN": ["Base", "Emitter", "Collector"],
    "PNP": ["Base", "Emitter", "Collector"],
    "Res": ["Pos", "Neg"],
    "Cap": ["Pos", "Neg"],
    "Ind": ["Pos", "Neg"],
    "Diode": ["In", "Out"],
    "Voltage": ["Positive", "Negative"],
    "Current": ["In", "Out"],
    "Diso_amp": ["InN", "InP", "Out"],
    "Siso_amp": ["In", "Out"],
    "Dido_amp": ["InN", "InP", "OutN", "OutP"],
}
OPTIONAL_PORTS = {"NMOS": {"Body"}, "PMOS": {"Body"}}

PROMPT = f"""You are given an image of an analog circuit schematic. Convert it into a netlist.

Return ONLY a JSON object (no prose, no markdown fences) of this exact form:
{{
  "ckt_netlist": [
    {{"component_type": "NMOS", "port_connection": {{"Drain": "net1", "Gate": "VIN", "Source": "GND"}}}},
    {{"component_type": "Res",  "port_connection": {{"Pos": "VDD", "Neg": "net1"}}}}
  ]
}}

Allowed component_type values and their port names (use these names exactly):
{json.dumps(DEVICE_PORTS, indent=2)}

Meaning of the types:
- NMOS / PMOS: MOSFETs. Include "Body" only if a separate bulk/body terminal is drawn; otherwise omit it.
- NPN / PNP: bipolar transistors.
- Res, Cap, Ind: resistor, capacitor, inductor.
- Diode: "In" = anode, "Out" = cathode.
- Voltage: independent voltage source. Current: independent current source.
- Diso_amp: differential-input, single-output op-amp / comparator.
- Siso_amp: single-input, single-output amplifier or buffer.
- Dido_amp: fully differential amplifier (differential in, differential out).

Rules:
- List every device in the schematic exactly once.
- Ground and supply symbols are NOT components: connect those ports to nets named "GND" and "VDD" (use the label written in the image, e.g. "VSS", if one is given).
- Each net is one electrically connected wire region. Wires that cross without a junction dot are NOT connected unless the drawing style clearly shows otherwise.
- Use the net label from the image when a wire is labelled (e.g. "VOUT", "VB1"); otherwise name nets net1, net2, ...
- Every port of every device must be connected to a net.
"""


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.split("/")[-1])


def extract_json(text: str):
    """Return the model's final netlist from a reply.

    Tolerates markdown fences and prose, and handles self-corrections: models sometimes
    emit a netlist, then "Wait, I need to correct...", then a second netlist. Every
    top-level JSON object is decoded and the LAST one containing "ckt_netlist" wins.
    """
    decoder = json.JSONDecoder()
    found, i = [], text.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)  # not a valid object start; try the next brace
            continue
        if isinstance(obj, dict):
            found.append(obj)
        i = text.find("{", end)
    netlists = [o for o in found if "ckt_netlist" in o]
    if netlists or found:
        return (netlists or found)[-1]
    raise json.JSONDecodeError("no JSON object found in reply", text, 0)


def validate(netlist) -> list[str]:
    """Return a list of schema problems (empty = matches the GED tool's schema)."""
    problems = []
    if not isinstance(netlist, dict) or not isinstance(netlist.get("ckt_netlist"), list):
        return ["top level must be {'ckt_netlist': [...]}"]
    for i, comp in enumerate(netlist["ckt_netlist"]):
        ctype = comp.get("component_type") if isinstance(comp, dict) else None
        ports = comp.get("port_connection") if isinstance(comp, dict) else None
        if ctype not in DEVICE_PORTS:
            problems.append(f"#{i}: unknown component_type {ctype!r}")
            continue
        if not isinstance(ports, dict):
            problems.append(f"#{i} {ctype}: port_connection must be an object")
            continue
        allowed = set(DEVICE_PORTS[ctype])
        required = allowed - OPTIONAL_PORTS.get(ctype, set())
        if extra := set(ports) - allowed:
            problems.append(f"#{i} {ctype}: unknown ports {sorted(extra)}")
        if missing := required - set(ports):
            problems.append(f"#{i} {ctype}: missing ports {sorted(missing)}")
    return problems


def media_type(data: bytes) -> str:
    """Detect the real image type from magic bytes (some CI2N '.png' files are JPEGs)."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError(f"unsupported image format (first bytes: {data[:8]!r})")


def query(image_path: Path, api_key: str) -> dict:
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode()
    body = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type(data)};base64,{b64}"}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "usage": {"include": True},  # ask OpenRouter to report cost
    }
    if REASONING_EFFORT:
        body["reasoning"] = {"effort": REASONING_EFFORT}
    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=REQUEST_TIMEOUT_S,
    )
    if not resp.ok:
        # Keep OpenRouter's error body - it says *why* (e.g. provider message).
        raise requests.HTTPError(f"{resp.status_code}: {resp.text[:500]}", response=resp)
    return resp.json()


def reparse(out: Path, stems: list[str]):
    """Re-parse saved raw replies for images without a prediction; fix their log rows in place.

    Keeps the original seconds/tokens/cost, so time and cost metrics stay correct.
    """
    log_path = out / "log.csv"
    with log_path.open() as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    col = {name: i for i, name in enumerate(header)}
    last = {r[col["id"]]: r for r in body}  # same objects as in body -> edits apply in place

    fixed = 0
    for stem in stems:
        raw, pred_path = out / "raw" / f"{stem}.txt", out / "preds" / f"{stem}.json"
        if pred_path.exists() or not raw.exists() or not raw.read_text().strip():
            continue
        try:
            netlist = extract_json(raw.read_text())
        except json.JSONDecodeError as e:
            print(f"{stem}: still unparseable - {e}")
            continue
        problems = validate(netlist)
        pred_path.write_text(json.dumps(netlist, indent=4))
        if stem in last:
            row = last[stem]
            row[col["status"]] = "schema_warning" if problems else "ok"
            row[col["n_components"]] = str(len(netlist.get("ckt_netlist", [])))
            row[col["problems"]] = " | ".join(problems)
        fixed += 1
        print(f"{stem}: recovered, {len(netlist.get('ckt_netlist', []))} comps"
              + (f" - {problems[0]}" if problems else ""))

    with log_path.open("w", newline="") as f:
        csv.writer(f).writerows([header, *body])
    print(f"\nRe-parsed {fixed} image(s). Re-run evaluate.py to update metrics.")


def main():
    api_key = OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not REPARSE_ONLY:
        sys.exit("Set OPENROUTER_API_KEY at the top of run.py or in the environment.")

    dataset = (HERE / DATASET_DIR).resolve()
    images = sorted((dataset / "images").glob("*.png"))
    if NUM_FILES is not None:
        images = images[:NUM_FILES]
    if not images:
        sys.exit(f"No images found in {dataset / 'images'}")

    out = HERE / RESULTS_DIR / slug(MODEL)
    for sub in ("preds", "raw", "golden"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    if REPARSE_ONLY:
        reparse(out, [img.stem for img in images])
        return

    log_path = out / "log.csv"
    new_log = not log_path.exists()
    log_file = log_path.open("a", newline="")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["id", "model", "status", "seconds", "prompt_tokens",
                      "completion_tokens", "cost_usd", "n_components", "problems"])

    print(f"Model: {MODEL}\nImages: {len(images)}\nOutput: {out}\n")
    for n, img in enumerate(images, 1):
        stem = img.stem
        golden = dataset / "golden" / f"{stem}.json"
        if golden.exists():
            shutil.copy(golden, out / "golden" / golden.name)
        pred_path = out / "preds" / f"{stem}.json"
        if SKIP_EXISTING and pred_path.exists():
            print(f"[{n}/{len(images)}] {stem}: exists, skipped")
            continue

        status, problems, usage, n_comp, text = "ok", [], {}, 0, ""
        t0 = time.time()
        try:
            data = query(img, api_key)
            seconds = time.time() - t0
            usage = data.get("usage") or {}
            text = data["choices"][0]["message"].get("content") or ""
            netlist = extract_json(text)
            problems = validate(netlist)
            if problems:
                status = "schema_warning"
            n_comp = len(netlist.get("ckt_netlist", [])) if isinstance(netlist, dict) else 0
            pred_path.write_text(json.dumps(netlist, indent=4))
        except json.JSONDecodeError as e:
            seconds, status, problems = time.time() - t0, "invalid_json", [str(e)]
        except (requests.RequestException, KeyError, IndexError, ValueError) as e:
            seconds, status, problems = time.time() - t0, "request_error", [str(e)]

        (out / "raw" / f"{stem}.txt").write_text(text)
        log.writerow([stem, MODEL, status, f"{seconds:.2f}", usage.get("prompt_tokens"),
                      usage.get("completion_tokens"), usage.get("cost"), n_comp,
                      " | ".join(problems)])
        log_file.flush()
        print(f"[{n}/{len(images)}] {stem}: {status} {seconds:.1f}s "
              f"{n_comp} comps" + (f" - {problems[0]}" if problems else ""))

    log_file.close()
    print(f"\nDone. Score with (from {dataset}):\n"
          f"  uv run python calc_ged.py --input_dir {out / 'preds'} "
          f"--golden_dir {out / 'golden'} --output_dir {out / 'ged'}")


if __name__ == "__main__":
    main()
