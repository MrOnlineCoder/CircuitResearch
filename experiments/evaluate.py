"""
Evaluate one experiment run against the CI2N golden netlists and write metrics JSON.

Works for any experiment that produces the layout written by 01-vllm-only/run.py:
    <RESULTS_DIR>/preds/<id>.json    predicted netlists (CI2N JSON format)
    <RESULTS_DIR>/golden/<id>.json   golden netlists for the images that were attempted
    <RESULTS_DIR>/log.csv            optional: per-image seconds / cost (last row per id wins)

Usage (any Python - it re-launches itself in the dataset's validation .venv):
    python3 experiments/evaluate.py                                    # uses RESULTS_DIR below
    python3 experiments/evaluate.py experiments/01-vllm-only/results/claude-sonnet-5.5

Output: <RESULTS_DIR>/metrics.json

GED/NED come from ged_tools.py: the dataset's own GED code plus a lower bound and a
heuristic upper bound, so each image gets an exact value or a proven [lower, upper] range.
"""

# ============================ EDITABLE CONSTANTS ============================

RESULTS_DIR = "01-vllm-only/results/claude-sonnet-5.5"  # relative to this file, or absolute
VALIDATION_DIR = "../datasets/ci2n_datasets-main/validation"  # relative to this file
GED_TIMEOUT_S = 60  # per image search budget after the bounds, like calc_ged.py --timeout
WORKERS = 6  # parallel GED processes

# ============================================================================

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VALIDATION = (HERE / VALIDATION_DIR).resolve()
VENV_PY = VALIDATION / ".venv" / "bin" / "python"

# Re-launch inside the validation venv (needs torch, networkx, scipy, ... from uv sync).
try:
    import scipy  # noqa: F401
    import torch  # noqa: F401
except ImportError:
    if VENV_PY.exists() and Path(sys.executable).resolve() != VENV_PY.resolve():
        os.execv(str(VENV_PY), [str(VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]])
    sys.exit(f"Missing deps. Run `uv sync` in {VALIDATION} first.")

import csv
import json
import statistics
import time
from collections import Counter
from multiprocessing import Pool

import ged_tools

ged_tools.load(VALIDATION)  # module level, so spawned worker processes load it too


def ged_task(args):
    stem, pred, gold = args
    return stem, ged_tools.compute_ged(pred, gold, timeout=GED_TIMEOUT_S)


def component_counts(netlist) -> Counter:
    comps = netlist.get("ckt_netlist", []) if isinstance(netlist, dict) else []
    return Counter(c.get("component_type") for c in comps if isinstance(c, dict))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def summarize(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return dict(mean=round(statistics.mean(values), 4), median=round(statistics.median(values), 4),
                min=round(min(values), 4), max=round(max(values), 4), total=round(sum(values), 4))


def read_log(path: Path) -> dict:
    if not path.exists():
        return {}
    rows = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            rows[row["id"]] = row  # last attempt per image wins
    return rows


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    results = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (HERE / RESULTS_DIR).resolve()
    golden_dir, pred_dir = results / "golden", results / "preds"
    if not golden_dir.is_dir():
        sys.exit(f"No golden/ folder in {results}")
    stems = sorted(p.stem for p in golden_dir.glob("*.json"))
    print(f"Evaluating {len(stems)} images in {results}")

    preds, golds, missing = {}, {}, []
    for s in stems:
        golds[s] = json.loads((golden_dir / f"{s}.json").read_text())
        try:
            preds[s] = json.loads((pred_dir / f"{s}.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            preds[s] = {"ckt_netlist": []}  # no usable output -> scored as an empty netlist
            missing.append(s)

    t0 = time.time()
    with Pool(min(WORKERS, len(stems))) as pool:
        ged = dict(pool.imap_unordered(ged_task, [(s, preds[s], golds[s]) for s in stems]))
    print(f"GED done in {time.time() - t0:.0f}s")

    log = read_log(results / "log.csv")
    per_image, type_tp, type_fp, type_fn = [], Counter(), Counter(), Counter()
    for s in stems:
        g = ged[s]
        pc, gc = component_counts(preds[s]), component_counts(golds[s])
        tp = sum((pc & gc).values())
        fp, fn = sum(pc.values()) - tp, sum(gc.values()) - tp
        for t in pc | gc:
            m = min(pc[t], gc[t])
            type_tp[t] += m
            type_fp[t] += pc[t] - m
            type_fn[t] += gc[t] - m

        row = log.get(s, {})
        per_image.append(dict(
            id=s, ged=g["ged"], ged_lower_bound=g["ged_lb"], ged_exact=g["exact"],
            ged_method=g["method"], ned=round(g["ged"] / g["denom"], 4),
            ned_lower_bound=round(g["ged_lb"] / g["denom"], 4),
            success=g["exact"] and g["ged"] == 0, ged_error=g["error"],
            iso_check_quirk=g["iso_check_quirk"], prediction_missing=s in missing,
            n_pred=sum(pc.values()), n_gold=sum(gc.values()), tp=tp, fp=fp, fn=fn,
            component_f1=prf(tp, fp, fn)[2],
            status=row.get("status"), seconds=num(row.get("seconds")), cost_usd=num(row.get("cost_usd")),
        ))

    n = len(per_image)
    tp, fp, fn = (sum(r[k] for r in per_image) for k in ("tp", "fp", "fn"))
    micro = prf(tp, fp, fn)
    costs = [r["cost_usd"] for r in per_image]
    total_cost = sum(c for c in costs if c is not None)
    n_success = sum(r["success"] for r in per_image)
    neds = [r["ned"] for r in per_image]
    ned_lbs = [r["ned_lower_bound"] for r in per_image]
    exact = [r for r in per_image if r["ged_exact"]]

    metrics = {
        "results_dir": str(results),
        "model": next((row.get("model") for row in log.values() if row.get("model")), None),
        "n_images": n,
        "ned": {
            "mean": round(statistics.mean(neds), 4),
            "median": round(statistics.median(neds), 4),
            "lower_bound_mean": round(statistics.mean(ned_lbs), 4),
            "lower_bound_median": round(statistics.median(ned_lbs), 4),
            "n_exact": len(exact),
            "n_bounded": n - len(exact),
            "note": "NED = GED / (devices + nets + ports of golden). mean/median use the best "
                    "known GED (exact, or a proven upper bound); the true value lies between "
                    "lower_bound_* and these. Missing/unparseable predictions are scored as an "
                    "empty netlist (NED = 1).",
        },
        "success_rate": round(n_success / n, 4),
        "n_success": n_success,
        "component_count": {
            "micro": dict(zip(("precision", "recall", "f1"), micro)),
            "macro_f1_over_images": round(statistics.mean(r["component_f1"] for r in per_image), 4),
            "per_type": {t: dict(zip(("precision", "recall", "f1"),
                                     prf(type_tp[t], type_fp[t], type_fn[t])),
                                 support=type_tp[t] + type_fn[t])
                         for t in sorted(type_tp | type_fp | type_fn, key=str)},
        },
        "time_and_cost": {
            "seconds_per_image": summarize([r["seconds"] for r in per_image]),
            "usd_per_image": summarize(costs),
            "usd_total": round(total_cost, 4),
            "usd_per_successful_netlist": round(total_cost / n_success, 4) if n_success else None,
        },
        "reliability": {
            "prediction_missing_or_invalid_json": len(missing),
            "ged_parse_errors": sum(r["ged_error"] is not None for r in per_image),
            "ged_method_counts": dict(Counter(r["ged_method"] for r in per_image)),
            "dataset_iso_check_false_positives": sum(r["iso_check_quirk"] for r in per_image),
            "status_counts": dict(Counter(r["status"] for r in per_image if r["status"])),
        },
        "settings": {"ged_timeout_s": GED_TIMEOUT_S},
        "per_image": per_image,
    }

    out = results / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    summary = {k: metrics[k] for k in ("model", "n_images", "success_rate")}
    summary["ned"] = {k: v for k, v in metrics["ned"].items() if k != "note"}
    summary["component_f1_micro"] = metrics["component_count"]["micro"]["f1"]
    summary["seconds_per_image_mean"] = (metrics["time_and_cost"]["seconds_per_image"] or {}).get("mean")
    summary["usd_per_image_mean"] = (metrics["time_and_cost"]["usd_per_image"] or {}).get("mean")
    summary["ged_methods"] = metrics["reliability"]["ged_method_counts"]
    print(json.dumps(summary, indent=2))
    print(f"Full metrics: {out}")


if __name__ == "__main__":
    main()
