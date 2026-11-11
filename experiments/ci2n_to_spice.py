#!/usr/bin/env python3
"""Convert CI2N netlist JSON (golden or predicted) into an LTspice-compatible SPICE netlist.

Input format (as produced by the experiments and the CI2N dataset):

    {"ckt_netlist": [{"component_type": "NMOS",
                      "port_connection": {"Drain": "net1", "Gate": "net2", "Source": "GND"}},
                     ...],
     "ckt_type": "DISO-Amplifier"}          # optional

CI2N netlists are topology only, so every device gets a placeholder value or model.
Those are written as .param / .model lines at the top of the file, so you can edit
them in one place.

Usage:
    python ci2n_to_spice.py preds/001.json                 # print to stdout
    python ci2n_to_spice.py preds/001.json -o 001.cir      # write one file
    python ci2n_to_spice.py results/run/preds -o out_dir/  # convert a whole folder
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Net names treated as SPICE ground (node 0); compared case-insensitively.
GROUND_NAMES = {"0", "gnd", "gnd!", "agnd", "gnda", "dgnd", "ground"}
# Supply-rail patterns. Used only when --supplies is on and no Voltage source drives the net.
POS_SUPPLY_RE = re.compile(r"^[ad]?v(dd|cc)[a-z0-9_]*!?$", re.I)
NEG_SUPPLY_RE = re.compile(r"^[ad]?v(ss|ee)[a-z0-9_]*!?$", re.I)

PREAMBLE = """\
* ---- placeholder values: CI2N netlists carry topology only; edit as needed ----
.param VDD=1.8 VSS=0
.param Rval=10k Cval=1p Ival=10u Vval=0.9
.param Wn=1u Ln=180n Wp=2u Lp=180n Aol=1e5
.model NMOS_CI2N NMOS (level=1 VTO=0.45 KP=200u LAMBDA=0.05)
.model PMOS_CI2N PMOS (level=1 VTO=-0.45 KP=80u LAMBDA=0.05)
.model NPN_CI2N NPN (Is=1e-16 Bf=100 Vaf=50)
.model PNP_CI2N PNP (Is=1e-16 Bf=50 Vaf=50)
.model D_CI2N D (Is=1e-14)
"""

# Ideal amplifier blocks for the CI2N amplifier symbols (VCVS with gain Aol, output referenced to ground).
SUBCKTS = {
    "Siso_amp": """\
.subckt SISO_AMP in out
E1 out 0 in 0 {Aol}
.ends SISO_AMP""",
    "Diso_amp": """\
.subckt DISO_AMP inp inn out
E1 out 0 inp inn {Aol}
.ends DISO_AMP""",
    "Dido_amp": """\
.subckt DIDO_AMP inp inn outp outn
E1 outp 0 inp inn {Aol/2}
E2 outn 0 inn inp {Aol/2}
.ends DIDO_AMP""",
}

# component_type -> (SPICE prefix, ordered CI2N port names, trailing model/value text)
# Port order follows SPICE: R/C n+ n-, D anode cathode, Q C B E, I/V n+ n-, M D G S B.
# For "Current", SPICE pushes current from n+ through the source to n-, which matches CI2N In -> Out.
DEVICES = {
    "Res":      ("R", ["Pos", "Neg"], "{Rval}"),
    "Cap":      ("C", ["Pos", "Neg"], "{Cval}"),
    "Diode":    ("D", ["In", "Out"], "D_CI2N"),
    "NPN":      ("Q", ["Collector", "Base", "Emitter"], "NPN_CI2N"),
    "PNP":      ("Q", ["Collector", "Base", "Emitter"], "PNP_CI2N"),
    "Current":  ("I", ["In", "Out"], "{Ival}"),
    "Voltage":  ("V", ["Positive", "Negative"], "{Vval}"),
    "NMOS":     ("M", ["Drain", "Gate", "Source", "Body"], "NMOS_CI2N L={Ln} W={Wn}"),
    "PMOS":     ("M", ["Drain", "Gate", "Source", "Body"], "PMOS_CI2N L={Lp} W={Wp}"),
    "Siso_amp": ("X", ["In", "Out"], "SISO_AMP"),
    "Diso_amp": ("X", ["InP", "InN", "Out"], "DISO_AMP"),
    "Dido_amp": ("X", ["InP", "InN", "OutP", "OutN"], "DIDO_AMP"),
}


class NetNamer:
    """Maps CI2N net names to legal, unique LTspice node names.

    LTspice node names are case-insensitive, so two nets differing only in case
    ("OUT" vs "out") would merge silently. Those get distinct names instead, since
    the source JSON treats them as different nets.
    """

    def __init__(self) -> None:
        self.map: dict[str, str] = {}
        self.used: set[str] = {"0"}

    def __call__(self, net: str) -> str:
        net = str(net).strip()
        if net in self.map:
            return self.map[net]
        if net.lower() in GROUND_NAMES:
            self.map[net] = "0"
            return "0"
        name = net.replace("+", "_p").replace("-", "_n")
        name = re.sub(r"[^A-Za-z0-9_]", "_", name) or "net"
        base, i = name, 1
        while name.lower() in self.used:
            i += 1
            name = f"{base}_{i}"
        self.used.add(name.lower())
        self.map[net] = name
        return name


def convert(data: dict, title: str = "CI2N circuit", supplies: bool = True,
            bulk: str = "source", analysis: str = ".op") -> str:
    """Return LTspice netlist text for one parsed CI2N JSON object."""
    comps = data.get("ckt_netlist", []) if isinstance(data, dict) else []
    nets = NetNamer()
    counters: dict[str, int] = defaultdict(int)
    lines: list[str] = []
    warnings: list[str] = []
    used_subckts: list[str] = []
    driven_by_vsource: set[str] = set()
    all_nets: list[str] = []  # original names, first-seen order

    for idx, comp in enumerate(comps):
        ctype = comp.get("component_type")
        ports = comp.get("port_connection", {}) or {}
        if ctype not in DEVICES:
            warnings.append(f"component #{idx}: unknown type {ctype!r}, skipped ({ports})")
            continue
        prefix, order, tail = DEVICES[ctype]
        pins = list(order)
        if prefix == "M" and "Body" not in ports:
            if bulk == "source":
                ports = {**ports, "Body": ports.get("Source")}
            else:  # tie bulk to a rail
                ports = {**ports, "Body": "VDD" if ctype == "PMOS" else "GND"}
        missing = [p for p in pins if ports.get(p) in (None, "")]
        if missing:
            warnings.append(f"component #{idx} ({ctype}): missing port(s) {missing}, skipped")
            continue
        extra = set(ports) - set(pins)
        if extra:
            warnings.append(f"component #{idx} ({ctype}): ignored unknown port(s) {sorted(extra)}")
        for p in pins:
            if ports[p] not in all_nets:
                all_nets.append(ports[p])
        counters[prefix] += 1
        name = f"{prefix}{counters[prefix]}"
        nodes = " ".join(nets(ports[p]) for p in pins)
        lines.append(f"{name} {nodes} {tail}")
        if prefix == "V":
            driven_by_vsource.update({ports["Positive"], ports["Negative"]})
        if prefix == "X" and ctype not in used_subckts:
            used_subckts.append(ctype)

    # Supply rails: drive VDD-like nets that no Voltage component already sets, so .op can run.
    supply_lines = []
    if supplies:
        for n in all_nets:
            node = nets(n)
            if node == "0" or n in driven_by_vsource:
                continue
            if POS_SUPPLY_RE.match(n):
                supply_lines.append(f"Vsup_{node} {node} 0 {{VDD}}")
            elif NEG_SUPPLY_RE.match(n):
                supply_lines.append(f"Vsup_{node} {node} 0 {{VSS}}")

    out = [f"* {title}"]
    if isinstance(data, dict) and data.get("ckt_type"):
        out.append(f"* ckt_type: {data['ckt_type']}")
    out.append("* generated by ci2n_to_spice.py")
    renamed = {k: v for k, v in nets.map.items() if k != v and v != "0"}
    if renamed:
        out.append("* renamed nets: " + ", ".join(f"{k} -> {v}" for k, v in renamed.items()))
    out += [f"* WARNING: {w}" for w in warnings]
    out.append(PREAMBLE)
    out += [SUBCKTS[t] for t in used_subckts]
    if used_subckts:
        out.append("")
    out += lines
    if supply_lines:
        out.append("* supply rails (added; not part of the CI2N netlist)")
        out += supply_lines
    if analysis:
        out.append(analysis)
    out.append(".end")
    return "\n".join(out) + "\n"


def load_netlist(path: Path) -> dict | None:
    """Load a CI2N netlist JSON; return None for files that are not netlists (labelme, edit logs)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"skip {path}: not valid JSON ({e})", file=sys.stderr)
        return None
    if not isinstance(data, dict) or "ckt_netlist" not in data:
        print(f"skip {path}: no 'ckt_netlist' key", file=sys.stderr)
        return None
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="CI2N JSON file(s) or folder(s) of them")
    ap.add_argument("-o", "--output", type=Path,
                    help="output .cir file (single input) or folder; default: stdout for one file")
    ap.add_argument("--ext", default=".cir", help="extension for written netlists (default .cir)")
    ap.add_argument("--no-supplies", dest="supplies", action="store_false",
                    help="do not add voltage sources for VDD/VCC/VSS/VEE-like nets")
    ap.add_argument("--bulk", choices=["source", "rail"], default="source",
                    help="where to tie the body of 3-terminal MOSFETs (default: source)")
    ap.add_argument("--analysis", default=".op",
                    help="analysis directive to append, e.g. '.tran 1u' or '' for none (default .op)")
    args = ap.parse_args(argv)

    files: list[Path] = []
    for p in args.inputs:
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    if not files:
        ap.error("no JSON files found")

    to_dir = args.output is not None and (len(files) > 1 or args.output.is_dir() or args.output.suffix == "")
    if args.output is None and len(files) > 1:
        ap.error("several inputs need -o OUTPUT_DIR")
    if to_dir:
        args.output.mkdir(parents=True, exist_ok=True)
    elif args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    for f in files:
        data = load_netlist(f)
        if data is None:
            continue
        text = convert(data, title=f"CI2N circuit from {f}", supplies=args.supplies,
                       bulk=args.bulk, analysis=args.analysis)
        if args.output is None:
            sys.stdout.write(text)
        else:
            dest = args.output / (f.stem + args.ext) if to_dir else args.output
            dest.write_text(text, encoding="utf-8")
        written += 1
    if args.output is not None:
        print(f"wrote {written} netlist(s) to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
