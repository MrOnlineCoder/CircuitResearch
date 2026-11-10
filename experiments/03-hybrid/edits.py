"""Edit operations the LLM may request, and applying them to the pipeline's netlist.

The LLM never rewrites the netlist. It returns a list of edits on the pipeline's components
(by ID, e.g. "M3") and nets (by name, e.g. "net5"). Invalid edits are rejected one by one
and logged; the rest are applied in order.
"""

import copy

# Same port schema as experiment 01 / the dataset's GED tool (validation/device_def.toml).
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

# Pin correspondence when a type changes within a family (other changes need explicit pins).
_FAMILY = {
    frozenset({"NMOS", "PMOS", "NPN", "PNP"}): {
        "Drain": "Collector", "Gate": "Base", "Source": "Emitter",
        "Collector": "Drain", "Base": "Gate", "Emitter": "Source",
    },
    frozenset({"Res", "Cap", "Ind"}): {},
    frozenset({"Diode", "Current"}): {},
}

OPS_HELP = """\
{"op": "set_pin", "component": "M3", "pin": "Drain", "net": "net5"}
    Connect one pin to a net. Use an existing net name, or a NEW name (e.g. "new1") to give
    the pin a net of its own / to split a net (give every pin that moves the same new name).
{"op": "swap_pins", "component": "M3", "pins": ["Drain", "Source"]}
    Exchange the nets of two pins of one component (e.g. drain/source, InP/InN, Pos/Neg).
{"op": "merge_nets", "nets": ["net2", "net7"]}
    The listed nets are one net (wires that are actually connected). The first name is kept.
{"op": "change_type", "component": "Q1", "type": "PNP", "pins": {"Collector": "net1", ...}}
    Wrong component type. "pins" is optional when the pins map one-to-one
    (NMOS/PMOS/NPN/PNP: Drain<->Collector, Gate<->Base, Source<->Emitter;
    Res/Cap/Ind; Diode/Current); otherwise give the full pin -> net mapping for the new type.
{"op": "add_component", "type": "Res", "pins": {"Pos": "net1", "Neg": "GND"}}
    A component the pipeline missed.
{"op": "remove_component", "component": "R4"}
    Not a real component (e.g. a detection on text or a symbol that is not a device).
{"op": "remove_pin", "component": "M2", "pin": "Body"}
    Only for the optional MOSFET Body pin."""


def _family_map(old: str, new: str):
    for fam, mapping in _FAMILY.items():
        if old in fam and new in fam:
            return mapping
    return None


def apply_edits(components: list[dict], edits: list) -> tuple[dict, list[dict]]:
    """components: [{"id", "type", "pins": {pin: net}}] from data/context/<id>.json.

    Returns (CI2N netlist, log) where log has one entry per edit: {"edit", "ok", "error"}.
    """
    comps = {c["id"]: {"type": c["type"], "pins": dict(c["pins"])} for c in copy.deepcopy(components)}
    order = [c["id"] for c in components]
    log, added = [], 0

    def fail(e, msg):
        log.append({"edit": e, "ok": False, "error": msg})

    for e in edits if isinstance(edits, list) else []:
        if not isinstance(e, dict):
            fail(e, "not an object")
            continue
        op, cid = e.get("op"), e.get("component")
        c = comps.get(cid)
        try:
            if op in ("set_pin", "swap_pins", "change_type", "remove_component", "remove_pin") and c is None:
                raise ValueError(f"unknown component {cid!r}")
            if op == "set_pin":
                pin, net = e.get("pin"), e.get("net")
                if pin not in DEVICE_PORTS.get(c["type"], c["pins"]):
                    raise ValueError(f"{c['type']} has no pin {pin!r}")
                if not isinstance(net, str) or not net:
                    raise ValueError("net must be a non-empty string")
                c["pins"][pin] = net
            elif op == "swap_pins":
                a, b = (e.get("pins") or [None, None])[:2]
                if a not in c["pins"] or b not in c["pins"]:
                    raise ValueError(f"pins {a!r}/{b!r} not on {cid}")
                c["pins"][a], c["pins"][b] = c["pins"][b], c["pins"][a]
            elif op == "merge_nets":
                nets = [n for n in (e.get("nets") or []) if isinstance(n, str)]
                if len(nets) < 2:
                    raise ValueError("merge_nets needs at least two net names")
                keep = nets[0]
                for other in comps.values():
                    for p, n in other["pins"].items():
                        if n in nets[1:]:
                            other["pins"][p] = keep
            elif op == "change_type":
                new = e.get("type")
                if new not in DEVICE_PORTS:
                    raise ValueError(f"unknown type {new!r}")
                if isinstance(e.get("pins"), dict):
                    c["pins"] = {p: n for p, n in e["pins"].items()}
                else:
                    mapping = _family_map(c["type"], new)
                    if mapping is None:
                        raise ValueError(f"{c['type']} -> {new} needs an explicit 'pins' mapping")
                    # keep pin names the new type has (NPN -> PNP); translate only the others
                    # (NMOS -> NPN: Drain -> Collector, ...)
                    renamed = {(p if p in DEVICE_PORTS[new] else mapping.get(p, p)): n
                               for p, n in c["pins"].items()}
                    c["pins"] = {p: n for p, n in renamed.items() if p in DEVICE_PORTS[new]}
                c["type"] = new
            elif op == "add_component":
                t, pins = e.get("type"), e.get("pins")
                if t not in DEVICE_PORTS or not isinstance(pins, dict):
                    raise ValueError("add_component needs a known 'type' and a 'pins' object")
                added += 1
                nid = f"new{added}"
                comps[nid] = {"type": t, "pins": dict(pins)}
                order.append(nid)
            elif op == "remove_component":
                del comps[cid]
                order.remove(cid)
            elif op == "remove_pin":
                if e.get("pin") not in OPTIONAL_PORTS.get(c["type"], ()) or e.get("pin") not in c["pins"]:
                    raise ValueError("only an existing optional pin (MOSFET Body) can be removed")
                del c["pins"][e["pin"]]
            else:
                raise ValueError(f"unknown op {op!r}")
            log.append({"edit": e, "ok": True, "error": None})
        except (ValueError, TypeError) as err:
            fail(e, str(err))

    netlist = {"ckt_netlist": [{"component_type": comps[i]["type"], "port_connection": comps[i]["pins"]}
                               for i in order if i in comps]}
    return netlist, log


def validate(netlist) -> list[str]:
    """Schema problems (empty = matches the GED tool's schema). Same check as experiment 01."""
    problems = []
    for i, comp in enumerate(netlist.get("ckt_netlist", [])):
        ctype, ports = comp.get("component_type"), comp.get("port_connection") or {}
        if ctype not in DEVICE_PORTS:
            problems.append(f"#{i}: unknown component_type {ctype!r}")
            continue
        allowed = set(DEVICE_PORTS[ctype])
        required = allowed - OPTIONAL_PORTS.get(ctype, set())
        if extra := set(ports) - allowed:
            problems.append(f"#{i} {ctype}: unknown ports {sorted(extra)}")
        if missing := required - set(ports):
            problems.append(f"#{i} {ctype}: missing ports {sorted(missing)}")
    return problems


if __name__ == "__main__":  # quick self-test
    comps = [{"id": "M1", "type": "NMOS", "pins": {"Drain": "net1", "Gate": "net2", "Source": "GND"}},
             {"id": "R1", "type": "Res", "pins": {"Pos": "VDD", "Neg": "net3"}}]
    nl, lg = apply_edits(comps, [
        {"op": "merge_nets", "nets": ["net1", "net3"]},
        {"op": "swap_pins", "component": "M1", "pins": ["Drain", "Source"]},
        {"op": "change_type", "component": "M1", "type": "NPN"},
        {"op": "set_pin", "component": "R1", "pin": "Pos", "net": "new1"},
        {"op": "add_component", "type": "Cap", "pins": {"Pos": "net2", "Neg": "GND"}},
        {"op": "set_pin", "component": "X9", "pin": "Pos", "net": "a"},  # rejected
    ])
    assert nl["ckt_netlist"][0] == {"component_type": "NPN",
                                    "port_connection": {"Collector": "GND", "Base": "net2", "Emitter": "net1"}}
    assert nl["ckt_netlist"][1]["port_connection"] == {"Pos": "new1", "Neg": "net1"}
    assert len(nl["ckt_netlist"]) == 3 and [x["ok"] for x in lg] == [True] * 5 + [False]
    assert not validate(nl)
    # same-family changes keep the pins (bug found on image 069: NPN -> PNP lost all pins)
    bjt = [{"id": "Q1", "type": "NPN", "pins": {"Collector": "a", "Base": "b", "Emitter": "c"}},
           {"id": "M1", "type": "NMOS", "pins": {"Drain": "d", "Gate": "g", "Source": "s", "Body": "s"}}]
    nl, lg = apply_edits(bjt, [{"op": "change_type", "component": "Q1", "type": "PNP"},
                               {"op": "change_type", "component": "M1", "type": "PMOS"}])
    assert nl["ckt_netlist"][0]["port_connection"] == {"Collector": "a", "Base": "b", "Emitter": "c"}
    assert nl["ckt_netlist"][1]["port_connection"] == {"Drain": "d", "Gate": "g", "Source": "s", "Body": "s"}
    assert not validate(nl)
    print("edits.py self-test ok")
