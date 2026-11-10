"""Image -> CI2N netlist JSON.

Stages:
  1. detect components, gnd/vdd symbols, terminals and hop arcs (YOLO)
  2. detect crossings (YOLO-OBB) and decide which are jumpers (drawing-style rule)    [v1+]
  3. erase components + jumpers, label connected wire regions (OpenCV)
  4. reconnect wires across jumpers (opposite leads), merge regions under gnd/vdd symbols
  5. find each component's contacts and name its pins (geometry or classifiers [v2])
  6. assign net names and write the JSON
"""

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from . import wires as W
from .config import BRIDGES, DEVICE_LABELS, NET_SYMBOLS, PINS, TERMINALS, Settings
from .detect import Box, Models
from .pins import assign_pins


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, a):
        self.parent.setdefault(a, a)
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass
class Trace:
    """Intermediate results, for debugging overlays and analysis."""

    ink_mode: str = "otsu"  # threshold actually used (see Settings.ink_threshold)
    components: list = field(default_factory=list)  # (Box, contacts, pins)
    symbols: list = field(default_factory=list)
    terminals: list = field(default_factory=list)
    jumpers: list = field(default_factory=list)
    connected_crossings: list = field(default_factory=list)
    guessed_inputs: list = field(default_factory=list)  # amp boxes whose +/- inputs were guessed
    labels: np.ndarray = None
    region_net: dict = field(default_factory=dict)  # wire region -> net name
    timings: dict = field(default_factory=dict)


def _inside(b: Box, other: Box) -> bool:
    cx, cy = b.center
    return other.x0 <= cx <= other.x1 and other.y0 <= cy <= other.y1


def _choose_jumpers(crossings: list[Box], bridges: list[Box], components: list[Box], s: Settings):
    """Which 4-way crossings are jumpers (not connected), from the drawing's style.

    Hop arcs are always jumpers. A flat crossing is a connection in drawings that use hop
    arcs for non-connections, a jumper in drawings that use dots for connections, and
    `flat_crossing_default` when the drawing has neither.
    """
    has_bridge = bool(bridges)
    has_dot = any(c.label == "dot" for c in crossings)
    jumpers, connected = list(bridges), []
    for c in crossings:
        if c.label != "cross" or any(_inside(c, comp) for comp in components):
            continue
        if any(_inside(c, b) or _inside(b, c) for b in bridges):
            continue  # already handled as a hop arc
        if has_bridge:
            connected.append(c)
        elif has_dot or s.flat_crossing_default == "jumper":
            jumpers.append(c)
        else:
            connected.append(c)
    return jumpers, connected


def _apply_feedback_rule(components, k, uf) -> None:
    """Op-amp inputs without a classifier: the input fed back from the output is InN.

    An input has feedback if it is on the output net, or another component has one pin on
    the output net and one on that input's net (a feedback resistor, a source follower...).
    Swaps InP/InN only when exactly one input has feedback.
    """
    comp, contacts, pins = components[k]

    def net(i):
        return uf.find(contacts[i].region) if contacts[i].region else None

    if not all(pins.get(p) for p in ("InP", "InN", "Out")):
        return
    nets = {p: net(pins[p][0]) for p in ("InP", "InN", "Out")}
    if None in nets.values():
        return
    linked = set()
    for j, (_, cs, ps) in enumerate(components):
        if j == k:
            continue
        on = {uf.find(cs[i].region) for idx in ps.values() for i in idx if cs[i].region}
        if nets["Out"] in on:
            linked |= on
    fed = [p for p in ("InP", "InN") if nets[p] == nets["Out"] or nets[p] in linked]
    if fed == ["InP"]:
        pins["InP"], pins["InN"] = pins["InN"], pins["InP"]


class Pipeline:
    def __init__(self, models_dir, settings: Settings):
        self.s = settings
        self.models = Models(models_dir, settings)

    def run(self, image_path) -> tuple[dict, Trace]:
        """Image -> (CI2N netlist, trace). With ink_threshold "auto": Otsu first; if most pins
        come out unconnected, the wires were lost (light-grey drawing), so redo with the
        background threshold."""
        mode = self.s.ink_threshold
        netlist, tr = self._run(image_path, "otsu" if mode == "auto" else mode)
        if mode == "auto":
            pins = [n for c in netlist["ckt_netlist"] for n in c["port_connection"].values()]
            if pins and sum(n.startswith("nc") for n in pins) / len(pins) > self.s.ink_fallback_nc_frac:
                netlist, tr = self._run(image_path, "background")
        return netlist, tr

    def _run(self, image_path, ink_mode: str) -> tuple[dict, Trace]:
        s, t, tr = self.s, time.time(), Trace()
        tr.ink_mode = ink_mode
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)  # decodes by content (JPEG-as-.png ok)
        if bgr is None:
            bgr = cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        # 1-2. detection
        dets = self.models.devices(bgr)
        comps = [d for d in dets if d.label in DEVICE_LABELS]
        tr.symbols = [d for d in dets if d.label in NET_SYMBOLS]
        tr.terminals = [d for d in dets if d.label in TERMINALS]
        bridges = [d for d in dets if d.label in BRIDGES]
        ink = W.binarize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), ink_mode, s.ink_contrast)
        sw = W.stroke_width(ink)
        if s.crossings == "model":
            crossings = self.models.crossings(bgr)
        elif s.crossings == "geometry":
            crossings = W.find_crossings(W.erase(ink, comps, s), sw)
        else:
            crossings = []
        tr.jumpers, tr.connected_crossings = _choose_jumpers(crossings, bridges, comps, s)
        tr.timings["detect"] = time.time() - t

        # 3. wire regions
        wires = W.erase(ink, comps + tr.jumpers, s)
        labels, extent = W.label_regions(wires)
        tr.labels = labels
        uf = _UnionFind()

        # 4a. jumpers: lead i connects to the opposite lead i + N/2
        for j in tr.jumpers:
            cs = [c for c in W.find_contacts(j, ink, wires, labels, s, sw) if c.region]
            cx, cy = j.center
            cs.sort(key=lambda c: np.arctan2(c.y - cy, c.x - cx))
            if len(cs) >= 4 and len(cs) % 2 == 0:
                half = len(cs) // 2
                for i in range(half):
                    uf.union(cs[i].region, cs[i + half].region)
            else:  # unclear geometry: treat as a junction
                for c in cs[1:]:
                    uf.union(cs[0].region, c.region)

        # 4b. ground / supply symbols name the region they sit on
        supply_regions = {"GND": [], "VDD": []}
        for sym in tr.symbols:
            regs = W.regions_in_box(sym, labels, s)
            if regs:
                supply_regions[NET_SYMBOLS[sym.label]].append(regs[0])
        for name, regs in supply_regions.items():
            if name == "VDD" and not s.merge_supply_symbols:
                continue
            for r in regs[1:]:
                uf.union(regs[0], r)

        def named(region):
            if not region:
                return None
            root = uf.find(region)
            for name, regs in supply_regions.items():
                if any(uf.find(r) == root for r in regs):
                    return name
            return None

        # 5. components: contacts -> pins
        def orient_fn(box: Box):
            def orient(task, rot_k=0):
                if not s.use_orientation_models:
                    return None
                pad = 0.1 * max(box.w, box.h)
                crop = pil.crop((int(box.x0 - pad), int(box.y0 - pad), int(box.x1 + pad), int(box.y1 + pad)))
                if rot_k:
                    crop = crop.rotate(90 * rot_k, expand=True, fillcolor="white")
                res = self.models.classify(task, crop)
                return res[0] if res and res[1] >= s.orientation_min_conf else None
            return orient

        all_contacts = [W.find_contacts(comp, ink, wires, labels, s, sw) for comp in comps]

        # Weak contacts: small regions (text, label fragments) touching only one component
        # lead and no symbol/terminal. They add nothing to the topology but can be mistaken
        # for leads, so pin naming uses them only when nothing better is there.
        anchored = {r for b in tr.symbols + tr.terminals for r in W.regions_in_box(b, labels, s)}
        uses = {}  # region -> number of distinct components it touches
        for cs in all_contacts:
            for r in {c.region for c in cs}:
                uses[r] = uses.get(r, 0) + 1
        small = max(s.weak_region_strokes * sw, s.weak_region_frac * min(ink.shape))
        for cs in all_contacts:
            for c in cs:
                c.extent = int(extent[c.region]) if c.region else 0
                c.weak = (not c.region or (uses[c.region] == 1 and c.region not in anchored
                                           and c.extent < small))

        guessed_amps = []
        for comp, contacts in zip(comps, all_contacts):
            dtype, kind = DEVICE_LABELS[comp.label]
            pins = assign_pins(dtype, kind, comp, contacts, named, orient_fn(comp))
            if pins.pop("_inputs_guessed", None) is not None:
                guessed_amps.append(len(tr.components))
            for idx in pins.values():  # e.g. both gate leads of a *-cross device
                regs = [contacts[i].region for i in idx if contacts[i].region]
                for r in regs[1:]:
                    uf.union(regs[0], r)
            tr.components.append((comp, contacts, pins))
        for k in guessed_amps:
            _apply_feedback_rule(tr.components, k, uf)
        tr.guessed_inputs = [tr.components[k][0] for k in guessed_amps]
        tr.timings["wires_pins"] = time.time() - t - tr.timings["detect"]

        # 6. net names and JSON
        net_of_root, counter = {}, {"net": 0, "nc": 0}

        def net_name(region):
            if not region:
                counter["nc"] += 1
                return f"nc{counter['nc']}"  # unconnected pin: its own net
            root = uf.find(region)
            if root not in net_of_root:
                name = named(region)
                if name is None:
                    counter["net"] += 1
                    name = f"net{counter['net']}"
                net_of_root[root] = name
            return net_of_root[root]

        netlist = []
        for comp, contacts, pins in sorted(tr.components, key=lambda x: (x[0].y0, x[0].x0)):
            dtype, _ = DEVICE_LABELS[comp.label]
            ports = {}
            for pin in PINS[dtype] + (["Body"] if "Body" in pins else []):
                idx = pins.get(pin, [])
                ports[pin] = net_name(contacts[idx[0]].region if idx else 0)
            netlist.append({"component_type": dtype, "port_connection": ports})
        tr.region_net = {r: n for r, n in ((r, net_of_root.get(uf.find(r))) for r in np.unique(labels)
                                           if r) if n}
        tr.timings["total"] = time.time() - t
        return {"ckt_netlist": netlist}, tr
