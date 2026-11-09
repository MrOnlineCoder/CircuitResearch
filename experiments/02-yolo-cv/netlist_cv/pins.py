"""Pin naming: map each device's contacts to CI2N port names.

Each rule works in box-normalised coordinates (u = offset from the box centre divided by
the half-size, so the box spans -1..1). Orientation comes from a classifier when available
(v2) and from geometry otherwise (v0/v1). Fallbacks for the remaining ambiguities use
conventions measured on the CI2N golden netlists: NMOS source / NPN emitter / current-source
Out / voltage-source Negative on ground; PMOS source / PNP emitter / current-source In on
the supply.

assign_pins() returns {pin: [contact indices]}; several contacts on one pin (the two gate
leads of a *-cross device) are joined into one net by the assembler. Pins left out get a
fresh unconnected net.
"""

from typing import Callable, Optional

import numpy as np

from .detect import Box
from .wires import Contact

SIDES = {"l": (-1.0, 0.0), "r": (1.0, 0.0), "u": (0.0, -1.0), "d": (0.0, 1.0)}

# orient(task, rot_k) -> classifier label, or None (no model / low confidence / disabled).
# rot_k = number of 90-degree counter-clockwise turns applied to the crop first.
Orient = Callable[[str, int], Optional[str]]


def ccw(v):  # 90 degrees counter-clockwise on screen (y axis points down)
    return (v[1], -v[0])


def cw(v):
    return (-v[1], v[0])


def turns_to(src, dst) -> int:
    """Counter-clockwise quarter turns that rotate direction src onto dst."""
    v = src
    for k in range(4):
        if np.allclose(v, dst):
            return k
        v = ccw(v)
    raise ValueError((src, dst))


def snap(v):
    """Nearest axis direction."""
    return (float(np.sign(v[0]) or 1.0), 0.0) if abs(v[0]) >= abs(v[1]) else (0.0, float(np.sign(v[1]) or 1.0))


class _Geo:
    def __init__(self, box: Box, contacts: list[Contact]):
        cx, cy = box.center
        hw, hh = max(box.w / 2, 1.0), max(box.h / 2, 1.0)
        self.u = {i: ((c.x - cx) / hw, (c.y - cy) / hh) for i, c in enumerate(contacts)}
        self.c = contacts

    def proj(self, i, v):
        return self.u[i][0] * v[0] + self.u[i][1] * v[1]

    def argmax(self, idx, v):
        return max(idx, key=lambda i: self.proj(i, v))

    def argmin(self, idx, v):
        return min(idx, key=lambda i: self.proj(i, v))


def _select(contacts: list[Contact], n: int) -> list[int]:
    """Best n contacts: connected wires before isolated text blobs, real leads first, then larger blobs."""
    def key(i):
        c = contacts[i]
        # among weak (probably text) contacts, longer regions are more likely real wires
        return (c.weak, -c.extent if c.weak else 0, not c.lead_ok, -c.size)

    return sorted(sorted(range(len(contacts)), key=key)[:n])


def _edge(u) -> str:
    """Box side a contact sits on: 'h' (left/right) or 'v' (top/bottom)."""
    return "h" if abs(u[0]) >= abs(u[1]) else "v"


def _odd_one(g: _Geo, idx):
    """For 3 contacts: the gate/base (odd one) and the drain/source or collector/emitter pair.

    First rule: the gate is the only lead on a left/right side while the other two leave
    through top/bottom (or the same rotated by 90 degrees). Fallback: the pair sharing an axis.
    """
    edges = {i: _edge(g.u[i]) for i in idx}
    for e in ("h", "v"):
        alone = [i for i in idx if edges[i] == e]
        if len(alone) == 1:
            odd = alone[0]
            pair = [i for i in idx if i != odd]
            mid = np.mean([g.u[i] for i in pair], axis=0)
            return odd, pair, snap(np.subtract(g.u[odd], mid))
    best = None
    for k in idx:
        pair = [i for i in idx if i != k]
        (ax, ay), (bx, by) = g.u[pair[0]], g.u[pair[1]]
        score = min(abs(ax - bx), abs(ay - by))
        if best is None or score < best[0]:
            best = (score, k, pair)
    _, odd, pair = best
    mid = np.mean([g.u[i] for i in pair], axis=0)
    return odd, pair, snap(np.subtract(g.u[odd], mid))


def _odd_of_four(g: _Geo, idx):
    """For 4 contacts (D, S, Body + Gate): the one off the line of the other three."""
    best = None
    for k in idx:
        rest = [i for i in idx if i != k]
        pts = np.array([g.u[i] for i in rest])
        sx, sy = pts[:, 0].std(), pts[:, 1].std()
        axis = 0 if sx < sy else 1  # rest share this coordinate
        score = abs(g.u[k][axis] - pts[:, axis].mean()) - min(sx, sy)
        if best is None or score > best[0]:
            best = (score, k, rest)
    _, k, rest = best
    return k, rest, snap(np.subtract(g.u[k], np.mean([g.u[i] for i in rest], axis=0)))


def _pick_by(g, pair, named, prefer_net, lower_wins: bool):
    """Choose one of two contacts (NMOS source / NPN emitter: lower; PMOS / PNP: upper).

    Position decides when the two are clearly one above the other (the usual drawing
    convention). Otherwise the one on the preferred net (GND / VDD) wins, if exactly one is.
    Net names come second because a "vdd"-style bar is sometimes drawn for other rails.
    """
    ys = {i: g.u[i][1] for i in pair}
    by_height = max(pair, key=ys.get) if lower_wins else min(pair, key=ys.get)
    if abs(ys[pair[0]] - ys[pair[1]]) > 0.5:
        return by_height
    hits = [i for i in pair if named(g.c[i].region) == prefer_net]
    return hits[0] if len(hits) == 1 else by_height


def _other(pair, i):
    return pair[1] if pair[0] == i else pair[0]


def _side(label: Optional[str]):
    return SIDES.get(label) if label else None


def assign_pins(dtype: str, kind: str, box: Box, contacts: list[Contact],
                named: Callable[[int], Optional[str]], orient: Orient) -> dict[str, list[int]]:
    """named(region) -> "GND" / "VDD" / None for the wire region a contact sits on."""
    g = _Geo(box, contacts)
    n = len(contacts)
    if n == 0:
        return {}

    if kind in ("mos", "bjt"):
        idx = _select(contacts, 3)
        if len(idx) < 3:
            return {}
        task = "orient_mos" if kind == "mos" else "orient_bjt_d"
        side = _side(orient(task, 0))
        if side:
            ctrl = g.argmax(idx, side)
            pair = [i for i in idx if i != ctrl]
        else:
            ctrl, pair, side = _odd_one(g, idx)
        if kind == "mos":
            low_net = "GND" if dtype == "NMOS" else "VDD"
            src = _pick_by(g, pair, named, low_net, lower_wins=dtype == "NMOS")
            return {"Gate": [ctrl], "Source": [src], "Drain": [_other(pair, src)]}
        mirror = orient("orient_bjt_m", turns_to(side, SIDES["l"]))
        if mirror in ("0", "1"):
            up = cw(side)  # "top" in the base-left frame, mapped back to the image
            emitter = g.argmax(pair, up if mirror == "0" else (-up[0], -up[1]))
        else:
            e_net = "GND" if dtype == "NPN" else "VDD"
            emitter = _pick_by(g, pair, named, e_net, lower_wins=dtype == "NPN")
        return {"Base": [ctrl], "Emitter": [emitter], "Collector": [_other(pair, emitter)]}

    if kind == "mos4":
        idx = _select(contacts, 4)
        if len(idx) < 4:
            return assign_pins(dtype, "mos", box, contacts, named, orient)
        side = _side(orient("orient_mos", 0))
        if side:
            gate = g.argmax(idx, side)
            rest = [i for i in idx if i != gate]
        else:
            gate, rest, side = _odd_of_four(g, idx)
        along = cw(side)
        rest = sorted(rest, key=lambda i: g.proj(i, along))
        body, pair = rest[1], [rest[0], rest[2]]
        same_as_body = [i for i in pair if contacts[i].region and contacts[i].region == contacts[body].region]
        if len(same_as_body) == 1:
            src = same_as_body[0]
        else:
            src = _pick_by(g, pair, named, "GND" if dtype == "NMOS" else "VDD", lower_wins=dtype == "NMOS")
        return {"Gate": [gate], "Source": [src], "Drain": [_other(pair, src)], "Body": [body]}

    if kind in ("mos_cross", "bjt_cross"):
        idx = _select(contacts, 4)
        if len(idx) < 4:
            return assign_pins(dtype, "mos" if kind == "mos_cross" else "bjt", box, contacts, named, orient)
        axis = (0.0, 1.0) if box.h >= box.w else (1.0, 0.0)  # channel runs along the long side
        ends = sorted(idx, key=lambda i: g.proj(i, axis))
        pair, ctrls = [ends[0], ends[-1]], ends[1:3]
        if kind == "mos_cross":
            src = _pick_by(g, pair, named, "GND" if dtype == "NMOS" else "VDD", lower_wins=dtype == "NMOS")
            return {"Gate": ctrls, "Source": [src], "Drain": [_other(pair, src)]}
        em = _pick_by(g, pair, named, "GND" if dtype == "NPN" else "VDD", lower_wins=dtype == "NPN")
        return {"Base": ctrls, "Emitter": [em], "Collector": [_other(pair, em)]}

    if kind in ("two", "diode", "current", "voltage", "battery", "siso"):
        idx = _select(contacts, 2)
        if len(idx) < 2:
            return {}
        a, b = idx
        if kind == "two":
            first = min(idx, key=lambda i: (g.u[i][1], g.u[i][0]))
            return {"Pos": [first], "Neg": [_other(idx, first)]}
        spread = np.abs(np.subtract(g.u[a], g.u[b]))
        flow = (0.0, 1.0) if spread[1] >= spread[0] else (1.0, 0.0)  # top->bottom / left->right
        if kind == "diode":
            side = _side(orient("orient_diode", 0)) or flow
            out = g.argmax(idx, side)
            return {"In": [_other(idx, out)], "Out": [out]}
        if kind == "current":
            vdd = [i for i in idx if named(contacts[i].region) == "VDD"]
            gnd = [i for i in idx if named(contacts[i].region) == "GND"]
            inp = vdd[0] if len(vdd) == 1 else _other(idx, gnd[0]) if len(gnd) == 1 else g.argmin(idx, flow)
            return {"In": [inp], "Out": [_other(idx, inp)]}
        if kind in ("voltage", "battery"):
            side = _side(orient("orient_voltagelines", 0)) if kind == "battery" else None
            if side:
                pos = g.argmax(idx, side)
            else:
                gnd = [i for i in idx if named(contacts[i].region) == "GND"]
                vdd = [i for i in idx if named(contacts[i].region) == "VDD"]
                pos = _other(idx, gnd[0]) if len(gnd) == 1 else vdd[0] if len(vdd) == 1 else g.argmin(idx, flow)
            return {"Positive": [pos], "Negative": [_other(idx, pos)]}
        side = _side(orient("orient_amp_d", 0)) or flow  # siso: tip = output
        out = g.argmax(idx, side)
        return {"In": [_other(idx, out)], "Out": [out]}

    if kind in ("diso", "dido"):
        n_out = 1 if kind == "diso" else 2
        strong = [i for i in range(n) if not contacts[i].weak]
        idx = strong if len(strong) >= n_out + 2 else list(range(n))
        tip = _side(orient("orient_amp_d", 0))
        if tip is None:
            tip = _amp_tip(g, idx, n_out)
        if tip is None:
            return {}
        outs = sorted(idx, key=lambda i: -g.proj(i, tip))[:n_out]
        ins = sorted([i for i in idx if i not in outs], key=lambda i: g.proj(i, tip))[:2]
        if len(ins) < 2:
            return {"Out": outs[:1]} if kind == "diso" else {}
        mirror = orient("orient_amp_m", turns_to(tip, SIDES["d"]))
        left = cw(tip)  # "left" in the tip-down frame, mapped back to the image
        # Without a classifier: "+" sits counter-clockwise of the tip (on top when the tip
        # points right) - true in all ground-truth-box test cases; the pipeline may still
        # swap it with the feedback rule (the input fed back from the output is "-").
        plus = g.argmax(ins, left) if mirror == "0" else g.argmin(ins, left)
        pins = {"InP": [plus], "InN": [_other(ins, plus)]}
        if mirror is None:
            pins["_inputs_guessed"] = []
        if kind == "diso":
            pins["Out"] = outs[:1]
        elif len(outs) == 2:
            # Fully differential: the output on the "+" input's side is the inverting one.
            out_n = g.argmax(outs, left) if plus == g.argmax(ins, left) else g.argmin(outs, left)
            pins.update(OutN=[out_n], OutP=[_other(outs, out_n)])
        return pins

    raise ValueError(f"unknown pin kind {kind!r}")


def _amp_tip(g: _Geo, idx, n_out):
    """Geometry guess of the output direction: n_out contacts on one side, 2 opposite."""
    candidates = []
    for name in ("r", "d", "l", "u"):  # signal usually flows right / down: prefer those on ties
        v = SIDES[name]
        outs = [i for i in idx if g.proj(i, v) > 0.4]
        ins = [i for i in idx if g.proj(i, v) < -0.4]
        if len(outs) == n_out and len(ins) >= 2:
            candidates.append(v)
    if candidates:
        return candidates[0]
    if len(idx) >= 3 and n_out == 1:  # tip = contact farthest from the others' midpoint
        far = max(idx, key=lambda i: np.linalg.norm(
            np.subtract(g.u[i], np.mean([g.u[j] for j in idx if j != i], axis=0))))
        return snap(np.subtract(g.u[far], np.mean([g.u[j] for j in idx if j != far], axis=0)))
    return None
