"""Label mappings, pin definitions and pipeline settings."""

from dataclasses import dataclass

# --- Device detector labels (CI2N device_identification) -> (netlist type, pin "kind") ---
# kind selects the pin-naming rule in pins.py.
DEVICE_LABELS = {
    "nmos": ("NMOS", "mos"),
    "pmos": ("PMOS", "mos"),
    "nmos-bulk": ("NMOS", "mos4"),
    "pmos-bulk": ("PMOS", "mos4"),
    "nmos-cross": ("NMOS", "mos_cross"),  # two parallel gate leads = one Gate pin
    "pmos-cross": ("PMOS", "mos_cross"),
    "npn": ("NPN", "bjt"),
    "pnp": ("PNP", "bjt"),
    "npn-cross": ("NPN", "bjt_cross"),
    "pnp-cross": ("PNP", "bjt_cross"),
    "resistor": ("Res", "two"),  # US zigzag
    "resistor2": ("Res", "two"),  # European box
    "resistor2_3": ("Res", "two"),
    "capacitor": ("Cap", "two"),
    "capacitor-3": ("Cap", "two"),
    "inductor": ("Ind", "two"),
    "inductor-3": ("Ind", "two"),
    "switch": ("Switch", "two"),
    "switch-3": ("Switch", "two"),
    "diode": ("Diode", "diode"),
    "current": ("Current", "current"),
    "voltage": ("Voltage", "voltage"),  # circle symbol
    "voltage-lines": ("Voltage", "battery"),  # striped (battery) symbol
    "single-end-amp": ("Diso_amp", "diso"),
    "single-input-single-end-amp": ("Siso_amp", "siso"),
    "diff-amp": ("Dido_amp", "dido"),
}
NET_SYMBOLS = {"gnd": "GND", "vdd": "VDD"}  # symbols that name the net they touch
TERMINALS = {"port", "antenna"}  # I/O terminals: part of a net, not a component
BRIDGES = {"cross-line-curved"}  # hop-over crossing: never a connection

# Full detector class list (all labels found in device_identification), fixed order.
DEVICE_CLASSES = [
    "port", "resistor", "gnd", "capacitor", "nmos", "pmos", "npn", "vdd", "resistor2",
    "current", "pnp", "cross-line-curved", "switch", "voltage", "diode", "single-end-amp",
    "inductor", "nmos-bulk", "pmos-bulk", "pmos-cross", "nmos-cross", "voltage-lines",
    "single-input-single-end-amp", "resistor2_3", "diff-amp", "inductor-3", "npn-cross",
    "pnp-cross", "switch-3", "antenna", "capacitor-3",
]

# --- Crossing detector (CI2N jumper_identification, YOLO-OBB) ---
# 0: 4-way crossing without a dot (flat "+" or hop arc) - the ambiguous case
# 1: junction with a dot (connected)   2: T-junction (connected)
JUMPER_CLASSES = ["cross", "dot", "tee"]

# --- Orientation classifiers (CI2N device_orientation) ---
# Label meaning, verified on the training crops:
#   orient_mos         u/d/l/r = side the gate is on
#   orient_bjt_d       u/d/l/r = side the base is on
#   orient_bjt_m       on a crop rotated so the base is LEFT: 0 = emitter at top, 1 = at bottom
#   orient_amp_d       u/d/l/r = direction the output (triangle tip) points
#   orient_amp_m       on a crop rotated so the tip points DOWN: 0 = "+" input left, 1 = "-" left
#   orient_diode       u/d/l/r = side of the cathode (bar)
#   orient_voltagelines u/d/l/r = side of the positive (long) plate
ORIENTATION_TASKS = {
    "orient_mos": "train_mos",
    "orient_bjt_d": "train_bjt_d",
    "orient_bjt_m": "train_bjt_m",
    "orient_amp_d": "train_amp_d",
    "orient_amp_m": "train_amp_m",
    "orient_diode": "train_diode",
    "orient_voltagelines": "train_voltagelines",
}

# Pins in the order the CI2N golden netlists use them.
PINS = {
    "NMOS": ["Drain", "Gate", "Source"], "PMOS": ["Drain", "Gate", "Source"],
    "NPN": ["Collector", "Base", "Emitter"], "PNP": ["Collector", "Base", "Emitter"],
    "Res": ["Pos", "Neg"], "Cap": ["Pos", "Neg"], "Ind": ["Pos", "Neg"], "Switch": ["Pos", "Neg"],
    "Diode": ["In", "Out"], "Current": ["In", "Out"], "Voltage": ["Positive", "Negative"],
    "Diso_amp": ["InP", "InN", "Out"], "Siso_amp": ["In", "Out"],
    "Dido_amp": ["InP", "InN", "OutP", "OutN"],
}


@dataclass
class Settings:
    """Pipeline switches. The three experiment variants differ only in the first two."""

    # Crossing detection: "model" (YOLO-OBB), "geometry" (model-free, wires.find_crossings)
    # or "none" (every crossing connects, except hop arcs from the device detector).
    crossings: str = "model"
    use_orientation_models: bool = True  # v2: classifiers name the pins (else geometry rules)
    # Flat 4-way crossing in a drawing that has neither dots nor hop arcs:
    flat_crossing_default: str = "jumper"  # "jumper" (not connected) or "connected"
    merge_supply_symbols: bool = True  # all vdd symbols form one net (no OCR to tell apart)

    device_conf: float = 0.25
    jumper_conf: float = 0.25
    orientation_min_conf: float = 0.5  # below this, fall back to the geometry rule
    imgsz_devices: int = 1024
    imgsz_jumpers: int = 1024

    # Ink threshold:
    #   "otsu"       global Otsu. Ignores light-grey annotations (current arrows, maths text),
    #                but loses light-grey wires when the text is darker (validation image 050).
    #   "background" anything >= ink_contrast grey levels from the background. Finds grey wires,
    #                but grey annotations then touch wires and merge nets (v2: 29.5 % -> 23.0 %).
    #   "auto"       Otsu; redo with "background" when more than ink_fallback_nc_frac of all
    #                pins are unconnected (only 050 in validation: 93 %, next highest 12.5 %).
    ink_threshold: str = "auto"
    ink_contrast: int = 45
    ink_fallback_nc_frac: float = 0.5

    # Erase padding around boxes. 0 measured best: padding cuts wires running along boxes
    # (ground-truth-box test on 46 circuits: 8.7 % -> 15.2 % exact netlists going 4 % -> 0).
    box_pad_frac: float = 0.0
    min_box_pad_px: int = 0
    # Regions that touch only one component (and no symbol/terminal) and are smaller than
    # max(weak_region_strokes x stroke width, weak_region_frac x image side) are treated as
    # text (e.g. "M7" next to a transistor), not leads.
    weak_region_strokes: float = 8.0
    weak_region_frac: float = 0.06
    contact_ring_strokes: float = 2.0  # ring width outside a box, in stroke widths
    min_contact_ring_px: int = 3


VARIANTS = {
    "v0": Settings(crossings="none", use_orientation_models=False),
    "v1g": Settings(crossings="geometry", use_orientation_models=False),  # model-free crossings
    "v1": Settings(crossings="model", use_orientation_models=False),
    "v2": Settings(crossings="model", use_orientation_models=True),
    "v2g": Settings(crossings="geometry", use_orientation_models=True),
}
