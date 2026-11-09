"""Debug overlay: wires coloured by net, boxes, contacts and pin names."""

import colorsys

import cv2
import numpy as np

from .pipeline import Trace

_PIN_ABBR = {"Drain": "D", "Gate": "G", "Source": "S", "Body": "B", "Base": "B", "Emitter": "E",
             "Collector": "C", "Positive": "+", "Negative": "-", "InP": "+", "InN": "-",
             "OutP": "o+", "OutN": "o-"}


def _color(i: int):
    r, g, b = colorsys.hsv_to_rgb((i * 0.618033988) % 1.0, 0.85, 0.85)
    return int(b * 255), int(g * 255), int(r * 255)


def overlay(image_path, trace: Trace, out_path) -> None:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    canvas = (img * 0.35 + 255 * 0.65).astype(np.uint8)  # faded original
    nets = sorted(set(trace.region_net.values()))
    colors = {n: _color(i) for i, n in enumerate(nets)}
    for region, net in trace.region_net.items():
        canvas[trace.labels == region] = colors[net]

    def rect(b, color, text=None, thick=1):
        cv2.rectangle(canvas, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), color, thick)
        if text:
            cv2.putText(canvas, text, (int(b.x0), max(10, int(b.y0) - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, color, 1, cv2.LINE_AA)

    for b in trace.jumpers:
        rect(b, (0, 0, 255), "J")
    for b in trace.connected_crossings:
        rect(b, (0, 160, 0), "X")
    for b in trace.symbols + trace.terminals:
        rect(b, (120, 120, 120), b.label)
    for box, contacts, pins in trace.components:
        rect(box, (40, 40, 40), box.label, 1)
        pin_of = {i: p for p, idx in pins.items() for i in idx}
        for i, c in enumerate(contacts):
            used = i in pin_of
            cv2.circle(canvas, (int(c.x), int(c.y)), 3, (0, 0, 0) if used else (0, 0, 255), -1)
            if used:
                cv2.putText(canvas, _PIN_ABBR.get(pin_of[i], pin_of[i]), (int(c.x) + 3, int(c.y) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)
