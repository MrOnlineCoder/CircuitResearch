"""Wire extraction: binarize, erase component/jumper boxes, label wire regions, find contacts.

A net is a connected region of wire pixels once every component box is erased. A contact
is a wire crossing the thin ring just outside a box: one contact per lead.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Settings
from .detect import Box


@dataclass
class Contact:
    x: float  # centroid, image coordinates
    y: float
    region: int  # wire region label (0 = none)
    size: int  # pixels of the contact blob in the ring
    lead_ok: bool  # the lead continues into the box (not a wire passing by)
    weak: bool = False  # small region touching nothing else (likely text); used only if needed
    extent: int = 0  # size of the wire region (max of bbox width/height)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Ink mask (True = drawn line). Otsu threshold; inverts dark-background images."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = ink > 0
    if ink.mean() > 0.5:  # mostly "ink" -> it is a dark background
        ink = ~ink
    return ink


def stroke_width(ink: np.ndarray) -> float:
    """Typical line thickness in pixels (2 x median distance-to-background of ink pixels)."""
    if not ink.any():
        return 1.0
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    return max(1.0, 2.0 * float(np.median(dist[ink])))


def padded(box: Box, s: Settings, shape) -> tuple[int, int, int, int]:
    """Integer box enlarged by the erase padding, clipped to the image."""
    pad = max(s.min_box_pad_px, s.box_pad_frac * max(box.w, box.h))
    h, w = shape
    return (max(0, int(box.x0 - pad)), max(0, int(box.y0 - pad)),
            min(w, int(np.ceil(box.x1 + pad))), min(h, int(np.ceil(box.y1 + pad))))


def erase(ink: np.ndarray, boxes, s: Settings) -> np.ndarray:
    wires = ink.copy()
    for b in boxes:
        x0, y0, x1, y1 = padded(b, s, ink.shape)
        wires[y0:y1, x0:x1] = False
    return wires


def label_regions(wires: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Region label image and each region's extent (max of bounding-box width/height)."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(wires.astype(np.uint8), connectivity=8)
    extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    return labels, extent


def find_contacts(box: Box, ink: np.ndarray, wires: np.ndarray, labels: np.ndarray,
                  s: Settings, sw: float) -> list[Contact]:
    """Wire leads touching the (padded) box: blobs of wire ink in a ring around it."""
    h, w = ink.shape
    x0, y0, x1, y1 = padded(box, s, ink.shape)
    r = max(s.min_contact_ring_px, int(round(s.contact_ring_strokes * sw)))
    X0, Y0, X1, Y1 = max(0, x0 - r), max(0, y0 - r), min(w, x1 + r), min(h, y1 + r)

    ring = np.ones((Y1 - Y0, X1 - X0), bool)
    ring[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
    blobs_mask = (wires[Y0:Y1, X0:X1] & ring).astype(np.uint8)
    n, blob_lab = cv2.connectedComponents(blobs_mask, connectivity=8)

    # Inner band: just inside the box on the ORIGINAL ink, used to confirm a lead enters it.
    inner = np.zeros_like(ring)
    band = r + 2
    inner[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = True
    core = np.zeros_like(ring)
    core[min(y1 - Y0, y0 - Y0 + band):max(y0 - Y0, y1 - Y0 - band),
         min(x1 - X0, x0 - X0 + band):max(x0 - X0, x1 - X0 - band)] = True
    inner &= ~core
    inner_ink = inner & ink[Y0:Y1, X0:X1]
    kernel = np.ones((2 * band + 1, 2 * band + 1), np.uint8)

    contacts = []
    for i in range(1, n):
        m = blob_lab == i
        ys, xs = np.nonzero(m)
        regions = labels[Y0:Y1, X0:X1][m]
        regions = regions[regions > 0]
        region = int(np.bincount(regions).argmax()) if regions.size else 0
        near = cv2.dilate(m.astype(np.uint8), kernel) > 0
        contacts.append(Contact(x=float(xs.mean() + X0), y=float(ys.mean() + Y0), region=region,
                                size=int(m.sum()), lead_ok=bool((near & inner_ink).any())))
    return contacts


def find_crossings(wires: np.ndarray, sw: float) -> list[Box]:
    """Model-free crossing/junction detector for Manhattan (horizontal/vertical) wiring.

    Long horizontal and vertical strokes are extracted with morphological opening; where they
    overlap there is a corner, T-junction or crossing. Counting the arms around each overlap
    separates 4-way crossings (label "cross") from the rest; a filled blob at the centre marks
    a dot junction ("dot"). Returns boxes in the same labels as the YOLO-OBB crossing model.
    """
    img = wires.astype(np.uint8)
    L = max(9, int(round(5 * sw)))
    horiz = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    vert = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    both = (horiz & vert).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(both, connectivity=8)
    h, w = wires.shape
    half_band = max(1, int(round(sw / 2)))
    out = []
    for i in range(1, n):
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = cents[i]
        reach = int(max(bw, bh) / 2 + 2 * sw + 2)
        arms = 0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x, y = int(round(cx + dx * reach)), int(round(cy + dy * reach))
            if not (0 <= x < w and 0 <= y < h):
                continue
            ys = slice(max(0, y - half_band), y + half_band + 1) if dx else slice(y, y + 1)
            xs = slice(max(0, x - half_band), x + half_band + 1) if dy else slice(x, x + 1)
            arms += bool(wires[ys, xs].any())
        if arms < 3:
            continue  # corner or line end
        r = int(round(1.5 * sw)) + 1
        patch = wires[max(0, int(cy) - r):int(cy) + r + 1, max(0, int(cx) - r):int(cx) + r + 1]
        is_dot = patch.size and patch.mean() > 0.75
        if is_dot:
            label = "dot"
        elif arms == 4:
            label = "cross"
        else:
            continue  # plain T-junction: connected anyway
        s = max(bw, bh) / 2 + sw
        out.append(Box(label, 1.0, cx - s, cy - s, cx + s, cy + s))
    return out


def regions_in_box(box: Box, labels: np.ndarray, s: Settings) -> list[int]:
    """Wire regions overlapping a box (for gnd/vdd symbols and terminals), largest first."""
    x0, y0, x1, y1 = padded(box, s, labels.shape)
    vals = labels[y0:y1, x0:x1].ravel()
    vals = vals[vals > 0]
    if not vals.size:
        return []
    counts = np.bincount(vals)
    return [int(v) for v in np.argsort(-counts) if counts[v] > 0]
