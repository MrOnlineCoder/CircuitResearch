"""
Convert one circuit diagram image into a CI2N netlist JSON.

    python infer.py path/to/schematic.png                 # JSON to stdout
    python infer.py img.png -o netlist.json --debug overlay.png --variant v1
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from netlist_cv import VARIANTS, Pipeline  # noqa: E402
from netlist_cv.debug import overlay  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("-o", "--out", help="write JSON here instead of stdout")
    ap.add_argument("--variant", default="v2", choices=sorted(VARIANTS))
    ap.add_argument("--models", default=str(HERE / "models"))
    ap.add_argument("--debug", help="save an overlay PNG (nets coloured, pins labelled)")
    args = ap.parse_args()

    netlist, trace = Pipeline(args.models, VARIANTS[args.variant]).run(args.image)
    text = json.dumps(netlist, indent=4)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)
    if args.debug:
        overlay(args.image, trace, args.debug)
    t = trace.timings
    print(f"{len(netlist['ckt_netlist'])} components in {t['total']:.2f}s "
          f"(detect {t['detect']:.2f}s, wires+pins {t['wires_pins']:.2f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
