"""Check a prepared robot bundle after copying it to another machine or directory.

    python scripts/check_asset_bundle.py --manifest /path/to/generated/asset_manifest.json

Run with the Isaac Lab Python environment (for pxr). No simulator or GPU is needed.
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from a0509pp.models import load_manifest
from a0509pp.sim.asset_builder import validate_asset_bundle

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest", default=os.path.join(ROOT, "assets", "generated", "asset_manifest.json"))
args = parser.parse_args()

manifest = load_manifest(args.manifest)
if not manifest:
    parser.error(f"asset manifest not found: {args.manifest}")
root = os.path.dirname(manifest["_path"])
names = ("arm_usd", "robotiq_usd", "robot_usd", "robot_usd_no_camera")
missing = [name for name in names if not manifest.get(name)]
if missing:
    parser.error(f"manifest lacks {missing}; regenerate assets with scripts/prepare_assets.py")
report = validate_asset_bundle(root, [manifest[name] for name in names])
print(f"[bundle] OK: {root}")
for name, counts in report.items():
    print(f"  {name}: {counts['layers']} USD layers, {counts['assets']} other assets")
