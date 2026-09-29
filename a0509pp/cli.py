"""Shared command-line handling for the scripts (no Isaac imports here)."""

from __future__ import annotations

import json
import os

import numpy as np

from .config import load_cfg

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def add_common_args(parser, num_envs=None, camera_mode=True):
    parser.add_argument("--config", default=None, help="task config JSON (defaults + overrides)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. physics.dt=0.0020833")
    parser.add_argument("--num-envs", type=int, default=num_envs, help="parallel environment copies")
    if camera_mode:
        parser.add_argument("--camera-mode", default=None, choices=["fixed", "wrist", "both", "none"])
    parser.add_argument("--manifest", default=None, help="asset manifest (default from config)")
    return parser


def cfg_from_args(args):
    cfg = load_cfg(args.config, args.set)
    if getattr(args, "num_envs", None):
        cfg.num_envs = int(args.num_envs)
    dev = getattr(args, "device", None)
    if dev:
        cfg.device = dev
    if getattr(args, "manifest", None):
        cfg.asset_manifest = args.manifest
    if not os.path.isabs(cfg.asset_manifest) and not os.path.exists(cfg.asset_manifest):
        cand = os.path.join(ROOT, cfg.asset_manifest)
        if os.path.exists(cand):
            cfg.asset_manifest = cand
    if getattr(args, "camera_mode", None):
        cfg.camera.mode = args.camera_mode
    return cfg


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def write_report(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=1)
    print(f"[report] {path}")


def check(results, name, ok, detail=""):
    results.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return ok


def save_png(path, img):
    """RGB uint8 or depth in metres (NaN = invalid) to PNG; matplotlib if available, else Pillow."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img = np.asarray(img)
    if img.ndim == 2:
        d = img.astype(float)
        v = np.isfinite(d)
        lo, hi = np.percentile(d[v], [1, 99]) if v.any() else (0.0, 1.0)
        g = np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1)
        img = np.where(v[..., None], np.stack([g, g, g], -1) * 255, 0).astype(np.uint8)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.imsave(path, img[..., :3])
    except ImportError:
        from PIL import Image

        Image.fromarray(img[..., :3].astype(np.uint8)).save(path)
