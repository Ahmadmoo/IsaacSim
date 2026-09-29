"""Fit conservative bounding spheres to the A0509 collision meshes (one-off, no Isaac needed).

python scripts/fit_arm_spheres.py --doosan /path/to/doosan-robot2
"""

import argparse
import json
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from a0509pp.kinematics import parse_urdf  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--doosan", required=True)
parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "assets", "models", "a0509_spheres.json"))
parser.add_argument("--max_radius", type=float, default=0.07)
args = parser.parse_args()

desc = os.path.join(args.doosan, "dsr_description2")
urdf = os.path.join(desc, "urdf", "a0509.urdf")
import xml.etree.ElementTree as ET  # noqa: E402

root = ET.parse(urdf).getroot()
spheres = {}
for link in root.findall("link"):
    name = link.get("name")
    pts = []
    for col in link.findall("collision"):
        mesh = col.find("geometry/mesh")
        if mesh is None:
            continue
        fname = mesh.get("filename").replace("package://dsr_description2", desc)
        scale = np.array([float(s) for s in mesh.get("scale", "1 1 1").split()])
        m = trimesh.load(fname, force="mesh")
        o = col.find("origin")
        xyz = np.array([float(v) for v in o.get("xyz").split()]) if o is not None else np.zeros(3)
        piece = m.vertices * scale + xyz
        pts.append(piece)
    link_spheres = []
    for piece in pts:
        c = piece.mean(0)
        _, _, vt = np.linalg.svd(piece - c, full_matrices=False)
        axis = vt[0]
        t = (piece - c) @ axis
        radial = np.linalg.norm((piece - c) - np.outer(t, axis), axis=1)
        length = t.max() - t.min()
        r_guess = min(max(np.percentile(radial, 99), 0.01), args.max_radius)
        n = max(1, int(np.ceil(length / (0.5 * r_guess))))
        edges = np.linspace(t.min(), t.max(), n + 1)
        for k in range(n):
            sel = (t >= edges[k] - 1e-9) & (t <= edges[k + 1] + 1e-9)
            if not np.any(sel):
                continue
            sub = piece[sel]
            lo, hi = sub.min(0), sub.max(0)
            ctr = (lo + hi) / 2.0
            r = float(np.max(np.linalg.norm(sub - ctr, axis=1)))
            link_spheres.append([*np.round(ctr, 5).tolist(), round(r + 0.002, 5)])
    if link_spheres:
        spheres[name] = link_spheres

model = parse_urdf(urdf)
out = {"source": model["source"], "note": "spheres [x, y, z, r] in URDF link frames, 2 mm padding", "spheres": spheres,
       # The J2 housing sits about 21 mm above the base for |q2| < 2.3 rad (hull distance constant), so the sphere
       # pair gives false contacts; the pair is replaced by a joint limit rule.
       "disabled_pairs": ["base_link|link_2"],
       "disabled_pairs_note": "J2 housing sits ~21.5 mm above the base for |q2| < 2.3 rad (hull distance constant); replaced by joint_rules",
       "joint_rules": [{"joint": 1, "abs_max": 2.2, "pair": "base_link|link_2"}]}
with open(args.out, "w") as f:
    json.dump(out, f, indent=1)
print({k: len(v) for k, v in spheres.items()})

# Per link-pair offsets: the sphere model is conservative near joints; calibrate the self-distance bias
# against the URDF collision meshes' convex hulls (the shapes PhysX collides) with python-fcl.
try:
    import fcl  # noqa: F401
    from trimesh.collision import CollisionManager
except ImportError:
    print("python-fcl not installed: pair offsets skipped (self-distances stay fully conservative)")
    raise SystemExit(0)

from a0509pp.kinematics import ArmModel  # noqa: E402

arm = ArmModel(model)
names = ["base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"]
hulls = {}
for link in root.findall("link"):
    ps = []
    for col in link.findall("collision"):
        mesh = col.find("geometry/mesh")
        fname = mesh.get("filename").replace("package://dsr_description2", desc)
        ps.append(trimesh.load(fname, force="mesh").apply_scale(float(mesh.get("scale", "1").split()[0])).convex_hull)
    if ps:
        hulls[link.get("name")] = ps
rng = np.random.default_rng(0)
S = {n: np.array(spheres[n]) for n in names}
bias = {}
for _ in range(1500):
    q = rng.uniform(-np.pi, np.pi, 6)
    q[2] = rng.uniform(-2.79, 2.79)
    Ts = arm.fk_all(q)
    W = {n: S[n][:, :3] @ Ts[i][:3, :3].T + Ts[i][:3, 3] for i, n in enumerate(names)}
    for i in range(7):
        for j in range(i + 2, 7):
            a, b = names[i], names[j]
            d = np.linalg.norm(W[a][:, None] - W[b][None], axis=2) - S[a][:, 3][:, None] - S[b][:, 3][None]
            ds = float(d.min())
            if ds > 0.12:
                continue
            ca, cb = CollisionManager(), CollisionManager()
            for k, h in enumerate(hulls[a]):
                ca.add_object(f"a{k}", h, transform=Ts[i])
            for k, h in enumerate(hulls[b]):
                cb.add_object(f"b{k}", h, transform=Ts[j])
            dt = float(ca.min_distance_other(cb))
            if dt <= 0.0:
                continue
            bias.setdefault(f"{a}|{b}", []).append(dt - ds)
offsets = {k: round(max(0.0, float(np.percentile(v, 2))), 4) for k, v in bias.items() if len(v) >= 20}
out["pair_offsets"] = offsets
out["pair_offsets_note"] = "2nd percentile of (hull distance - sphere distance) over random postures with sphere distance < 0.12 m"
with open(args.out, "w") as f:
    json.dump(out, f, indent=1)
print("pair offsets:", offsets)
