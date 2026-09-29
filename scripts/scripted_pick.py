"""Milestone 2: scripted pick-and-place. Target: at least 90 of 100 trials succeed on the default scene.

Each trial resets the scene (default block, pose jittered or sampled), perceives it (camera or privileged), plans
candidates, executes one with nominal physics, and scores the section 11 task label.

    python scripts/scripted_pick.py --trials 100
    python scripts/scripted_pick.py --trials 100 --random-pose
    python scripts/scripted_pick.py --perception privileged --camera-mode none
    python scripts/scripted_pick.py --trials 3 --viz kit
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

from a0509pp.cli import add_common_args, cfg_from_args  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
add_common_args(parser, num_envs=1)
parser.add_argument("--trials", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--random-pose", action="store_true", help="sample the block pose over the whole range")
parser.add_argument("--jitter", type=float, nargs=2, default=[0.01, 10.0], metavar=("M", "DEG"))
parser.add_argument("--perception", default=None, choices=["camera", "privileged"])
parser.add_argument("--select", default="shortest", choices=["shortest", "first", "clearance"])
parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "m2"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
cfg = cfg_from_args(args)
args.enable_cameras = cfg.camera.mode != "none"  # Isaac Lab 3.0 has no --enable_cameras flag
app = AppLauncher(args).app

import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from a0509pp.cli import write_report  # noqa: E402
from a0509pp.scene_spec import default_spec  # noqa: E402
from a0509pp.sim.env import PickPlaceEnv  # noqa: E402

src = args.perception or ("privileged" if cfg.camera.mode == "none" else cfg.perception.mode)
env = PickPlaceEnv(cfg, num_envs=1, camera_mode=cfg.camera.mode)
rng = np.random.default_rng(args.seed)
pick = {"first": lambda cs: cs[0], "shortest": lambda cs: min(cs, key=lambda c: c.duration),
        "clearance": lambda cs: max(cs, key=lambda c: min(c.clearance.get("env_min", 0.0), c.clearance.get("self_min", 0.0)))}
rows, fails = [], {}
t_start = time.time()
for i in range(args.trials):
    if args.random_pose:
        spec = env.sample_spec(i, args.seed, families=[], num_realizations=1)
    else:
        spec = default_spec(cfg, scene_id=f"m2_{args.seed:03d}_{i:04d}")
        spec.seed = args.seed * 100000 + i
        d = rng.normal(size=2)
        d *= rng.uniform(0.0, args.jitter[0]) / max(np.linalg.norm(d), 1e-9)
        spec.target.xy = (np.asarray(spec.target.xy) + d).tolist()
        spec.target.yaw = math.radians(rng.uniform(-args.jitter[1], args.jitter[1]))
        spec.target_hint_xy = list(spec.target.xy)
    t0 = time.time()
    env.reset(spec)
    percep = env.perceive(src)
    cands, pool = env.propose_candidates(percep)
    row = {"trial": i, "scene_id": spec.scene_id, "target_xy": spec.target.xy, "target_yaw": spec.target.yaw,
           "n_candidates": len(cands), "pool": pool["counts"],
           "perception_error_mm": None if percep.target is None else float(1000 * np.linalg.norm(np.asarray(percep.target.center[:2]) - np.asarray(spec.target.xy)))}
    if cands:
        c = pick[args.select](cands)
        r = env.run_candidate(c, env.nominal_realization(), record_traj=False)
        row.update({"success": r["labels"]["y_task"], "exec": r["labels"]["y_exec"], "failure": r["labels"]["failure_type"],
                    "duration": c.duration, "route": c.route, "branch": c.branch_name,
                    "max_track_err": float(np.max(r["monitor"]["max_track_err"])), "max_slip": r["monitor"]["max_slip"], "final": r["final"]})
    else:
        row.update({"success": 0, "failure": "no_candidate"})
    row["wall_s"] = time.time() - t0
    rows.append(row)
    if not row["success"]:
        fails[row["failure"]] = fails.get(row["failure"], 0) + 1
    print(f"trial {i + 1:3d}: {'ok  ' if row['success'] else 'FAIL'} {row['failure'] or ''} ({len(cands)} candidates, {row['wall_s']:.1f} s)"
          f" -> {sum(r['success'] for r in rows)}/{i + 1}", flush=True)
n_ok = sum(r["success"] for r in rows)
passed = n_ok >= math.ceil(0.9 * len(rows))
print(f"M2: {n_ok}/{len(rows)} successful, target >= 90%: {'PASS' if passed else 'FAIL'}"
      + (" | failures: " + ", ".join(f"{k} {v}" for k, v in sorted(fails.items(), key=lambda kv: -kv[1])) if fails else ""))
write_report(os.path.join(args.out, "scripted_pick.json"), {
    "successes": n_ok, "trials": len(rows), "pass": passed, "failures": fails, "perception": src, "camera_mode": env.camera_mode,
    "select": args.select, "random_pose": args.random_pose, "wall_s": time.time() - t_start, "rows": rows})
app.close()
