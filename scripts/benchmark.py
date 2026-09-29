"""Throughput for 1/8/16/32 parallel envs (one subprocess per size): control ticks per second while holding,
rollout throughput on planned candidates, and reset + camera capture time.

    python scripts/benchmark.py --num-envs-list 1 8 16 32
    python scripts/benchmark.py --num-envs-list 16 --camera-mode fixed
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num-envs-list", type=int, nargs="+", default=[1, 8, 16, 32])
parser.add_argument("--ticks", type=int, default=600)
parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "benchmark"))
known, extra = parser.parse_known_args()

if len(known.num_envs_list) > 1:
    rows = []
    for n in known.num_envs_list:
        subprocess.run([sys.executable, os.path.abspath(__file__), "--num-envs-list", str(n), "--ticks", str(known.ticks),
                        "--out", known.out] + extra, check=False)
        path = os.path.join(known.out, f"bench_{n}.json")
        if os.path.exists(path):
            rows.append(json.load(open(path)))
    print(" envs | ticks/s | env-ticks/s | rollouts/h | reset s")
    for r in rows:
        print(f" {r['num_envs']:4d} | {r['ticks_per_s']:7.1f} | {r['env_ticks_per_s']:11.1f} | {r['rollouts_per_hour']:10.0f} | {r['reset_s']:.2f}")
    with open(os.path.join(known.out, "benchmark.json"), "w") as f:
        json.dump(rows, f, indent=1)
    sys.exit(0)

from isaaclab.app import AppLauncher  # noqa: E402

from a0509pp.cli import add_common_args, cfg_from_args  # noqa: E402

add_common_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
cfg = cfg_from_args(args)
args.enable_cameras = cfg.camera.mode != "none"  # Isaac Lab 3.0 has no --enable_cameras flag
app = AppLauncher(args).app

import time  # noqa: E402

from a0509pp.cli import write_report  # noqa: E402
from a0509pp.sim.env import PickPlaceEnv  # noqa: E402

n = args.num_envs_list[0]
t0 = time.time()
env = PickPlaceEnv(cfg, num_envs=n, camera_mode=cfg.camera.mode, verbose=False)
t_build = time.time() - t0
t0 = time.time()
env.reset()
t_reset = time.time() - t0
t0 = time.time()
env.hold(args.ticks)
dt_hold = time.time() - t0
env.perceive("privileged")
cands, _ = env.propose_candidates()
reals = []
while cands and (not reals or len(cands) * len(reals) < n):
    r = env.nominal_realization()
    r.rid = len(reals)
    reals.append(r)
t0 = time.time()
res = env.run_candidates(cands, reals, record_traj=False) if cands else []
dt_roll = time.time() - t0
out = {"num_envs": n, "camera_mode": env.camera_mode, "build_s": t_build, "reset_s": t_reset,
       "ticks_per_s": args.ticks / dt_hold, "env_ticks_per_s": n * args.ticks / dt_hold,
       "realtime_factor": args.ticks * env.control_dt / dt_hold, "rollouts": len(res), "rollout_wall_s": dt_roll,
       "rollouts_per_hour": 3600.0 * len(res) / dt_roll if dt_roll > 0 else 0.0}
print(json.dumps(out, indent=1))
write_report(os.path.join(args.out, f"bench_{n}.json"), out)
app.close()
