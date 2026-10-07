"""Milestone 2: scripted pick-and-place. Target: at least 90 of 100 trials succeed on the default scene.

Each trial resets the scene (default block, pose jittered or sampled), perceives it (camera or privileged), plans
candidates, executes one with nominal physics, and scores the section 11 task label.

    python scripts/scripted_pick.py --trials 100
    python scripts/scripted_pick.py --trials 100 --random-pose
    python scripts/scripted_pick.py --perception privileged --camera-mode none
    python scripts/scripted_pick.py --trials 3 --viz kit
    python scripts/scripted_pick.py --trials 3 --perception privileged --camera-mode both --viz kit --show-cams
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
parser.add_argument("--show-cams", action="store_true", help="live RGB | depth | events window of every camera (PNGs if OpenCV has no GUI)")
parser.add_argument("--show-hz", type=float, default=30.0, help="camera frames per simulated second (also the event frame rate)")
parser.add_argument("--event-threshold", type=float, default=0.15, help="log-intensity contrast threshold C")
parser.add_argument("--events", default="simple", choices=["simple", "evis"],
                    help="event model: built-in frame difference, or the EVIS core (pip install -e <evis repo> --no-deps)")
parser.add_argument("--event-noise", action="store_true", help="EVIS sensor noise (threshold mismatch, leak, shot, hot pixels)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
cfg = cfg_from_args(args)
args.enable_cameras = cfg.camera.mode != "none"  # Isaac Lab 3.0 has no --enable_cameras flag
app = AppLauncher(args).app

import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from a0509pp.cli import save_png, write_report  # noqa: E402
from a0509pp.scene_spec import default_spec  # noqa: E402
from a0509pp.sim.env import PickPlaceEnv  # noqa: E402

src = args.perception or ("privileged" if cfg.camera.mode == "none" else cfg.perception.mode)
env = PickPlaceEnv(cfg, num_envs=1, camera_mode=cfg.camera.mode)
rng = np.random.default_rng(args.seed)
if args.show_cams:
    if not env.cams:
        parser.error("--show-cams needs --camera-mode fixed, wrist or both")
    try:
        import cv2
    except ImportError:
        cv2 = None

    ref = {}
    if args.events == "evis":
        import torch
        from dvs_gen.dvs import BatchedMultiCamProcessor, DVSNoiseCfg, DVSNoiseModel, GeneralDVSRecorder

        recorder = GeneralDVSRecorder(os.path.join(args.out, "events"))
        procs = {c: BatchedMultiCamProcessor(recorder, c, args.event_threshold, DVSNoiseModel(
            DVSNoiseCfg(intensity_scale=255.0), args.event_threshold) if args.event_noise else None) for c in env.cams}
        for proc in procs.values():
            proc.stash_events = True

    def show(frames):
        rows = []
        for name, f in frames.items():
            d = f["depth"].astype(np.float32)
            v = np.isfinite(d) & (d > 0)
            lo, hi = np.percentile(d[v], [1, 99]) if v.any() else (0.0, 1.0)
            g = np.where(v, 255 - np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) * 255, 0).astype(np.uint8)
            dimg = cv2.applyColorMap(g, cv2.COLORMAP_TURBO)[..., ::-1] if cv2 else np.repeat(g[..., None], 3, 2)
            if args.events == "evis":
                procs[name](torch.from_numpy(f["rgb"][None].astype(np.float32)), env._tick * env.control_dt)
                m = procs[name].last_masks
                pos, neg = (m[0][0].cpu().numpy(), m[1][0].cpu().numpy()) if m else (np.zeros(g.shape, bool),) * 2
                n, d = (pos | neg) * 3.0, pos.astype(float) - neg
            else:
                L = np.log(f["rgb"].astype(np.float32) @ np.float32([0.299, 0.587, 0.114]) + 1.0)
                d = L - ref.setdefault(name, L.copy())
                n = np.floor(np.abs(d) / args.event_threshold)
                ref[name] += np.sign(d) * n * args.event_threshold
            a = (np.minimum(n, 3) / 3 * 255).astype(np.uint8)
            z = np.zeros_like(a)
            ev = np.where((d > 0)[..., None], np.stack([a, z, z], -1), np.stack([z, a // 2, a], -1))
            rows.append(np.hstack([f["rgb"], dimg, ev]))
        img = np.vstack(rows)
        try:
            cv2.imshow("cameras: RGB | depth | events", img[..., ::-1])
            cv2.waitKey(1)
        except Exception:
            save_png(os.path.join(args.out, "cams", f"{show.n:05d}.png"), img)
        show.n += 1

    show.n = 0
    env.on_frame, env.live_hz = show, args.show_hz
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
    if args.show_cams:
        ref.clear()
        if args.events == "evis":
            for proc in procs.values():
                proc.reset_envs(torch.tensor([0]))
    percep = env.perceive(src)
    cands, pool = env.propose_candidates(percep)
    row = {"trial": i, "scene_id": spec.scene_id, "target_xy": spec.target.xy, "target_yaw": spec.target.yaw,
           "n_candidates": len(cands), "pool": pool["counts"],
           "perception_error_mm": None if percep.target is None else float(1000 * np.linalg.norm(np.asarray(percep.target.center[:2]) - np.asarray(spec.target.xy)))}
    if cands:
        c = pick[args.select](cands)
        r = env.run_candidate(c, env.nominal_realization(), record_traj=False)
        if args.show_cams and args.events == "evis":
            recorder.flush_episode(0, i)
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
