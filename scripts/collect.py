"""Milestone 4: counterfactual data collection.

Per base scene: sample the layout and hidden physics, settle, capture camera observations, perceive, plan up to 8
distinct candidates, compute nominal and oracle features, then execute every candidate under every realization
(batched over the parallel envs, all from the same captured initial condition). One HDF5 record per (scene,
observation group) and one index.csv row per rollout. Re-running the same command resumes.

Views: an observation group sees one camera subset (fixed, wrist or both). With --candidates shared, one candidate set
(planned from --plan-view) and its rollouts are recorded under every view, which isolates selection. With
--candidates regenerated (default), each view runs its own perception, planning and rollouts (full-system test).

    python scripts/collect.py --config configs/pilot.json
    python scripts/collect.py --num-scenes 100 --num-envs 24 --out data/pilot
    python scripts/collect.py --views fixed wrist both --candidates shared --plan-view both --out data/views
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

from a0509pp.cli import add_common_args, cfg_from_args  # noqa: E402
from a0509pp.scene_spec import make_object_pool  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
add_common_args(parser)
parser.add_argument("--num-scenes", type=int, default=None)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--out", default=None)
parser.add_argument("--families", nargs="*", default=None, help="randomization families (a0509pp/scene_spec.py)")
parser.add_argument("--realizations", type=int, default=None)
parser.add_argument("--perception", nargs="*", default=None, choices=["camera", "privileged"])
parser.add_argument("--views", nargs="*", default=None, choices=["fixed", "wrist", "both"])
parser.add_argument("--candidates", default="regenerated", choices=["regenerated", "shared"])
parser.add_argument("--plan-view", default=None, choices=["fixed", "wrist", "both"])
parser.add_argument("--fixed-pose", action="store_true")
parser.add_argument("--object-pool", type=int, default=6, help="block sizes when the 'object' family is on")
parser.add_argument("--shard-size", type=int, default=50)
parser.add_argument("--no-traj", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

VIEW_CAMS = {"fixed": ["fixed_camera"], "wrist": ["wrist_camera"], "both": ["fixed_camera", "wrist_camera"]}

cfg = cfg_from_args(args)
C = cfg.collect
C.num_scenes = args.num_scenes or C.num_scenes
C.seed = C.seed if args.seed is None else args.seed
C.out_dir = args.out or C.out_dir
C.realizations = args.realizations or C.realizations
C.perception_sources = list(args.perception or C.perception_sources)
if args.families is not None:
    cfg.randomization.enabled = list(args.families)
if "object" in cfg.randomization.enabled and len(cfg.layout.object_pool_dims) == 1:
    cfg.layout.object_pool_dims = make_object_pool(cfg, k=args.object_pool, seed=C.seed)
views = list(args.views or ([] if cfg.camera.mode == "none" else [cfg.camera.mode]))
view_cams = {c for v in views for c in VIEW_CAMS[v]}
plan_view = args.plan_view or ("both" if len(view_cams) == 2 else (views[0] if views else None))
needed = {c for v in views + ([plan_view] if plan_view else []) for c in VIEW_CAMS[v]}
cfg.camera.mode = "both" if len(needed) == 2 else ("fixed" if "fixed_camera" in needed else "wrist" if needed else "none")
if not views:
    C.perception_sources = ["privileged"]
    views = ["none"]
out = C.out_dir if os.path.isabs(C.out_dir) else os.path.join(ROOT, C.out_dir)
args.enable_cameras = cfg.camera.mode != "none"  # Isaac Lab 3.0 has no --enable_cameras flag
app = AppLauncher(args).app

import csv  # noqa: E402
import time  # noqa: E402

from a0509pp.cli import write_report  # noqa: E402
from a0509pp.config import DATASET_VERSION, PROVISIONAL  # noqa: E402
from a0509pp.features import FEATURE_NAMES  # noqa: E402
from a0509pp.recording import DatasetWriter  # noqa: E402
from a0509pp.sim.env import PickPlaceEnv, _stable_hash  # noqa: E402
from a0509pp.trajectory import PHASES  # noqa: E402

env = PickPlaceEnv(cfg, num_envs=cfg.num_envs, camera_mode=cfg.camera.mode)
man = env.manifest
writer = DatasetWriter(out, shard_size=args.shard_size, manifest={
    "dataset_version": DATASET_VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": cfg.to_dict(),
    "provisional": PROVISIONAL, "feature_names": FEATURE_NAMES, "phases": PHASES, "families": cfg.randomization.enabled,
    "camera_mode": env.camera_mode, "views": views, "candidates": args.candidates, "plan_view": plan_view,
    "perception_sources": C.perception_sources,
    "asset_manifest": {k: man.get(k) for k in ("isaac_sim_version", "isaaclab_version", "sources", "tcp_definition",
                                               "gripper_calibration", "kinematics_max_diff_vs_snapshot")},
    "home": {"q": env.home_q.tolist(), **env.home_info}, "gripper": env.gripper.summary(),
    "labels": {"y_task": "inside tray footprint with margin, supported, settled, released, withdrawn, within 20 s, no execution "
                         "violation or simulator error",
               "y_exec": "reference completed without forbidden contact, tracking/joint-limit violation, stall or timeout"}})
done = set()
if os.path.exists(os.path.join(out, "index.csv")):
    with open(os.path.join(out, "index.csv")) as f:
        done = {(r["scene_id"], r["og_id"]) for r in csv.DictReader(f)}

stats = {"scenes": 0, "groups": 0, "rollouts": 0, "y_task": 0, "y_exec": 0, "no_candidate": 0, "sim_error": 0, "failures": {},
         "t_reset": 0.0, "t_plan": 0.0, "t_feat": 0.0, "t_roll": 0.0}


def og_name(view, src):
    shared = src == "privileged" or args.candidates == "shared"
    return f"{view}_{src}" + (f"_shared-{plan_view}" if shared and src == "camera" else "")


def run_group(src, plan_cams, video):
    t1 = time.time()
    cands, pool = env.propose_candidates(env.perceive(src, plan_cams))
    t2 = time.time()
    feats, _, metas = env.evaluate_physics(cands, "nominal")
    feats_o, _, _ = env.evaluate_physics(cands, "oracle")
    t3 = time.time()
    results = env.run_candidates(cands, spec.realizations, record_traj=not args.no_traj, video=video)
    t4 = time.time()
    timing = {"planning_s": t2 - t1, "features_s": t3 - t2, "rollouts_s": t4 - t3}
    for k, v in (("t_plan", t2 - t1), ("t_feat", t3 - t2), ("t_roll", t4 - t3)):
        stats[k] += v
    stats["no_candidate"] += int(not cands)
    for r in results:
        lab = r["labels"]
        stats["rollouts"] += 1
        stats["y_task"] += lab["y_task"]
        stats["y_exec"] += lab["y_exec"]
        stats["sim_error"] += int(lab["simulator_error"])
        if lab["failure_type"]:
            stats["failures"][lab["failure_type"]] = stats["failures"].get(lab["failure_type"], 0) + 1
    print(f"    {src} (planned from {plan_cams or 'hidden state'}): {len(cands)} candidates, task "
          f"{sum(r['labels']['y_task'] for r in results)}/{len(results)} | plan {t2 - t1:.1f} s, features {t3 - t2:.1f} s, "
          f"sim {t4 - t3:.1f} s", flush=True)
    return cands, pool, results, feats, feats_o, metas, timing


t_all = time.time()
print(f"[collect] {C.num_scenes} scenes, <= {cfg.candidates.max_candidates} candidates x {C.realizations} realizations, "
      f"{env.num_envs} envs, views {views}, candidates {args.candidates}, perception {C.perception_sources}, "
      f"families {cfg.randomization.enabled}")
for i in range(args.start, args.start + C.num_scenes):
    spec = env.sample_spec(i, C.seed, cfg.randomization.enabled, args.fixed_pose, C.realizations)
    todo = [(v, s) for s in C.perception_sources for v in views if (spec.scene_id, og_name(v, s)) not in done]
    if not todo:
        continue
    video = C.video_fraction > 0 and _stable_hash(spec.scene_id) % 10000 < C.video_fraction * 10000
    t0 = time.time()
    env.reset(spec)
    stats["t_reset"] += time.time() - t0
    print(f"[{i - args.start + 1}/{C.num_scenes}] {spec.scene_id} ({spec.split})", flush=True)
    for src in C.perception_sources:
        vs = [v for v, s in todo if s == src]
        if not vs:
            continue
        if src == "privileged" or args.candidates == "shared":
            g = run_group(src, VIEW_CAMS.get(plan_view) if src == "camera" else None, video)
            for v in vs:
                cs = "privileged" if src == "privileged" else f"shared:{plan_view}"
                writer.write_scene(env.record_episode(*g[:6], og_name(v, src), views=VIEW_CAMS.get(v, []), candidate_set=cs, timing=g[6]))
                stats["groups"] += 1
        else:
            for v in vs:
                g = run_group(src, VIEW_CAMS[v], video)
                writer.write_scene(env.record_episode(*g[:6], og_name(v, src), views=VIEW_CAMS[v], candidate_set="regenerated", timing=g[6]))
                stats["groups"] += 1
    stats["scenes"] += 1
    if stats["scenes"] % 10 == 0:
        el = time.time() - t_all
        print(f"[collect] {el / 60:.1f} min elapsed, ~{(C.num_scenes - (i - args.start + 1)) * el / stats['scenes'] / 60:.1f} min left, "
              f"task rate so far {stats['y_task'] / max(stats['rollouts'], 1):.3f}", flush=True)
writer.close()
stats["wall_s"] = time.time() - t_all
stats["task_rate"] = stats["y_task"] / max(stats["rollouts"], 1)
stats["exec_rate"] = stats["y_exec"] / max(stats["rollouts"], 1)
write_report(os.path.join(out, "collect_summary.json"), stats)
print(f"[collect] {stats['scenes']} scenes, {stats['rollouts']} rollouts, task {stats['task_rate']:.3f}, exec {stats['exec_rate']:.3f}, "
      f"{stats['wall_s'] / 60:.1f} min")
app.close()
