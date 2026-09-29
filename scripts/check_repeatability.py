"""Repeatability, restore/replay, and time-step sensitivity.

    python scripts/check_repeatability.py --mode same --num-envs 8
    python scripts/check_repeatability.py --mode replay --dataset data/pilot --max-groups 5
    python scripts/check_repeatability.py --mode save --save outputs/dt240.json
    python scripts/check_repeatability.py --mode save --save outputs/dt480.json \\
        --set physics.dt=0.00208333333 physics.control_decimation=4 physics.camera_decimation=16
    python scripts/check_repeatability.py --mode compare --compare outputs/dt240.json outputs/dt480.json

same: identical jobs twice from the captured initial condition. replay: restore the recorded initial condition of
dataset records (validated) and re-run the stored references and realizations. save/compare: outcomes at two
physics rates on the same planned candidates.
"""

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from a0509pp.cli import add_common_args  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--mode", default="same", choices=["same", "replay", "save", "compare"])
parser.add_argument("--scenes", type=int, default=3)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--families", nargs="*", default=["object_friction", "pad_friction"])
parser.add_argument("--dataset", default=None)
parser.add_argument("--max-groups", type=int, default=5)
parser.add_argument("--save", default=None)
parser.add_argument("--compare", nargs=2, default=None)
parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "repeatability"))
add_common_args(parser, num_envs=8, camera_mode=False)
app = None
if "compare" not in sys.argv:
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
else:
    args = parser.parse_args()

import types  # noqa: E402

import numpy as np  # noqa: E402

from a0509pp.cli import cfg_from_args, write_report  # noqa: E402
from a0509pp.scene_spec import EpisodeSpec, Realization  # noqa: E402
from a0509pp.trajectory import Reference  # noqa: E402


def outcome(r, sid):
    return {"scene_id": sid, "cid": r["cid"], "rid": r["rid"], "y_task": r["labels"]["y_task"], "y_exec": r["labels"]["y_exec"],
            "failure": r["labels"]["failure_type"], "object_pose": r["final"]["object_pose"], "time": r["final"]["time"]}


def compare(a, b):
    B = {(o["scene_id"], o["cid"], o["rid"]): o for o in b}
    rows = [(o, B[k]) for o in a if (k := (o["scene_id"], o["cid"], o["rid"])) in B]
    if not rows:
        return {"n": 0}
    d = [1000 * float(np.linalg.norm(np.asarray(x["object_pose"][:3]) - np.asarray(y["object_pose"][:3]))) for x, y in rows]
    return {"n": len(rows), "task_agreement": float(np.mean([x["y_task"] == y["y_task"] for x, y in rows])),
            "exec_agreement": float(np.mean([x["y_exec"] == y["y_exec"] for x, y in rows])),
            "median_final_pos_diff_mm": float(np.median(d)), "max_final_pos_diff_mm": float(np.max(d))}


if args.mode == "compare":
    a, b = (json.load(open(p)) for p in args.compare)
    cmp = compare(a["outcomes"], b["outcomes"])
    print(json.dumps({**cmp, "dt": [a.get("dt"), b.get("dt")]}, indent=1))
    write_report(os.path.join(args.out, "compare.json"), {**cmp, "files": args.compare})
    sys.exit(0)

from a0509pp.sim.env import PickPlaceEnv  # noqa: E402

cfg = cfg_from_args(args)
env = PickPlaceEnv(cfg, num_envs=cfg.num_envs, camera_mode="none")

if args.mode in ("same", "save"):
    rows, outs = [], []
    for i in range(args.scenes):
        spec = env.sample_spec(i, args.seed, args.families, False, 2)
        env.reset(spec)
        env.perceive("privileged")
        cands, _ = env.propose_candidates()
        first = env.run_candidates(cands[:4], spec.realizations)
        outs += [outcome(r, spec.scene_id) for r in first]
        if args.mode == "save":
            continue
        second = env.run_candidates(cands[:4], spec.realizations)
        for x, y in zip(first, second):
            n = min(len(x["traj"]["t"]), len(y["traj"]["t"]))
            row = {"scene": spec.scene_id, "cid": x["cid"], "rid": x["rid"], "labels_equal": x["labels"] == y["labels"],
                   "max_joint_diff": float(np.max(np.abs(x["traj"]["q"][:n] - y["traj"]["q"][:n]))),
                   "max_object_diff": float(np.max(np.abs(x["traj"]["obj_pose"][:n, :3] - y["traj"]["obj_pose"][:n, :3])))}
            rows.append(row)
            print(f"{spec.scene_id} c{row['cid']} r{row['rid']}: labels equal {row['labels_equal']}, "
                  f"max |dq| {row['max_joint_diff']:.2e} rad, max |dp| {row['max_object_diff']:.2e} m")
    if args.mode == "save":
        write_report(args.save or os.path.join(args.out, "outcomes.json"),
                     {"dt": cfg.physics.dt, "decimation": cfg.physics.control_decimation, "outcomes": outs})
    else:
        agree = float(np.mean([r["labels_equal"] for r in rows])) if rows else float("nan")
        print(f"label agreement {agree:.3f} over {len(rows)} job pairs (GPU PhysX is not bit-exact; "
              "physics.enhanced_determinism=true tightens it)")
        write_report(os.path.join(args.out, "same.json"), {"agreement": agree, "rows": rows})

if args.mode == "replay":
    import h5py

    from a0509pp.recording import read_attr, read_index

    keys = []
    for r in read_index(args.dataset):
        k = (r["file"], r["scene_id"], r["og_id"])
        if r["valid"] == 1 and k not in keys:
            keys.append(k)
    report = []
    for fname, sid, og in keys[: args.max_groups]:
        with h5py.File(os.path.join(args.dataset, fname), "r") as f:
            sg = f[f"scenes/{sid}"]
            spec = EpisodeSpec.from_json(read_attr(sg, "spec"))
            g = sg[f"obs/{og}"]
            ic = read_attr(g["annotation"], "hidden_state")["initial_condition"]
            cands, stored, reals = [], [], {}
            for name, cg in g["candidates"].items():
                if not isinstance(cg, h5py.Group):
                    continue
                summ = read_attr(cg, "summary")
                ref = Reference(t=cg["ref_t"][()], q=cg["ref_q"][()].astype(float), qd=cg["ref_qd"][()].astype(float),
                                aperture=cg["ref_aperture"][()].astype(float), phase=cg["ref_phase"][()].astype(int))
                cands.append(types.SimpleNamespace(cid=summ["cid"], reference=ref, duration=ref.duration,
                                                   grasp=types.SimpleNamespace(width=summ["grasp"]["width"])))
                for rg in cg.get("rollouts", {}).values():
                    rz = Realization(**read_attr(rg, "realization"))
                    reals[rz.rid] = rz
                    lab, fin = read_attr(rg, "labels"), read_attr(rg, "final")
                    stored.append({"scene_id": sid, "cid": summ["cid"], "rid": rz.rid, "y_task": lab["y_task"],
                                   "y_exec": lab["y_exec"], "object_pose": fin["object_pose"]})
        env.reset(spec)
        v = env.restore_initial_condition(ic)
        env.ic = ic
        now = [outcome(r, sid) for r in env.run_candidates(cands, [reals[k] for k in sorted(reals)], record_traj=False)]
        cmp = compare(stored, now)
        print(f"{sid}/{og}: restore validation {v}; replay vs stored {cmp}")
        report.append({"scene_id": sid, "og_id": og, "ic_validation": v, **cmp})
    write_report(os.path.join(args.out, "replay.json"), report)

app.close()
