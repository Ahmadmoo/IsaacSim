"""Check the simulated joint servo: step responses per joint and tracking error on planned references.

The arm servo is an implicit PD (PhysX joint drive) with velocity feed-forward and ideal gravity compensation.
The gains are provisional; compare scalings and adjust robot.servo_stiffness / servo_damping before collecting
data (runtime monitor threshold: 0.05 rad for 0.1 s).

    python scripts/tune_servo.py --num-envs 8
    python scripts/tune_servo.py --scales 0.5 1 2
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

from a0509pp.cli import add_common_args  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
add_common_args(parser, num_envs=8, camera_mode=False)
parser.add_argument("--step", type=float, default=0.05, help="step size [rad]")
parser.add_argument("--scales", type=float, nargs="*", default=[1.0])
parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "servo"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402

from a0509pp.cli import cfg_from_args, write_report  # noqa: E402
from a0509pp.sim.env import PickPlaceEnv  # noqa: E402

cfg = cfg_from_args(args)
env = PickPlaceEnv(cfg, num_envs=args.num_envs, camera_mode="none")
E = env.num_envs
env.reset()
report = {"gains": {"stiffness": cfg.robot.servo_stiffness, "damping": cfg.robot.servo_damping, "armature": cfg.robot.servo_armature},
          "steps": [], "tracking": []}

combos = [(j, s) for s in args.scales for j in range(6)]
for b in range(0, len(combos), E):
    chunk = combos[b:b + E]
    env.restore_initial_condition(env.ic, validate=False)
    scales = np.ones((E, 6))
    q_cmd = np.repeat(env.home_q[None], E, 0)
    for e, (j, s) in enumerate(chunk):
        scales[e] = s
        q_cmd[e, j] += args.step
    env._set_servo(list(range(E)), scales)
    env._cur_real = [None] * E
    y = np.array([env.step(q_cmd)["q"] - env.home_q[None] for _ in range(int(1.0 / env.control_dt))])
    t = (np.arange(len(y)) + 1) * env.control_dt
    for e, (j, s) in enumerate(chunk):
        r = y[:, e, j] / args.step
        t10, t90 = t[np.argmax(r >= 0.1)], t[np.argmax(r >= 0.9)]
        out = np.flatnonzero(np.abs(r - 1.0) > 0.02)
        m = {"joint": j + 1, "scale": s, "final": float(r[-1]), "overshoot": float(max(0.0, r.max() - 1.0)),
             "rise_time": float(t90 - t10) if r.max() >= 0.9 else float("nan"),
             "settling_time": float(t[out[-1]]) if len(out) and out[-1] < len(t) - 1 else float("nan"),
             "coupling_max": float(np.abs(np.delete(y[:, e], j, axis=1)).max())}
        report["steps"].append(m)
        print(f"joint {j + 1} x{s:.2f}: rise {1000 * m['rise_time']:6.1f} ms, overshoot {100 * m['overshoot']:5.1f} %, "
              f"settle {1000 * m['settling_time']:6.1f} ms, final {m['final']:.4f}")
env._set_servo(list(range(E)), np.ones((E, 6)))
env._cur_real = [None] * E

env.perceive("privileged")
cands, pool = env.propose_candidates()
reals = []
for i, s in enumerate(args.scales):
    r = env.nominal_realization()
    r.servo_scale, r.rid = [s] * 6, i
    reals.append(r)
for res in env.run_candidates(cands, reals, record_traj=False):
    m = res["monitor"]
    row = {"cid": res["cid"], "scale": reals[res["rid"]].servo_scale[0], "max_err": m["max_track_err"], "rms_err": m["rms_track_err"],
           "y_exec": res["labels"]["y_exec"], "y_task": res["labels"]["y_task"], "failure": res["labels"]["failure_type"]}
    report["tracking"].append(row)
    print(f"candidate {row['cid']} x{row['scale']:.2f}: max error {np.max(row['max_err']):.4f} rad (joint "
          f"{int(np.argmax(row['max_err'])) + 1}), exec {row['y_exec']}, task {row['y_task']} {row['failure']}")
worst = max((np.max(r["max_err"]) for r in report["tracking"] if r["scale"] == 1.0), default=float("nan"))
thr = cfg.monitor.tracking_error
report["worst_nominal_tracking_error"] = worst
report["verdict"] = "ok" if worst < 0.5 * thr else "marginal" if worst < thr else "too large"
print(f"worst nominal tracking error {worst:.4f} rad vs threshold {thr} rad: {report['verdict']}")
write_report(os.path.join(args.out, "servo_report.json"), report)
app.close()
