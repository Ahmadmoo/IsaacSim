"""Milestone 3: cameras and contact sensors.

Cameras: depth geometry against the true scene (table plane, block top), perception error, 30 Hz frame spacing.
Contacts: resting block weight, finger loads during a real grasp, and detection of a deliberate forbidden contact
(TCP pushed into the table) by the runtime monitor.

    python scripts/check_sensors.py --camera-mode both
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

from a0509pp.cli import add_common_args  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
add_common_args(parser, num_envs=2)
parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "m3"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.camera_mode = args.camera_mode or "both"
args.enable_cameras = args.camera_mode != "none"  # Isaac Lab 3.0 has no --enable_cameras flag
app = AppLauncher(args).app

import types  # noqa: E402

import numpy as np  # noqa: E402

from a0509pp.candidates import tcp_rot  # noqa: E402
from a0509pp.cli import cfg_from_args, check, save_png, write_report  # noqa: E402
from a0509pp.geometry import make_T, quat_to_mat  # noqa: E402
from a0509pp.perception import backproject  # noqa: E402
from a0509pp.sim.env import PickPlaceEnv  # noqa: E402
from a0509pp.trajectory import PHASE_ID, JointPath, Segment, build_reference, time_scale  # noqa: E402

cfg = cfg_from_args(args)
env = PickPlaceEnv(cfg, num_envs=max(2, args.num_envs or 2), camera_mode=args.camera_mode)
res, rep = [], {}
env.reset()
spec, obs = env.spec, env.observe()
ann, dims = obs["annotation"], np.asarray(env.spec.target.dims)

for name in env.cams:
    pol = obs["policy"]
    save_png(os.path.join(args.out, f"{name}_rgb.png"), pol[f"{name}_rgb"][-1])
    save_png(os.path.join(args.out, f"{name}_depth.png"), pol[f"{name}_depth"][-1].astype(np.float32))
    view = next(v for v in obs["views"] if v["name"] == name)
    P = backproject(view["depth"], ann[f"{name}_K_full"], ann[f"{name}_T_world_cam_true"][-1], view["valid"], stride=2)
    op = ann["object_pose"]
    loc = (P - op[:3]) @ quat_to_mat(op[3:])
    near = np.all(np.abs(loc[:, :2]) < dims[:2] / 2 + 0.02, axis=1)
    tray = (np.abs(P[:, 0] - spec.tray_xy[0]) < cfg.layout.tray_interior[0] / 2 + 0.03) & (np.abs(P[:, 1] - spec.tray_xy[1]) < cfg.layout.tray_interior[1] / 2 + 0.03)
    table = (np.abs(P[:, 2]) < 0.02) & ~near & ~tray & (P[:, 0] > 0.2)
    if table.sum() > 100:
        z = P[table, 2]
        check(res, f"{name}: table plane from depth", abs(np.median(z)) < 0.002 and np.std(z) < 0.004,
              f"median {1000 * np.median(z):.2f} mm, std {1000 * np.std(z):.2f} mm")
    top = np.all(np.abs(loc[:, :2]) < dims[:2] / 2 - 0.005, axis=1) & (P[:, 2] > 0.5 * dims[2])
    if top.sum() > 20:
        zb = np.median(P[top, 2])
        check(res, f"{name}: block top height", abs(zb - dims[2]) < 0.003, f"{1000 * zb:.1f} vs {1000 * dims[2]:.1f} mm")
    ft = pol[f"{name}_frame_t"]
    if len(ft) > 1:
        check(res, f"{name}: 30 Hz frames", np.allclose(np.diff(ft), 1.0 / 30.0, atol=1e-3), str(np.round(np.diff(ft), 4).tolist()))
if env.cams:
    pc = env.perceive("camera")
    check(res, "camera perception finds the block", pc.target is not None)
    if pc.target is not None:
        e = np.linalg.norm(pc.target.center[:2] - ann["object_pose"][:2])
        check(res, "camera perception: block centre", e < 0.005, f"{1000 * e:.2f} mm")
        check(res, "camera perception: block size", np.max(np.abs(pc.target.dims - dims)) < 0.006, f"{np.round(1000 * pc.target.dims, 1).tolist()} mm")
        rep["perception"] = pc.to_dict()

env.hold(30)
st = env.read_state(env.k_target)
w = env.nominal_realization().object_mass * abs(cfg.physics.gravity[2])
f = float(st["obj_env_force"][0].mean())
check(res, "block weight carried by the table", abs(f - w) < 0.15 * w + 0.05, f"{f:.3f} N vs m*g {w:.3f} N")

env.perceive("privileged")
cands, _ = env.propose_candidates()
check(res, "planner returns candidates", len(cands) > 0, str(len(cands)))
if cands:
    r = env.run_candidate(min(cands, key=lambda c: c.duration), env.nominal_realization())
    tr = r["traj"]
    carry = (tr["phase"] >= PHASE_ID["lift"]) & (tr["phase"] <= PHASE_ID["transfer"])
    if carry.any():
        fm = float(np.median(tr["pad_obj_force"][carry]))
        check(res, "both fingers load the block while carrying", fm > cfg.monitor.grasp_min_force, f"median min-side {fm:.2f} N")
        tau = env.grip_torque * cfg.gripper.grip_force / max(fm, 1e-6)
        rep["grip_force"] = {"measured_N": fm, "setting_N": cfg.gripper.grip_force, "torque_limit": env.grip_torque,
                             "torque_for_setting": tau}
        print(f"  grip force {fm:.1f} N at torque limit {env.grip_torque:.2f} N*m; for {cfg.gripper.grip_force:.0f} N use "
              f"--set gripper.max_torque={tau:.2f}")
    check(res, "grasp executes without forbidden contact", r["labels"]["y_exec"] == 1, f"{r['labels']['failure_type'] or 'ok'}, task {r['labels']['y_task']}")
    rep["grasp"] = {k: r[k] for k in ("labels", "monitor", "final")}

pl = env.models.planner
a = cfg.gripper.open_aperture
T_top = make_T(tcp_rot(0.0), [0.35, -0.30, 0.12])
T_bot = make_T(tcp_rot(0.0), [0.35, -0.30, env.gripper.pad_bottom_extent() - 0.01])
q_top = pl.ik_near(T_top, env.home_q)
Q = None if q_top is None else pl.cartesian(T_top, T_bot, q_top)
if Q is not None:
    segs = [Segment("transition", JointPath(np.array([env.home_q, q_top])), 0.0, a, a), Segment("descend", JointPath(Q), 0.0, a, a)]
    for s in segs:
        s.profile = time_scale(s.path, env.arm, env.T_flange_tcp, pl.caps, env.control_dt)
        s.duration = s.profile.T
    ref = build_reference(segs, cfg.gripper.speed, env.control_dt)
    bad = types.SimpleNamespace(cid=99, reference=ref, grasp=types.SimpleNamespace(width=0.04), duration=ref.duration)
    r = env.run_candidate(bad, env.nominal_realization(), record_traj=False)
    kinds = [e["kind"] for e in r["monitor"]["events"]]
    check(res, "monitor flags the TCP pushed into the table", "forbidden_contact" in kinds, str(kinds))
    rep["press"] = r["monitor"]
else:
    check(res, "press-test reference", False, "IK failed")

write_report(os.path.join(args.out, "report.json"), {**rep, "checks": res})
print(f"M3: {sum(r['pass'] for r in res)}/{len(res)} checks passed")
app.close()
