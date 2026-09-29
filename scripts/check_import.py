"""Milestone 1: the robot USD imports as one articulation, holds still, and matches the kinematic model.

Checks joint/body inventory, limits and drive settings, masses, drift and self-contact while holding home, forward
kinematics of the simulated gripper base versus the numpy model (home + random postures), the 2F-85 mimic
coupling, and camera images. Writes outputs/m1/report.json and PNG snapshots.

    python scripts/check_import.py
    python scripts/check_import.py --viz kit
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
parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "m1"))
parser.add_argument("--postures", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
cfg = cfg_from_args(args)
args.enable_cameras = cfg.camera.mode != "none"  # Isaac Lab 3.0 has no --enable_cameras flag
app = AppLauncher(args).app

import numpy as np  # noqa: E402

from a0509pp.cli import check, save_png, write_report  # noqa: E402
from a0509pp.geometry import pose7_to_T_batch, so3_log  # noqa: E402
from a0509pp.sim.env import PickPlaceEnv, _np, _torch  # noqa: E402


def body_T(env, names):
    ids = env.robot.find_bodies(names, preserve_order=True)[0]
    p = _np(_torch(env.robot.data.body_link_pose_w)[0, ids]).astype(np.float64)
    p[:, :3] -= env.origins[0]
    return pose7_to_T_batch(p)


env = PickPlaceEnv(cfg, num_envs=1, camera_mode=cfg.camera.mode)
res = []
rd = env.robot.data

check(res, "six arm joints", len(env.arm_ids) == 6, str([env.joint_names[i] for i in env.arm_ids]))
check(res, "finger + passive joints", len(env.passive_ids) == len(cfg.gripper.passive_joints),
      str([env.joint_names[env.finger_id]] + [env.joint_names[i] for i in env.passive_ids]))
check(res, "mimic coupling read from USD", bool(env.manifest.get("gripper_mimic")))
lim = _np(_torch(rd.joint_pos_limits)[0])[env.arm_ids]
check(res, "arm joint limits match URDF", np.allclose(lim[:, 0], env.arm.q_min, atol=1e-3) and np.allclose(lim[:, 1], env.arm.q_max, atol=1e-3),
      str(np.round(lim, 4).tolist()))
stiff = _np(_torch(rd.joint_stiffness)[0])
eff = _np(_torch(rd.joint_effort_limits)[0])
check(res, "arm drive gains applied", np.allclose(stiff[env.arm_ids], cfg.robot.servo_stiffness, rtol=1e-3), str(stiff[env.arm_ids].round(1).tolist()))
check(res, "finger torque limit", abs(eff[env.finger_id] - env.grip_torque) < 1e-3, f"{eff[env.finger_id]:.3f} N*m for {cfg.gripper.grip_force} N")
masses = _np(_torch(rd.body_mass)[0])
want = env.manifest.get("prim_paths", {}).get("gripper_base_mass", {}).get("mass")
if want:
    m_gb = masses[env.body_names.index(cfg.gripper.base_link)]
    check(res, "gripper base assembly mass", abs(m_gb - want) < 1e-3, f"{m_gb:.4f} kg (expected {want:.4f})")
arm_bodies = [cfg.robot.arm_base_link] + [f"link_{i}" for i in range(1, 7)]
sim_m = np.array([masses[env.body_names.index(b)] if b in env.body_names else np.nan for b in arm_bodies])
check(res, "arm link masses match URDF", np.allclose(sim_m, env.arm.masses, rtol=0.02), f"sim {np.round(sim_m, 3).tolist()}")

env.reset()
check(res, "object settles", env.settle_info["settled"], str(env.settle_info))
check(res, "initial condition broadcast", env.ic_report["ok"], str(env.ic_report))
peak = np.zeros(len(env.cs_names))
for _ in range(240):
    env._physics_steps()
    st = env.read_state(env.k_target)
    peak = np.maximum(peak, st["link_force"][0].max(0))
drift = float(np.max(np.abs(st["q"][0] - env.home_q)))
check(res, "home drift < 2e-3 rad over 2 s", drift < 2e-3, f"{drift:.2e} rad")
nb = [i for i, c in enumerate(env.body_classes) if c != "base"]
check(res, "no robot contact at home", peak[nb].max() < cfg.monitor.contact_force,
      f"max {peak[nb].max():.2f} N on {env.cs_names[nb[int(np.argmax(peak[nb]))]]}")
err = np.linalg.norm(st["obj_pose"][0, :2] - np.asarray(env.spec.target.xy))
check(res, "block stays at its spawn pose", err < 0.002, f"{err * 1000:.2f} mm")

rng = np.random.default_rng(0)
qs = [env.home_q] + [np.clip(env.home_q + rng.uniform(-1.2, 1.2, 6), env.arm.q_min, env.arm.q_max) for _ in range(args.postures)]
errs = []
for q in qs:
    env.write_robot_state(q, 0.0)
    env.sim.forward()
    env.scene.update(0.0)
    T_sim = body_T(env, [cfg.gripper.base_link])[0]
    T_mod = env.arm.fk(q) @ env.gripper.T_flange_base()
    errs.append((np.linalg.norm(T_sim[:3, 3] - T_mod[:3, 3]), np.linalg.norm(so3_log(T_mod[:3, :3].T @ T_sim[:3, :3]))))
errs = np.array(errs)
check(res, "FK position error < 0.5 mm", errs[:, 0].max() < 5e-4, f"max {errs[:, 0].max() * 1000:.3f} mm")
check(res, "FK rotation error < 1 mrad", errs[:, 1].max() < 1e-3, f"max {errs[:, 1].max() * 1000:.3f} mrad")

env.restore_initial_condition(env.ic)
for _ in range(60):
    env.step(env.home_q, None, cfg.gripper.open_aperture)
tips0, base0 = body_T(env, ["left_fingertip", "right_fingertip"]), body_T(env, [cfg.gripper.base_link])[0]
th_cmd = 0.5 * cfg.gripper.joint_upper
for _ in range(120):
    st = env.step(env.home_q, None, float(env.gripper.aperture(th_cmd)))
jp = st["joint_pos"][0]
th = jp[env.finger_id]
coupling = np.abs(jp[env.passive_ids] - (th * env.mimic_coef + env.mimic_off))
check(res, "finger joint tracks the aperture command", abs(th - th_cmd) < 0.02, f"{th:.4f} vs {th_cmd:.4f} rad")
check(res, "passive joints on the mimic coupling", coupling.max() < 0.01, f"max {coupling.max():.4f} rad")
tips1, base1 = body_T(env, ["left_fingertip", "right_fingertip"]), body_T(env, [cfg.gripper.base_link])[0]
tilt = [np.linalg.norm(so3_log((base0[:3, :3].T @ a[:3, :3]).T @ (base1[:3, :3].T @ b[:3, :3]))) for a, b in zip(tips0, tips1)]
check(res, "pads stay parallel while closing", max(tilt) < 0.02, f"{np.round(tilt, 4).tolist()} rad")

obs = env.observe()
for name in env.cams:
    pol = obs["policy"]
    save_png(os.path.join(args.out, f"{name}_rgb.png"), pol[f"{name}_rgb"][-1])
    save_png(os.path.join(args.out, f"{name}_depth.png"), pol[f"{name}_depth"][-1].astype(np.float32))
    frac = float(pol[f"{name}_depth_valid"][-1].mean())
    check(res, f"{name}: depth mostly valid", frac > 0.5, f"{frac * 100:.1f}% valid")
    check(res, f"{name}: image not blank", float(pol[f"{name}_rgb"][-1].std()) > 3.0)

write_report(os.path.join(args.out, "report.json"), {
    "checks": res, "joint_names": env.joint_names, "body_names": env.body_names, "contact_bodies": env.cs_names,
    "contact_classes": env.body_classes, "home_q": env.home_q, "home_info": env.home_info,
    "body_masses": dict(zip(env.body_names, masses.tolist())), "hold_peak_contact": dict(zip(env.cs_names, peak.tolist())),
    "fk_errors": errs})
print(f"M1: {sum(r['pass'] for r in res)}/{len(res)} checks passed")
app.close()
