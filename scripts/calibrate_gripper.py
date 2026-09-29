"""Measure the 2F-85 aperture map in simulation and register it in the asset manifest.

Sweeps finger_joint from open to closed in free space (arm at home), reads both fingertip bodies relative to the
gripper base, and converts them to pad inner-face separation and pinch-centre height using the pad faces measured
from the USD meshes by prepare_assets.py. The table (theta -> aperture, pad-centre z) defines the command mapping
(aperture in metres -> finger_joint) and the TCP (pinch centre at the reference aperture).

    python scripts/calibrate_gripper.py
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

from a0509pp.cli import add_common_args  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
add_common_args(parser, num_envs=1, camera_mode=False)
parser.add_argument("--steps", type=int, default=48)
parser.add_argument("--settle", type=float, default=0.4, help="hold time per sample [s]")
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from a0509pp.cli import cfg_from_args, write_report  # noqa: E402
from a0509pp.geometry import inv_T, pose7_to_T_batch, so3_log  # noqa: E402
from a0509pp.gripper_model import GripperModel  # noqa: E402
from a0509pp.models import Models, load_manifest, save_manifest_field  # noqa: E402
from a0509pp.sim.env import PickPlaceEnv, _np, _torch  # noqa: E402

cfg = cfg_from_args(args)
cfg.gripper.calibration_file = ""
env = PickPlaceEnv(cfg, num_envs=1, camera_mode="none")
geom = env.manifest.get("gripper_geometry", {}).get("bodies", {})
if "left_fingertip" not in geom or "right_fingertip" not in geom:
    sys.exit("manifest has no fingertip bounds; re-run scripts/prepare_assets.py")
L, R = geom["left_fingertip"], geom["right_fingertip"]
pL0 = np.array([(L["min"][0] + L["max"][0]) / 2, L["max"][1], (L["min"][2] + L["max"][2]) / 2, 1.0])
pR0 = np.array([(R["min"][0] + R["max"][0]) / 2, R["min"][1], (R["min"][2] + R["max"][2]) / 2, 1.0])
ids = env.robot.find_bodies(["left_fingertip", "right_fingertip", cfg.gripper.base_link], preserve_order=True)[0]

env.reset()
env.set_targets(env.home_q[None], np.zeros((1, 6)), [cfg.gripper.open_aperture])
n_hold = int(round(args.settle / env.control_dt))
TL0 = TR0 = None
rows = []
for th in np.linspace(0.0, cfg.gripper.joint_upper * 0.995, args.steps):
    val = torch.full((1, 1), float(th), device=env.device)
    if env._tc is not None:
        env._tc.set_position_index(value=val, joint_ids=env.wp_finger)
    else:
        env.robot.set_joint_position_target_index(target=val, joint_ids=env.wp_finger)
    env.hold(n_hold)
    th_meas = float(_np(_torch(env.robot.data.joint_pos)[0, env.finger_id]))
    T = pose7_to_T_batch(_np(_torch(env.robot.data.body_link_pose_w)[0, ids]).astype(np.float64))
    TL, TR = inv_T(T[2]) @ T[0], inv_T(T[2]) @ T[1]
    if TL0 is None:
        TL0, TR0 = TL, TR
    pL, pR = TL @ inv_T(TL0) @ pL0, TR @ inv_T(TR0) @ pR0
    tilt = max(np.linalg.norm(so3_log(TL0[:3, :3].T @ TL[:3, :3])), np.linalg.norm(so3_log(TR0[:3, :3].T @ TR[:3, :3])))
    rows.append((th_meas, pR[1] - pL[1], 0.5 * (pL[2] + pR[2]), tilt))
rows = np.array(sorted(rows))
rows = rows[np.concatenate([[True], np.diff(rows[:, 0]) > 1e-5])]
cal = {"theta": rows[:, 0].tolist(), "aperture": rows[:, 1].tolist(), "pad_center_z": rows[:, 2].tolist(),
       "pad_tilt": rows[:, 3].tolist(), "source": "calibrate_gripper.py: simulated sweep, pad faces from USD mesh bounds"}
if np.any(np.diff(rows[:, 1]) > 1e-4):
    print("[calibrate] WARNING: aperture is not monotonic in theta; check the mimic coupling (check_import.py)")
if rows[:, 3].max() > 0.02:
    print("[calibrate] WARNING: pads rotate while closing; the parallel-grip variant should keep them parallel")
out = args.out or os.path.join(os.path.dirname(env.manifest["_path"]), "gripper_calibration.json")
write_report(out, cal)
g = GripperModel(cfg.gripper, cfg.robot, cal)
print(f"[calibrate] aperture {1000 * rows[0, 1]:.1f} -> {1000 * rows[-1, 1]:.1f} mm, TCP z {g.tcp_z():.5f} m, "
      f"finger torque limit {g.max_torque():.3f} N*m for {cfg.gripper.grip_force:.0f} N")
save_manifest_field(env.manifest["_path"], "gripper_calibration", out)
cfg.gripper.calibration_file = out
man = load_manifest(env.manifest["_path"])
man.pop("home", None)
models = Models(cfg, man)
q_home, info = models.solve_home()
save_manifest_field(env.manifest["_path"], "home", {"q": q_home.tolist(), "info": info, "tcp_pos": list(cfg.robot.home_tcp_pos),
                                                  "tcp_z": models.gripper.tcp_z()})
print(f"[calibrate] home q {np.round(q_home, 4).tolist()} ({info.get('branch')}) saved to the manifest")
app.close()
