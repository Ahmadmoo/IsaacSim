"""Milestone 0: build the robot USD and the asset manifest.

Converts the official Doosan A0509 URDF to USD, bundles the Robotiq asset, mounts it through a flange adapter,
adds the wrist-camera housing, and writes assets/generated/asset_manifest.json. The resulting directory can be
copied as a unit and reused without either source checkout. Then run scripts/calibrate_gripper.py.

    python scripts/prepare_assets.py --doosan ~/src/doosan-robot2 --robotiq ~/src/IsaacSim-assets
"""

import argparse
import hashlib
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--doosan", required=True, help="checkout of github.com/DoosanRobotics/doosan-robot2")
parser.add_argument("--robotiq", required=True, help="folder that contains grippers/Robotiq_2F_85 (IsaacSim-assets checkout)")
parser.add_argument("--out", default=os.path.join(ROOT, "assets", "generated"))
parser.add_argument("--config", default=None)
parser.add_argument("--set", nargs="*", default=[])
parser.add_argument("--collision", default="Convex Hull", choices=["Convex Hull", "Convex Decomposition"])
parser.add_argument("--no-instanceable", action="store_true", help="convert the arm without USD instancing")
parser.add_argument("--skip-convert", action="store_true", help="reuse an existing converted arm USD")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

from a0509pp.config import DATASET_VERSION, load_cfg  # noqa: E402
from a0509pp.kinematics import ArmModel, parse_urdf  # noqa: E402
from a0509pp.sim import asset_builder as ab  # noqa: E402


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def rel(path, base):
    return os.path.relpath(os.path.abspath(path), os.path.abspath(base))


cfg = load_cfg(args.config, args.set)
out = os.path.abspath(args.out)
os.makedirs(out, exist_ok=True)
t0 = time.time()

urdf_path, urdf_src = ab.prepare_urdf(args.doosan, os.path.join(out, "urdf"), base_name=cfg.robot.arm_base_link)
robotiq_bundle = ab.bundle_robotiq(args.robotiq, out)

usd_dir = os.path.join(out, "usd")
arm_usd = os.path.join(usd_dir, "a0509_arm", "a0509_arm.usda")
if not (args.skip_convert and os.path.exists(arm_usd)):
    arm_usd = UrdfConverter(UrdfConverterCfg(
        asset_path=urdf_path, usd_dir=usd_dir, fix_base=True, merge_fixed_joints=True, self_collision=True,
        collision_type=args.collision, force_usd_conversion=True, make_instanceable=not args.no_instanceable,
        physics_variant="physx", robot_type="Manipulator",
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force", target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=cfg.robot.servo_stiffness[0], damping=cfg.robot.servo_damping[0])),
    )).usd_path
print(f"[prepare] arm USD {arm_usd}")

variants = {}
for with_cam in (True, False):
    path = os.path.join(usd_dir, "a0509_2f85_cam.usda" if with_cam else "a0509_2f85.usda")
    paths = ab.build_combined(arm_usd, robotiq_bundle, path, cfg.robot, cfg.gripper, cfg.camera, with_camera=with_cam)
    paths["pad_material"] = ab.bind_pad_material(path, paths, cfg.gripper.pad_links, cfg.physics.pad_static_friction,
                                                 cfg.physics.pad_dynamic_friction, cfg.physics.restitution)
    variants[with_cam] = (path, paths)
    print(f"[prepare] built {path}")

usd_cam, paths = variants[True]
geom = ab.measure_gripper_geometry(usd_cam, paths)
mimic = ab.read_mimic(usd_cam, paths["gripper_joints"])
missing = [j for j in cfg.gripper.passive_joints if j not in mimic]
if missing:
    print(f"[prepare] WARNING: no PhysX mimic data for {missing}; the env falls back to the default coupling signs")

models_dir = os.path.join(out, "models")
os.makedirs(models_dir, exist_ok=True)
shipped = ab.copy_models(models_dir)
kin_new = os.path.join(models_dir, "a0509_kinematics_from_urdf.json")
ArmModel(parse_urdf(urdf_src)).to_json(kin_new)
a, b = json.load(open(kin_new)), json.load(open(shipped["kinematics_json"]))
kin_diff = max(float(np.max(np.abs(np.array(ja[k]) - np.array(jb[k])))) for ja, jb in zip(a["joints"], b["joints"])
               for k in ("xyz", "rpy", "axis", "lower", "upper"))
if kin_diff > 1e-6:
    print(f"[prepare] WARNING: your URDF differs from the shipped kinematics snapshot by {kin_diff:.2e}; using the "
          "URDF-derived model. Re-run scripts/fit_arm_spheres.py for matching collision spheres.")
kinematics_json = kin_new if kin_diff > 1e-6 else shipped["kinematics_json"]
dependencies = ab.validate_asset_bundle(out, [arm_usd, os.path.join(robotiq_bundle, ab.ROBOTIQ_CFG_REL),
                                               variants[True][0], variants[False][0]])

try:
    from isaaclab.utils.version import get_isaac_sim_version

    sim_version = str(get_isaac_sim_version())
except Exception:
    sim_version = "unknown"
import isaaclab  # noqa: E402

manifest = {
    "dataset_version": DATASET_VERSION,
    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "isaac_sim_version": sim_version,
    "isaaclab_version": getattr(isaaclab, "__version__", "unknown"),
    "robot_usd": rel(variants[True][0], out),
    "robot_usd_no_camera": rel(variants[False][0], out),
    "arm_usd": rel(arm_usd, out),
    "robotiq_usd": rel(os.path.join(robotiq_bundle, ab.ROBOTIQ_CFG_REL), out),
    "usd_dependencies": dependencies,
    "prim_paths": paths,
    "prim_paths_no_camera": variants[False][1],
    "arm_joint_names": list(cfg.robot.arm_joint_names),
    "gripper_joint_names": sorted(paths["gripper_joints"]),
    "gripper_mimic": mimic,
    "gripper_geometry": geom,
    "gripper_calibration": "",
    "kinematics_json": rel(kinematics_json, out),
    "arm_spheres_json": rel(shipped["arm_spheres_json"], out),
    "kinematics_max_diff_vs_snapshot": kin_diff,
    "tcp_definition": "fixed frame on the gripper base; origin at the pad pinch centre at the reference aperture "
                      f"({cfg.gripper.tcp_reference_aperture * 1000:.0f} mm); +z approach; +y closing axis",
    "sources": {
        "doosan": {"repo": os.path.abspath(args.doosan), "commit": ab.git_rev(args.doosan), "urdf": urdf_src,
                   "urdf_sha256": sha256(urdf_src)},
        "robotiq": {"repo": os.path.abspath(args.robotiq), "commit": ab.git_rev(args.robotiq),
                    "usd": os.path.join(args.robotiq, ab.ROBOTIQ_CFG_REL), "variant": "Physx_parallel_grip, standard fingertips"},
    },
    "config_at_build": {"robot": cfg.to_dict()["robot"], "gripper": cfg.to_dict()["gripper"]},
}
man_path = os.path.join(out, "asset_manifest.json")
ab.write_json(man_path, manifest)
pads = geom.get("pads", {})
print(f"[prepare] wrote {man_path} ({time.time() - t0:.0f} s); pads: " + ", ".join(f"{k}={v:.4f}" for k, v in pads.items()))
app.close()
