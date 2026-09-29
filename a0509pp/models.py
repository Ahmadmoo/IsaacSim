"""Bundle of nominal models used by planning, features and the environment."""

from __future__ import annotations

import json
import math
import os

import numpy as np

from .candidates import Planner, tcp_rot
from .collision import RobotSpheres, world_boxes
from .geometry import inv_T, make_T
from .gripper_model import GripperModel
from .kinematics import ArmModel


MANIFEST_PATH_KEYS = ("robot_usd", "robot_usd_no_camera", "arm_usd", "gripper_calibration", "kinematics_json",
                      "arm_spheres_json")


def load_manifest(path):
    """Read the asset manifest; file entries are stored relative to the manifest and returned absolute."""
    if not path:
        return {}
    if not os.path.isabs(path) and not os.path.exists(path):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", path)
        path = alt if os.path.exists(alt) else path
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        man = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    for k in MANIFEST_PATH_KEYS:
        v = man.get(k)
        if v and not os.path.isabs(v):
            man[k] = os.path.normpath(os.path.join(base, v))
    man["_path"] = os.path.abspath(path)
    return man


def save_manifest_field(path, key, value):
    """Update one manifest entry; file paths are stored relative to the manifest."""
    with open(path) as f:
        man = json.load(f)
    if key in MANIFEST_PATH_KEYS and value:
        value = os.path.relpath(os.path.abspath(value), os.path.dirname(os.path.abspath(path)))
    man[key] = value
    with open(path, "w") as f:
        json.dump(man, f, indent=1)


class Models:
    def __init__(self, cfg, manifest=None):
        self.cfg = cfg
        man = manifest if manifest is not None else load_manifest(cfg.asset_manifest)
        self.manifest = man
        self.arm = ArmModel(man.get("kinematics_json") or None)
        cal = cfg.gripper.calibration_file or man.get("gripper_calibration", "")
        pads = (man.get("gripper_geometry") or {}).get("pads") or {}
        if pads:
            # Pad faces measured from the USD meshes (prepare_assets.py) replace the provisional defaults in the
            # effective config, so the planner, features and dataset manifest all use the same values.
            for k in ("pad_half_height", "pad_half_width", "pad_thickness") + (() if cal else ("pad_inner_y_open", "pad_center_z_open")):
                if k in pads:
                    setattr(cfg.gripper, k, float(pads[k]))
        self.gripper = GripperModel(cfg.gripper, cfg.robot, cal)
        if pads and not cal:
            self.gripper.calibration_source = "USD mesh bounds + linkage model (run calibrate_gripper.py)"
        m, c, I = self.gripper.tool_inertial()
        self.arm.set_tool(m, c, I)
        self.spheres = RobotSpheres(self.arm, self.gripper, man.get("arm_spheres_json") or None)
        self.planner = Planner(cfg, self.arm, self.gripper, self.spheres)

    @property
    def T_flange_tcp(self):
        return self.gripper.T_flange_tcp()

    def tcp_pose(self, q):
        return self.arm.fk(q, self.T_flange_tcp)

    def scaled_jacobian(self, q):
        D_task = np.diag(self.cfg.features.task_scale)
        D_joint = np.diag([self.cfg.robot.joint_speed_cap] * 6)
        return np.linalg.inv(D_task) @ self.arm.jacobian(q, self.T_flange_tcp) @ D_joint

    def solve_home(self):
        """IK for the home TCP (approach down), collision-free, best joint + capability margins."""
        r = self.cfg.robot
        if r.home_q is not None:
            return np.array(r.home_q, float), {"source": "config"}
        saved = self.manifest.get("home")
        if saved and saved.get("tcp_pos") is not None and np.allclose(saved["tcp_pos"], r.home_tcp_pos) \
                and saved.get("tcp_z") is not None and abs(saved["tcp_z"] - self.gripper.tcp_z()) < 1e-6:
            return np.array(saved["q"], float), {**saved.get("info", {}), "source": "asset manifest"}
        boxes = world_boxes(self.cfg.layout, self.cfg.layout.tray_center_xy)
        best = None
        for yaw in (0.0, math.pi / 2, math.pi, -math.pi / 2):
            T = make_T(tcp_rot(yaw), r.home_tcp_pos)
            for q, flags in self.arm.ik_all(T @ inv_T(self.T_flange_tcp), np.zeros(6)):
                name = self.planner.branch_name(q, flags)
                if not name.startswith("front-up"):
                    continue
                C, rad, tags = self.spheres.compute(q[None], 0.0)
                from .collision import sphere_box_distance

                D = sphere_box_distance(C, rad, boxes)
                D[:, [t == "base_link" for t in tags], :] = np.inf
                if D.min() < 0.02:
                    continue
                span = self.arm.q_max - self.arm.q_min
                margin = float(np.min(np.minimum(q - self.arm.q_min, self.arm.q_max - q) / span))
                sv = np.linalg.svd(self.scaled_jacobian(q), compute_uv=False)
                score = margin + 0.25 * sv[-1] / sv[0] - 0.02 * (abs(q[3]) + abs(q[5]))
                if best is None or score > best[0]:
                    best = (score, q, {"yaw": yaw, "branch": name, "joint_margin": margin, "sigma_ratio": float(sv[-1] / sv[0])})
        if best is None:
            raise RuntimeError("no collision-free home configuration; adjust robot.home_tcp_pos")
        return best[1], best[2]
