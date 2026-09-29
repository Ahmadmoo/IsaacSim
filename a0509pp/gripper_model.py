"""Robotiq 2F-85 (Physx_parallel_grip) kinematic model, TCP definition, and tool inertia.

Linkage anchors come from the Robotiq payload USDA files. In the parallel-grip variant the outer finger is
welded to the outer knuckle and the inner finger/pad translate (parallelogram), so the pad motion is a pure
translation Delta(theta) of the finger pivot about the knuckle pivot. Pad-face geometry is provisional until
scripts/calibrate_gripper.py measures it from the USD meshes and writes a calibration JSON.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from .geometry import make_T, rot_x, rot_z

# Masses [kg] and CoMs [m] in the gripper base frame at finger_joint = 0 (Robotiq payload USDA).
_BODIES = {
    "base_link": (0.77744, (0.0, 0.0, 0.035508), (0.001, 0.001, 0.001)),
    "outer_knuckle": (0.02, (0.0, 0.048356, 0.056017), (1e-5, 8e-6, 4e-6)),
    "outer_finger": (0.039179202, (0.0, 0.06595047, 0.06895309), (1.32e-5, 4.29e-6, 1.09e-5)),
    "inner_finger": (0.01327301, (0.0, 0.0594077, 0.10376717), (8.96e-7, 1.21e-6, 1.44e-6)),
    "inner_knuckle": (0.027306942, (0.0, 0.03222066, 0.083420366), (1.18e-5, 4.49e-6, 8.0e-6)),
    "fingertip": (0.02598171, (0.0, 0.05063447, 0.12616345), (1.39e-6, 3.31e-6, 3.73e-6)),
}
_INNER_KNUCKLE_PIVOT = (0.0, 0.0127, 0.06118)
_INNER_KNUCKLE_TIP = (0.0, 0.04986, 0.1046)
_OUTER_KNUCKLE_ELBOW = (0.0, 0.06213, 0.0509)


class GripperModel:
    def __init__(self, gcfg, rcfg, calibration: str | dict | None = None):
        self.g = gcfg
        self.r = rcfg
        p1 = np.array(gcfg.knuckle_pivot)
        p2 = np.array(gcfg.finger_pivot)
        self.v = p2 - p1
        self.L = float(np.linalg.norm(self.v[1:]))
        self.phi0 = math.atan2(self.v[1], self.v[2])
        self.pad_inner_y_open = gcfg.pad_inner_y_open
        self.pad_center_z_open = gcfg.pad_center_z_open
        self.table = None
        self.calibration_source = "linkage model (provisional)"
        if isinstance(calibration, str) and calibration and os.path.exists(calibration):
            with open(calibration) as f:
                calibration = json.load(f)
        if isinstance(calibration, dict) and calibration:
            self.load_calibration(calibration)
        self.theta_max = gcfg.joint_upper

    def load_calibration(self, cal: dict):
        """cal: {'theta': [...], 'aperture': [...], 'pad_center_z': [...], 'source': str}."""
        th = np.asarray(cal["theta"], dtype=float)
        order = np.argsort(th)
        self.table = {k: np.asarray(cal[k], dtype=float)[order] for k in ("theta", "aperture", "pad_center_z")}
        self.pad_inner_y_open = float(np.interp(0.0, self.table["theta"], self.table["aperture"]) / 2.0)
        self.pad_center_z_open = float(np.interp(0.0, self.table["theta"], self.table["pad_center_z"]))
        self.calibration_source = cal.get("source", "calibration file")

    # ------------------------------------------------------------------ aperture mapping
    def delta(self, theta):
        """Pad translation (dy toward the centre, dz toward the tip) of the left finger for joint angle theta."""
        vy, vz = abs(self.v[1]), self.v[2]
        dy = vy * (1.0 - np.cos(theta)) + vz * np.sin(theta)
        dz = vy * np.sin(theta) - vz * (1.0 - np.cos(theta))
        return dy, dz

    def aperture(self, theta):
        """Pad inner-face separation [m] at finger_joint = theta [rad]."""
        theta = np.asarray(theta, dtype=float)
        if self.table is not None:
            return np.interp(theta, self.table["theta"], self.table["aperture"])
        dy, _ = self.delta(theta)
        return 2.0 * self.pad_inner_y_open - 2.0 * dy

    def theta(self, aperture):
        """Inverse of aperture(), clipped to the joint range."""
        a = np.asarray(aperture, dtype=float)
        if self.table is not None:
            th = np.interp(-a, -self.table["aperture"], self.table["theta"])
        else:
            dy = (2.0 * self.pad_inner_y_open - a) / 2.0
            s = np.clip(math.sin(self.phi0) - dy / self.L, -1.0, 1.0)
            th = self.phi0 - np.arcsin(s)
        return np.clip(th, 0.0, self.theta_max)

    def pad_center_z(self, theta):
        if self.table is not None:
            return np.interp(theta, self.table["theta"], self.table["pad_center_z"])
        return self.pad_center_z_open + self.delta(theta)[1]

    def lever_arm(self, theta):
        """d(pad y)/d(theta) per finger [m]; from the measured aperture map when calibrated."""
        if self.table is not None:
            th, ap = self.table["theta"], self.table["aperture"]
            return np.interp(theta, th, -0.5 * np.gradient(ap, th))
        vy, vz = abs(self.v[1]), self.v[2]
        return vy * np.sin(theta) + vz * np.cos(theta)

    def max_torque(self):
        """finger_joint torque limit for the grip-force setting: tau = 2 F r, r averaged over 30-60 mm."""
        if self.g.max_torque is not None:
            return float(self.g.max_torque)
        th = self.theta(np.linspace(0.030, 0.060, 16))
        return float(2.0 * self.g.grip_force * np.mean(self.lever_arm(th)))

    # ------------------------------------------------------------------ frames
    def T_flange_base(self):
        """Gripper base frame in the flange (link_6) frame: adapter offset + yaw on flange."""
        return make_T(rot_z(self.r.gripper_yaw_on_flange), [0.0, 0.0, self.r.adapter_thickness])

    def tcp_z(self):
        """Fixed TCP origin along the gripper z axis: pinch centre at the reference aperture."""
        return float(self.pad_center_z(self.theta(self.g.tcp_reference_aperture)))

    def T_base_tcp(self):
        return make_T(None, [0.0, 0.0, self.tcp_z()])

    def T_flange_tcp(self):
        return self.T_flange_base() @ self.T_base_tcp()

    def pinch_offset(self, aperture):
        """Pinch-centre position relative to the TCP along TCP z at a given aperture."""
        return float(self.pad_center_z(self.theta(aperture)) - self.tcp_z())

    def T_base_wrist_cam(self, look_at=None, pos=None):
        from .geometry import look_at_ros

        pos = np.asarray(pos, dtype=float)
        R = look_at_ros(pos, np.asarray(look_at, dtype=float), up=(1.0, 0.0, 0.0))
        return make_T(R, pos)

    # ------------------------------------------------------------------ inertia
    def tool_inertial(self, theta=0.0):
        """Mass, CoM and inertia (about CoM) of adapter + gripper + camera assembly in the flange frame."""
        parts = []
        t = self.r.adapter_thickness
        m, rad = self.r.adapter_mass, self.r.adapter_radius
        parts.append((m, np.array([0.0, 0.0, t / 2.0]), np.diag([m * (3 * rad**2 + t**2) / 12.0] * 2 + [m * rad**2 / 2.0])))
        T_fb = self.T_flange_base()
        R_fb = T_fb[:3, :3]
        dy, dz = self.delta(theta)
        for name, (mass, com, inertia) in _BODIES.items():
            sides = [1.0] if name == "base_link" else [-1.0, 1.0]
            for s in sides:
                c = np.array([com[0], s * com[1], com[2]])
                if name in ("inner_finger", "fingertip"):
                    c = c + np.array([0.0, -s * dy, dz])
                parts.append((mass, T_fb[:3, :3] @ c + T_fb[:3, 3], R_fb @ np.diag(inertia) @ R_fb.T))
        if self.r.wrist_camera_hardware:
            mc = self.r.wrist_camera_mass + self.r.wrist_bracket_mass
            parts.append((mc, R_fb @ np.array([0.055, 0.0, 0.070]) + T_fb[:3, 3], np.eye(3) * mc * 0.02**2 / 6.0))
        M = sum(p[0] for p in parts)
        C = sum(p[0] * p[1] for p in parts) / M
        I = np.zeros((3, 3))
        for m, c, Ic in parts:
            d = c - C
            I += Ic + m * (d @ d * np.eye(3) - np.outer(d, d))
        return M, C, I

    # ------------------------------------------------------------------ collision spheres
    def spheres(self, theta=0.0):
        """Conservative spheres [x, y, z, r] in the gripper base frame at finger angle theta, with link tags."""
        dy, dz = self.delta(theta)
        out, tags = [], []

        def add(p, r, tag):
            out.append([p[0], p[1], p[2], r])
            tags.append(tag)

        for x in (-0.018, 0.018):
            for y in (-0.025, 0.025):
                for z in (0.022, 0.055):
                    add((x, y, z), 0.036, "gripper_base")
        add((0.0, 0.0, -self.r.adapter_thickness / 2.0), 0.040, "gripper_base")
        if self.r.wrist_camera_hardware:
            for y in (-0.010, 0.010):
                add((0.060, y, 0.070), 0.027, "wrist_camera")
            add((0.044, 0.0, 0.070), 0.016, "wrist_camera")
        pad_y = self.aperture(theta) / 2.0
        pad_z = self.pad_center_z(theta)
        r_pad = 0.012
        span = max(self.g.pad_half_height - r_pad, 0.003)
        n_pad = max(2, int(math.ceil(2.0 * span / r_pad)) + 1)
        pad_offsets = np.linspace(-span, span, n_pad)
        for s, side in ((-1.0, "left"), (1.0, "right")):
            R = rot_x(s * theta)
            p1 = np.array([0.0, s * self.g.knuckle_pivot[1], self.g.knuckle_pivot[2]])
            k0 = np.array([0.0, s * _OUTER_KNUCKLE_ELBOW[1], _OUTER_KNUCKLE_ELBOW[2]])
            p20 = np.array([0.0, s * self.g.finger_pivot[1], self.g.finger_pivot[2]])
            k = p1 + R @ (k0 - p1)
            p2 = p1 + R @ (p20 - p1)
            b = np.array([0.0, s * _INNER_KNUCKLE_PIVOT[1], _INNER_KNUCKLE_PIVOT[2]])
            c0 = np.array([0.0, s * _INNER_KNUCKLE_TIP[1], _INNER_KNUCKLE_TIP[2]])
            c = b + R @ (c0 - b)
            for t in (0.3, 0.8):
                add(p1 + t * (k - p1), 0.013, f"{side}_outer_knuckle")
            for t in (0.1, 0.5, 0.9):
                add(k + t * (p2 - k), 0.012, f"{side}_outer_finger")
            for t in (0.2, 0.55, 0.9):
                add(b + t * (c - b), 0.011, f"{side}_inner_knuckle")
            pad_c = np.array([0.0, s * (pad_y + r_pad), pad_z])
            top = pad_c - np.array([0.0, 0.0, self.g.pad_half_height])
            for t in (0.35, 0.75):
                add(p2 + t * (top - p2), 0.011, f"{side}_inner_finger")
            for dzp in pad_offsets:
                add(pad_c + np.array([0.0, 0.0, dzp]), r_pad, f"{side}_fingertip")
        return np.array(out), tags

    def pad_bottom_extent(self):
        """Distance from the pad centre to the lowest point of the pad spheres along the approach axis."""
        return max(float(self.g.pad_half_height), 0.015)

    def summary(self):
        th_ref = float(self.theta(self.g.tcp_reference_aperture))
        return {
            "calibration_source": self.calibration_source,
            "pad_inner_y_open": self.pad_inner_y_open,
            "pad_center_z_open": self.pad_center_z_open,
            "tcp_reference_aperture": self.g.tcp_reference_aperture,
            "tcp_reference_theta": th_ref,
            "tcp_z_in_gripper_base": self.tcp_z(),
            "T_flange_tcp": self.T_flange_tcp().tolist(),
            "max_torque_finger_joint": self.max_torque(),
            "grip_force_setting": self.g.grip_force,
            "tcp_definition": "origin at pad pinch centre at reference aperture; +z approach (palm->object); +y finger closing axis",
        }
