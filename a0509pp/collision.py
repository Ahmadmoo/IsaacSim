"""Nominal collision model: robot spheres vs oriented boxes, and robot self-distance (numpy)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .geometry import make_T, rot_z

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPHERES = os.path.join(_HERE, "..", "assets", "models", "a0509_spheres.json")
ARM_LINKS = ["base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"]
GRIPPER_TAGS_FINGER = ("outer_knuckle", "outer_finger", "inner_knuckle", "inner_finger", "fingertip")


@dataclass
class Box:
    name: str
    center: np.ndarray
    R: np.ndarray
    half: np.ndarray
    margin: float = 0.0
    kind: str = "fixture"
    confidence: float = 1.0
    meta: dict = field(default_factory=dict)

    @staticmethod
    def from_xyzyaw(name, center, yaw, dims, margin=0.0, kind="fixture", confidence=1.0):
        return Box(name, np.asarray(center, float), rot_z(yaw), np.asarray(dims, float) / 2.0, margin, kind, confidence)


def world_boxes(layout, tray_center_xy=None, obstacles=(), fixture_margin=0.002, include_floor=True):
    """Table, tray parts, floor and obstacle boxes (obstacles: list of Box already inflated as required)."""
    boxes = [Box("table", np.array(layout.table_center, float), np.eye(3), np.array(layout.table_size) / 2.0, fixture_margin, "table")]
    boxes += tray_boxes(layout, tray_center_xy, fixture_margin)
    if include_floor:
        boxes.append(Box("floor", np.array([0.45, 0.0, layout.floor_z - 0.05]), np.eye(3), np.array([3.0, 3.0, 0.05]), 0.0, "floor"))
    boxes += list(obstacles)
    return boxes


def tray_boxes(layout, tray_center_xy=None, margin=0.002):
    cx, cy = tray_center_xy if tray_center_xy is not None else layout.tray_center_xy
    ix, iy = layout.tray_interior
    w, fl, h = layout.tray_wall, layout.tray_floor, layout.tray_rim_height
    ox, oy = ix + 2 * w, iy + 2 * w
    parts = [
        ("tray_floor", (cx, cy, fl / 2.0), (ox, oy, fl)),
        ("tray_wall_xp", (cx + ix / 2.0 + w / 2.0, cy, h / 2.0), (w, oy, h)),
        ("tray_wall_xn", (cx - ix / 2.0 - w / 2.0, cy, h / 2.0), (w, oy, h)),
        ("tray_wall_yp", (cx, cy + iy / 2.0 + w / 2.0, h / 2.0), (ix, w, h)),
        ("tray_wall_yn", (cx, cy - iy / 2.0 - w / 2.0, h / 2.0), (ix, w, h)),
    ]
    return [Box(n, np.array(c, float), np.eye(3), np.array(d, float) / 2.0, margin, "tray") for n, c, d in parts]


def sphere_box_distance(C, r, boxes):
    """Signed clearance (N, S, K) between spheres (N, S, 3)/(S,) and boxes, net of box margins.

    Boxes are assumed to rotate about z only (tables, trays, yawed obstacles)."""
    if not boxes:
        return np.full(C.shape[:2] + (0,), np.inf)
    Bc = np.stack([b.center for b in boxes])
    cy = np.array([b.R[0, 0] for b in boxes])
    sy = np.array([b.R[1, 0] for b in boxes])
    Bh = np.stack([b.half for b in boxes])
    Bm = np.array([b.margin for b in boxes])
    dx = C[:, :, None, 0] - Bc[None, None, :, 0]
    dy = C[:, :, None, 1] - Bc[None, None, :, 1]
    dz = C[:, :, None, 2] - Bc[None, None, :, 2]
    ex = np.abs(cy * dx + sy * dy) - Bh[:, 0]
    ey = np.abs(-sy * dx + cy * dy) - Bh[:, 1]
    ez = np.abs(dz) - Bh[:, 2]
    outside = np.sqrt(np.maximum(ex, 0.0) ** 2 + np.maximum(ey, 0.0) ** 2 + np.maximum(ez, 0.0) ** 2)
    inside = np.minimum(np.maximum(np.maximum(ex, ey), ez), 0.0)
    return outside + inside - r[None, :, None] - Bm[None, None, :]


class RobotSpheres:
    """World-frame spheres for the arm, gripper, and an optional held object."""

    def __init__(self, arm, gripper, spheres_json=None):
        self.arm = arm
        self.gripper = gripper
        with open(spheres_json or DEFAULT_SPHERES) as f:
            raw = json.load(f)
        data = raw["spheres"]
        self.pair_offsets = raw.get("pair_offsets", {})
        self.disabled_pairs = set(raw.get("disabled_pairs", []))
        self.joint_rules = raw.get("joint_rules", [])
        self.arm_local, self.arm_link, self.arm_tags = [], [], []
        for li, name in enumerate(ARM_LINKS):
            for s in data.get(name, []):
                self.arm_local.append(s[:3])
                self.arm_link.append(li)
                self.arm_tags.append(name)
        self.arm_local = np.array(self.arm_local)
        self.arm_radius = np.array([s[3] for name in ARM_LINKS for s in data.get(name, [])])
        self.arm_link = np.array(self.arm_link)
        g0, gtags = gripper.spheres(0.0)
        self.grip_tags = gtags
        self.tags = self.arm_tags + gtags
        self._self_pairs = None

    def gripper_local(self, theta):
        s, _ = self.gripper.spheres(theta)
        return s

    def compute(self, Q, theta, obj_spheres_tcp=None):
        """Q (N,6), theta (N,) finger angle. obj_spheres_tcp: (M,4) spheres in the TCP frame or None.

        Returns centers (N, S, 3), radii (S,), tags (S,)."""
        Q = np.atleast_2d(Q)
        N = Q.shape[0]
        theta = np.broadcast_to(np.asarray(theta, float), (N,))
        Ts = self.arm.fk_batch(Q)
        Tl = Ts[:, self.arm_link]
        arm_c = np.einsum("nsij,sj->nsi", Tl[:, :, :3, :3], self.arm_local) + Tl[:, :, :3, 3]
        T_base = Ts[:, 6] @ self.gripper.T_flange_base()
        uq, inv = np.unique(np.round(theta, 4), return_inverse=True)
        gl = np.stack([self.gripper_local(t) for t in uq])
        gsel = gl[inv]
        grip_c = np.einsum("nij,nsj->nsi", T_base[:, :3, :3], gsel[:, :, :3]) + T_base[:, None, :3, 3]
        C = np.concatenate([arm_c, grip_c], axis=1)
        r = np.concatenate([self.arm_radius, gl[0, :, 3]])
        tags = list(self.tags)
        if obj_spheres_tcp is not None and len(obj_spheres_tcp):
            T_tcp = T_base @ self.gripper.T_base_tcp()
            oc = np.einsum("nij,sj->nsi", T_tcp[:, :3, :3], obj_spheres_tcp[:, :3]) + T_tcp[:, None, :3, 3]
            C = np.concatenate([C, oc], axis=1)
            r = np.concatenate([r, obj_spheres_tcp[:, 3]])
            tags += ["held_object"] * len(obj_spheres_tcp)
        return C, r, tags

    def self_pairs(self, tags):
        """Sphere index pairs of non-adjacent parts and a per-pair distance offset.

        Arm-arm offsets are calibrated against the URDF convex hulls (fit_arm_spheres.py); tool and held-object
        pairs use no offset (fully conservative)."""
        key = tuple(tags)
        if self._self_pairs is not None and self._self_pairs[0] == key:
            return self._self_pairs[1], self._self_pairs[2]

        def group(t):
            if t in ARM_LINKS:
                return ARM_LINKS.index(t)
            if t == "held_object":
                return 9
            if t == "wrist_camera":
                return 8
            return 7

        g = np.array([group(t) for t in tags])
        I, J = np.triu_indices(len(tags), 1)
        gi, gj = g[I], g[J]
        lo, hi = np.minimum(gi, gj), np.maximum(gi, gj)
        arm_arm = (hi <= 6) & (hi - lo >= 2)
        tool_arm = (hi >= 7) & (hi <= 8) & (lo <= 4)
        obj_arm = (hi == 9) & (lo <= 5)
        keep = arm_arm | tool_arm | obj_arm
        pairs = np.stack([I[keep], J[keep]], axis=1)
        offs = np.zeros(len(pairs))
        drop = np.zeros(len(pairs), bool)
        for k, (a, b) in enumerate(pairs):
            ga, gb = g[a], g[b]
            if ga <= 6 and gb <= 6:
                name = f"{ARM_LINKS[min(ga, gb)]}|{ARM_LINKS[max(ga, gb)]}"
                offs[k] = self.pair_offsets.get(name, 0.0)
                drop[k] = name in self.disabled_pairs
        pairs, offs = pairs[~drop], offs[~drop]
        gp = np.stack([g[pairs[:, 0]], g[pairs[:, 1]]], 1)
        order = np.lexsort((gp[:, 1], gp[:, 0]))
        pairs, offs, gp = pairs[order], offs[order], gp[order]
        blocks = []
        if len(gp):
            cut = np.flatnonzero(np.any(np.diff(gp, axis=0) != 0, axis=1)) + 1
            starts = np.concatenate([[0], cut])
            ends = np.concatenate([cut, [len(gp)]])
            blocks = [(int(gp[a, 0]), int(gp[a, 1]), int(a), int(b)) for a, b in zip(starts, ends)]
        self._self_pairs = (key, pairs, offs, blocks, g)
        return pairs, offs

    def self_distance(self, C, r, tags, broad=0.08):
        """Per-sample min self clearance (N,) and argmin pair; broad phase on per-group bounding spheres."""
        pairs, offs = self.self_pairs(tags)
        _, _, _, blocks, g = self._self_pairs
        N = C.shape[0]
        best = np.full(N, np.inf)
        best_pair = np.zeros(N, int)
        cache = {}

        def bound(grp):
            if grp not in cache:
                idx = np.flatnonzero(g == grp)
                ctr = C[:, idx].mean(1)
                rad = (np.linalg.norm(C[:, idx] - ctr[:, None], axis=2) + r[idx][None]).max(1)
                cache[grp] = (ctr, rad)
            return cache[grp]

        for ga, gb, a, b in blocks:
            ca, ra = bound(ga)
            cb, rb = bound(gb)
            if np.min(np.linalg.norm(ca - cb, axis=1) - ra - rb + offs[a:b].max()) > broad:
                continue
            P = pairs[a:b]
            d = np.linalg.norm(C[:, P[:, 0]] - C[:, P[:, 1]], axis=2) - r[P[:, 0]] - r[P[:, 1]] + offs[a:b][None]
            k = d.argmin(1)
            dm = d[np.arange(N), k]
            upd = dm < best
            best[upd] = dm[upd]
            best_pair[upd] = a + k[upd]
        return best, best_pair

    def joint_rule_margin(self, Q):
        """Min over joint rules of (abs_max - |q_j|) per sample (inf if no rules)."""
        Q = np.atleast_2d(Q)
        m = np.full(len(Q), np.inf)
        for rule in self.joint_rules:
            m = np.minimum(m, rule["abs_max"] - np.abs(Q[:, rule["joint"]]))
        return m


def object_spheres(dims, grid=None):
    """Spheres [x, y, z, r] covering a box of dims centred at the origin (box frame)."""
    dims = np.asarray(dims, float)
    if grid is None:
        grid = np.maximum(1, np.ceil(dims / 0.025).astype(int))
    cell = dims / grid
    r = float(np.linalg.norm(cell) / 2.0)
    axes = [(np.arange(n) + 0.5) * c - d / 2.0 for n, c, d in zip(grid, cell, dims)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return np.concatenate([pts, np.full((len(pts), 1), r)], axis=1)


def transform_spheres(S, T):
    out = S.copy()
    out[:, :3] = S[:, :3] @ T[:3, :3].T + T[:3, 3]
    return out


def T_box(center, yaw):
    return make_T(rot_z(yaw), center)
