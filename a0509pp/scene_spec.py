"""Episode specification and scene sampling with validity checks (GPU-free)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import box_corners, rot_z

FAMILIES = [
    "object", "object_shape", "object_color", "object_mass", "obstacles", "object_friction", "pad_friction", "lighting", "depth", "calibration",
    "fixed_camera", "joint_noise", "command_delay", "camera_delay", "servo",
]
HIDDEN_FAMILIES = {"object_friction", "pad_friction", "servo", "command_delay", "object_mass"}


@dataclass
class ObjectSpec:
    pool_index: int
    dims: list
    xy: list
    yaw: float
    color: list = field(default_factory=lambda: [0.85, 0.20, 0.15])
    shape: str = "box"


@dataclass
class ObstacleSpec:
    pool_index: int
    dims: list
    xy: list
    yaw: float


@dataclass
class Realization:
    """Hidden physics shared by every candidate of a rollout group."""
    rid: int = 0
    object_mass: float = 0.10
    object_static_friction: float = 0.6
    object_dynamic_friction: float = 0.5
    pad_static_friction: float = 0.9
    pad_dynamic_friction: float = 0.7
    servo_scale: list = field(default_factory=lambda: [1.0] * 6)
    command_delay: int = 0


@dataclass
class EpisodeSpec:
    scene_id: str
    seed: int
    target: ObjectSpec
    obstacles: list = field(default_factory=list)
    distractors: list = field(default_factory=list)
    tray_xy: list = field(default_factory=lambda: [0.45, 0.23])
    light_scale: float = 1.0
    light_yaw: float = 0.0
    depth_sigma: float = 0.0
    depth_dropout: float = 0.0
    camera_delay: int = 0
    joint_noise: float = 0.0
    fixed_cam_delta: list = field(default_factory=lambda: [0.0] * 6)
    calib_delta_fixed: list = field(default_factory=lambda: [0.0] * 6)
    calib_delta_wrist: list = field(default_factory=lambda: [0.0] * 6)
    target_hint_xy: list = field(default_factory=lambda: [0.45, -0.15])
    realizations: list = field(default_factory=list)
    families: list = field(default_factory=list)
    split: str = "train"

    def to_json(self):
        return json.dumps(dataclasses.asdict(self))

    @staticmethod
    def from_json(s):
        d = json.loads(s) if isinstance(s, str) else s
        d["target"] = ObjectSpec(**d["target"])
        d["obstacles"] = [ObstacleSpec(**o) for o in d["obstacles"]]
        d["distractors"] = [ObjectSpec(**o) for o in d.get("distractors", [])]
        d["realizations"] = [Realization(**r) for r in d["realizations"]]
        return EpisodeSpec(**d)


def split_of(scene_id: str, fractions=(0.7, 0.1, 0.1, 0.1)):
    """Deterministic split by base-scene id: train / val / calib / test."""
    h = int(hashlib.sha256(scene_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    edges = np.cumsum(fractions)
    names = ["train", "val", "calib", "test"]
    return names[int(np.searchsorted(edges, h, side="right").clip(0, 3))]


def _quant(x, q):
    return float(round(x / q) * q)


def _rect_overlap(c1, yaw1, d1, c2, yaw2, d2, gap):
    """Separating-axis test for two rectangles inflated by gap/2 each."""
    corners = []
    for c, yaw, d in ((c1, yaw1, d1), (c2, yaw2, d2)):
        R = rot_z(yaw)[:2, :2]
        h = (np.asarray(d[:2]) + gap) / 2.0
        pts = np.array([[sx * h[0], sy * h[1]] for sx in (-1, 1) for sy in (-1, 1)]) @ R.T + np.asarray(c[:2])
        corners.append((pts, R))
    for pts_a, Ra in corners:
        for axis in Ra.T:
            p1 = corners[0][0] @ axis
            p2 = corners[1][0] @ axis
            if p1.max() < p2.min() or p2.max() < p1.min():
                return False
    return True


def _inside_table(layout, xy, yaw, dims, margin=0.02):
    c = np.array([xy[0], xy[1], 0.0])
    pts = box_corners(c, rot_z(yaw), dims)[:, :2]
    tx, ty = layout.table_center[0], layout.table_center[1]
    hx, hy = layout.table_size[0] / 2 - margin, layout.table_size[1] / 2 - margin
    return bool(np.all(np.abs(pts[:, 0] - tx) <= hx) and np.all(np.abs(pts[:, 1] - ty) <= hy))


def _in_keepout(layout, xy, dims):
    return math.hypot(xy[0], xy[1]) - 0.5 * math.hypot(dims[0], dims[1]) < layout.robot_keepout_radius


def tray_outer(layout, tray_xy):
    return [layout.tray_interior[0] + 2 * layout.tray_wall, layout.tray_interior[1] + 2 * layout.tray_wall, layout.tray_rim_height]


def valid_layout(layout, spec: EpisodeSpec):
    items = [(spec.target.xy, spec.target.yaw, spec.target.dims, "target")]
    items += [(o.xy, o.yaw, o.dims, f"obstacle_{i}") for i, o in enumerate(spec.obstacles)]
    items += [(o.xy, o.yaw, o.dims, f"distractor_{i}") for i, o in enumerate(spec.distractors)]
    tray = (spec.tray_xy, 0.0, tray_outer(layout, spec.tray_xy))
    for xy, yaw, dims, name in items:
        if not _inside_table(layout, xy, yaw, dims):
            return False, f"{name} outside table"
        if _in_keepout(layout, xy, dims):
            return False, f"{name} in robot keepout"
        if _rect_overlap(xy, yaw, dims, tray[0], tray[1], tray[2], layout.spawn_gap):
            return False, f"{name} overlaps tray"
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if _rect_overlap(a[0], a[1], a[2], b[0], b[1], b[2], layout.spawn_gap):
                return False, f"{a[3]} overlaps {b[3]}"
    return True, ""


class SceneSampler:
    def __init__(self, cfg, object_pool_dims=None, obstacle_pool_dims=None):
        self.cfg = cfg
        self.L = cfg.layout
        self.R = cfg.randomization
        self.object_pool = [list(d) for d in (object_pool_dims or self.L.object_pool_dims)]
        shapes = list(self.L.object_pool_shapes or [])
        self.object_shapes = shapes if len(shapes) == len(self.object_pool) else ["box"] * len(self.object_pool)
        self.obstacle_pool = [list(d) for d in (obstacle_pool_dims or self.L.obstacle_pool_dims)]

    def realization(self, rng, rid, families):
        R, P = self.R, self.cfg.physics
        r = Realization(rid=rid, object_mass=self.cfg.layout.target_default_mass,
                        object_static_friction=P.object_static_friction, object_dynamic_friction=P.object_dynamic_friction,
                        pad_static_friction=P.pad_static_friction, pad_dynamic_friction=P.pad_dynamic_friction)
        if "object_mass" in families:
            r.object_mass = float(rng.uniform(*R.object_mass))
        if "object_friction" in families:
            mu = _quant(rng.uniform(*R.object_static_friction), R.friction_quantum)
            r.object_static_friction, r.object_dynamic_friction = mu, _quant(mu * R.dynamic_ratio, R.friction_quantum)
        if "pad_friction" in families:
            mu = _quant(rng.uniform(*R.pad_static_friction), R.friction_quantum)
            r.pad_static_friction, r.pad_dynamic_friction = mu, _quant(mu * R.dynamic_ratio, R.friction_quantum)
        if "servo" in families:
            r.servo_scale = rng.uniform(*R.servo_scale, size=6).round(4).tolist()
        if "command_delay" in families:
            r.command_delay = int(rng.integers(R.command_delay[0], R.command_delay[1] + 1))
        return r

    def sample(self, index: int, seed: int, families=None, fixed_pose=False, num_realizations=1, max_tries=200):
        families = list(self.R.enabled if families is None else families)
        unknown = set(families) - set(FAMILIES)
        if unknown:
            raise ValueError(f"unknown randomization families: {unknown}")
        rng = np.random.default_rng([seed, index])
        scene_id = f"s{seed:05d}_{index:06d}"
        for _ in range(max_tries):
            kinds = list(dict.fromkeys(self.object_shapes))
            kind = kinds[int(rng.integers(len(kinds)))] if "object_shape" in families else kinds[0]
            ks = [i for i, s in enumerate(self.object_shapes) if s == kind]
            k = ks[int(rng.integers(len(ks)))] if "object" in families else ks[0]
            dims = self.object_pool[k]
            if fixed_pose:
                xy, yaw = list(self.L.target_default_xy), 0.0
            else:
                xy = [float(rng.uniform(*self.L.target_x_range)), float(rng.uniform(*self.L.target_y_range))]
                yaw = float(rng.uniform(0.0, 2 * math.pi))
            color = rng.uniform(0.1, 0.95, 3).round(3).tolist() if "object_color" in families else [0.85, 0.20, 0.15]
            target = ObjectSpec(k, list(dims), xy, yaw, color, self.object_shapes[k])
            obstacles = []
            if "obstacles" in families:
                n_obs = int(rng.integers(self.L.num_obstacles[0], self.L.num_obstacles[1] + 1))
                idx = rng.choice(len(self.obstacle_pool), size=min(n_obs, len(self.obstacle_pool)), replace=False)
                for j in idx:
                    obstacles.append(ObstacleSpec(int(j), list(self.obstacle_pool[j]),
                                                  [float(rng.uniform(*self.L.obstacle_x_range)), float(rng.uniform(*self.L.obstacle_y_range))],
                                                  float(rng.uniform(0.0, math.pi))))
            spec = EpisodeSpec(scene_id=scene_id, seed=seed, target=target, obstacles=obstacles,
                               tray_xy=list(self.L.tray_center_xy), families=families,
                               split=split_of(scene_id, self.cfg.collect.split))
            ok, _ = valid_layout(self.L, spec)
            if ok:
                break
        else:
            raise RuntimeError(f"could not sample a valid layout for scene {scene_id}")
        R = self.R
        if "lighting" in families:
            spec.light_scale = float(rng.uniform(*R.light_scale))
            spec.light_yaw = float(rng.uniform(-math.pi, math.pi))
        if "depth" in families:
            spec.depth_sigma = float(rng.uniform(*R.depth_sigma))
            spec.depth_dropout = float(rng.uniform(*R.depth_dropout))
        if "camera_delay" in families:
            spec.camera_delay = int(rng.integers(R.camera_delay[0], R.camera_delay[1] + 1))
        if "joint_noise" in families:
            spec.joint_noise = float(rng.uniform(*R.joint_noise))
        if "fixed_camera" in families:
            spec.fixed_cam_delta = _delta6(rng, R.fixed_cam_trans, math.radians(R.fixed_cam_rot_deg))
        if "calibration" in families:
            spec.calib_delta_fixed = _delta6(rng, R.calib_trans, math.radians(R.calib_rot_deg))
            spec.calib_delta_wrist = _delta6(rng, R.calib_trans, math.radians(R.calib_rot_deg))
        hint = np.array(spec.target.xy) + rng.normal(0.0, self.cfg.perception.target_hint_sigma, 2)
        spec.target_hint_xy = hint.round(5).tolist()
        spec.realizations = [self.realization(rng, r, families) for r in range(num_realizations)]
        return spec


def _delta6(rng, trans_max, rot_max):
    d = rng.normal(size=3)
    d *= rng.uniform(0.0, trans_max) / max(np.linalg.norm(d), 1e-12)
    a = rng.normal(size=3)
    a *= rng.uniform(0.0, rot_max) / max(np.linalg.norm(a), 1e-12)
    return np.concatenate([d, a]).round(6).tolist()


def delta_to_T(delta6):
    from .geometry import axis_angle_to_mat, make_T

    d = np.asarray(delta6, float)
    ang = np.linalg.norm(d[3:])
    R = axis_angle_to_mat(d[3:] / ang, ang) if ang > 1e-12 else np.eye(3)
    return make_T(R, d[:3])


def default_spec(cfg, scene_id="default"):
    L = cfg.layout
    t = ObjectSpec(0, list(L.object_pool_dims[0]), list(L.target_default_xy), 0.0, shape=SceneSampler(cfg).object_shapes[0])
    spec = EpisodeSpec(scene_id=scene_id, seed=0, target=t, tray_xy=list(L.tray_center_xy),
                       target_hint_xy=list(L.target_default_xy))
    spec.realizations = [SceneSampler(cfg).realization(np.random.default_rng(0), 0, [])]
    return spec


def make_object_pool(cfg, k=6, seed=0, shapes=("box",)):
    """Deterministic pool of pickable solids: for each shape, the default size first, then k-1 sizes spanning the
    randomisation ranges. Returns (dims, shapes)."""
    from .geometry import shape_dims

    rng = np.random.default_rng([seed, 7919])
    R, w0, d0, h0 = cfg.randomization, *cfg.layout.target_default_dims
    dims, kinds = [], []
    for s in shapes:
        dims.append([round(float(x), 4) for x in shape_dims(s, w0, d0, h0)])
        kinds.append(s)
        for _ in range(k - 1):
            w, d, h = (float(rng.uniform(*R.object_width)), float(rng.uniform(*R.object_width)), float(rng.uniform(*R.object_height)))
            dims.append([round(float(x), 4) for x in shape_dims(s, w, d, h)])
            kinds.append(s)
    return dims, kinds
