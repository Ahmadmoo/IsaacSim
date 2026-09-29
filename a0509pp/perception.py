"""Policy-visible scene estimates from depth (target box, obstacle boxes) and a privileged oracle."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

from .collision import tray_boxes


@dataclass
class BoxEstimate:
    center: np.ndarray
    yaw: float
    dims: np.ndarray
    confidence: float = 1.0
    n_points: int = 0
    kind: str = "object"
    source: str = "camera"

    def to_dict(self):
        return {"center": np.asarray(self.center).tolist(), "yaw": float(self.yaw), "dims": np.asarray(self.dims).tolist(),
                "confidence": float(self.confidence), "n_points": int(self.n_points), "kind": self.kind, "source": self.source}


@dataclass
class PerceptionResult:
    target: BoxEstimate | None
    obstacles: list = field(default_factory=list)
    source: str = "camera"
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self):
        return {"target": None if self.target is None else self.target.to_dict(),
                "obstacles": [o.to_dict() for o in self.obstacles], "source": self.source, "diagnostics": self.diagnostics}


def backproject(depth, K, T_world_cam, valid=None, stride=1):
    """Optical-axis depth image -> world points (M, 3)."""
    H, W = depth.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[::stride, ::stride]
    ok = np.isfinite(z) & (z > 0)
    if valid is not None:
        ok &= valid[::stride, ::stride]
    u, v, z = u[ok], v[ok], z[ok]
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    return pc @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]


def min_area_rect(xy, trim=0.5):
    """Minimum-area rectangle: (center(2), yaw, dims(2)), yaw in [-pi/2, pi/2); extents trimmed at percentiles."""
    if len(xy) < 3:
        c = xy.mean(0) if len(xy) else np.zeros(2)
        return c, 0.0, np.array([0.0, 0.0])
    try:
        hull = xy[ConvexHull(xy).vertices]
    except Exception:
        hull = xy
    best = None
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        a = np.arctan2(e[1], e[0])
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, s], [-s, c]])
        p = hull @ R.T
        lo, hi = p.min(0), p.max(0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            ctr = ((lo + hi) / 2.0) @ R
            best = (area, ctr, a, hi - lo)
    _, ctr, a, dims = best
    if trim > 0 and len(xy) >= 20:
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, s], [-s, c]])
        p = xy @ R.T
        lo, hi = np.percentile(p, [trim, 100 - trim], axis=0)
        ctr, dims = ((lo + hi) / 2.0) @ R, hi - lo
    a = (a + np.pi / 2) % np.pi - np.pi / 2
    return ctr, float(a), dims


def _robot_mask(points, spheres_c, spheres_r, margin):
    if spheres_c is None or len(points) == 0:
        return np.zeros(len(points), bool)
    keep = np.zeros(len(points), bool)
    for i in range(0, len(spheres_c), 32):
        c = spheres_c[i : i + 32]
        r = spheres_r[i : i + 32] + margin
        d = np.linalg.norm(points[:, None, :] - c[None], axis=2)
        keep |= (d <= r[None]).any(1)
    return keep


def _inside_boxes(points, boxes, margin):
    m = np.zeros(len(points), bool)
    for b in boxes:
        q = (points - b.center) @ b.R
        m |= np.all(np.abs(q) <= b.half + margin, axis=1)
    return m


def estimate(views, layout, tray_xy, hint_xy, pcfg, robot_spheres=None):
    """views: list of dicts {depth, K, T_world_cam, valid}. robot_spheres: (centers, radii) at the measured q."""
    pts = [backproject(v["depth"], v["K"], v["T_world_cam"], v.get("valid")) for v in views]
    P = np.concatenate(pts, 0) if pts else np.zeros((0, 3))
    diag = {"raw_points": int(len(P))}
    tx, ty = layout.table_center[:2]
    hx, hy = layout.table_size[0] / 2, layout.table_size[1] / 2
    keep = (np.abs(P[:, 0] - tx) < hx) & (np.abs(P[:, 1] - ty) < hy) & (P[:, 2] > pcfg.table_band) & (P[:, 2] < 0.5)
    P = P[keep]
    if robot_spheres is not None:
        P = P[~_robot_mask(P, robot_spheres[0], robot_spheres[1], pcfg.robot_margin)]
    P = P[~_inside_boxes(P, tray_boxes(layout, tray_xy, 0.0), pcfg.tray_margin)]
    diag["workspace_points"] = int(len(P))
    if len(P) == 0:
        return PerceptionResult(None, [], "camera", diag)
    cell = pcfg.voxel
    ij = np.floor((P[:, :2] - np.array([tx - hx, ty - hy])) / cell).astype(int)
    shape = (int(np.ceil(2 * hx / cell)) + 1, int(np.ceil(2 * hy / cell)) + 1)
    ij = np.clip(ij, 0, np.array(shape) - 1)
    occ = np.zeros(shape, bool)
    occ[ij[:, 0], ij[:, 1]] = True
    occ = ndimage.binary_closing(occ, structure=np.ones((3, 3)))
    lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
    pl = lab[ij[:, 0], ij[:, 1]]
    boxes = []
    for k in range(1, n + 1):
        Q = P[pl == k]
        if len(Q) < pcfg.min_cluster_points:
            continue
        z_top = float(np.percentile(Q[:, 2], 98))
        if z_top < pcfg.min_height:
            continue
        top = Q[Q[:, 2] >= z_top - pcfg.top_band]
        c_top, yaw_top, d_top = min_area_rect(top[:, :2])
        c_all, yaw_all, d_all = min_area_rect(Q[:, :2])
        expected = max(d_top[0] * d_top[1], 1e-6)
        coverage = min(1.0, len(top) * cell * cell / expected / 1.5)
        agree = float(np.exp(-np.linalg.norm(np.asarray(d_all) - np.asarray(d_top)) / 0.01))
        boxes.append({"top": (c_top, yaw_top, d_top), "all": (c_all, yaw_all, d_all), "z_top": z_top,
                      "n": int(len(Q)), "conf": float(np.clip(0.5 * coverage + 0.5 * agree, 0.0, 1.0))})
    diag["clusters"] = len(boxes)
    target = None
    obstacles = []
    if boxes and hint_xy is not None:
        dist = [np.linalg.norm(b["top"][0] - np.asarray(hint_xy)) for b in boxes]
        i = int(np.argmin(dist))
        if dist[i] < 0.08:
            b = boxes.pop(i)
            c, yaw, d = b["top"]
            target = BoxEstimate(np.array([c[0], c[1], b["z_top"] / 2.0]), yaw, np.array([d[0], d[1], b["z_top"]]),
                                 b["conf"], b["n"], "target")
            diag["target_hint_distance"] = float(dist[i])
    for b in boxes:
        c, yaw, d = b["all"]
        obstacles.append(BoxEstimate(np.array([c[0], c[1], b["z_top"] / 2.0]), yaw, np.array([d[0], d[1], b["z_top"]]),
                                     b["conf"], b["n"], "obstacle"))
    return PerceptionResult(target, obstacles, "camera", diag)


def privileged(spec, true_target_pose=None):
    """Oracle estimates from the annotation channel (explicitly labelled privileged)."""
    t = spec.target
    if true_target_pose is not None:
        c, yaw = true_target_pose
    else:
        c, yaw = np.array([t.xy[0], t.xy[1], t.dims[2] / 2.0]), t.yaw
    target = BoxEstimate(np.asarray(c, float), float(yaw), np.asarray(t.dims, float), 1.0, 0, "target", "privileged")
    obs = [BoxEstimate(np.array([o.xy[0], o.xy[1], o.dims[2] / 2.0]), o.yaw, np.asarray(o.dims, float), 1.0, 0, "obstacle", "privileged")
           for o in spec.obstacles + spec.distractors]
    return PerceptionResult(target, obs, "privileged", {})
