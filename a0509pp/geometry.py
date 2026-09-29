"""Rigid-transform helpers. Quaternions are persisted as (x, y, z, w)."""

from __future__ import annotations

import numpy as np


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_to_mat(rpy):
    """URDF convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return rot_z(rpy[2]) @ rot_y(rpy[1]) @ rot_x(rpy[0])


def axis_angle_to_mat(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = skew(axis)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def make_T(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def inv_T(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def quat_to_mat(q):
    """q = (x, y, z, w)."""
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_to_mat_batch(q):
    """(N, 4) quaternions (x, y, z, w) -> (N, 3, 3)."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def pose7_to_T_batch(p):
    """(N, 7) [x, y, z, qx, qy, qz, qw] -> (N, 4, 4)."""
    p = np.asarray(p, dtype=float)
    T = np.zeros(p.shape[:-1] + (4, 4))
    T[..., :3, :3] = quat_to_mat_batch(p[..., 3:7])
    T[..., :3, 3] = p[..., :3]
    T[..., 3, 3] = 1.0
    return T


def mat_to_quat(R):
    """Returns (x, y, z, w) with w >= 0."""
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0.0:
        s = 2.0 * np.sqrt(t + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return q if q[3] >= 0.0 else -q


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_from_yaw(yaw):
    return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)])


def yaw_from_mat(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def so3_log(R):
    """Rotation vector of R."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    a = np.arccos(c)
    if a < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    if a > np.pi - 1e-6:
        A = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(A)))
        v = A[:, i] / np.sqrt(max(A[i, i], 1e-12))
        return v * a
    return a / (2.0 * np.sin(a)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def pose_error(T_cur, T_goal):
    """6-vector [dp, dtheta] in the base frame that moves T_cur toward T_goal."""
    dp = T_goal[:3, 3] - T_cur[:3, 3]
    dr = so3_log(T_goal[:3, :3] @ T_cur[:3, :3].T)
    return np.concatenate([dp, dr])


def so3_log_batch(R):
    """Rotation vectors of (N, 3, 3) rotations (accurate away from angle pi)."""
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    a = np.arccos(c)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    s = np.sin(a)
    k = np.where(a < 1e-9, 0.5, a / (2.0 * np.maximum(s, 1e-12)))
    out = v * k[:, None]
    big = a > np.pi - 1e-6
    if np.any(big):
        out[big] = np.stack([so3_log(r) for r in R[big]])
    return out


def pose_error_batch(T_cur, T_goal):
    dp = T_goal[:, :3, 3] - T_cur[:, :3, 3]
    dr = so3_log_batch(T_goal[:, :3, :3] @ np.transpose(T_cur[:, :3, :3], (0, 2, 1)))
    return np.concatenate([dp, dr], 1)


def pose7(T):
    """(x, y, z, qx, qy, qz, qw)."""
    return np.concatenate([T[:3, 3], mat_to_quat(T[:3, :3])])


def T_from_pose7(p):
    return make_T(quat_to_mat(p[3:7]), p[:3])


def look_at_ros(eye, target, up=(0.0, 0.0, 1.0)):
    """Rotation of an optical frame (+x right, +y down, +z forward) looking from eye to target."""
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-9:
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


SHAPES = ("box", "cylinder", "hex_prism")


def shape_dims(shape, width, depth, height):
    """Bounding-box dims (x, y, z). Cylinder: diameter = width. Hexagonal prism: across-flats = width along x."""
    if shape == "cylinder":
        return [width, width, height]
    if shape == "hex_prism":
        return [width, width * 2.0 / np.sqrt(3.0), height]
    return [width, depth, height]


def shape_vertices(shape, dims, n=24):
    """Local-frame vertices of the solid's convex hull (upright, centred)."""
    x, y, z = np.asarray(dims, float) / 2.0
    if shape == "cylinder":
        a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        ring = np.stack([x * np.cos(a), x * np.sin(a)], 1)
    elif shape == "hex_prism":
        a = np.radians(30.0 + 60.0 * np.arange(6))
        ring = np.stack([y * np.cos(a), y * np.sin(a)], 1)
    else:
        ring = np.array([[-x, -y], [x, -y], [x, y], [-x, y]])
    return np.concatenate([np.c_[ring, np.full(len(ring), -z)], np.c_[ring, np.full(len(ring), z)]])


def shape_inertia(shape, m, dims):
    """Principal inertia (Ixx, Iyy, Izz) about the centre of mass."""
    a, b, h = dims
    if shape == "cylinder":
        r = a / 2.0
        return np.array([m * (3 * r * r + h * h) / 12.0] * 2 + [m * r * r / 2.0])
    if shape == "hex_prism":
        R = b / 2.0
        return np.array([m * (5 * R * R / 24.0 + h * h / 12.0)] * 2 + [5 * m * R * R / 12.0])
    return np.array([m / 12.0 * (b * b + h * h), m / 12.0 * (a * a + h * h), m / 12.0 * (a * a + b * b)])


def solid_corners(center, R, dims, shape="box"):
    return np.asarray(center) + shape_vertices(shape, dims) @ np.asarray(R).T


def box_corners(center, R, dims):
    h = np.asarray(dims, dtype=float) / 2.0
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    return center + (signs * h) @ np.asarray(R).T


def wrap_to_pi(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def perturb_T(T, trans_max, rot_max_rad, rng):
    """Random rigid perturbation with uniform translation norm <= trans_max and rotation <= rot_max_rad."""
    d = rng.normal(size=3)
    d *= rng.uniform(0.0, trans_max) / max(np.linalg.norm(d), 1e-12)
    a = rng.normal(size=3)
    a /= max(np.linalg.norm(a), 1e-12)
    R = axis_angle_to_mat(a, rng.uniform(0.0, rot_max_rad))
    return make_T(R @ T[:3, :3], T[:3, 3] + d)
