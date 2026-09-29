"""A0509 kinematics and nominal dynamics (numpy, GPU-free).

The chain is read from the Doosan URDF (or the JSON snapshot in assets/models). Analytic IK uses the
ideal geometry (exact right angles, spherical wrist) and is refined by damped least squares on the exact
URDF chain, which uses 1.571 rad rather than pi/2.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET

import numpy as np

from .geometry import inv_T, make_T, pose_error, pose_error_batch, rot_x, rot_z, rpy_to_mat

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(_HERE, "..", "assets", "models", "a0509_kinematics.json")
LINKS = ["base", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"]


def parse_urdf(path: str) -> dict:
    root = ET.parse(path).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    links = {l.get("name"): l for l in root.findall("link")}
    out = {"source": os.path.abspath(path), "joints": [], "inertials": {}}
    for i in range(1, 7):
        j = joints[f"joint_{i}"]
        o = j.find("origin")
        lim = j.find("limit")
        out["joints"].append(
            {
                "name": j.get("name"),
                "parent": j.find("parent").get("link"),
                "child": j.find("child").get("link"),
                "xyz": [float(v) for v in o.get("xyz").split()],
                "rpy": [float(v) for v in o.get("rpy").split()],
                "axis": [float(v) for v in j.find("axis").get("xyz").split()],
                "lower": float(lim.get("lower")),
                "upper": float(lim.get("upper")),
                "velocity": float(lim.get("velocity")),
                "effort": float(lim.get("effort")),
            }
        )
    for name, l in links.items():
        inert = l.find("inertial")
        if inert is None:
            continue
        o = inert.find("origin")
        I = inert.find("inertia")
        g = lambda k: float(I.get(k))
        out["inertials"][name] = {
            "mass": float(inert.find("mass").get("value")),
            "com": [float(v) for v in o.get("xyz").split()] if o is not None else [0.0, 0.0, 0.0],
            "inertia": [[g("ixx"), g("ixy"), g("ixz")], [g("ixy"), g("iyy"), g("iyz")], [g("ixz"), g("iyz"), g("izz")]],
        }
    return out


class ArmModel:
    """Serial chain base -> link_6 (flange). Frames follow the URDF."""

    def __init__(self, model: dict | str | None = None, T_world_base=None):
        if model is None:
            model = DEFAULT_MODEL
        if isinstance(model, str):
            with open(model) as f:
                model = json.load(f)
        self.model = model
        js = model["joints"]
        self.T_origin = np.stack([make_T(rpy_to_mat(j["rpy"]), j["xyz"]) for j in js])
        self.axis = np.array([np.array(j["axis"]) / np.linalg.norm(j["axis"]) for j in js])
        self.q_min = np.array([j["lower"] for j in js])
        self.q_max = np.array([j["upper"] for j in js])
        self.v_max = np.array([j["velocity"] for j in js])
        self.effort = np.array([j["effort"] for j in js])
        self.T_world_base = np.eye(4) if T_world_base is None else np.asarray(T_world_base, dtype=float)
        self._axis_z = [bool(np.allclose(a, [0.0, 0.0, 1.0])) for a in self.axis]
        self._T_joint_frames = None
        # Ideal geometry for analytic IK.
        self.d1 = js[0]["xyz"][2]
        self.a2 = js[2]["xyz"][0]
        self.d4 = -js[3]["xyz"][1]
        self.d6 = -js[5]["xyz"][1]
        inert = model["inertials"]
        base_name = js[0]["parent"]
        self.link_names = [base_name] + [j["child"] for j in js]
        self.masses = np.array([inert[n]["mass"] for n in self.link_names])
        self.coms = np.array([inert[n]["com"] for n in self.link_names])
        self.inertias = np.array([inert[n]["inertia"] for n in self.link_names])
        self.tool_mass = 0.0
        self.tool_com = np.zeros(3)
        self.tool_inertia = np.zeros((3, 3))

    # ------------------------------------------------------------------ forward kinematics
    def _joint_rot(self, i, q):
        if self._axis_z[i]:
            c, s = np.cos(q), np.sin(q)
            return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        from .geometry import axis_angle_to_mat

        return axis_angle_to_mat(self.axis[i], q)

    def fk_all(self, q):
        """World transforms of [base, link_1..link_6] (7, 4, 4)."""
        Ts = np.empty((7, 4, 4))
        T = self.T_world_base.copy()
        Ts[0] = T
        J = np.eye(4)
        for i in range(6):
            J[:3, :3] = self._joint_rot(i, q[i])
            T = T @ self.T_origin[i] @ J
            Ts[i + 1] = T
        return Ts

    def fk(self, q, T_flange_tool=None):
        T = self.fk_all(q)[6]
        return T if T_flange_tool is None else T @ T_flange_tool

    def fk_batch(self, Q):
        """World transforms of all links for N configurations: (N, 7, 4, 4). Assumes z joint axes."""
        Q = np.atleast_2d(Q)
        N = Q.shape[0]
        out = np.zeros((N, 7, 4, 4))
        T = np.broadcast_to(self.T_world_base, (N, 4, 4)).copy()
        out[:, 0] = T
        for i in range(6):
            c, s = np.cos(Q[:, i]), np.sin(Q[:, i])
            Rj = np.zeros((N, 4, 4))
            Rj[:, 0, 0] = c
            Rj[:, 0, 1] = -s
            Rj[:, 1, 0] = s
            Rj[:, 1, 1] = c
            Rj[:, 2, 2] = 1.0
            Rj[:, 3, 3] = 1.0
            T = T @ self.T_origin[i] @ Rj
            out[:, i + 1] = T
        return out

    def jacobian(self, q, T_flange_tool=None):
        """Geometric Jacobian of the tool frame origin, rows [vx vy vz wx wy wz] in the world frame."""
        return self.jacobian_batch(np.asarray(q, float)[None], T_flange_tool)[0]

    def jacobian_batch(self, Q, T_flange_tool=None):
        Ts = self.fk_batch(Q)
        T_tool = Ts[:, 6] if T_flange_tool is None else Ts[:, 6] @ T_flange_tool
        p = T_tool[:, :3, 3]
        J = np.zeros((Q.shape[0], 6, 6))
        for i in range(6):
            Tj = Ts[:, i] @ self.T_origin[i]
            z = Tj[:, :3, 2]
            J[:, :3, i] = np.cross(z, p - Tj[:, :3, 3])
            J[:, 3:, i] = z
        return J

    # ------------------------------------------------------------------ inverse kinematics
    def ik_analytic(self, T_flange):
        """Up to 8 ideal-geometry solutions for a flange pose, each tagged with branch flags."""
        T_b = inv_T(self.T_world_base) @ T_flange
        R6, p6 = T_b[:3, :3], T_b[:3, 3]
        pw = p6 - self.d6 * R6[:, 2]
        a, b = self.a2, self.d4
        sols = []
        for shoulder in (0, 1):
            q1 = np.arctan2(pw[1], pw[0]) + (np.pi if shoulder else 0.0)
            r = np.hypot(pw[0], pw[1]) * (-1.0 if shoulder else 1.0)
            u, v = r, pw[2] - self.d1
            c3 = (u * u + v * v - a * a - b * b) / (2 * a * b)
            if abs(c3) > 1.0 + 1e-9:
                continue
            c3 = np.clip(c3, -1.0, 1.0)
            for elbow in (0, 1):
                q3 = np.arccos(c3) * (1.0 if elbow == 0 else -1.0)
                q2 = np.arctan2(u, v) - np.arctan2(b * np.sin(q3), a + b * np.cos(q3))
                M = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
                R03 = rot_z(q1) @ M @ rot_z(q2) @ rot_z(np.pi / 2 + q3)
                W = (R03 @ rot_x(np.pi / 2)).T @ R6
                s5 = np.hypot(W[0, 2], W[1, 2])
                for wrist in (0, 1):
                    if wrist == 0:
                        q5 = np.arctan2(s5, W[2, 2])
                        if s5 > 1e-9:
                            q4 = np.arctan2(W[1, 2], W[0, 2])
                            q6 = np.arctan2(W[2, 1], -W[2, 0])
                        else:
                            q4 = 0.0
                            q6 = np.arctan2(W[1, 0], W[0, 0])
                    else:
                        q5 = np.arctan2(-s5, W[2, 2])
                        if s5 > 1e-9:
                            q4 = np.arctan2(-W[1, 2], -W[0, 2])
                            q6 = np.arctan2(-W[2, 1], W[2, 0])
                        else:
                            q4 = np.pi
                            q6 = np.arctan2(W[1, 0], W[0, 0]) - np.pi
                    sols.append((np.array([q1, q2, q3, q4, q5, q6]), (shoulder, elbow, wrist)))
        return sols

    def ik_refine(self, T_flange, q0, iters=30, tol_p=1e-6, tol_r=1e-6, damping=1e-4):
        q = np.array(q0, dtype=float)
        for _ in range(iters):
            e = pose_error(self.fk(q), T_flange)
            en = np.linalg.norm(e)
            if np.linalg.norm(e[:3]) < tol_p and np.linalg.norm(e[3:]) < tol_r:
                return q, True
            J = self.jacobian(q)
            # Levenberg-Marquardt style: damping shrinks with the residual so steps become Gauss-Newton.
            lam = damping * min(1.0, en / 1e-2) ** 2 + 1e-12
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            step = np.max(np.abs(dq))
            if step > 0.3:
                dq *= 0.3 / step
            q = q + dq
        e = pose_error(self.fk(q), T_flange)
        return q, bool(np.linalg.norm(e[:3]) < tol_p * 10 and np.linalg.norm(e[3:]) < tol_r * 10)

    def closest_equivalent(self, q, q_ref):
        """Shift each joint by multiples of 2 pi toward q_ref while staying inside limits."""
        q = np.array(q, dtype=float)
        for i in range(6):
            best, best_d = None, np.inf
            for k in range(-2, 3):
                c = q[i] + 2 * np.pi * k
                if self.q_min[i] - 1e-9 <= c <= self.q_max[i] + 1e-9 and abs(c - q_ref[i]) < best_d:
                    best, best_d = c, abs(c - q_ref[i])
            if best is None:
                return None
            q[i] = best
        return q

    def ik_all(self, T_flange, q_ref, tol_p=1e-5, tol_r=1e-5):
        """All refined solutions near q_ref, deduplicated. Returns list of (q, branch_flags)."""
        out = []
        for q, flags in self.ik_analytic(T_flange):
            qr, ok = self.ik_refine(T_flange, q, tol_p=tol_p, tol_r=tol_r)
            if not ok:
                continue
            qr = self.closest_equivalent(qr, q_ref)
            if qr is None:
                continue
            if any(np.max(np.abs(qr - o[0])) < 1e-4 for o in out):
                continue
            out.append((qr, flags))
        return out

    def ik_track(self, T_list, q_start, max_jump=0.15, iters=8, tol=1e-6):
        """Follow flange poses from q_start. Batched Newton from joint-space seeds; sequential fallback."""
        T_goal = np.asarray(T_list)
        N = len(T_goal)
        q_start = np.asarray(q_start, float)
        q_end, ok = self.ik_refine(T_goal[-1], q_start, iters=50)
        if ok:
            w = np.linspace(0.0, 1.0, N + 1)[1:, None]
            Q = q_start[None] + w * (q_end - q_start)[None]
            for _ in range(iters):
                E = pose_error_batch(self.fk_batch(Q)[:, 6], T_goal)
                if np.max(np.abs(E)) < tol:
                    break
                J = self.jacobian_batch(Q)
                Q = Q + np.linalg.solve(J, E[..., None])[..., 0]
            E = pose_error_batch(self.fk_batch(Q)[:, 6], T_goal)
            steps = np.abs(np.diff(np.vstack([q_start[None], Q]), axis=0))
            if np.max(np.abs(E)) < 1e-5 and steps.max() <= max_jump and np.all(Q >= self.q_min) and np.all(Q <= self.q_max):
                return Q
        qs = []
        q = q_start.copy()
        for T in T_goal:
            qn, ok = self.ik_refine(T, q, iters=50)
            if not ok or np.max(np.abs(qn - q)) > max_jump:
                return None
            if np.any(qn < self.q_min) or np.any(qn > self.q_max):
                return None
            qs.append(qn)
            q = qn
        return np.array(qs)

    # ------------------------------------------------------------------ dynamics
    def set_tool(self, mass, com, inertia):
        """Rigid tool attached to the flange: mass, CoM and inertia (about CoM) in the flange frame."""
        self.tool_mass = float(mass)
        self.tool_com = np.asarray(com, dtype=float)
        self.tool_inertia = np.asarray(inertia, dtype=float)

    def rnea(self, q, qd, qdd, gravity=(0.0, 0.0, -9.81), payload=None):
        """Joint torques from Newton-Euler recursion. payload = (mass, com_in_flange, inertia_about_com)."""
        g = np.asarray(gravity, dtype=float)
        Ts = self.fk_all(q)
        masses = list(self.masses[1:])
        coms = list(self.coms[1:])
        inertias = list(self.inertias[1:])
        extra = [(self.tool_mass, self.tool_com, self.tool_inertia)]
        if payload is not None:
            extra.append(payload)
        m6, c6, I6 = masses[5], coms[5], inertias[5]
        for m, c, I in extra:
            if m <= 0:
                continue
            mt = m6 + m
            ct = (m6 * c6 + m * np.asarray(c)) / mt
            d1 = c6 - ct
            d2 = np.asarray(c) - ct
            It = I6 + m6 * (d1 @ d1 * np.eye(3) - np.outer(d1, d1)) + np.asarray(I) + m * (d2 @ d2 * np.eye(3) - np.outer(d2, d2))
            m6, c6, I6 = mt, ct, It
        masses[5], coms[5], inertias[5] = m6, c6, I6
        w = np.zeros(3)
        wd = np.zeros(3)
        a = -g
        z_axes, origins, pc_list, F, N_ = [], [], [], [], []
        for i in range(6):
            Tj = Ts[i] @ self.T_origin[i]
            z = Tj[:3, :3] @ self.axis[i]
            o = Tj[:3, 3]
            if i > 0:
                a = a + np.cross(wd, o - origins[-1]) + np.cross(w, np.cross(w, o - origins[-1]))
            w_new = w + z * qd[i]
            wd = wd + z * qdd[i] + np.cross(w, z * qd[i])
            w = w_new
            R = Ts[i + 1][:3, :3]
            pc = Ts[i + 1][:3, 3] + R @ coms[i]
            ac = a + np.cross(wd, pc - o) + np.cross(w, np.cross(w, pc - o))
            Iw = R @ inertias[i] @ R.T
            F.append(masses[i] * ac)
            N_.append(Iw @ wd + np.cross(w, Iw @ w))
            z_axes.append(z)
            origins.append(o)
            pc_list.append(pc)
        tau = np.zeros(6)
        f = np.zeros(3)
        n = np.zeros(3)
        for i in reversed(range(6)):
            o = origins[i]
            n_new = N_[i] + np.cross(pc_list[i] - o, F[i]) + n + (np.cross(origins[i + 1] - o, f) if i < 5 else 0.0)
            f = F[i] + f
            n = n_new
            tau[i] = n @ z_axes[i]
        return tau

    def rnea_batch(self, Q, Qd, Qdd, gravity=(0.0, 0.0, -9.81), payload=None, payload_mask=None):
        """Vectorised rnea over N samples. payload applies where payload_mask is True."""
        Q, Qd, Qdd = (np.atleast_2d(x) for x in (Q, Qd, Qdd))
        N = Q.shape[0]
        g = np.asarray(gravity, dtype=float)
        Ts = self.fk_batch(Q)
        masses = np.tile(self.masses[1:], (N, 1))
        coms = np.tile(self.coms[1:][None], (N, 1, 1))
        inertias = np.tile(self.inertias[1:][None], (N, 1, 1, 1))
        extras = [(self.tool_mass, self.tool_com, self.tool_inertia, np.ones(N, bool))]
        if payload is not None:
            extras.append((*payload, np.ones(N, bool) if payload_mask is None else np.asarray(payload_mask, bool)))
        for m, c, I, sel in extras:
            if m <= 0 or not np.any(sel):
                continue
            m6, c6, I6 = masses[sel, 5], coms[sel, 5], inertias[sel, 5]
            mt = m6 + m
            ct = (m6[:, None] * c6 + m * np.asarray(c)[None]) / mt[:, None]
            d1 = c6 - ct
            d2 = np.asarray(c)[None] - ct
            eye = np.eye(3)[None]
            It = (I6 + m6[:, None, None] * (np.einsum("ni,ni->n", d1, d1)[:, None, None] * eye - np.einsum("ni,nj->nij", d1, d1))
                  + np.asarray(I)[None] + m * (np.einsum("ni,ni->n", d2, d2)[:, None, None] * eye - np.einsum("ni,nj->nij", d2, d2)))
            masses[sel, 5], coms[sel, 5], inertias[sel, 5] = mt, ct, It
        w = np.zeros((N, 3))
        wd = np.zeros((N, 3))
        a = np.broadcast_to(-g, (N, 3)).copy()
        Z, O, PC, F, Nn = [], [], [], [], []
        for i in range(6):
            Tj = Ts[:, i] @ self.T_origin[i]
            z = Tj[:, :3, :3] @ self.axis[i]
            o = Tj[:, :3, 3]
            if i > 0:
                r = o - O[-1]
                a = a + np.cross(wd, r) + np.cross(w, np.cross(w, r))
            zq = z * Qd[:, i : i + 1]
            wd = wd + z * Qdd[:, i : i + 1] + np.cross(w, zq)
            w = w + zq
            R = Ts[:, i + 1, :3, :3]
            pc = Ts[:, i + 1, :3, 3] + np.einsum("nij,nj->ni", R, coms[:, i])
            rc = pc - o
            ac = a + np.cross(wd, rc) + np.cross(w, np.cross(w, rc))
            Iw = R @ inertias[:, i] @ np.transpose(R, (0, 2, 1))
            F.append(masses[:, i : i + 1] * ac)
            Nn.append(np.einsum("nij,nj->ni", Iw, wd) + np.cross(w, np.einsum("nij,nj->ni", Iw, w)))
            Z.append(z)
            O.append(o)
            PC.append(pc)
        tau = np.zeros((N, 6))
        f = np.zeros((N, 3))
        n = np.zeros((N, 3))
        for i in reversed(range(6)):
            o = O[i]
            n_new = Nn[i] + np.cross(PC[i] - o, F[i]) + n + (np.cross(O[i + 1] - o, f) if i < 5 else 0.0)
            f = F[i] + f
            n = n_new
            tau[:, i] = np.einsum("ni,ni->n", n, Z[i])
        return tau

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.model, f, indent=1)
