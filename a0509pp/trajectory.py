"""Path time-parameterisation under joint and TCP caps, and 120 Hz reference resampling."""

from __future__ import annotations

from dataclasses import dataclass, field

import math

import numpy as np
from scipy.interpolate import CubicSpline

PHASES = ["transition", "descend", "close", "lift", "transfer", "lower", "open", "withdraw", "hold"]
PHASE_ID = {p: i for i, p in enumerate(PHASES)}


def quintic(tau):
    tau = np.clip(tau, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3
    return s, ds, dds


class JointPath:
    """C2 spline q(s), s in [0, 1], through joint waypoints parameterised by joint-space arc length."""

    def __init__(self, Q):
        Q = np.asarray(Q, dtype=float)
        if len(Q) == 1:
            Q = np.vstack([Q, Q])
        seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        keep = np.concatenate([[True], seg > 1e-9])
        Q = Q[keep]
        if len(Q) == 1:
            Q = np.vstack([Q, Q])
            seg = np.array([0.0])
        else:
            seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        self.length = float(seg.sum())
        self.Q = Q
        if self.length < 1e-9:
            self.s_knots = np.linspace(0.0, 1.0, len(Q))
            self.static = True
        else:
            self.s_knots = np.concatenate([[0.0], np.cumsum(seg)]) / self.length
            self.static = False
        bc = "natural" if len(Q) > 2 else "not-a-knot"
        self.spline = CubicSpline(self.s_knots, Q, axis=0, bc_type=bc) if len(Q) > 2 else None

    def eval(self, s):
        s = np.clip(np.asarray(s, float), 0.0, 1.0)
        if self.static:
            return np.broadcast_to(self.Q[0], s.shape + (6,)).copy(), np.zeros(s.shape + (6,)), np.zeros(s.shape + (6,))
        if self.spline is None:
            d = self.Q[1] - self.Q[0]
            q = self.Q[0] + s[..., None] * d
            return q, np.broadcast_to(d, q.shape).copy(), np.zeros_like(q)
        return self.spline(s), self.spline(s, 1), self.spline(s, 2)


@dataclass
class Segment:
    phase: str
    path: JointPath | None
    duration: float
    aperture_start: float
    aperture_end: float
    aperture_hold_extra: float = 0.0
    note: str = ""
    profile: object = None

    @property
    def q_start(self):
        return self.path.Q[0]

    @property
    def q_end(self):
        return self.path.Q[-1]


class Profile:
    """Trapezoidal path speed with cycloidal (sin) ramps: C2 position, continuous acceleration, zero at the ends."""

    def __init__(self, sd_max, sdd_max):
        self.sd_m = float(sd_max)
        self.ta = math.pi * self.sd_m / (2.0 * sdd_max)
        if self.ta * self.sd_m > 1.0:
            self.sd_m = math.sqrt(2.0 * sdd_max / math.pi)
            self.ta = 1.0 / self.sd_m
        self.T = 1.0 / self.sd_m + self.ta

    def stretch(self, factor):
        """Uniform time dilation (keeps shape): speeds / factor, accelerations / factor^2."""
        self.sd_m /= factor
        self.ta *= factor
        self.T *= factor

    def _up(self, t):
        w = math.pi / self.ta
        s = self.sd_m / 2.0 * (t - np.sin(w * t) / w)
        sd = self.sd_m / 2.0 * (1.0 - np.cos(w * t))
        sdd = self.sd_m * w / 2.0 * np.sin(w * t)
        return s, sd, sdd

    def eval(self, t):
        t = np.clip(np.asarray(t, float), 0.0, self.T)
        s = np.empty_like(t)
        sd = np.empty_like(t)
        sdd = np.empty_like(t)
        a = t < self.ta
        c = t > self.T - self.ta
        m = ~a & ~c
        s[a], sd[a], sdd[a] = self._up(t[a])
        s[m] = self.sd_m * self.ta / 2.0 + self.sd_m * (t[m] - self.ta)
        sd[m] = self.sd_m
        sdd[m] = 0.0
        su, sdu, sddu = self._up(self.T - t[c])
        s[c], sd[c], sdd[c] = 1.0 - su, sdu, -sddu
        return np.clip(s, 0.0, 1.0), sd, sdd


def time_scale(path: JointPath, arm, T_flange_tool, caps, dt_round=1.0 / 120.0, samples=200):
    """Fastest Profile satisfying joint speed/accel and TCP linear/angular speed caps along the path."""
    if path.static:
        return None
    s = np.linspace(0.0, 1.0, samples)
    q, dq, ddq = path.eval(s)
    J = arm.jacobian_batch(q, T_flange_tool)
    v_lin = np.linalg.norm(np.einsum("nij,nj->ni", J[:, :3], dq), axis=1)
    v_ang = np.linalg.norm(np.einsum("nij,nj->ni", J[:, 3:], dq), axis=1)
    eps = 1e-12
    sd_m = min(
        caps["joint_speed"] / max(np.abs(dq).max(), eps),
        caps["tcp_lin"] / max(v_lin.max(), eps),
        caps["tcp_ang"] / max(v_ang.max(), eps),
    )
    for _ in range(60):
        head = caps["joint_accel"] - np.abs(ddq) * sd_m**2
        if np.all(head > 0.05 * caps["joint_accel"]):
            sdd_m = float(np.min(head / np.maximum(np.abs(dq), eps)))
            break
        sd_m *= 0.9
    else:
        sdd_m = caps["joint_accel"] / max(np.abs(dq).max(), eps)
    prof = Profile(sd_m, sdd_m)
    prof.stretch(1.01)
    T_round = math.ceil(prof.T / dt_round) * dt_round
    prof.stretch(T_round / prof.T)
    return prof


@dataclass
class Reference:
    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    aperture: np.ndarray
    phase: np.ndarray
    segment_bounds: list = field(default_factory=list)

    @property
    def duration(self):
        return float(self.t[-1])


def build_reference(segments, gripper_speed, dt=1.0 / 120.0):
    """Concatenate segments into 120 Hz arrays. Gripper segments hold the arm and ramp the aperture."""
    ts, qs, qds, aps, phs, bounds = [], [], [], [], [], []
    t0 = 0.0
    for seg in segments:
        n = int(round(seg.duration / dt))
        if n <= 0:
            continue
        tl = np.arange(1, n + 1) * dt
        if seg.phase in ("close", "open"):
            q = np.broadcast_to(seg.q_start, (n, 6)).copy()
            qd = np.zeros((n, 6))
            ramp = abs(seg.aperture_end - seg.aperture_start) / max(gripper_speed, 1e-6)
            frac = np.clip(tl / max(ramp, 1e-9), 0.0, 1.0)
            ap = seg.aperture_start + frac * (seg.aperture_end - seg.aperture_start)
        else:
            S, sd, _ = seg.profile.eval(tl)
            q, dq, _ = seg.path.eval(S)
            qd = dq * sd[:, None]
            ap = np.full(n, seg.aperture_end)
        bounds.append((seg.phase, sum(len(x) for x in ts), n))
        ts.append(t0 + tl)
        qs.append(q)
        qds.append(qd)
        aps.append(ap)
        phs.append(np.full(n, PHASE_ID[seg.phase]))
        t0 += n * dt
    q0 = segments[0].q_start
    a0 = segments[0].aperture_start
    ref = Reference(
        t=np.concatenate([[0.0]] + ts),
        q=np.vstack([q0[None]] + qs),
        qd=np.vstack([np.zeros((1, 6))] + qds),
        aperture=np.concatenate([[a0]] + aps),
        phase=np.concatenate([[PHASE_ID[segments[0].phase]]] + phs),
        segment_bounds=[(p, s + 1, n) for p, s, n in bounds],
    )
    return ref


def check_caps(ref: Reference, arm, T_flange_tool, caps, tol=1.02):
    """Finite-difference verification of caps on the 120 Hz reference. Returns (ok, utilisation dict)."""
    dt = np.diff(ref.t)
    qd = np.diff(ref.q, axis=0) / dt[:, None]
    qdd = np.diff(qd, axis=0) / dt[1:, None]
    Ts = arm.fk_batch(ref.q)[:, 6] @ T_flange_tool
    v_lin = np.linalg.norm(np.diff(Ts[:, :3, 3], axis=0), axis=1) / dt
    R = Ts[:, :3, :3]
    rel = np.einsum("nji,njk->nik", R[:-1], R[1:])
    ang = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)) / dt
    util = {
        "joint_speed": np.max(np.abs(qd), axis=0) / caps["joint_speed"],
        "joint_accel": np.max(np.abs(qdd), axis=0) / caps["joint_accel"] if len(qdd) else np.zeros(6),
        "tcp_lin": float(v_lin.max() / caps["tcp_lin"]) if len(v_lin) else 0.0,
        "tcp_ang": float(ang.max() / caps["tcp_ang"]) if len(ang) else 0.0,
    }
    ok = (
        util["joint_speed"].max() <= tol
        and util["joint_accel"].max() <= tol * 1.05
        and util["tcp_lin"] <= tol
        and util["tcp_ang"] <= tol
    )
    return bool(ok), util


def knots(ref: Reference, n=32):
    """(n, 15) = [q(6), qd(6), aperture, elapsed_time, phase_id] at time-normalised samples."""
    T = ref.duration
    tk = np.linspace(0.0, T, n)
    q = np.stack([np.interp(tk, ref.t, ref.q[:, i]) for i in range(6)], 1)
    qd = np.stack([np.interp(tk, ref.t, ref.qd[:, i]) for i in range(6)], 1)
    ap = np.interp(tk, ref.t, ref.aperture)
    idx = np.clip(np.searchsorted(ref.t, tk), 0, len(ref.t) - 1)
    ph = ref.phase[idx].astype(float)
    return np.concatenate([q, qd, ap[:, None], tk[:, None], ph[:, None]], axis=1).astype(np.float32)
