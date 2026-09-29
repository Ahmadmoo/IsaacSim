"""Candidate grasp-and-motion plans: grasp hypotheses x IK branches x transport routes, with rejection reasons."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .collision import Box, object_spheres, sphere_box_distance, world_boxes
from .geometry import inv_T, make_T, rot_z
from .trajectory import PHASE_ID, JointPath, Segment, build_reference, check_caps, knots, time_scale

REJECT_REASONS = ["ik_failure", "transition_failure", "collision", "joint_velocity_violation", "timeout", "grasp_geometry"]


def tcp_rot(yaw):
    """TCP rotation: +z down (approach), +y (finger closing axis) at world yaw."""
    y = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([np.cross(y, z), y, z])


@dataclass
class Grasp:
    gid: int
    axis: int
    width: float
    depth: float
    height_mode: str
    z_pad: float
    tcp_yaw: float
    close_aperture: float
    T_tcp: np.ndarray
    T_tcp_object: np.ndarray
    obj_dims: tuple = (0.04, 0.04, 0.05)

    def to_dict(self):
        return {"gid": self.gid, "axis": self.axis, "width": self.width, "depth": self.depth, "height_mode": self.height_mode,
                "z_pad": self.z_pad, "tcp_yaw": self.tcp_yaw, "close_aperture": self.close_aperture,
                "T_world_tcp_grasp": self.T_tcp.tolist(), "T_tcp_object": self.T_tcp_object.tolist(), "obj_dims": list(self.obj_dims)}


@dataclass
class Candidate:
    cid: int
    grasp: Grasp
    branch: tuple
    branch_name: str
    route: str
    valid: bool = False
    rejection: str = ""
    rejection_detail: str = ""
    segments: list = field(default_factory=list)
    reference: object = None
    knots: np.ndarray | None = None
    T_tcp_place: np.ndarray | None = None
    place_yaw: float = 0.0
    clearance: dict = field(default_factory=dict)
    source: str = "planner_ik"

    @property
    def duration(self):
        return 0.0 if self.reference is None else self.reference.duration

    def summary(self):
        return {"cid": self.cid, "valid": self.valid, "rejection": self.rejection, "detail": self.rejection_detail,
                "grasp": self.grasp.to_dict(), "branch": list(self.branch), "branch_name": self.branch_name,
                "route": self.route, "duration": self.duration, "source": self.source,
                "T_world_tcp_place": None if self.T_tcp_place is None else self.T_tcp_place.tolist(),
                "place_yaw": self.place_yaw, "clearance": self.clearance}


class Planner:
    def __init__(self, cfg, arm, gripper, spheres):
        self.cfg = cfg
        self.C = cfg.candidates
        self.arm = arm
        self.g = gripper
        self.rs = spheres
        self.T_ft = gripper.T_flange_tcp()
        self.T_tf = inv_T(self.T_ft)
        r = cfg.robot
        self.caps = {"joint_speed": r.joint_speed_cap, "joint_accel": r.joint_accel_cap,
                     "tcp_lin": r.tcp_lin_speed_cap, "tcp_ang": r.tcp_ang_speed_cap}
        self.dt = cfg.control_dt
        self.z_palm = 0.075

    # ------------------------------------------------------------------ helpers
    def flange(self, T_tcp):
        return T_tcp @ self.T_tf

    def branch_name(self, q, flags):
        """shoulder front/back, elbow above/below the shoulder-wrist line, wrist sign of q5."""
        Ts = self.arm.fk_all(q)
        sh, el, wc = Ts[2][:3, 3], Ts[3][:3, 3], Ts[4][:3, 3]
        d = wc - sh
        t = float((el - sh) @ d / max(d @ d, 1e-12))
        up = el[2] > sh[2] + t * d[2]
        return f"{'front' if flags[0] == 0 else 'back'}-{'up' if up else 'down'}-{'w+' if q[4] >= 0 else 'w-'}"

    def cartesian(self, T_a, T_b, q_start):
        dist = np.linalg.norm(T_b[:3, 3] - T_a[:3, 3])
        n = max(2, int(math.ceil(dist / self.C.cartesian_step)) + 1)
        Ts = []
        for s in np.linspace(0.0, 1.0, n)[1:]:
            T = T_a.copy()
            T[:3, 3] = T_a[:3, 3] + s * (T_b[:3, 3] - T_a[:3, 3])
            Ts.append(self.flange(T))
        Q = self.arm.ik_track(Ts, q_start, max_jump=self.C.max_joint_jump)
        return None if Q is None else np.vstack([q_start[None], Q])

    def ik_near(self, T_tcp, q_ref, flags=None):
        sols = self.arm.ik_all(self.flange(T_tcp), q_ref, self.C.ik_tol_pos, self.C.ik_tol_rot)
        if flags is not None:
            sols = [s for s in sols if s[1][:2] == flags[:2]] or sols
        if not sols:
            return None
        return min(sols, key=lambda s: np.max(np.abs(s[0] - q_ref)))[0]

    # ------------------------------------------------------------------ grasps
    def grasps(self, target, q_ref):
        C, g = self.C, self.g
        dims = np.asarray(target.dims, float)
        h = float(dims[2])
        R_obj = rot_z(target.yaw)
        out = []
        square = abs(dims[0] - dims[1]) < 0.002
        axes = [0] if square else [0, 1]
        for axis in axes:
            w, d = float(dims[axis]), float(dims[1 - axis])
            if w + 2 * C.min_open_clearance > self.g.g.open_aperture or w < 0.01:
                continue
            a_close = max(0.0, w - self.cfg.gripper.close_margin)
            th_close = float(g.theta(a_close))
            z_min = g.pad_bottom_extent() + 0.002 + C.min_pad_table_clearance
            modes = {}
            for mode in C.grasp_height_options:
                z = h / 2.0 if mode == "mid" else h - self.cfg.gripper.pad_half_height - 0.004
                z = max(z, z_min)
                if all(abs(z - v) > 0.005 for v in modes.values()):
                    modes[mode] = z
            u = R_obj[:, axis]
            base_yaw = math.atan2(u[1], u[0])
            yaws = [base_yaw, base_yaw + math.pi]
            if square:
                yaws += [base_yaw + math.pi / 2, base_yaw - math.pi / 2]
            if C.wrist_sym_dedupe:
                yaws = [min(yaws, key=lambda y: self._wrist_cost(y, target, q_ref))]
            for mode, z in modes.items():
                pad_z_close = float(g.pad_center_z(th_close))
                if (h - z) > pad_z_close - self.z_palm - 0.005:
                    continue
                for yaw in yaws:
                    p = np.array([target.center[0], target.center[1], z + g.pinch_offset(a_close)])
                    T_tcp = make_T(tcp_rot(yaw), p)
                    T_obj = make_T(R_obj, [target.center[0], target.center[1], h / 2.0])
                    out.append(Grasp(len(out), axis, w, d, mode, z, float(yaw), a_close, T_tcp, inv_T(T_tcp) @ T_obj,
                                     tuple(float(x) for x in dims)))
        return out

    def _wrist_cost(self, yaw, target, q_ref):
        """Wrist rotation needed relative to the current TCP yaw carried around by the base joint."""
        R = self.arm.fk(q_ref, self.T_ft)[:3, :3]
        offset = math.atan2(R[1, 1], R[0, 1]) - q_ref[0]
        neutral = math.atan2(target.center[1], target.center[0]) + offset
        return abs(math.remainder(yaw - neutral, 2 * math.pi))

    # ------------------------------------------------------------------ place pose
    def place_pose(self, grasp, target, tray_xy):
        L = self.cfg.layout
        dyaw = math.atan2(tray_xy[1], tray_xy[0]) - math.atan2(target.center[1], target.center[0])
        floor_top = L.tray_floor
        z_pad = floor_top + self.C.place_drop_gap + grasp.z_pad
        z_tcp = z_pad + self.g.pinch_offset(grasp.close_aperture)
        for extra in (dyaw, 0.0, dyaw + math.pi / 2, dyaw - math.pi / 2):
            yaw_tcp = grasp.tcp_yaw + extra
            T = make_T(tcp_rot(yaw_tcp), [tray_xy[0], tray_xy[1], z_tcp])
            T_obj = T @ grasp.T_tcp_object
            corners = np.array([[sx, sy] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5)]) * np.asarray(target.dims[:2])
            xy = corners @ T_obj[:2, :2].T + T_obj[:2, 3]
            margin = self.cfg.labels.footprint_margin + 0.004
            if np.all(np.abs(xy[:, 0] - tray_xy[0]) <= L.tray_interior[0] / 2 - margin) and np.all(
                np.abs(xy[:, 1] - tray_xy[1]) <= L.tray_interior[1] / 2 - margin
            ):
                return T, float(yaw_tcp)
        return None, 0.0

    # ------------------------------------------------------------------ collision evaluation
    def clearance(self, ref, grasp, target_box, boxes, T_tcp_place=None, dims=None):
        """Min env/self clearance over the reference with phase-dependent allowed contacts."""
        sub = max(1, int(round(self.cfg.features.sample_dt / self.dt)))
        idx = np.arange(0, len(ref.t), sub)
        idx = np.unique(np.concatenate([idx, [len(ref.t) - 1]]))
        res = self._clearance_at(ref, idx, grasp, target_box, boxes, T_tcp_place, dims)
        k = self.cfg.features.refine_factor
        if res["env_min"] < self.cfg.features.refine_clearance or res["self_min"] < self.cfg.features.refine_clearance:
            j = res["_argmins"]
            extra = []
            for jj in j:
                lo, hi = max(0, jj - sub), min(len(ref.t) - 1, jj + sub)
                extra.append(np.arange(lo, hi + 1, max(1, sub // k)))
            idx2 = np.unique(np.concatenate([idx] + extra))
            res = self._clearance_at(ref, idx2, grasp, target_box, boxes, T_tcp_place, dims)
        res.pop("_argmins")
        return res

    def _clearance_at(self, ref, idx, grasp, target_box, boxes, T_tcp_place, dims):
        ph = ref.phase[idx]
        th = self.g.theta(ref.aperture[idx])
        held = (ph >= PHASE_ID["lift"]) & (ph <= PHASE_ID["open"])
        closing = ph == PHASE_ID["close"]
        obj_s = object_spheres(dims)
        obj_tcp = obj_s.copy()
        obj_tcp[:, :3] = obj_s[:, :3] @ grasp.T_tcp_object[:3, :3].T + grasp.T_tcp_object[:3, 3]
        C, r, tags = self.rs.compute(ref.q[idx], th, obj_tcp)
        tags = np.array(tags)
        is_obj = tags == "held_object"
        is_finger = np.array([any(k in t for k in ("fingertip", "inner_finger", "outer_finger", "inner_knuckle")) for t in tags])
        is_base = tags == "base_link"
        all_boxes = list(boxes)
        names = [b.name for b in all_boxes]
        tgt_i = None
        if target_box is not None:
            all_boxes.append(target_box)
            tgt_i = len(all_boxes) - 1
        placed_i = None
        if T_tcp_place is not None:
            T_obj = T_tcp_place @ grasp.T_tcp_object
            T_obj[2, 3] -= self.C.place_drop_gap
            all_boxes.append(Box("placed_object", T_obj[:3, 3], T_obj[:3, :3], np.asarray(dims) / 2.0, 0.0, "target"))
            placed_i = len(all_boxes) - 1
        names = [b.name for b in all_boxes]
        D = sphere_box_distance(C, r, all_boxes)
        mask = np.zeros(D.shape, bool)
        mask[:, is_base, :] = True
        mask[np.ix_(~held, is_obj, np.arange(D.shape[2]))] = True
        kinds = np.array([b.kind for b in all_boxes])
        if tgt_i is not None:
            before = ph <= PHASE_ID["close"]
            mask[np.ix_(~before, np.ones(len(tags), bool), [tgt_i])] = True
            mask[np.ix_(closing, is_finger, [tgt_i])] = True
        if placed_i is not None:
            before_release = ph < PHASE_ID["withdraw"]
            mask[np.ix_(before_release, np.ones(len(tags), bool), [placed_i])] = True
        lift = ph == PHASE_ID["lift"]
        tbl = [i for i, b in enumerate(all_boxes) if b.kind == "table"]
        mask[np.ix_(lift, is_obj, tbl)] = True
        tray_floor = [i for i, n in enumerate(names) if n == "tray_floor"]
        lowering = (ph == PHASE_ID["lower"]) | (ph == PHASE_ID["open"])
        mask[np.ix_(lowering, is_obj, tray_floor)] = True
        D = np.where(mask, np.inf, D)
        flat = D.reshape(len(idx), -1)
        env_t = flat.min(1)
        jt = int(np.argmin(env_t))
        js = int(np.argmin(flat[jt]))
        si, bi = divmod(js, D.shape[2])
        pairs, _ = self.rs.self_pairs(list(tags))
        Cs = C.copy()
        Cs[np.ix_(~held, is_obj, np.arange(3))] = 1e3
        self_t, self_k = self.rs.self_distance(Cs, r, list(tags))
        rule = self.rs.joint_rule_margin(ref.q[idx])
        kt = int(np.argmin(self_t))
        kp = int(self_k[kt])
        conf = float(all_boxes[bi].confidence)
        return {
            "env_min": float(env_t[jt]), "env_time": float(ref.t[idx[jt]]), "env_phase": int(ph[jt]),
            "env_pair": [str(tags[si]), names[bi]], "env_box_kind": str(kinds[bi]), "env_confidence": conf,
            "self_min": float(self_t[kt]), "self_time": float(ref.t[idx[kt]]), "self_phase": int(ph[kt]),
            "self_pair": [str(tags[pairs[kp, 0]]), str(tags[pairs[kp, 1]])] if len(pairs) else ["", ""],
            "per_phase_env_min": [float(env_t[ph == p].min()) if np.any(ph == p) else float("inf") for p in range(9)],
            "joint_rule_min": float(rule.min()),
            "_argmins": [int(idx[jt]), int(idx[kt])],
        }

    # ------------------------------------------------------------------ main entry
    def propose(self, q_now, percep, tray_xy, rng=None, true_boxes=None):
        """Returns (selected candidates list, pool summary dict). Obstacles come from percep (policy-visible)."""
        rng = rng or np.random.default_rng(0)
        q_now = np.asarray(q_now, float)
        pool_summary = {"counts": {}, "rejected": []}
        if percep.target is None:
            pool_summary["counts"]["no_target"] = 1
            return [], pool_summary
        target = percep.target
        pc = self.cfg.perception
        obstacles = [Box.from_xyzyaw(f"obstacle_{i}", o.center, o.yaw, o.dims, pc.planning_buffer + pc.measurement_allowance,
                                     "obstacle", o.confidence) for i, o in enumerate(percep.obstacles)]
        boxes = world_boxes(self.cfg.layout, tray_xy, obstacles)
        target_box = Box.from_xyzyaw("target", target.center, target.yaw, target.dims, 0.0, "target", target.confidence)
        routes = [r for r in self.C.routes if r in ("direct", "high") or obstacles]
        cands = []
        cid = 0
        for grasp in self.grasps(target, q_now):
            T_place, place_yaw = self.place_pose(grasp, target, tray_xy)
            sols = self.arm.ik_all(self.flange(grasp.T_tcp), q_now, self.C.ik_tol_pos, self.C.ik_tol_rot)
            if not sols:
                c = Candidate(cid, grasp, (-1, -1, -1), "none", "-", rejection="ik_failure", rejection_detail="grasp pose unreachable")
                cands.append(c)
                cid += 1
                continue
            for q_g, flags in sols:
                bname = self.branch_name(q_g, flags)
                base = self._grasp_side(q_now, grasp, q_g, target, boxes, target_box)
                for route in routes:
                    c = Candidate(cid, grasp, tuple(int(f) for f in flags), bname, route, T_tcp_place=T_place, place_yaw=place_yaw)
                    cid += 1
                    cands.append(c)
                    if T_place is None:
                        c.rejection, c.rejection_detail = "grasp_geometry", "object does not fit tray at any place yaw"
                        continue
                    if isinstance(base, str):
                        c.rejection, c.rejection_detail = base.split(":", 1)[0], base
                        continue
                    self._complete(c, base, target, boxes, target_box, tray_xy)
                    if sum(x.valid for x in cands) >= self.C.pool_limit:
                        break
        for c in cands:
            key = c.rejection if not c.valid else "valid"
            pool_summary["counts"][key] = pool_summary["counts"].get(key, 0) + 1
        pool_summary["rejected"] = [c.summary() for c in cands if not c.valid][:64]
        valid = self._dedupe([c for c in cands if c.valid])
        pool_summary["counts"]["valid_after_dedupe"] = len(valid)
        selected = self._select(valid, rng)
        for i, c in enumerate(selected):
            c.cid = i
        return selected, pool_summary

    def _grasp_side(self, q_now, grasp, q_g, target, boxes, target_box):
        C = self.C
        T_pre = grasp.T_tcp.copy()
        T_pre[2, 3] += C.pregrasp_height
        up = self.cartesian(grasp.T_tcp, T_pre, q_g)
        if up is None:
            return "ik_failure: descend path"
        Q_desc = up[::-1]
        q_pre = Q_desc[0]
        obs_top = max([b.center[2] + b.half[2] + b.margin for b in boxes if b.kind == "obstacle"] + [0.0])
        lift_h = max(C.lift_height_min, obs_top + C.obstacle_clearance) if obs_top > 0 else C.lift_height_min
        T_lift = grasp.T_tcp.copy()
        T_lift[2, 3] += lift_h
        Q_lift = self.cartesian(grasp.T_tcp, T_lift, q_g)
        if Q_lift is None:
            return "ik_failure: lift path"
        q_start = self.arm.closest_equivalent(q_now, q_now)
        trans = [q_start, q_pre]
        seg = Segment("transition", JointPath(np.array(trans)), 0.0, self.g.g.open_aperture, self.g.g.open_aperture)
        seg.profile = time_scale(seg.path, self.arm, self.T_ft, self.caps, self.dt)
        seg.duration = 0.0 if seg.profile is None else seg.profile.T
        gc = self.cfg.gripper
        grip_t = abs(gc.open_aperture - grasp.close_aperture) / gc.speed + gc.settle_time
        # Lower bound for the rest: descend and lift at the TCP cap, two gripper actions, lower + withdraw.
        rest = 2 * grip_t + (2 * C.pregrasp_height + lift_h + C.withdraw_height) / self.caps["tcp_lin"] + 1.0
        if seg.duration + rest > C.max_duration:
            return f"timeout: transition {seg.duration:.1f} s + remaining lower bound {rest:.1f} s"
        return {"Q_desc": Q_desc, "Q_lift": Q_lift, "T_lift": T_lift, "q_start": q_start, "transition": seg, "lift_h": lift_h}

    def _transfer_paths(self, route, q_lift, T_lift, T_preplace, grasp, tray_xy, flags):
        C = self.C
        z_travel = T_lift[2, 3]
        if route == "direct":
            q_pp = self.ik_near(T_preplace, q_lift, flags)
            return None if q_pp is None else [q_lift, q_pp]
        if route == "high":
            zh = max(z_travel, T_preplace[2, 3]) + C.route_high_extra
            W1 = T_lift.copy()
            W1[2, 3] = zh
            W2 = T_preplace.copy()
            W2[2, 3] = zh
            q1 = self.ik_near(W1, q_lift, flags)
            if q1 is None:
                return None
            q2 = self.ik_near(W2, q1, flags)
            if q2 is None:
                return None
            q_pp = self.ik_near(T_preplace, q2, flags)
            return None if q_pp is None else [q_lift, q1, q2, q_pp]
        if route in ("left", "right"):
            a, b = T_lift[:2, 3], T_preplace[:2, 3]
            mid = (a + b) / 2.0
            d = b - a
            n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
            sgn = 1.0 if route == "left" else -1.0
            W = T_lift.copy()
            W[:2, 3] = mid + sgn * C.route_lateral_offset * n
            W[2, 3] = max(z_travel, T_preplace[2, 3])
            yaw_mid = math.atan2(T_lift[1, 1], T_lift[0, 1]) + 0.5 * math.remainder(
                math.atan2(T_preplace[1, 1], T_preplace[0, 1]) - math.atan2(T_lift[1, 1], T_lift[0, 1]), 2 * math.pi)
            W[:3, :3] = tcp_rot(yaw_mid)
            q1 = self.ik_near(W, q_lift, flags)
            if q1 is None:
                return None
            q_pp = self.ik_near(T_preplace, q1, flags)
            return None if q_pp is None else [q_lift, q1, q_pp]
        return None

    def _complete(self, c, base, target, boxes, target_box, tray_xy):
        C, gc = self.C, self.cfg.gripper
        grasp = c.grasp
        a_open = gc.open_aperture
        a_close = grasp.close_aperture
        Q_desc, Q_lift, T_lift = base["Q_desc"], base["Q_lift"], base["T_lift"]
        q_g = Q_desc[-1]
        T_preplace = c.T_tcp_place.copy()
        T_preplace[2, 3] += C.pregrasp_height
        way = self._transfer_paths(c.route, Q_lift[-1], T_lift, T_preplace, grasp, tray_xy, c.branch)
        if way is None:
            c.rejection, c.rejection_detail = "ik_failure", f"transfer waypoints ({c.route})"
            return
        q_pp = way[-1]
        Q_lower = self.cartesian(T_preplace, c.T_tcp_place, q_pp)
        if Q_lower is None:
            c.rejection, c.rejection_detail = "ik_failure", "lower path"
            return
        T_wd = c.T_tcp_place.copy()
        T_wd[2, 3] += C.withdraw_height
        Q_wd = self.cartesian(c.T_tcp_place, T_wd, Q_lower[-1])
        if Q_wd is None:
            c.rejection, c.rejection_detail = "ik_failure", "withdraw path"
            return
        close_t = abs(a_open - a_close) / gc.speed + gc.settle_time
        open_t = abs(a_open - a_close) / gc.speed + gc.settle_time
        segs = [
            base["transition"],
            Segment("descend", JointPath(Q_desc), 0.0, a_open, a_open),
            Segment("close", JointPath(q_g[None]), close_t, a_open, a_close),
            Segment("lift", JointPath(Q_lift), 0.0, a_close, a_close),
            Segment("transfer", JointPath(np.array(way)), 0.0, a_close, a_close),
            Segment("lower", JointPath(Q_lower), 0.0, a_close, a_close),
            Segment("open", JointPath(Q_lower[-1][None]), open_t, a_close, a_open),
            Segment("withdraw", JointPath(Q_wd), 0.0, a_open, a_open),
        ]
        for s in segs:
            if s.phase not in ("close", "open") and s.profile is None:
                s.profile = time_scale(s.path, self.arm, self.T_ft, self.caps, self.dt)
                s.duration = 0.0 if s.profile is None else s.profile.T
            if np.any(s.path.Q < self.arm.q_min - 1e-6) or np.any(s.path.Q > self.arm.q_max + 1e-6):
                c.rejection, c.rejection_detail = "joint_velocity_violation", f"joint limit in {s.phase}"
                return
        for a, b in zip(segs[:-1], segs[1:]):
            if np.max(np.abs(a.path.Q[-1] - b.path.Q[0])) > 1e-6:
                c.rejection, c.rejection_detail = "transition_failure", f"discontinuity {a.phase}->{b.phase}"
                return
        c.segments = segs
        ref = build_reference(segs, gc.speed, self.dt)
        if ref.duration > C.max_duration:
            c.rejection, c.rejection_detail = "timeout", f"nominal duration {ref.duration:.1f} s"
            c.reference = ref
            return
        ok, util = check_caps(ref, self.arm, self.T_ft, self.caps)
        if not ok:
            c.rejection, c.rejection_detail = "joint_velocity_violation", str({k: np.max(v) for k, v in util.items()})
            return
        c.reference = ref
        c.clearance = self.clearance(ref, grasp, target_box, boxes, c.T_tcp_place, target.dims)
        if c.clearance["env_min"] < C.min_clearance_env:
            phase = int(c.clearance["env_phase"])
            c.rejection = "transition_failure" if phase == PHASE_ID["transition"] else "collision"
            c.rejection_detail = f"env {c.clearance['env_pair']} {c.clearance['env_min']:.3f} m"
            return
        if c.clearance["joint_rule_min"] < 0.0:
            c.rejection, c.rejection_detail = "collision", f"self joint rule {c.clearance['joint_rule_min']:.3f} rad"
            return
        if c.clearance["self_min"] < C.min_clearance_self:
            c.rejection, c.rejection_detail = "collision", f"self {c.clearance['self_pair']} {c.clearance['self_min']:.3f} m"
            return
        c.knots = knots(ref, C.knots)
        c.valid = True

    def _dedupe(self, cands):
        out = []
        for c in cands:
            dup = False
            for o in out:
                if o.grasp.gid == c.grasp.gid and np.max(np.abs(o.knots[:, :6] - c.knots[:, :6])) < self.C.dedupe_joint_tol:
                    dup = True
                    break
            if not dup:
                out.append(c)
        return out

    def _select(self, cands, rng):
        """Farthest-point selection in time-normalised joint-knot space (diversity, not quality)."""
        k = self.C.max_candidates
        if len(cands) <= k:
            return list(cands)
        X = np.stack([c.knots[:, :6].ravel() for c in cands])
        chosen = [int(rng.integers(len(cands)))]
        d = np.linalg.norm(X - X[chosen[0]], axis=1)
        while len(chosen) < k:
            i = int(np.argmax(d))
            chosen.append(i)
            d = np.minimum(d, np.linalg.norm(X - X[i], axis=1))
        return [cands[i] for i in sorted(chosen)]


def validity_mask(cands, k):
    m = np.zeros(k, bool)
    m[: len(cands)] = True
    return m
