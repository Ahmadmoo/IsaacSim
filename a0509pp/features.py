"""Motion-dependent physical features for each candidate (section 9). Values carry their model provenance."""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from .collision import Box, world_boxes
from .trajectory import PHASE_ID

FEATURE_NAMES = (
    [f"jl_margin_{i}" for i in range(1, 7)] + ["jl_margin_min"]
    + [f"vel_util_{i}" for i in range(1, 7)] + [f"acc_util_{i}" for i in range(1, 7)]
    + ["tcp_lin_util", "tcp_ang_util", "vel_util_max", "acc_util_max"]
    + ["sigma_min", "cond_max", "sigma_min_phase", "sigma_min_time_frac"]
    + ["cap_residual_max", "cap_util_max", "cap_residual_phase"]
    + ["env_clear_min", "env_clear_phase", "env_clear_conf", "self_clear_min", "self_clear_phase", "joint_rule_min"]
    + [f"env_clear_phase_{p}" for p in range(8)]
    + [f"track_err_pred_{i}" for i in range(1, 7)] + ["track_err_pred_max"]
    + [f"torque_util_{i}" for i in range(1, 7)] + ["torque_util_max"]
    + ["duration", "joint_path_len", "tcp_path_len", "transition_dist", "intervention_est"]
    + ["grasp_width", "grasp_depth", "close_aperture", "open_clearance", "grasp_height_frac", "grasp_height_mode_high",
       "grasp_axis", "target_conf", "target_height", "place_yaw_delta", "route_direct", "route_high", "route_lateral"]
)


def _finite(x, cap=1.0):
    return float(np.clip(x, -cap, cap)) if np.isfinite(x) else float(cap)


class FeatureComputer:
    def __init__(self, cfg, models):
        self.cfg = cfg
        self.m = models
        self.arm = models.arm
        self.T_ft = models.T_flange_tcp
        self.D_task = np.diag(cfg.features.task_scale)
        self.D_joint = np.diag([cfg.robot.joint_speed_cap] * 6)
        self.caps = {"joint_speed": cfg.robot.joint_speed_cap, "joint_accel": cfg.robot.joint_accel_cap,
                     "tcp_lin": cfg.robot.tcp_lin_speed_cap, "tcp_ang": cfg.robot.tcp_ang_speed_cap}

    def compute(self, cand, percep, tray_xy, model="nominal", spec=None):
        """model: 'nominal' (perceived scene, nominal robot model) or 'oracle' (true scene from spec)."""
        ref = cand.reference
        f = {}
        prov = {}
        q, qd, t, ph = ref.q, ref.qd, ref.t, ref.phase
        span = self.arm.q_max - self.arm.q_min
        m = np.minimum(q - self.arm.q_min, self.arm.q_max - q) / span
        for i in range(6):
            f[f"jl_margin_{i + 1}"] = float(m[:, i].min())
        f["jl_margin_min"] = float(m.min())
        dt = np.diff(t)
        qdd = np.diff(qd, axis=0) / dt[:, None]
        vu = np.abs(qd).max(0) / self.caps["joint_speed"]
        au = np.abs(qdd).max(0) / self.caps["joint_accel"]
        for i in range(6):
            f[f"vel_util_{i + 1}"] = float(vu[i])
            f[f"acc_util_{i + 1}"] = float(au[i])
        Tt = self.arm.fk_batch(q)[:, 6] @ self.T_ft
        v_lin = np.linalg.norm(np.diff(Tt[:, :3, 3], axis=0), axis=1) / dt
        R = Tt[:, :3, :3]
        rel = np.einsum("nji,njk->nik", R[:-1], R[1:])
        ang = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)) / dt
        f["tcp_lin_util"] = float(v_lin.max() / self.caps["tcp_lin"])
        f["tcp_ang_util"] = float(ang.max() / self.caps["tcp_ang"])
        f["vel_util_max"] = float(vu.max())
        f["acc_util_max"] = float(au.max())
        prov["limits"] = "reference trajectory vs configured caps"

        sub = max(1, int(round(self.cfg.features.sample_dt / (t[1] - t[0]))))
        idx = np.arange(0, len(t), sub)
        J = self.arm.jacobian_batch(q[idx], self.T_ft)
        Js = np.linalg.inv(self.D_task)[None] @ J @ self.D_joint[None]
        sv = np.linalg.svd(Js, compute_uv=False)
        smin = sv[:, -1]
        cond = sv[:, 0] / np.maximum(smin, 1e-12)
        k = int(np.argmin(smin))
        f["sigma_min"] = float(smin[k])
        f["cond_max"] = float(min(cond.max(), 1e6))
        f["sigma_min_phase"] = float(ph[idx[k]])
        f["sigma_min_time_frac"] = float(t[idx[k]] / t[-1])
        prov["singularity"] = "J_scaled = inv(D_task) J D_joint, D_task=diag(0.15,0.15,0.15,0.5,0.5,0.5), D_joint=0.5 rad/s"

        res, util, res_phase = self._capability(q, qd, ph)
        f["cap_residual_max"], f["cap_util_max"], f["cap_residual_phase"] = res, util, res_phase
        prov["capability"] = "bounded differential IK for the requested twist direction at the task cap"

        clear = cand.clearance if model == "nominal" else self._oracle_clearance(cand, spec, tray_xy)
        f["env_clear_min"] = _finite(clear["env_min"])
        f["env_clear_phase"] = float(clear["env_phase"])
        f["env_clear_conf"] = float(clear.get("env_confidence", 1.0))
        f["self_clear_min"] = _finite(clear["self_min"])
        f["self_clear_phase"] = float(clear["self_phase"])
        f["joint_rule_min"] = _finite(clear.get("joint_rule_min", 1.0), 3.0)
        for p in range(8):
            f[f"env_clear_phase_{p}"] = _finite(clear["per_phase_env_min"][p])
        prov["clearance"] = f"{model}: robot spheres vs {'perceived' if model == 'nominal' else 'true'} boxes; " \
                            f"obstacles inflated {self.cfg.perception.planning_buffer + self.cfg.perception.measurement_allowance:.3f} m"

        te, tu = self._dynamics(cand, q, qd, t, ph)
        for i in range(6):
            f[f"track_err_pred_{i + 1}"] = float(te[i])
            f[f"torque_util_{i + 1}"] = float(tu[i])
        f["track_err_pred_max"] = float(te.max())
        f["torque_util_max"] = float(tu.max())
        prov["tracking"] = "static PD model: |inertial + payload torque| / Kp (robot gravity compensated in sim)"
        prov["dynamics"] = "RNEA with URDF inertials + tool + nominal payload; URDF effort limits (unverified, provisional)"

        f["duration"] = float(t[-1])
        f["joint_path_len"] = float(np.abs(np.diff(q, axis=0)).sum())
        f["tcp_path_len"] = float(np.linalg.norm(np.diff(Tt[:, :3, 3], axis=0), axis=1).sum())
        tr = ph == PHASE_ID["transition"]
        f["transition_dist"] = float(np.abs(np.diff(q[tr], axis=0)).sum()) if tr.sum() > 1 else 0.0
        f["intervention_est"] = 0.0
        g = cand.grasp
        f["grasp_width"] = g.width
        f["grasp_depth"] = g.depth
        f["close_aperture"] = g.close_aperture
        f["open_clearance"] = (self.cfg.gripper.open_aperture - g.width) / 2.0
        h = float(percep.target.dims[2]) if percep.target is not None else 0.05
        f["grasp_height_frac"] = g.z_pad / max(h, 1e-6)
        f["grasp_height_mode_high"] = float(g.height_mode == "high")
        f["grasp_axis"] = float(g.axis)
        f["target_conf"] = float(percep.target.confidence) if percep.target is not None else 0.0
        f["target_height"] = h
        f["place_yaw_delta"] = float(abs(np.remainder(cand.place_yaw - g.tcp_yaw + np.pi, 2 * np.pi) - np.pi))
        f["route_direct"] = float(cand.route == "direct")
        f["route_high"] = float(cand.route == "high")
        f["route_lateral"] = float(cand.route in ("left", "right"))
        vec = np.array([f[n] for n in FEATURE_NAMES], dtype=np.float32)
        per_joint = {"jl_margin": m.min(0), "vel_util": vu, "acc_util": au}
        return vec, f, {"model": model, "provenance": prov, "effort_limit_verified": self.cfg.robot.effort_limit_verified,
                        "per_joint": {k: v.tolist() for k, v in per_joint.items()}, "clearance_detail": clear}

    def _capability(self, q, qd, ph):
        n = self.cfg.features.capability_samples
        moving = np.flatnonzero(np.linalg.norm(qd, axis=1) > 1e-4)
        if len(moving) == 0:
            return 0.0, 0.0, -1.0
        sel = moving[np.linspace(0, len(moving) - 1, min(n, len(moving))).astype(int)]
        Dinv = np.linalg.inv(self.D_task)
        vcap = self.cfg.robot.joint_speed_cap
        dt = self.cfg.features.capability_dt
        worst_res, worst_util, worst_phase = 0.0, 0.0, -1.0
        J = self.arm.jacobian_batch(q[sel], self.T_ft)
        for k, i in enumerate(sel):
            v = J[k] @ qd[i]
            vs = Dinv @ v
            nv = np.linalg.norm(vs)
            if nv < 1e-9:
                continue
            v_req = v / nv
            A = Dinv @ J[k]
            b = Dinv @ v_req
            lo = np.maximum(-vcap, (self.arm.q_min - q[i]) / dt)
            hi = np.minimum(vcap, (self.arm.q_max - q[i]) / dt)
            sol = lsq_linear(A, b, bounds=(lo, hi), method="bvls")
            res = float(np.linalg.norm(A @ sol.x - b))
            qd_free = np.linalg.lstsq(J[k], v_req, rcond=None)[0]
            util = float(np.abs(qd_free).max() / vcap)
            if res > worst_res or (res == worst_res and util > worst_util):
                worst_res, worst_phase = res, float(ph[i])
            worst_util = max(worst_util, util)
        return worst_res, worst_util, worst_phase

    def _dynamics(self, cand, q, qd, t, ph):
        sub = max(1, int(round(self.cfg.features.sample_dt / (t[1] - t[0]))))
        idx = np.arange(1, len(t) - 1, sub)
        qdd = (qd[idx + 1] - qd[idx - 1]) / (t[idx + 1] - t[idx - 1])[:, None]
        g = np.array(self.cfg.physics.gravity)
        held = (ph[idx] >= PHASE_ID["lift"]) & (ph[idx] <= PHASE_ID["open"])
        mass = self.cfg.perception.nominal_object_mass
        T_fo = self.T_ft @ cand.grasp.T_tcp_object
        a, b, c = cand.grasp.obj_dims
        I_obj = mass / 12.0 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])
        payload = (mass, T_fo[:3, 3], T_fo[:3, :3] @ I_obj @ T_fo[:3, :3].T)
        J_pay = _translation(T_fo[:3, 3])
        Kp = np.array(self.cfg.robot.servo_stiffness)
        lim = np.array(self.cfg.robot.effort_limit)
        Qs, Qds = q[idx], qd[idx]
        tau_servo = self.arm.rnea_batch(Qs, Qds, qdd, gravity=(0.0, 0.0, 0.0), payload=payload, payload_mask=held)
        if np.any(held):
            Jp = self.arm.jacobian_batch(Qs[held], J_pay)[:, :3]
            tau_servo[held] -= np.einsum("nij,i->nj", Jp, mass * g)
        tau_full = self.arm.rnea_batch(Qs, Qds, qdd, gravity=tuple(g), payload=payload, payload_mask=held)
        return np.abs(tau_servo).max(0) / Kp, np.abs(tau_full).max(0) / lim

    def _oracle_clearance(self, cand, spec, tray_xy):
        """Clearance against the true scene geometry (annotation channel)."""
        obs = [Box.from_xyzyaw(f"obstacle_{i}", [o.xy[0], o.xy[1], o.dims[2] / 2.0], o.yaw, o.dims, 0.0, "obstacle")
               for i, o in enumerate(spec.obstacles)]
        boxes = world_boxes(self.cfg.layout, tray_xy, obs)
        t = spec.target
        tgt = Box.from_xyzyaw("target", [t.xy[0], t.xy[1], t.dims[2] / 2.0], t.yaw, t.dims, 0.0, "target")
        return self.m.planner.clearance(cand.reference, cand.grasp, tgt, boxes, cand.T_tcp_place, t.dims)


def _translation(p):
    T = np.eye(4)
    T[:3, 3] = p
    return T
