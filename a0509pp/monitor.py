"""Runtime monitor (120 Hz, vectorised over environments) and failure event bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .trajectory import PHASE_ID

EXEC_FAILURES = {"forbidden_contact", "tracking_error", "joint_limit", "timeout", "no_progress", "object_collision"}
TASK_FAILURES = {"missed_grasp", "object_lost", "object_disturbed", "wrong_object", "placement_outside", "not_released",
                 "not_withdrawn", "not_settled"}
SIM_ERRORS = {"sim_error"}


@dataclass
class Event:
    time: float
    kind: str
    phase: int
    detail: str = ""

    def to_dict(self):
        return {"time": self.time, "kind": self.kind, "phase": self.phase, "detail": self.detail}


@dataclass
class EnvMonitorState:
    events: list = field(default_factory=list)
    aborted: bool = False
    abort_reason: str = ""
    abort_time: float = -1.0
    task_failed: bool = False
    exec_failed: bool = False
    sim_error: bool = False
    max_track_err: np.ndarray = field(default_factory=lambda: np.zeros(6))
    sq_track_err: np.ndarray = field(default_factory=lambda: np.zeros(6))
    n_track: int = 0
    max_slip: float = 0.0
    max_contact_force: float = 0.0
    grasp_rel: np.ndarray | None = None
    release_tcp_z: float | None = None
    grasp_checked: bool = False


class Monitor:
    """Feed step() once per control tick with batched readings. All forces in N, distances in m."""

    def __init__(self, cfg, num_envs, arm_q_min, arm_q_max, control_dt, physics_dt, body_classes):
        """body_classes: list per robot contact body: 'base', 'arm', or 'finger' (finger links without object filter)."""
        self.c = cfg.monitor
        self.cfg = cfg
        self.E = num_envs
        self.q_min = np.asarray(arm_q_min)
        self.q_max = np.asarray(arm_q_max)
        self.dt = control_dt
        self.pdt = physics_dt
        self.body_classes = np.array(body_classes)
        self.reset(np.arange(num_envs))

    def reset(self, env_ids):
        if not hasattr(self, "state"):
            self.state = [EnvMonitorState() for _ in range(self.E)]
            self.track_cnt = np.zeros(self.E, int)
            self.contact_cnt = {}
            self.loss_cnt = np.zeros(self.E, int)
        for e in env_ids:
            self.state[e] = EnvMonitorState()
            self.track_cnt[e] = 0
            self.loss_cnt[e] = 0
        for k in list(self.contact_cnt):
            self.contact_cnt[k][env_ids] = 0
        if hasattr(self, "_np_n"):
            self._np_n[np.asarray(env_ids, int)] = 0

    def _event(self, e, t, kind, phase, detail="", abort=None):
        st = self.state[e]
        if st.aborted:
            return
        st.events.append(Event(float(t), kind, int(phase), detail))
        if kind in EXEC_FAILURES:
            st.exec_failed = True
        if kind in TASK_FAILURES:
            st.task_failed = True
        if kind in SIM_ERRORS:
            st.sim_error = True
        if abort is None:
            abort = kind in EXEC_FAILURES or kind in SIM_ERRORS or (self.c.abort_on_task_failure and kind in TASK_FAILURES)
        if abort:
            st.aborted = True
            st.abort_reason = kind
            st.abort_time = float(t)

    def _sustained(self, key, cond):
        """cond: (E, H) booleans over physics substeps; returns (E,) True when a run of contact_steps is reached."""
        if key not in self.contact_cnt:
            self.contact_cnt[key] = np.zeros(self.E, int)
        cnt = self.contact_cnt[key]
        hit = np.zeros(self.E, bool)
        for h in range(cond.shape[1]):
            cnt[:] = np.where(cond[:, h], cnt + 1, 0)
            hit |= cnt >= self.c.contact_steps
        return hit

    def step(self, t, r):
        """r: dict of batched readings for this control tick (see env for the contract)."""
        c = self.c
        active = np.array([not s.aborted for s in self.state]) & r["active"]
        ph = r["phase"]
        q, q_cmd = r["q"], r["q_cmd"]
        finite = np.isfinite(q).all(1) & np.isfinite(r["obj_pos"]).all(1) & np.isfinite(r["qd"]).all(1)
        wild = (np.abs(r["qd"]).max(1) > 20.0) | (np.linalg.norm(r["obj_vel"], axis=1) > 20.0)
        for e in np.flatnonzero(active & (~finite | wild)):
            self._event(e, t, "sim_error", ph[e], "non-finite or exploding state")
        active &= finite & ~wild
        err = np.abs(q - q_cmd)
        over = (err > c.tracking_error).any(1)
        self.track_cnt = np.where(active & over, self.track_cnt + 1, 0)
        n_track = int(round(c.tracking_time / self.dt))
        for e in np.flatnonzero(active & (self.track_cnt > n_track)):
            j = int(np.argmax(err[e]))
            self._event(e, t, "tracking_error", ph[e], f"joint_{j + 1} error {err[e, j]:.3f} rad")
        for e in np.flatnonzero(active):
            st = self.state[e]
            st.max_track_err = np.maximum(st.max_track_err, err[e])
            st.sq_track_err += err[e] ** 2
            st.n_track += 1
        lim = (q <= self.q_min + c.joint_limit_margin) | (q >= self.q_max - c.joint_limit_margin)
        for e in np.flatnonzero(active & lim.any(1)):
            self._event(e, t, "joint_limit", ph[e], f"joint_{int(np.argmax(lim[e])) + 1}")

        # Robot contacts: link net normal forces (E, H, B); finger pads with object filter (E, H, F, 2): net, object.
        F = r["link_force"]
        cls = self.body_classes
        forb_links = (cls == "arm") | (cls == "finger")
        f_arm = F[:, :, forb_links]
        pad_other = r["pad_other_force"]
        allf = np.concatenate([f_arm, pad_other], axis=2) if pad_other.size else f_arm
        for e in np.flatnonzero(active):
            self.state[e].max_contact_force = max(self.state[e].max_contact_force, float(allf[e].max()) if allf.size else 0.0)
        sustained = self._sustained("robot", (allf > c.contact_force).any(2))
        impulse = (allf > c.impulse_force).any((1, 2))
        for e in np.flatnonzero(active & (sustained | impulse)):
            names = r["contact_names"]
            k = int(np.argmax(allf[e].max(0)))
            self._event(e, t, "forbidden_contact", ph[e], f"{names[k]} {allf[e].max():.1f} N")

        # Object contacts with obstacles, or with the table while carried.
        held = (ph >= PHASE_ID["lift"]) & (ph <= PHASE_ID["lower"])
        obs_f = r["obj_obstacle_force"]
        if obs_f.size:
            hit = self._sustained("obj_obs", (obs_f > c.contact_force).any(2)) | (obs_f > c.impulse_force).any((1, 2))
            for e in np.flatnonzero(active & hit):
                self._event(e, t, "object_collision", ph[e], f"obstacle {obs_f[e].max():.1f} N")
        carried = held & r["obj_grasped"] & (r["obj_bottom_z"] < 0.004) & (r["obj_env_force"].max(1) > c.contact_force)
        drag = self._sustained("obj_table", r["obj_env_force"] > c.contact_force) & carried
        for e in np.flatnonzero(active & drag & (ph > PHASE_ID["lift"])):
            self._event(e, t, "object_collision", ph[e], "carried object dragged on table")

        # Grasp checks at the end of the close phase, then slip / loss while carried.
        for e in np.flatnonzero(active):
            st = self.state[e]
            if not st.grasp_checked and r["close_done"][e]:
                st.grasp_checked = True
                if r["aperture"][e] < r["grasp_width"][e] - c.missed_grasp_margin or r["pad_obj_force"][e] < c.grasp_min_force:
                    self._event(e, t, "missed_grasp", ph[e], f"aperture {r['aperture'][e]:.4f} m, pad force {r['pad_obj_force'][e]:.2f} N")
                st.grasp_rel = r["obj_in_tcp"][e].copy()
            if st.grasp_rel is not None and held[e]:
                slip = float(np.linalg.norm(r["obj_in_tcp"][e] - st.grasp_rel))
                st.max_slip = max(st.max_slip, slip)
                lost = slip > c.object_loss_dist or r["pad_obj_force"][e] < 0.1 * c.grasp_min_force
                self.loss_cnt[e] = self.loss_cnt[e] + 1 if lost else 0
                if self.loss_cnt[e] > int(round(c.object_loss_time / self.dt)) and not any(ev.kind == "object_lost" for ev in st.events):
                    self._event(e, t, "object_lost", ph[e], f"slip {slip:.3f} m")
            if ph[e] == PHASE_ID["open"] and st.release_tcp_z is None:
                st.release_tcp_z = float(r["tcp_z"][e])
            if not st.grasp_checked and ph[e] < PHASE_ID["close"] and r["obj_disp"][e] > 0.01 and \
                    not any(ev.kind == "object_disturbed" for ev in st.events):
                self._event(e, t, "object_disturbed", ph[e], f"target moved {r['obj_disp'][e]:.3f} m before grasp", abort=False)
        # No progress: over a sliding window the reference TCP moved but the measured TCP did not follow.
        if "tcp_pos" in r and "ref_tcp_pos" in r:
            W = max(1, int(round(c.no_progress_time / self.dt)))
            if not hasattr(self, "_np_tcp"):
                self._np_tcp = np.zeros((W + 1, self.E, 3))
                self._np_ref = np.zeros((W + 1, self.E, 3))
                self._np_n = np.zeros(self.E, int)
                self._np_i = 0
            i = self._np_i % (W + 1)
            self._np_tcp[i] = r["tcp_pos"]
            self._np_ref[i] = r["ref_tcp_pos"]
            self._np_n = np.where(active, self._np_n + 1, self._np_n)
            old = (self._np_i + 1) % (W + 1)
            self._np_i += 1
            d_meas = np.linalg.norm(self._np_tcp[i] - self._np_tcp[old], axis=1)
            d_ref = np.linalg.norm(self._np_ref[i] - self._np_ref[old], axis=1)
            stall = active & (self._np_n > W) & (d_ref > 0.03) & (d_meas < 0.25 * d_ref)
            for e in np.flatnonzero(stall):
                self._event(e, t, "no_progress", ph[e], f"TCP moved {d_meas[e]:.3f} m of {d_ref[e]:.3f} m in {c.no_progress_time:.1f} s")
        # The episode limit applies to plan execution; a completed plan is judged by the task check instead.
        plan_done = r.get("plan_done", np.zeros(self.E, bool))
        for e in np.flatnonzero(active & ~plan_done & (t >= self.cfg.physics.episode_timeout - 1e-9)):
            self._event(e, t, "timeout", ph[e], "episode time limit")

    def summary(self, e):
        st = self.state[e]
        n = max(st.n_track, 1)
        return {
            "events": [ev.to_dict() for ev in st.events], "aborted": st.aborted, "abort_reason": st.abort_reason,
            "abort_time": st.abort_time, "exec_failed": st.exec_failed, "task_failed": st.task_failed,
            "sim_error": st.sim_error, "max_track_err": st.max_track_err.tolist(),
            "rms_track_err": np.sqrt(st.sq_track_err / n).tolist(), "max_slip": st.max_slip,
            "max_contact_force": st.max_contact_force, "release_tcp_z": st.release_tcp_z,
        }
