"""Kinematic stand-in for the Isaac Lab scene, for GPU-free tests of PickPlaceEnv's glue logic.

Joints follow the commands through a first-order lag with velocity feed-forward. The block is carried rigidly once
both pads close on it, slips out if the pad friction cannot hold its weight, and drops onto the support below
when released. Contacts are penetrations of the collision-sphere model into boxes. Cameras use the numpy depth
renderer. Everything the Isaac-specific methods of PickPlaceEnv do is replaced here; the rest (control loop,
monitor, labels, observation packing, features, record assembly) runs unchanged.
"""

from __future__ import annotations

import numpy as np

from a0509pp.collision import object_spheres, sphere_box_distance, world_boxes
from a0509pp.features import FeatureComputer
from a0509pp.geometry import inv_T, make_T, mat_to_quat, pose7_to_T_batch, quat_from_yaw, quat_to_mat, yaw_from_mat
from a0509pp.mock_render import intrinsics, render_depth
from a0509pp.models import Models
from a0509pp.scene_spec import SceneSampler
from a0509pp.sim.env import DEFAULT_MIMIC_SIGN, PickPlaceEnv
from a0509pp.sim.scene_cfg import obstacle_park_positions, object_park_positions

K_CONTACT = 3000.0  # N per metre of sphere-model penetration
SLIP_FACTOR = 20.0  # exaggerated load so the hidden friction/mass realizations change outcomes
H = 2


class _Stub:
    has_gui = False

    def forward(self):
        pass

    def reset(self, *a, **k):
        pass

    def update(self, *a, **k):
        pass

    def render(self):
        pass


class FakeEnv(PickPlaceEnv):
    def __init__(self, cfg, num_envs=4, camera_mode="fixed", manifest=None, lag=0.03):
        self.cfg, self.verbose, self.device = cfg, False, "cpu"
        E = self.num_envs = int(num_envs)
        self.manifest = manifest or {}
        self.models = Models(cfg, self.manifest)
        self.gripper, self.arm = self.models.gripper, self.models.arm
        self.features = FeatureComputer(cfg, self.models)
        self.sampler = SceneSampler(cfg)
        self.home_q, self.home_info = self.models.solve_home()
        self.grip_torque = self.gripper.max_torque()
        P = cfg.physics
        self.physics_dt, self.decimation = P.dt, P.control_decimation
        self.control_dt = P.dt * P.control_decimation
        self.cam_ticks = max(1, P.camera_decimation // P.control_decimation)
        self.gui_render_every = 0
        self.sim = self.scene = _Stub()
        self.camera_mode = camera_mode
        self.cams = {n: None for n in {"fixed": ["fixed_camera"], "wrist": ["wrist_camera"],
                                       "both": ["fixed_camera", "wrist_camera"], "none": []}[camera_mode]}
        g = cfg.gripper
        self.joint_names = list(cfg.robot.arm_joint_names) + [g.finger_joint] + list(g.passive_joints)
        self.J = len(self.joint_names)
        self.arm_ids, self.finger_id, self.passive_ids = list(range(6)), 6, list(range(7, self.J))
        self.mimic_coef = np.array([DEFAULT_MIMIC_SIGN.get(n, 0.0) for n in g.passive_joints])
        self.mimic_off = np.zeros(len(g.passive_joints))
        self.cs_names = [cfg.robot.arm_base_link] + [f"link_{i}" for i in range(1, 7)] + [g.base_link] + list(g.finger_links)
        self.body_names = list(self.cs_names)
        cls = []
        for n in self.cs_names:
            cls.append("base" if n == cfg.robot.arm_base_link else "pad" if n in g.contact_links else
                       "finger" if n in g.finger_links else "arm")
        self.body_classes = cls
        forb = [i for i, c in enumerate(cls) if c in ("arm", "finger")]
        self.contact_names = [self.cs_names[i] for i in forb] + list(g.contact_links)
        self.pad_side = np.array([-1 if l.startswith("left") else 1 for l in g.contact_links])
        self.left_pads = [i for i, s in enumerate(self.pad_side) if s < 0]
        self.right_pads = [i for i, s in enumerate(self.pad_side) if s > 0]
        self.nominal_stiffness = np.array(cfg.robot.servo_stiffness, float)
        self.nominal_damping = np.array(cfg.robot.servo_damping, float)
        self.T_base_tcp = self.gripper.T_base_tcp()
        self.T_flange_tcp = self.gripper.T_flange_tcp()
        self.origins = np.zeros((E, 3))
        self._tc = None
        self._cur_real = [None] * E
        self._tick = 0
        self.spec = self.ic = self.obs = self.percep = None
        self.lag = lag
        L = cfg.layout
        self.obj_park = object_park_positions(L, L.object_pool_dims)
        self.obs_park = obstacle_park_positions(L, L.obstacle_pool_dims)
        self.objects = list(range(len(L.object_pool_dims)))
        self.obstacles = list(range(len(L.obstacle_pool_dims)))
        self.q = np.repeat(self.home_q[None], E, 0)
        self.qd = np.zeros((E, 6))
        self.th = np.zeros(E)
        self.q_t, self.qd_t, self.th_t = self.q.copy(), np.zeros((E, 6)), np.zeros(E)
        self.obj_pose = np.zeros((E, len(self.objects), 7))
        self.obj_pose[:, :, 6] = 1.0
        self.obj_pose[:, :, :3] = np.asarray(self.obj_park)[None]
        self.obj_vel = np.zeros((E, len(self.objects), 6))
        self.obs_pose = np.zeros((E, len(self.obstacles), 7))
        self.obs_pose[:, :, 6] = 1.0
        self.obs_pose[:, :, :3] = np.asarray(self.obs_park)[None]
        self.active_obs = []
        self.grasped = np.zeros(E, bool)
        self.rel = np.repeat(np.eye(4)[None], E, 0)
        self.real = [self.nominal_realization()] * E
        self.k_target = 0
        self.static = world_boxes(L, L.tray_center_xy)
        self._sensed = None
        self._tag_body = self._map_tags()

    # ------------------------------------------------------------------ Isaac replacements
    def _map_tags(self):
        m = {}
        for t in self.models.spheres.tags:
            if t == "base_link":
                m[t] = self.cfg.robot.arm_base_link
            elif t in ("gripper_base", "wrist_camera"):
                m[t] = self.cfg.gripper.base_link
            else:
                m[t] = t
        return np.array([self.cs_names.index(m[t]) for t in self.models.spheres.tags])

    def set_targets(self, q_cmd, qd_cmd, aperture_cmd, env_ids=None):
        ids = np.arange(self.num_envs) if env_ids is None else np.asarray(env_ids)
        self.q_t[ids] = q_cmd
        self.qd_t[ids] = qd_cmd
        self.th_t[ids] = self.gripper.theta(np.asarray(aperture_cmd, float))

    def write_robot_state(self, q_arm, theta, env_ids=None, qd_arm=None):
        ids = np.arange(self.num_envs) if env_ids is None else np.asarray(env_ids)
        self.q[ids] = q_arm
        self.qd[ids] = 0.0 if qd_arm is None else qd_arm
        self.th[ids] = theta

    def apply_realizations(self, reals, k_target):
        self.real = list(reals)

    def _place_layout(self, spec):
        k_t = self._target_index(spec)
        L = self.cfg.layout
        self.obj_pose[:, :, :3] = np.asarray(self.obj_park)[None]
        self.obj_pose[:, :, 3:] = [0, 0, 0, 1]
        d = L.object_pool_dims[k_t]
        self.obj_pose[:, k_t] = [spec.target.xy[0], spec.target.xy[1], d[2] / 2.0] + list(quat_from_yaw(spec.target.yaw))
        self.obj_vel[:] = 0.0
        self.obs_pose[:, :, :3] = np.asarray(self.obs_park)[None]
        self.obs_pose[:, :, 3:] = [0, 0, 0, 1]
        self.active_obs = [o.pool_index for o in spec.obstacles]
        for o in spec.obstacles:
            dz = L.obstacle_pool_dims[o.pool_index][2]
            self.obs_pose[:, o.pool_index] = [o.xy[0], o.xy[1], dz / 2.0] + list(quat_from_yaw(o.yaw))
        self.grasped[:] = False
        self.k_target = k_t
        return k_t

    def _apply_visuals(self, spec):
        pass

    def _settle(self, k_t):
        self.hold(int(round(self.cfg.physics.settle_min / self.control_dt)))
        return {"settle_time": self.cfg.physics.settle_min, "settled": True, "wall_s": 0.0}

    def _obstacle_boxes(self, e=0):
        from a0509pp.collision import Box

        L = self.cfg.layout
        out = []
        for j in self.active_obs:
            p = self.obs_pose[e, j]
            out.append(Box(f"obstacle_{j}", p[:3], quat_to_mat(p[3:]), np.asarray(L.obstacle_pool_dims[j]) / 2.0, 0.0, "obstacle"))
        return out

    def _target_box(self, e=0):
        from a0509pp.collision import Box

        p = self.obj_pose[e, self.k_target]
        return Box("target", p[:3], quat_to_mat(p[3:]), np.asarray(self.cfg.layout.object_pool_dims[self.k_target]) / 2.0, 0.0, "target")

    def _render_cameras(self, read=True):
        if not read:
            return {}
        C = self.cfg.camera
        K = intrinsics(C.width, C.height, C.hfov_deg)
        boxes = [(b.center, b.R, b.half, i) for i, b in enumerate(self.static) if b.kind != "floor"]
        boxes += [(b.center, b.R, b.half, 20 + i) for i, b in enumerate(self._obstacle_boxes())]
        tb = self._target_box()
        boxes.append((tb.center, tb.R, tb.half, 10))
        Cs, rs, _ = self.models.spheres.compute(self.q[:1], self.th[:1])
        out = {}
        for name in self.cams:
            if name == "fixed_camera":
                T = self.fixed_camera_true(self.spec)
            else:
                T = self.arm.fk(self.q[0]) @ self.gripper.T_flange_base() @ self.T_base_wrist_cam()
            depth, ids = render_depth(K, T, C.width, C.height, boxes, (Cs[0], rs, 30))
            depth[depth > C.clip[1]] = np.inf
            shade = np.clip(1.2 - 0.5 * np.nan_to_num(depth, posinf=2.0), 0.1, 1.0)
            base = np.array([[0.55, 0.5, 0.44]])[np.zeros_like(ids)]
            base[ids == 10] = [0.85, 0.2, 0.15]
            base[ids >= 20] = [0.45, 0.45, 0.48]
            base[ids == 30] = [0.2, 0.2, 0.22]
            out[name] = {"rgb": (255 * base * shade[..., None]).astype(np.uint8), "depth": depth.astype(np.float32), "K": K}
        return out

    def _grab_frames(self, cam):
        C = self.cfg.camera
        from a0509pp.sim.env import _downsample_rgb

        img = self._render_cameras()[cam]["rgb"]
        return np.repeat(_downsample_rgb(img, C.net_width // 2, C.net_height // 2)[None], self.num_envs, 0)

    def _capture_frames(self, spec):
        C = self.cfg.camera
        n = C.history + int(max(self.cfg.randomization.camera_delay))
        frames = []
        for i in range(n):
            if i > 0:
                self.hold(self.cam_ticks)
            f = self._render_cameras()
            f["_q"] = self.q[0].copy()
            f["_T_gbase"] = self.arm.fk(self.q[0]) @ self.gripper.T_flange_base()
            f["_t"] = self._tick * self.control_dt
            frames.append(f)
        return frames

    def capture_initial_condition(self, env=0):
        jp = self.full_joint_pos(self.q[env][None], self.th[env])[0]
        jt = self.full_joint_pos(self.q_t[env][None], self.th_t[env])[0]
        return {"scene_id": None if self.spec is None else self.spec.scene_id, "joint_names": self.joint_names,
                "joint_pos": jp.tolist(), "joint_vel": np.concatenate([self.qd[env], np.zeros(self.J - 6)]).tolist(),
                "joint_pos_target": jt.tolist(),
                "objects": [{"pose": self.obj_pose[env, k].tolist(), "vel": self.obj_vel[env, k].tolist()} for k in self.objects],
                "obstacles": [{"pose": self.obs_pose[env, j].tolist()} for j in self.obstacles],
                "target_index": int(self.k_target), "physics_dt": self.physics_dt, "control_dt": self.control_dt,
                **self._ic_meta()}

    def restore_initial_condition(self, ic, env_ids=None, validate=True, tol=1e-4):
        ids = np.arange(self.num_envs) if env_ids is None else np.asarray(env_ids)
        jp, jt = np.asarray(ic["joint_pos"]), np.asarray(ic["joint_pos_target"])
        self.q[ids], self.th[ids] = jp[:6], jp[6]
        self.qd[ids] = np.asarray(ic["joint_vel"])[:6]
        self.q_t[ids], self.qd_t[ids], self.th_t[ids] = jt[:6], 0.0, jt[6]
        for k, s in enumerate(ic["objects"]):
            self.obj_pose[ids, k] = s["pose"]
            self.obj_vel[ids, k] = s["vel"]
        for j, s in enumerate(ic["obstacles"]):
            self.obs_pose[ids, j] = s["pose"]
        self.grasped[ids] = False
        self.rel[ids] = np.eye(4)
        self._sensed = None
        return {"ok": True, "validated": validate, "joint_err": 0.0, "object_err": 0.0}

    # ------------------------------------------------------------------ kinematic "physics"
    def _support_z(self, xy, h):
        L = self.cfg.layout
        tx, ty = self.spec.tray_xy
        if abs(xy[0] - tx) < L.tray_interior[0] / 2 and abs(xy[1] - ty) < L.tray_interior[1] / 2:
            return L.tray_floor + h / 2.0
        if abs(xy[0] - L.table_center[0]) < L.table_size[0] / 2 and abs(xy[1] - L.table_center[1]) < L.table_size[1] / 2:
            return h / 2.0
        return L.floor_z + h / 2.0

    def _physics_steps(self, n=None):
        cfg, dt, E = self.cfg, self.control_dt, self.num_envs
        g = self.gripper
        k = self.k_target
        dims = np.asarray(cfg.layout.object_pool_dims[k])
        scale = np.array([np.mean(r.servo_scale) for r in self.real])
        tau = self.lag / scale
        q_new = self.q + dt * (self.qd_t + (self.q_t - self.q) / tau[:, None])
        self.qd = (q_new - self.q) / dt
        self.q = q_new
        T_tcp = self.arm.fk_batch(self.q)[:, 6] @ self.T_flange_tcp
        rate = 1.2 * dt
        pad_f = np.zeros(E)
        prev_pose = self.obj_pose[:, k].copy()
        for e in range(E):
            T_obj = make_T(quat_to_mat(self.obj_pose[e, k, 3:]), self.obj_pose[e, k, :3])
            if not self.grasped[e]:
                rel = inv_T(T_tcp[e]) @ T_obj
                p, Rr = rel[:3, 3], rel[:3, :3]
                ext = np.abs(Rr) @ (dims / 2.0)
                pz = g.pinch_offset(float(g.aperture(self.th[e])))
                between = abs(p[0]) < ext[0] + 0.004 and abs(p[1]) < ext[1] + 0.02 and abs(p[2] - pz) < ext[2] + cfg.gripper.pad_half_height * 0.5
                a_c = 2.0 * (ext[1] + abs(p[1]))
                th_c = float(g.theta(a_c))
                th_next = self.th[e] + np.clip(self.th_t[e] - self.th[e], -rate, rate)
                if between and th_next >= th_c and self.th_t[e] > th_c:
                    self.th[e] = th_c
                    T_obj[:3, 3] -= T_tcp[e, :3, 1] * p[1]
                    self.rel[e] = inv_T(T_tcp[e]) @ T_obj
                    self.grasped[e] = True
                else:
                    self.th[e] = th_next
            else:
                rz = self.real[e]
                mu = 0.5 * (rz.pad_static_friction + rz.object_static_friction)
                hold_ok = 2.0 * mu * cfg.gripper.grip_force > 1.5 * rz.object_mass * 9.81 * SLIP_FACTOR
                a_c = float(g.aperture(self.th[e]))
                opening = self.th_t[e] < float(g.theta(a_c + 0.002))
                if opening or not hold_ok:
                    self.grasped[e] = False
                    T_obj = T_tcp[e] @ self.rel[e]
                    yaw = yaw_from_mat(T_obj[:3, :3])
                    xy = T_obj[:2, 3]
                    self.obj_pose[e, k] = [xy[0], xy[1], self._support_z(xy, dims[2])] + list(quat_from_yaw(yaw))
                    self.th[e] += np.clip(self.th_t[e] - self.th[e], -rate, rate)
                else:
                    pad_f[e] = cfg.gripper.grip_force / 2.0
                    T_obj = T_tcp[e] @ self.rel[e]
                    self.obj_pose[e, k] = np.concatenate([T_obj[:3, 3], mat_to_quat(T_obj[:3, :3])])
        self.obj_vel[:, k, :3] = (self.obj_pose[:, k, :3] - prev_pose[:, :3]) / dt
        self.obj_vel[:, k, 3:] = 0.0
        dropped = (~self.grasped) & (np.linalg.norm(self.obj_vel[:, k, :3], axis=1) > 0.5)
        self.obj_vel[dropped, k] = 0.0
        self._sense(T_tcp, pad_f)
        self._tick += 1

    def _sense(self, T_tcp, pad_f):
        cfg, E = self.cfg, self.num_envs
        k = self.k_target
        dims = np.asarray(cfg.layout.object_pool_dims[k])
        C, r, tags = self.models.spheres.compute(self.q, self.th)
        env_boxes = list(self.static) + self._obstacle_boxes()
        D = sphere_box_distance(C, r, env_boxes)
        D[:, np.asarray([t == "base_link" for t in tags]), :] = np.inf
        pen_env = np.maximum(-D, 0.0).max(2) * K_CONTACT
        Dt = sphere_box_distance(C, r, [self._target_box(0)])[:, :, 0]
        pen_tgt = np.maximum(-Dt, 0.0) * K_CONTACT
        B = len(self.cs_names)
        link = np.zeros((E, B))
        pad_other = np.zeros((E, len(cfg.gripper.contact_links)))
        for s, b in enumerate(self._tag_body):
            cls = self.body_classes[b]
            f = pen_env[:, s] + (np.where(self.grasped, 0.0, pen_tgt[:, s]) if cls != "pad" else 0.0)
            link[:, b] = np.maximum(link[:, b], f)
            if cls == "pad":
                i = list(cfg.gripper.contact_links).index(self.cs_names[b])
                pad_other[:, i] = np.maximum(pad_other[:, i], pen_env[:, s])
        obs_boxes = self._obstacle_boxes()
        n_obs = len(self.obstacles)
        obj_obs = np.zeros((E, n_obs))
        if obs_boxes:
            S = object_spheres(dims)
            for e in range(E):
                Tq = pose7_to_T_batch(self.obj_pose[e, k][None])[0]
                Cw = (S[:, :3] @ Tq[:3, :3].T + Tq[:3, 3])[None]
                Do = sphere_box_distance(Cw, S[:, 3], obs_boxes)[0]
                for jj, j in enumerate(self.active_obs):
                    obj_obs[e, j] = np.maximum(-Do[:, jj], 0.0).max() * K_CONTACT
        m = np.array([r.object_mass for r in self.real])
        env_f = np.where(self.grasped, 0.0, m * 9.81)
        self._sensed = {"link_force": np.repeat(link[:, None], H, 1), "pad_other_force": np.repeat(pad_other[:, None], H, 1),
                        "pad_obj_force": pad_f, "obj_obstacle_force": np.repeat(obj_obs[:, None], H, 1),
                        "obj_env_force": np.repeat(env_f[:, None], H, 1), "T_tcp": T_tcp}

    def read_state(self, k_target=None):
        E = self.num_envs
        if self._sensed is None:
            T_tcp = self.arm.fk_batch(self.q)[:, 6] @ self.T_flange_tcp
            self._sense(T_tcp, np.zeros(E))
        s = self._sensed
        jp = self.full_joint_pos(self.q, self.th)
        jv = np.zeros_like(jp)
        jv[:, :6] = self.qd
        T_gb = self.arm.fk_batch(self.q)[:, 6] @ self.gripper.T_flange_base()
        out = {"q": self.q.copy(), "qd": self.qd.copy(), "theta": self.th.copy(), "aperture": np.asarray(self.gripper.aperture(self.th)),
               "joint_pos": jp, "joint_vel": jv, "T_gbase": T_gb, "T_tcp": T_gb @ self.T_base_tcp, "link_force": s["link_force"]}
        if k_target is None:
            return out
        out.update({"obj_pose": self.obj_pose[:, k_target].copy(), "obj_vel": self.obj_vel[:, k_target].copy(),
                    "pad_obj_force": s["pad_obj_force"].copy(), "pad_obj_force_max": s["pad_obj_force"].copy(),
                    "pad_other_force": s["pad_other_force"], "pad_obj_friction": np.zeros(E),
                    "obj_obstacle_force": s["obj_obstacle_force"], "obj_env_force": s["obj_env_force"]})
        return out
