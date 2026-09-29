"""Numpy mocks of the Isaac Lab objects PickPlaceEnv touches (articulation, rigid objects, contact sensors, cameras,
scene, simulation context) plus torch/warp shims. They follow the Isaac Lab 3.0 shapes and keyword-only signatures,
so the Isaac-facing code paths of a0509pp/sim/env.py run offline. Physics is a first-order joint lag; FK places the
gripper base; the block does not move. This checks plumbing (indices, shapes, conventions), not physics.
"""

from __future__ import annotations

import types

import numpy as np

from a0509pp.geometry import mat_to_quat


class T(np.ndarray):
    """Tensor shim: ndarray with the few torch methods env.py uses."""

    def __new__(cls, a, dtype=None):
        return np.asarray(a, dtype=dtype).view(cls)

    def flip(self, dim):
        return np.flip(self, axis=dim).view(T)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def clone(self):
        return self.copy()

    def contiguous(self):
        return self

    def to(self, dtype=None, device=None, **k):
        return self.astype(dtype).view(T) if dtype is not None else self


torch = types.SimpleNamespace(
    Tensor=T, float32=np.float32, long=np.int64, int32=np.int32,
    as_tensor=lambda x, dtype=None, device=None: T(np.asarray(x), dtype=dtype),
    full=lambda shape, v, device=None: T(np.full(shape, v, dtype=np.float32)),
    linalg=types.SimpleNamespace(norm=lambda x, dim=-1: T(np.linalg.norm(np.asarray(x), axis=dim))),
    stack=lambda xs, dim=0: T(np.stack([np.asarray(x) for x in xs], axis=dim)),
    minimum=lambda a, b: T(np.minimum(a, b)), maximum=lambda a, b: T(np.maximum(a, b)),
)
wp = types.SimpleNamespace(
    int32=np.int32, float32=np.float32,
    array=lambda a, dtype=None, device=None: np.asarray(a, dtype=dtype),
    to_torch=lambda a: T(np.asarray(a)), from_torch=lambda t, dtype=None: np.asarray(t, dtype=dtype),
)


class Proxy:
    def __init__(self, arr):
        self._a = arr

    @property
    def torch(self):
        return T(self._a)


def _ids(env_ids, n):
    return np.arange(n) if env_ids is None else np.asarray(env_ids, int)


def _check(value, shape, name):
    v = np.asarray(value)
    if v.shape != tuple(shape):
        raise ValueError(f"{name}: shape {v.shape}, expected {tuple(shape)}")
    return v


class _TargetCommand:
    def __init__(self, art):
        self.a = art

    @property
    def position(self):
        return Proxy(self.a.q_t)

    def _set(self, buf, value, joint_ids, env_ids):
        e, j = _ids(env_ids, self.a.E), _ids(joint_ids, self.a.J)
        buf[np.ix_(e, j)] = _check(value, (len(e), len(j)), "target")

    def set_position_index(self, *, value, joint_ids=None, env_ids=None, full_data=False):
        self._set(self.a.q_t, value, joint_ids, env_ids)

    def set_velocity_index(self, *, value, joint_ids=None, env_ids=None, full_data=False):
        self._set(self.a.qd_t, value, joint_ids, env_ids)


class _View:
    def __init__(self, count, shapes, link_paths=None):
        self.count, self.max_shapes = count, shapes
        self.mats = np.tile(np.array([0.5, 0.5, 0.0], np.float32), (count, shapes, 1))
        self.link_paths = link_paths

    def get_material_properties(self):
        return self.mats.copy()

    def set_material_properties(self, mats, ids):
        ids = np.asarray(ids, int)
        self.mats[ids] = np.asarray(mats)[ids]


class MockArticulation:
    def __init__(self, E, joint_names, body_names, arm, T_flange_base, home, gbase):
        self.E, self.joint_names, self.body_names = E, list(joint_names), list(body_names)
        self.J, self.B = len(joint_names), len(body_names)
        self.arm, self.T_fb, self.gbase = arm, T_flange_base, gbase
        self.q = np.zeros((E, self.J))
        self.q[:, :6] = home
        self.qd = np.zeros((E, self.J))
        self.q_t, self.qd_t = self.q.copy(), np.zeros((E, self.J))
        self.stiff = np.full((E, self.J), 100.0)
        self.damp = np.full((E, self.J), 10.0)
        self.actuators = types.SimpleNamespace(target_command=_TargetCommand(self))
        paths = [f"/World/envs/env_0/Robot/x/{n}" for n in body_names]
        self.root_view = _View(E, 2 * self.B, [paths])
        self.data = types.SimpleNamespace()
        self._refresh()

    def _refresh(self):
        d = self.data
        d.joint_pos, d.joint_vel = Proxy(self.q), Proxy(self.qd)
        pose = np.zeros((self.E, self.B, 7))
        pose[:, :, 6] = 1.0
        for e in range(self.E):
            Tg = self.arm.fk(self.q[e, :6]) @ self.T_fb
            pose[e, self.gbase] = np.concatenate([Tg[:3, 3], mat_to_quat(Tg[:3, :3])])
        d.body_link_pose_w = Proxy(pose)
        d.joint_pos_target = Proxy(self.q_t)
        d.joint_pos_limits = Proxy(np.tile(np.array([-6.28, 6.28]), (self.E, self.J, 1)))
        d.joint_stiffness, d.joint_damping = Proxy(self.stiff), Proxy(self.damp)
        d.joint_effort_limits = Proxy(np.full((self.E, self.J), 50.0))
        d.body_mass = Proxy(np.ones((self.E, self.B)))

    def find_joints(self, names, preserve_order=True):
        names = [names] if isinstance(names, str) else list(names)
        return [self.joint_names.index(n) for n in names], names

    def find_bodies(self, names, preserve_order=True):
        names = [names] if isinstance(names, str) else list(names)
        return [self.body_names.index(n) for n in names], names

    def write_joint_state_to_sim_index(self, *, position, velocity, joint_ids=None, env_ids=None, **k):
        e = _ids(env_ids, self.E)
        self.q[e] = _check(position, (len(e), self.J), "position")
        self.qd[e] = _check(velocity, (len(e), self.J), "velocity")
        self._refresh()

    def write_joint_stiffness_to_sim_index(self, *, stiffness, joint_ids=None, env_ids=None, **k):
        e, j = _ids(env_ids, self.E), _ids(joint_ids, self.J)
        self.stiff[np.ix_(e, j)] = _check(stiffness, (len(e), len(j)), "stiffness")

    def write_joint_damping_to_sim_index(self, *, damping, joint_ids=None, env_ids=None, **k):
        e, j = _ids(env_ids, self.E), _ids(joint_ids, self.J)
        self.damp[np.ix_(e, j)] = _check(damping, (len(e), len(j)), "damping")

    def step(self, dt):
        self.qd = (self.q_t - self.q) / 0.03 + self.qd_t
        self.q = self.q + dt * self.qd
        self._refresh()


class MockRigidObject:
    def __init__(self, E, pos, kinematic=False):
        self.E, self.kinematic = E, kinematic
        self.pose = np.zeros((E, 7))
        self.pose[:, :3] = pos
        self.pose[:, 6] = 1.0
        self.vel = np.zeros((E, 6))
        self.mass = np.full((E, 1), 0.1)
        self.inertia = np.zeros((E, 1, 9))
        self.root_view = _View(E, 1)
        self.data = types.SimpleNamespace()
        self._refresh()

    def _refresh(self):
        self.data.root_link_pose_w = Proxy(self.pose)
        self.data.root_com_vel_w = Proxy(self.vel)

    def write_root_pose_to_sim_index(self, *, root_pose, env_ids=None, **k):
        e = _ids(env_ids, self.E)
        self.pose[e] = _check(root_pose, (len(e), 7), "root_pose")
        self._refresh()

    def write_root_velocity_to_sim_index(self, *, root_velocity, env_ids=None, **k):
        if self.kinematic:
            raise RuntimeError("velocity write on a kinematic body")
        e = _ids(env_ids, self.E)
        self.vel[e] = _check(root_velocity, (len(e), 6), "root_velocity")
        self._refresh()

    def set_masses_index(self, *, masses, body_ids=None, env_ids=None, **k):
        e = _ids(env_ids, self.E)
        self.mass[e] = _check(masses, (len(e), 1), "masses")

    def set_inertias_index(self, *, inertias, body_ids=None, env_ids=None, **k):
        e = _ids(env_ids, self.E)
        self.inertia[e] = _check(inertias, (len(e), 1, 9), "inertias")


class MockContactSensor:
    def __init__(self, E, H, body_names, n_filter=0, friction=False):
        self.body_names = list(body_names)
        S = len(body_names)
        self.data = types.SimpleNamespace(
            net_normal_forces_w_history=Proxy(np.zeros((E, H, S, 3))),
            normal_force_matrix_w_history=Proxy(np.zeros((E, H, S, n_filter, 3))) if n_filter else None,
            friction_force_matrix_w_history=Proxy(np.zeros((E, H, S, n_filter, 3))) if friction else None)


class MockCamera:
    def __init__(self, E, w, h, K):
        self.E, self.w, self.h = E, w, h
        self.K = np.repeat(K[None], E, 0)
        self.updates = 0
        self.data = types.SimpleNamespace(
            output={"rgb": Proxy(np.full((E, h, w, 4), 128, np.uint8)),
                    "distance_to_image_plane": Proxy(np.full((E, h, w, 1), 1.0, np.float32))},
            intrinsic_matrices=Proxy(self.K))

    def update(self, dt, force_recompute=False):
        self.updates += 1

    def set_world_poses(self, positions=None, orientations=None, env_ids=None, convention="ros"):
        _check(positions, (self.E, 3), "positions")
        _check(orientations, (self.E, 4), "orientations")


class MockScene:
    def __init__(self, entities, E, spacing=3.0):
        self.entities = entities
        self.env_origins = T(np.stack([np.array([spacing * i, 0.0, 0.0]) for i in range(E)]))
        self.resets = 0

    def __getitem__(self, k):
        return self.entities[k]

    def reset(self, env_ids=None):
        self.resets += 1

    def update(self, dt):
        pass

    def write_data_to_sim(self):
        pass


class MockSim:
    has_gui = False

    def __init__(self, robot, dt):
        self.robot, self.dt = robot, dt
        self.steps = 0

    def step(self, render=False):
        self.robot.step(self.dt)
        self.steps += 1

    def forward(self):
        self.robot._refresh()

    def render(self):
        pass


def build_mock_env(cfg, num_envs=3, camera_mode="fixed"):
    """PickPlaceEnv wired to the mocks (bypasses __init__, which needs the Kit app)."""
    import sys

    import a0509pp.sim.env as envmod
    from a0509pp.features import FeatureComputer
    from a0509pp.mock_render import intrinsics
    from a0509pp.models import Models
    from a0509pp.scene_spec import SceneSampler

    envmod.torch, envmod.wp = torch, wp
    physx_stub = types.ModuleType("isaaclab_physx.physics")
    physx_stub.PhysxManager = types.SimpleNamespace(get_physics_sim_view=lambda: types.SimpleNamespace(
        create_rigid_body_view=lambda path: types.SimpleNamespace(max_shapes=2)))
    sys.modules.setdefault("isaaclab_physx", types.ModuleType("isaaclab_physx"))
    sys.modules["isaaclab_physx.physics"] = physx_stub

    env = envmod.PickPlaceEnv.__new__(envmod.PickPlaceEnv)
    E = num_envs
    env.cfg, env.verbose, env.num_envs, env.device, env.manifest = cfg, False, E, "cpu", {}
    env.models = Models(cfg, {})
    env.gripper, env.arm = env.models.gripper, env.models.arm
    env.features = FeatureComputer(cfg, env.models)
    env.sampler = SceneSampler(cfg)
    env.home_q, env.home_info = env.models.solve_home()
    env.grip_torque = env.gripper.max_torque()
    P = cfg.physics
    env.physics_dt, env.decimation = P.dt, P.control_decimation
    env.control_dt = P.dt * P.control_decimation
    env.cam_ticks = max(1, P.camera_decimation // P.control_decimation)
    env.gui_render_every = 0
    g = cfg.gripper
    joints = list(cfg.robot.arm_joint_names) + [g.finger_joint] + list(g.passive_joints)
    bodies = [cfg.robot.arm_base_link] + [f"link_{i}" for i in range(1, 7)] + [g.base_link] + list(g.finger_links)
    robot = MockArticulation(E, joints, bodies, env.arm, env.gripper.T_flange_base(), env.home_q, bodies.index(g.base_link))
    L = cfg.layout
    ents = {"robot": robot, "robot_contacts": MockContactSensor(E, P.control_decimation, bodies)}
    objs = [f"object_{k}" for k in range(len(L.object_pool_dims))]
    obs = [f"obstacle_{j}" for j in range(len(L.obstacle_pool_dims))]
    for k, n in enumerate(objs):
        ents[n] = MockRigidObject(E, [0.2 + 0.1 * k, 0.3, -0.7])
        ents[f"{n}_contacts"] = MockContactSensor(E, P.control_decimation, [f"Object_{k}"], len(g.contact_links) + len(obs))
    for j, n in enumerate(obs):
        ents[n] = MockRigidObject(E, [0.2 + 0.1 * j, -0.3, -0.7], kinematic=True)
    for link in g.contact_links:
        ents[f"contact_{link}"] = MockContactSensor(E, P.control_decimation, [link], len(objs), friction=True)
    cams = {"fixed": ["fixed_camera"], "wrist": ["wrist_camera"], "both": ["fixed_camera", "wrist_camera"], "none": []}[camera_mode]
    C = cfg.camera
    for c in cams:
        ents[c] = MockCamera(E, C.width, C.height, intrinsics(C.width, C.height, C.hfov_deg))
    env.scene = MockScene(ents, E, cfg.env_spacing)
    env.sim = MockSim(robot, P.dt)
    env.info = {"objects": objs, "obstacles": obs, "cameras": cams, "pad_sensors": [f"contact_{l}" for l in g.contact_links],
                "object_sensors": [f"{n}_contacts" for n in objs], "camera_mode": camera_mode,
                "prim_paths": {"gripper_bodies": {n: f"x/{n}" for n in [g.base_link] + list(g.finger_links)}, "gripper_base": f"x/{g.base_link}"}}
    env.camera_mode = camera_mode
    env._init_handles()
    env._init_materials()
    env._tick = 0
    env.spec = env.ic = env.obs = env.percep = None
    env._cur_real = [None] * E
    return env
