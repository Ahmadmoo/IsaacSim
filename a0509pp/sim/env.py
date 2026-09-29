"""PickPlaceEnv: the Isaac Lab implementation of the section 15 interface.

Counterfactual batching: every environment copy holds the same base scene. After reset() settles the scene
(nominal physics) the state of env 0 is captured as the initial condition and copied to all envs. Each env then
runs one job = (candidate, hidden-physics realization); jobs beyond num_envs run in further batches from the same
restored initial condition. Observations are taken from env 0 only.

Import this module only after the Kit app is running (AppLauncher).
"""

from __future__ import annotations

import dataclasses
import json
import math
import time

import numpy as np

try:
    import torch
    import warp as wp
except ImportError:  # GPU-free tests (tests/fake_env.py) override every method that needs them
    torch = None
    wp = None

from ..candidates import validity_mask
from ..config import DATASET_VERSION, PROVISIONAL
from ..features import FEATURE_NAMES, FeatureComputer
from ..geometry import make_T, mat_to_quat, pose7_to_T_batch, quat_from_yaw, quat_to_mat, quat_to_mat_batch
from ..labels import assemble_labels, placement_check
from ..models import Models, load_manifest
from ..monitor import Monitor
from ..perception import estimate, privileged
from ..scene_spec import EpisodeSpec, Realization, SceneSampler, default_spec, delta_to_T
from ..trajectory import PHASE_ID
from .scene_cfg import (
    build_scene_cfg,
    fixed_camera_pose,
    obstacle_park_positions,
    object_park_positions,
    wrist_camera_pose,
)

HOLD = PHASE_ID["hold"]
DEFAULT_MIMIC_SIGN = {
    "right_outer_knuckle_joint": 1.0, "right_inner_finger_joint": 1.0, "left_inner_finger_joint": -1.0,
    "left_inner_finger_knuckle_joint": -1.0, "right_inner_finger_knuckle_joint": -1.0,
}


def _np(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _torch(x):
    return x.torch if hasattr(x, "torch") else x


def _box_inertia(m, dims):
    a, b, c = dims
    return np.array([m / 12.0 * (b * b + c * c), m / 12.0 * (a * a + c * c), m / 12.0 * (a * a + b * b)])


class PickPlaceEnv:
    def __init__(self, cfg, num_envs=None, camera_mode=None, device=None, manifest=None, gui_render_every=4, verbose=True):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

        self.cfg = cfg
        self.verbose = verbose
        self.num_envs = int(num_envs or cfg.num_envs)
        self.device = device or cfg.device
        self.manifest = manifest if manifest is not None else load_manifest(cfg.asset_manifest)
        if not self.manifest.get("robot_usd"):
            raise FileNotFoundError(f"asset manifest '{cfg.asset_manifest}' not found or incomplete; run scripts/prepare_assets.py")
        self.models = Models(cfg, self.manifest)
        self.gripper = self.models.gripper
        self.arm = self.models.arm
        self.features = FeatureComputer(cfg, self.models)
        self.sampler = SceneSampler(cfg)
        self.home_q, self.home_info = self.models.solve_home()
        self.grip_torque = self.gripper.max_torque()
        P = cfg.physics
        self.physics_dt = P.dt
        self.decimation = P.control_decimation
        self.control_dt = P.dt * P.control_decimation
        self.cam_ticks = max(1, P.camera_decimation // P.control_decimation)
        self.gui_render_every = gui_render_every

        physx = PhysxCfg(solver_type=P.solver_type, enable_stabilization=P.enable_stabilization,
                         enable_enhanced_determinism=P.enhanced_determinism, bounce_threshold_velocity=0.2)
        default_mat = PhysxRigidBodyMaterialCfg(static_friction=P.default_static_friction, dynamic_friction=P.default_dynamic_friction,
                                                restitution=P.restitution, friction_combine_mode=P.combine_mode,
                                                restitution_combine_mode=P.combine_mode)
        sim_cfg = sim_utils.SimulationCfg(dt=P.dt, device=self.device, gravity=tuple(P.gravity), physics=physx,
                                          physics_material=default_mat, render_interval=P.camera_decimation)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        scene_cfg, self.info = build_scene_cfg(cfg, self.manifest, self.home_q, self.num_envs, self.grip_torque, camera_mode)
        self.camera_mode = self.info["camera_mode"]
        self.scene = InteractiveScene(scene_cfg)
        self._author_ccd()
        if self.sim.has_gui:
            self.sim.set_camera_view(eye=(1.7, -1.5, 1.2), target=(0.45, 0.0, 0.1))
        self.sim.reset()
        self.scene.update(0.0)
        self._init_handles()
        self._init_materials()
        self._tick = 0
        self.spec = None
        self.ic = None
        self.obs = None
        self.percep = None
        self._cur_real = [None] * self.num_envs
        if self.cams:
            self._render_warmup(cfg.camera.warmup_renders)
        if verbose:
            print(f"[env] {self.num_envs} envs, camera mode '{self.camera_mode}', joints {self.joint_names}")
            print(f"[env] home q {np.round(self.home_q, 4).tolist()} ({self.home_info.get('branch')}), grip torque limit {self.grip_torque:.3f} N*m")

    # ================================================================== setup
    def _init_handles(self):
        cfg = self.cfg
        E = self.num_envs
        self.robot = self.scene["robot"]
        self.origins = _np(self.scene.env_origins).astype(np.float64)
        self.joint_names = list(self.robot.joint_names)
        self.J = len(self.joint_names)
        self.arm_ids = self.robot.find_joints(cfg.robot.arm_joint_names, preserve_order=True)[0]
        self.finger_id = self.robot.find_joints([cfg.gripper.finger_joint], preserve_order=True)[0][0]
        self.passive_ids = self.robot.find_joints(list(cfg.gripper.passive_joints), preserve_order=True)[0]
        mimic = self.manifest.get("gripper_mimic", {})
        self.mimic_coef = np.array([-mimic[n]["gearing"] if n in mimic else DEFAULT_MIMIC_SIGN.get(n, 0.0) for n in cfg.gripper.passive_joints])
        self.mimic_off = np.array([-mimic[n].get("offset", 0.0) if n in mimic else 0.0 for n in cfg.gripper.passive_joints])
        self.body_names = list(self.robot.body_names)
        self.gbase_body = self.robot.find_bodies([cfg.gripper.base_link], preserve_order=True)[0][0]
        self.link6_body = self.robot.find_bodies([cfg.robot.flange_link], preserve_order=True)[0][0]
        dev = self.device
        self.wp_arm = wp.array(np.asarray(self.arm_ids, np.int32), dtype=wp.int32, device=dev)
        self.wp_finger = wp.array(np.asarray([self.finger_id], np.int32), dtype=wp.int32, device=dev)
        self.wp_all_envs = wp.array(np.arange(E, dtype=np.int32), dtype=wp.int32, device=dev)
        self.objects = [self.scene[n] for n in self.info["objects"]]
        self.obstacles = [self.scene[n] for n in self.info["obstacles"]]
        self.obj_park = object_park_positions(cfg.layout, cfg.layout.object_pool_dims)
        self.obs_park = obstacle_park_positions(cfg.layout, cfg.layout.obstacle_pool_dims)
        self.robot_cs = self.scene["robot_contacts"]
        self.pad_cs = [self.scene[n] for n in self.info["pad_sensors"]]
        self.obj_cs = [self.scene[n] for n in self.info["object_sensors"]]
        self.cams = {n: self.scene[n] for n in self.info["cameras"]}
        self.cs_names = list(self.robot_cs.body_names)
        classes = []
        for n in self.cs_names:
            if n == cfg.robot.arm_base_link:
                classes.append("base")
            elif n in cfg.gripper.contact_links:
                classes.append("pad")
            elif n in cfg.gripper.finger_links:
                classes.append("finger")
            else:
                classes.append("arm")
        self.body_classes = classes
        forb = [i for i, c in enumerate(classes) if c in ("arm", "finger")]
        self.contact_names = [self.cs_names[i] for i in forb] + list(cfg.gripper.contact_links)
        self.pad_side = np.array([-1 if l.startswith("left") else 1 for l in cfg.gripper.contact_links])
        self.left_pads = [i for i, s in enumerate(self.pad_side) if s < 0]
        self.right_pads = [i for i, s in enumerate(self.pad_side) if s > 0]
        tc = getattr(self.robot, "actuators", None)
        self._tc = getattr(tc, "target_command", None) if tc is not None else None
        self.nominal_stiffness = np.array(cfg.robot.servo_stiffness, float)
        self.nominal_damping = np.array(cfg.robot.servo_damping, float)
        self.T_base_tcp = self.gripper.T_base_tcp()
        self.T_flange_tcp = self.gripper.T_flange_tcp()

    def _init_materials(self):
        """Shape index ranges for the pad bodies in the articulation material buffer (PhysX tensor API)."""
        from isaaclab_physx.physics import PhysxManager

        self._pad_shape_idx = None
        try:
            view = self.robot.root_view
            sim_view = PhysxManager.get_physics_sim_view()
            paths = list(view.link_paths[0])
            counts = [sim_view.create_rigid_body_view(p).max_shapes for p in paths]
            names = [p.split("/")[-1] for p in paths]
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)
            idx = []
            for link in self.cfg.gripper.pad_links:
                i = names.index(link)
                idx += list(range(starts[i], starts[i] + counts[i]))
            if sum(counts) != view.max_shapes:
                raise RuntimeError(f"shape count mismatch {sum(counts)} != {view.max_shapes}")
            self._pad_shape_idx = idx
        except Exception as exc:  # pad friction then stays at the USD-bound nominal value
            print(f"[env] WARNING: pad material indexing unavailable ({exc}); pad friction randomization disabled")

    def _author_ccd(self):
        """Speculative CCD on the small dynamic bodies (blocks, fingertips), as the Robotiq guide recommends."""
        if not self.cfg.physics.object_speculative_ccd:
            return
        import isaaclab.sim as sim_utils
        from pxr import Sdf

        stage = sim_utils.get_current_stage()
        paths = self.info["prim_paths"]
        for i in range(self.num_envs):
            prims = [f"/World/envs/env_{i}/Object_{k}" for k in range(len(self.info["objects"]))]
            prims += [f"/World/envs/env_{i}/Robot/" + paths["gripper_bodies"][l] for l in self.cfg.gripper.pad_links]
            for path in prims:
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid() or prim.IsInstanceProxy():
                    continue
                if "PhysxRigidBodyAPI" not in prim.GetAppliedSchemas():
                    prim.AddAppliedSchema("PhysxRigidBodyAPI")
                prim.CreateAttribute("physxRigidBody:enableSpeculativeCCD", Sdf.ValueTypeNames.Bool).Set(True)

    def _render_warmup(self, n):
        for _ in range(max(0, int(n))):
            self._render_cameras(read=False)

    # ================================================================== low-level helpers
    def _envs_wp(self, env_ids):
        if env_ids is None:
            return self.wp_all_envs
        return wp.array(np.asarray(env_ids, np.int32), dtype=wp.int32, device=self.device)

    def _t(self, x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    def full_joint_pos(self, q_arm, theta):
        """(N, J) joint vector from arm joints and finger angle, passive joints on their mimic coupling."""
        q_arm = np.atleast_2d(q_arm)
        theta = np.broadcast_to(np.asarray(theta, float), (q_arm.shape[0],))
        out = np.zeros((q_arm.shape[0], self.J))
        out[:, self.arm_ids] = q_arm
        out[:, self.finger_id] = theta
        out[:, self.passive_ids] = theta[:, None] * self.mimic_coef[None] + self.mimic_off[None]
        return out

    def set_targets(self, q_cmd, qd_cmd, aperture_cmd, env_ids=None):
        """Arm joint position + velocity targets and the finger-joint target from a commanded aperture [m]."""
        theta = np.asarray(self.gripper.theta(np.asarray(aperture_cmd, float)), float).reshape(-1, 1)
        ids = self._envs_wp(env_ids)
        q, qd, th = self._t(q_cmd), self._t(qd_cmd), self._t(theta)
        if self._tc is not None:
            self._tc.set_position_index(value=q, joint_ids=self.wp_arm, env_ids=ids)
            self._tc.set_velocity_index(value=qd, joint_ids=self.wp_arm, env_ids=ids)
            self._tc.set_position_index(value=th, joint_ids=self.wp_finger, env_ids=ids)
        else:  # older API
            self.robot.set_joint_position_target_index(target=q, joint_ids=self.wp_arm, env_ids=ids)
            self.robot.set_joint_velocity_target_index(target=qd, joint_ids=self.wp_arm, env_ids=ids)
            self.robot.set_joint_position_target_index(target=th, joint_ids=self.wp_finger, env_ids=ids)

    def _physics_steps(self, n=None):
        for _ in range(n or self.decimation):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
        self._tick += 1
        if self.sim.has_gui and self.gui_render_every and self._tick % self.gui_render_every == 0:
            self.sim.render()

    def write_robot_state(self, q_arm, theta, env_ids=None, qd_arm=None):
        n = self.num_envs if env_ids is None else len(env_ids)
        pos = self.full_joint_pos(np.broadcast_to(q_arm, (n, 6)), theta)
        vel = np.zeros_like(pos)
        if qd_arm is not None:
            vel[:, self.arm_ids] = qd_arm
        self.robot.write_joint_state_to_sim_index(position=self._t(pos), velocity=self._t(vel), env_ids=self._envs_wp(env_ids))

    def _write_pose(self, asset, local_pose, env_ids=None, vel=None, kinematic=False):
        ids = np.arange(self.num_envs) if env_ids is None else np.asarray(env_ids)
        lp = np.atleast_2d(np.asarray(local_pose, float))
        pose = np.broadcast_to(lp, (len(ids), 7)).copy()
        pose[:, :3] += self.origins[ids]
        w = self._envs_wp(ids)
        asset.write_root_pose_to_sim_index(root_pose=self._t(pose), env_ids=w)
        if not kinematic:  # PhysX rejects velocity writes on kinematic bodies
            v = np.zeros((len(ids), 6)) if vel is None else np.broadcast_to(np.atleast_2d(vel), (len(ids), 6))
            asset.write_root_velocity_to_sim_index(root_velocity=self._t(v), env_ids=w)

    # ================================================================== physical parameters
    def _set_object_physics(self, k, env_ids, masses, sf, df):
        obj = self.objects[k]
        dims = self.cfg.layout.object_pool_dims[k]
        ids = self._envs_wp(env_ids)
        m = np.asarray(masses, float).reshape(-1, 1)
        obj.set_masses_index(masses=self._t(m), env_ids=ids)
        inert = np.zeros((len(env_ids), 1, 9))
        for i, mi in enumerate(m[:, 0]):
            inert[i, 0, [0, 4, 8]] = _box_inertia(mi, dims)
        obj.set_inertias_index(inertias=self._t(inert), env_ids=ids)
        view = obj.root_view
        mats = wp.to_torch(view.get_material_properties()).clone()
        e = torch.as_tensor(np.asarray(env_ids), dtype=torch.long)
        mats[e, :, 0] = torch.as_tensor(np.asarray(sf, np.float32))[:, None]
        mats[e, :, 1] = torch.as_tensor(np.asarray(df, np.float32))[:, None]
        mats[e, :, 2] = float(self.cfg.physics.restitution)
        view.set_material_properties(wp.from_torch(mats.contiguous(), dtype=wp.float32),
                                     wp.from_torch(e.to(torch.int32).contiguous(), dtype=wp.int32))

    def _set_pad_friction(self, env_ids, sf, df):
        if self._pad_shape_idx is None:
            return
        view = self.robot.root_view
        mats = wp.to_torch(view.get_material_properties()).clone()
        e = torch.as_tensor(np.asarray(env_ids), dtype=torch.long)
        idx = torch.as_tensor(self._pad_shape_idx, dtype=torch.long)
        sf_t = torch.as_tensor(np.asarray(sf, np.float32))
        df_t = torch.as_tensor(np.asarray(df, np.float32))
        for i, ei in enumerate(e):
            mats[ei, idx, 0] = sf_t[i]
            mats[ei, idx, 1] = df_t[i]
            mats[ei, idx, 2] = float(self.cfg.physics.restitution)
        view.set_material_properties(wp.from_torch(mats.contiguous(), dtype=wp.float32),
                                     wp.from_torch(e.to(torch.int32).contiguous(), dtype=wp.int32))

    def _set_servo(self, env_ids, scales):
        s = np.asarray(scales, float).reshape(len(env_ids), 6)
        ids = self._envs_wp(env_ids)
        self.robot.write_joint_stiffness_to_sim_index(stiffness=self._t(s * self.nominal_stiffness), joint_ids=self.wp_arm, env_ids=ids)
        self.robot.write_joint_damping_to_sim_index(damping=self._t(s * self.nominal_damping), joint_ids=self.wp_arm, env_ids=ids)

    def nominal_realization(self):
        P, L = self.cfg.physics, self.cfg.layout
        return Realization(rid=-1, object_mass=L.target_default_mass, object_static_friction=P.object_static_friction,
                           object_dynamic_friction=P.object_dynamic_friction, pad_static_friction=P.pad_static_friction,
                           pad_dynamic_friction=P.pad_dynamic_friction)

    def apply_realizations(self, reals, k_target):
        """Per-env hidden physics (object mass/inertia/friction, pad friction, servo gains). Writes only changes."""
        E = self.num_envs
        assert len(reals) == E
        key = lambda r: (r.object_mass, r.object_static_friction, r.object_dynamic_friction, r.pad_static_friction,
                         r.pad_dynamic_friction, tuple(r.servo_scale), k_target)
        changed = [e for e in range(E) if self._cur_real[e] is None or key(self._cur_real[e]) != key(reals[e])]
        if not changed:
            return
        rs = [reals[e] for e in changed]
        self._set_object_physics(k_target, changed, [r.object_mass for r in rs], [r.object_static_friction for r in rs],
                                 [r.object_dynamic_friction for r in rs])
        self._set_pad_friction(changed, [r.pad_static_friction for r in rs], [r.pad_dynamic_friction for r in rs])
        self._set_servo(changed, [r.servo_scale for r in rs])
        for e in changed:
            self._cur_real[e] = dataclasses.replace(reals[e])

    # ================================================================== scene layout and visuals
    def _target_index(self, spec):
        k = int(spec.target.pool_index)
        dims = self.cfg.layout.object_pool_dims[k]
        if np.max(np.abs(np.asarray(dims) - np.asarray(spec.target.dims))) > 1e-6:
            raise ValueError(f"spec target dims {spec.target.dims} do not match object pool entry {k} {dims}")
        return k

    def _place_layout(self, spec):
        k_t = self._target_index(spec)
        for k, obj in enumerate(self.objects):
            dims = self.cfg.layout.object_pool_dims[k]
            if k == k_t:
                pose = [spec.target.xy[0], spec.target.xy[1], dims[2] / 2.0 + 5e-4] + list(quat_from_yaw(spec.target.yaw))
            else:
                pose = list(self.obj_park[k]) + [0.0, 0.0, 0.0, 1.0]
            self._write_pose(obj, pose)
        active = {o.pool_index: o for o in spec.obstacles}
        for j, ob in enumerate(self.obstacles):
            dims = self.cfg.layout.obstacle_pool_dims[j]
            if j in active:
                o = active[j]
                pose = [o.xy[0], o.xy[1], dims[2] / 2.0 + 2e-4] + list(quat_from_yaw(o.yaw))
            else:
                pose = list(self.obs_park[j]) + [0.0, 0.0, 0.0, 1.0]
            self._write_pose(ob, pose, kinematic=True)
        return k_t

    def _apply_visuals(self, spec):
        try:
            import isaaclab.sim as sim_utils
            from pxr import Gf, Usd, UsdShade

            stage = sim_utils.get_current_stage()
            for path, base in (("/World/DomeLight", 1200.0), ("/World/KeyLight", 2500.0)):
                prim = stage.GetPrimAtPath(path)
                if prim.IsValid():
                    attr = prim.GetAttribute("inputs:intensity")
                    if attr and attr.IsValid():
                        attr.Set(float(base * spec.light_scale))
            key = stage.GetPrimAtPath("/World/KeyLight")
            if key.IsValid():
                tilt = np.array([0.2706, 0.2706, 0.0, 0.9239])
                yaw = quat_from_yaw(spec.light_yaw)
                from ..geometry import quat_mul

                q = quat_mul(yaw, tilt)
                a = key.GetAttribute("xformOp:orient")
                if a and a.IsValid():
                    qt = Gf.Quatf if "quatf" in str(a.GetTypeName()) else Gf.Quatd
                    a.Set(qt(float(q[3]), float(q[0]), float(q[1]), float(q[2])))
            if "object" in spec.families:
                k = spec.target.pool_index
                if not hasattr(self, "_color_attrs"):
                    self._color_attrs = {}
                if k not in self._color_attrs:
                    attrs = []
                    for i in range(self.num_envs):
                        root = stage.GetPrimAtPath(f"/World/envs/env_{i}/Object_{k}")
                        for p in (Usd.PrimRange(root) if root.IsValid() else []):
                            if p.IsA(UsdShade.Shader):
                                a = p.GetAttribute("inputs:diffuseColor")
                                if a and a.IsValid():
                                    attrs.append(a)
                    self._color_attrs[k] = attrs
                for a in self._color_attrs[k]:
                    a.Set(Gf.Vec3f(*[float(c) for c in spec.target.color]))
        except Exception as exc:
            if self.verbose:
                print(f"[env] visual randomization skipped: {exc}")
        if "fixed_camera" in self.cams:
            T = self.fixed_camera_true(spec)
            pos = T[:3, 3][None] + self.origins
            q = np.broadcast_to(mat_to_quat(T[:3, :3]), (self.num_envs, 4))
            self.cams["fixed_camera"].set_world_poses(positions=self._t(pos), orientations=self._t(q), convention="ros")

    def fixed_camera_nominal(self):
        pos, q = fixed_camera_pose(self.cfg.camera)
        return make_T(quat_to_mat(q), pos)

    def fixed_camera_true(self, spec):
        return self.fixed_camera_nominal() @ delta_to_T(spec.fixed_cam_delta)

    def T_base_wrist_cam(self):
        pos, q = wrist_camera_pose(self.cfg.camera)
        return make_T(quat_to_mat(q), pos)

    # ================================================================== state readout
    def read_state(self, k_target=None):
        """Batched numpy readout of robot, target object and contact sensors (after the last physics step)."""
        rd = self.robot.data
        jp = _np(_torch(rd.joint_pos)).astype(np.float64)
        jv = _np(_torch(rd.joint_vel)).astype(np.float64)
        bp = _np(_torch(rd.body_link_pose_w)[:, self.gbase_body]).astype(np.float64)
        bp[:, :3] -= self.origins
        T_gb = pose7_to_T_batch(bp)
        T_tcp = T_gb @ self.T_base_tcp
        th = jp[:, self.finger_id]
        out = {
            "q": jp[:, self.arm_ids], "qd": jv[:, self.arm_ids], "theta": th, "aperture": np.asarray(self.gripper.aperture(th)),
            "joint_pos": jp, "joint_vel": jv, "T_gbase": T_gb, "T_tcp": T_tcp,
        }
        lf = _torch(self.robot_cs.data.net_normal_forces_w_history)
        out["link_force"] = _np(torch.linalg.norm(lf, dim=-1).flip(1)).astype(np.float64)
        if k_target is None:
            return out
        od = self.objects[k_target].data
        op = _np(_torch(od.root_link_pose_w)).astype(np.float64)
        op[:, :3] -= self.origins
        ov = _np(_torch(od.root_com_vel_w)).astype(np.float64)
        out["obj_pose"] = op
        out["obj_vel"] = ov
        # Pad sensors: force with the target object and with everything else.
        tgt_vec, other = [], []
        for s in self.pad_cs:
            d = s.data
            net = _torch(d.net_normal_forces_w_history)[:, :, 0]
            mat = _torch(d.normal_force_matrix_w_history)[:, :, 0]
            tgt_vec.append(mat[:, :, k_target])
            other.append(torch.linalg.norm(net - mat.sum(2), dim=-1))
        tv = torch.stack(tgt_vec, 2)  # (E, H, 4, 3), newest first
        side_l = tv[:, 0, self.left_pads].sum(1)
        side_r = tv[:, 0, self.right_pads].sum(1)
        fl, fr = torch.linalg.norm(side_l, dim=-1), torch.linalg.norm(side_r, dim=-1)
        out["pad_obj_force"] = _np(torch.minimum(fl, fr)).astype(np.float64)
        out["pad_obj_force_max"] = _np(torch.maximum(fl, fr)).astype(np.float64)
        out["pad_other_force"] = _np(torch.stack(other, 2).flip(1)).astype(np.float64)
        fric = []
        for s in self.pad_cs:
            fm = getattr(s.data, "friction_force_matrix_w_history", None)
            if fm is not None:
                fric.append(torch.linalg.norm(_torch(fm)[:, 0, 0, k_target], dim=-1))
        out["pad_obj_friction"] = _np(torch.stack(fric, 1).sum(1)) if fric else np.zeros(self.num_envs)
        # Target object sensor: obstacles and remaining environment contact.
        d = self.obj_cs[k_target].data
        net = _torch(d.net_normal_forces_w_history)[:, :, 0]
        mat = _torch(d.normal_force_matrix_w_history)[:, :, 0]
        n_pad = len(self.pad_cs)
        out["obj_obstacle_force"] = _np(torch.linalg.norm(mat[:, :, n_pad:], dim=-1).flip(1)).astype(np.float64)
        out["obj_env_force"] = _np(torch.linalg.norm(net - mat.sum(2), dim=-1).flip(1)).astype(np.float64)
        return out

    # ================================================================== initial conditions
    def capture_initial_condition(self, env=0):
        rd = self.robot.data
        tgt = _torch(self._tc.position) if self._tc is not None else _torch(rd.joint_pos_target)
        ic = {
            "dataset_version": DATASET_VERSION,
            "scene_id": None if self.spec is None else self.spec.scene_id,
            "joint_names": self.joint_names,
            "joint_pos": _np(_torch(rd.joint_pos)[env]).astype(np.float64).tolist(),
            "joint_vel": _np(_torch(rd.joint_vel)[env]).astype(np.float64).tolist(),
            "joint_pos_target": _np(tgt[env]).astype(np.float64).tolist(),
            "objects": [], "obstacles": [],
            "target_index": None if self.spec is None else int(self.spec.target.pool_index),
            "physics_dt": self.physics_dt, "control_dt": self.control_dt,
        }
        for obj in self.objects:
            p = _np(_torch(obj.data.root_link_pose_w)[env]).astype(np.float64)
            p[:3] -= self.origins[env]
            v = _np(_torch(obj.data.root_com_vel_w)[env]).astype(np.float64)
            ic["objects"].append({"pose": p.tolist(), "vel": v.tolist()})
        for ob in self.obstacles:
            p = _np(_torch(ob.data.root_link_pose_w)[env]).astype(np.float64)
            p[:3] -= self.origins[env]
            ic["obstacles"].append({"pose": p.tolist()})
        ic.update(self._ic_meta())
        return ic

    def _ic_meta(self):
        """Controller, queue and RNG state that the rollouts start from, and what restoring does not cover."""
        spec = self.spec
        return {
            "controller": {"type": "implicit PD + velocity feed-forward", "servo_scale": "per realization",
                           "command_queue": "empty; a rollout with delay d repeats the first reference sample for d ticks",
                           "monitor": "fresh per rollout"},
            "rng": None if spec is None else {"observation": [int(spec.seed), _stable_hash(spec.scene_id), 101],
                                              "planner": [int(spec.seed), 17], "scene": [int(spec.seed), spec.scene_id]},
            "restoration": "deterministic reconstruction: joint, body and target states are written through the PhysX "
                           "tensor API; solver warm-start and contact caches are not restored. Check with "
                           "scripts/check_repeatability.py.",
        }

    def restore_initial_condition(self, ic, env_ids=None, validate=True, tol=1e-4):
        """Write a captured initial condition into env_ids (default all); returns a validation report."""
        if list(ic["joint_names"]) != self.joint_names:
            raise ValueError("initial condition joint names do not match the articulation")
        ids = list(range(self.num_envs)) if env_ids is None else [int(e) for e in env_ids]
        n = len(ids)
        self.scene.reset(ids)
        pos = np.broadcast_to(np.asarray(ic["joint_pos"]), (n, self.J))
        vel = np.broadcast_to(np.asarray(ic["joint_vel"]), (n, self.J))
        w = self._envs_wp(ids)
        self.robot.write_joint_state_to_sim_index(position=self._t(pos), velocity=self._t(vel), env_ids=w)
        tgt = np.asarray(ic["joint_pos_target"])
        th = float(tgt[self.finger_id])
        self.set_targets(np.broadcast_to(tgt[self.arm_ids], (n, 6)), np.zeros((n, 6)),
                         np.full(n, float(self.gripper.aperture(th))), env_ids=ids)
        for obj, s in zip(self.objects, ic["objects"]):
            self._write_pose(obj, s["pose"], ids, s["vel"])
        for ob, s in zip(self.obstacles, ic["obstacles"]):
            self._write_pose(ob, s["pose"], ids, kinematic=True)
        self.sim.forward()
        self.scene.update(0.0)
        if not validate:
            return {"ok": True, "validated": False}
        rd = self.robot.data
        jp = _np(_torch(rd.joint_pos))[ids]
        rep = {"joint_err": float(np.max(np.abs(jp - pos))), "object_err": 0.0}
        for obj, s in zip(self.objects, ic["objects"]):
            p = _np(_torch(obj.data.root_link_pose_w))[ids].astype(np.float64)
            p[:, :3] -= self.origins[ids]
            dq = np.abs(np.abs(np.sum(p[:, 3:] * np.asarray(s["pose"][3:])[None], 1)) - 1.0)
            rep["object_err"] = max(rep["object_err"], float(np.max(np.abs(p[:, :3] - np.asarray(s["pose"][:3])))), float(np.max(dq)))
        rep["ok"] = rep["joint_err"] < tol and rep["object_err"] < tol
        rep["validated"] = True
        return rep

    # ================================================================== reset / settle / observe
    def sample_spec(self, index, seed, families=None, fixed_pose=False, num_realizations=None):
        return self.sampler.sample(index, seed, families, fixed_pose, num_realizations or self.cfg.collect.realizations)

    def hold(self, n_ticks):
        for _ in range(int(n_ticks)):
            self._physics_steps()

    def reset(self, spec: EpisodeSpec | None = None):
        """Lay out the scene in all envs, settle with nominal physics, capture observation and initial condition."""
        spec = spec or default_spec(self.cfg)
        self.spec = spec
        E = self.num_envs
        k_t = self._place_layout(spec)
        self.k_target = k_t
        self.apply_realizations([self.nominal_realization()] * E, k_t)
        self.scene.reset()
        self.write_robot_state(self.home_q, 0.0)
        self.set_targets(np.broadcast_to(self.home_q, (E, 6)), np.zeros((E, 6)), np.full(E, self.cfg.gripper.open_aperture))
        self._apply_visuals(spec)
        self.sim.forward()
        self.settle_info = self._settle(k_t)
        frames = self._capture_frames(spec) if self.cams else []
        ic = self.capture_initial_condition(0)
        rep = self.restore_initial_condition(ic)
        if not rep["ok"]:
            print(f"[env] WARNING: initial-condition broadcast validation {rep}")
        self.ic = ic
        self.ic_report = rep
        self.obs = self._make_observation(spec, frames)
        return self.obs

    def _settle(self, k_t):
        P = self.cfg.physics
        n_min = int(math.ceil(P.settle_min / self.control_dt))
        n_max = int(math.ceil(P.settle_max / self.control_dt))
        n_hold = max(1, int(math.ceil(P.settle_hold / self.control_dt)))
        calm = 0
        t0 = time.time()
        for k in range(n_max):
            self._physics_steps()
            v = _np(_torch(self.objects[k_t].data.root_com_vel_w)[0])
            calm = calm + 1 if (np.linalg.norm(v[:3]) < P.settle_lin_speed and np.linalg.norm(v[3:]) < P.settle_ang_speed) else 0
            if k + 1 >= n_min and calm >= n_hold:
                break
        return {"settle_time": (k + 1) * self.control_dt, "settled": calm >= n_hold, "wall_s": time.time() - t0}

    def _render_cameras(self, read=True):
        self.sim.render()
        out = {}
        for name, cam in self.cams.items():
            cam.update(0.0, force_recompute=True)
            d = cam.data
            if read:
                rgb = _np(_torch(d.output["rgb"])[0])[..., :3].astype(np.uint8)
                depth = _np(_torch(d.output["distance_to_image_plane"])[0]).astype(np.float32).reshape(rgb.shape[:2])
                K = _np(_torch(d.intrinsic_matrices)[0]).astype(np.float64)
                out[name] = {"rgb": rgb, "depth": depth, "K": K}
        return out

    def _capture_frames(self, spec):
        C = self.cfg.camera
        delay_max = max(self.cfg.randomization.camera_delay)
        n = C.history + int(delay_max)
        for _ in range(C.rerenders_on_reset):
            self._render_cameras(read=False)
        frames = []
        for i in range(n):
            if i > 0:
                self.hold(self.cam_ticks)
            f = self._render_cameras()
            rd = self.robot.data
            f["_q"] = _np(_torch(rd.joint_pos)[0][self.arm_ids]).astype(np.float64)
            bp = _np(_torch(rd.body_link_pose_w)[0, self.gbase_body]).astype(np.float64)
            bp[:3] -= self.origins[0]
            f["_T_gbase"] = pose7_to_T_batch(bp[None])[0]
            f["_t"] = self._tick * self.control_dt
            frames.append(f)
        return frames

    def _make_observation(self, spec, frames):
        cfg = self.cfg
        C = cfg.camera
        rng = np.random.default_rng([spec.seed, _stable_hash(spec.scene_id), 101])
        st = self.read_state(self.k_target)
        q_true = st["q"][0]
        noise = (lambda: rng.normal(0.0, spec.joint_noise, 6)) if spec.joint_noise > 0 else (lambda: np.zeros(6))
        q_meas = q_true + noise()
        dt_cam = self.cam_ticks * self.control_dt
        q_prev = (frames[-2]["_q"] if len(frames) >= 2 else q_true) + noise()
        policy, ann = {}, {}
        policy["q"] = q_meas
        policy["qd"] = (q_meas - q_prev) / dt_cam
        policy["gripper_aperture"] = np.float64(st["aperture"][0])
        policy["finger_joint"] = np.float64(st["theta"][0])
        policy["tray_xy"] = np.asarray(spec.tray_xy, float)
        policy["target_hint_xy"] = np.asarray(spec.target_hint_xy, float)
        ann["q_true"] = q_true
        ann["object_pose"] = st["obj_pose"][0]
        ann["obstacle_poses"] = np.array([s["pose"] for s in self.ic["obstacles"]]) if self.ic else np.zeros((0, 7))
        views, calib = [], {}
        d = int(spec.camera_delay)
        sel = frames[len(frames) - C.history - d: len(frames) - d] if frames else []
        if sel:
            policy["q_history"] = np.stack([f["_q"] + noise() for f in sel])
            policy["q_history_t"] = np.asarray([f["_t"] for f in sel])
        for name in self.cams:
            rgbs, depths, valids, Tr, Tt, times = [], [], [], [], [], []
            for f in sel:
                v = f[name]
                depth = v["depth"].copy()
                valid = np.isfinite(depth) & (depth > C.clip[0]) & (depth < C.clip[1])
                if spec.depth_sigma > 0:
                    depth = depth + rng.normal(0.0, spec.depth_sigma, depth.shape).astype(np.float32)
                if spec.depth_dropout > 0:
                    valid &= rng.random(depth.shape) >= spec.depth_dropout
                depth = np.where(valid, depth, np.nan).astype(np.float32)
                if name == "fixed_camera":
                    T_true = self.fixed_camera_true(spec)
                    T_rep = T_true @ delta_to_T(spec.calib_delta_fixed)
                else:
                    T_true = f["_T_gbase"] @ self.T_base_wrist_cam()
                    q_rep = f["_q"] + (q_meas - q_true)
                    T_rep = self.arm.fk(q_rep) @ self.gripper.T_flange_base() @ self.T_base_wrist_cam() @ delta_to_T(spec.calib_delta_wrist)
                rgbs.append(v["rgb"])
                depths.append(depth)
                valids.append(valid)
                Tr.append(T_rep)
                Tt.append(T_true)
                times.append(f["_t"])
            K = sel[-1][name]["K"]
            policy[f"{name}_rgb"] = np.stack([_downsample_rgb(x, C.net_width, C.net_height) for x in rgbs])
            dn = [_downsample_depth(x, C.net_width, C.net_height) for x in depths]
            policy[f"{name}_depth"] = np.stack([x[0] for x in dn]).astype(np.float32)
            policy[f"{name}_depth_valid"] = np.stack([x[1] for x in dn])
            Kn = K.copy()
            Kn[0] *= C.net_width / C.width
            Kn[1] *= C.net_height / C.height
            policy[f"{name}_K"] = Kn
            policy[f"{name}_T_world_cam"] = np.stack(Tr)
            policy[f"{name}_frame_t"] = np.asarray(times)
            ann[f"{name}_T_world_cam_true"] = np.stack(Tt)
            ann[f"{name}_K_full"] = K
            if cfg.collect.save_diag_frames:
                ann[f"{name}_depth_clean_last"] = sel[-1][name]["depth"].astype(np.float32)
            views.append({"depth": depths[-1], "K": K, "T_world_cam": Tr[-1], "valid": valids[-1], "name": name})
            calib[name] = {"K_full": K.tolist(), "K_net": Kn.tolist(), "T_world_cam_reported": Tr[-1].tolist(),
                           "resolution": [C.width, C.height], "net_resolution": [C.net_width, C.net_height],
                           "hfov_deg": C.hfov_deg, "depth": "distance to image plane (optical-axis z), NaN = invalid"}
        calib["tcp"] = {"T_flange_tcp": self.T_flange_tcp.tolist(), "source": self.gripper.calibration_source}
        return {"policy": policy, "annotation": ann, "views": views, "calibration": calib, "q_meas": q_meas,
                "frames_delay": d, "camera_mode": self.camera_mode}

    def observe(self):
        return self.obs

    # ================================================================== perception / planning / features
    def perceive(self, source=None, views=None):
        """Scene estimate from the enabled camera views (policy-visible) or the labelled privileged oracle."""
        source = source or self.cfg.perception.mode
        spec = self.spec
        vs = [v for v in self.obs["views"] if views is None or v["name"] in views]
        if source == "camera" and not vs:
            raise ValueError(f"camera perception requested but no camera views among {views or list(self.cams)}")
        if source == "privileged":
            p = self.obs["annotation"]["object_pose"]
            yaw = math.atan2(*quat_to_mat(p[3:])[[1, 0], 0])
            self.percep = privileged(spec, (np.array([p[0], p[1], spec.target.dims[2] / 2.0]), yaw))
        else:
            q = self.obs["q_meas"]
            th = float(self.gripper.theta(self.obs["policy"]["gripper_aperture"]))
            Cs, rs, _ = self.models.spheres.compute(q[None], th)
            self.percep = estimate(vs, self.cfg.layout, spec.tray_xy, spec.target_hint_xy, self.cfg.perception, (Cs[0], rs))
            self.percep.diagnostics["views"] = [v["name"] for v in vs]
        return self.percep

    def propose_candidates(self, percep=None, rng=None):
        percep = percep or self.percep or self.perceive()
        rng = rng or np.random.default_rng([self.spec.seed, 17])
        t0 = time.time()
        cands, pool = self.models.planner.propose(self.obs["q_meas"], percep, self.spec.tray_xy, rng=rng)
        pool["plan_wall_s"] = time.time() - t0
        self.cands, self.pool = cands, pool
        return cands, pool

    def evaluate_physics(self, cands, model="nominal", percep=None):
        """Section 9 motion-dependent features under the nominal (perceived) or oracle (true scene) model."""
        percep = percep or self.percep
        X, dicts, metas = [], [], []
        for c in cands:
            v, f, m = self.features.compute(c, percep, self.spec.tray_xy, model=model, spec=self.spec)
            X.append(v)
            dicts.append(f)
            metas.append(m)
        return (np.stack(X) if X else np.zeros((0, len(FEATURE_NAMES)), np.float32)), dicts, metas

    # ================================================================== execution
    def step(self, q_cmd, qd_cmd=None, aperture_cmd=None, env_ids=None, k_target=None):
        """One 120 Hz control tick with the given commands; returns read_state()."""
        n = self.num_envs if env_ids is None else len(env_ids)
        q_cmd = np.broadcast_to(np.asarray(q_cmd, float), (n, 6))
        qd_cmd = np.zeros((n, 6)) if qd_cmd is None else np.broadcast_to(np.asarray(qd_cmd, float), (n, 6))
        ap = np.full(n, self.cfg.gripper.open_aperture) if aperture_cmd is None else np.broadcast_to(np.asarray(aperture_cmd, float), (n,))
        self.set_targets(q_cmd, qd_cmd, ap, env_ids)
        self._physics_steps()
        return self.read_state(self.k_target if k_target is None else k_target)

    def run_candidate(self, cand, realization=None, record_traj=True):
        return self.run_candidates([cand], [realization or self.nominal_realization()], record_traj)[0]

    def run_candidates(self, cands, realizations, record_traj=True, progress=None, video=False):
        """Every candidate under every realization, batched over envs. Returns one result dict per job.
        video: also store camera frames of every rollout at collect.video_hz (diagnostic subset)."""
        jobs = [(c, r) for c in cands for r in realizations]
        results = []
        for b in range(0, len(jobs), self.num_envs):
            batch = jobs[b:b + self.num_envs]
            t0 = time.time()
            results += self._rollout_batch(batch, record_traj, video)
            if progress:
                progress(b + len(batch), len(jobs), time.time() - t0)
        return results

    def _job_arrays(self, jobs):
        E = self.num_envs
        ic_q = np.asarray(self.ic["joint_pos"])[self.arm_ids]
        refs = [c.reference for c, _ in jobs]
        T = max(len(r.t) for r in refs)
        Q = np.repeat(ic_q[None, None], E, 0).repeat(T, 1)
        QD = np.zeros((E, T, 6))
        AP = np.full((E, T), self.cfg.gripper.open_aperture)
        PH = np.full((E, T), HOLD, int)
        L = np.ones(E, int)
        for e, r in enumerate(refs):
            n = len(r.t)
            Q[e, :n], QD[e, :n], AP[e, :n], PH[e, :n] = r.q, r.qd, r.aperture, r.phase
            Q[e, n:], AP[e, n:] = r.q[-1], r.aperture[-1]
            L[e] = n
        Tt = self.arm.fk_batch(Q.reshape(-1, 6))[:, 6] @ self.T_flange_tcp
        TCP = Tt[:, :3, 3].reshape(E, T, 3)
        return Q, QD, AP, PH, L, TCP

    def _grab_frames(self, cam):
        """Current RGB of every env from one camera at half the network resolution (outcome frames, videos)."""
        self.sim.render()
        c = self.cams[cam]
        c.update(0.0, force_recompute=True)
        rgb = _np(_torch(c.data.output["rgb"]))[..., :3].astype(np.uint8)
        C = self.cfg.camera
        return np.stack([_downsample_rgb(x, C.net_width // 2, C.net_height // 2) for x in rgb])

    def _rollout_batch(self, jobs, record_traj=True, video=False):
        cfg = self.cfg
        E, n_jobs = self.num_envs, len(jobs)
        k_t = self.k_target
        dims = np.asarray(cfg.layout.object_pool_dims[k_t], float)
        reals = [r for _, r in jobs] + [self.nominal_realization()] * (E - n_jobs)
        self.restore_initial_condition(self.ic, validate=False)
        self.apply_realizations(reals, k_t)
        Q, QD, AP, PH, L, TCP = self._job_arrays(jobs)
        T = Q.shape[1]
        delays = np.array([int(r.command_delay) for r in reals])
        grasp_w = np.array([c.grasp.width for c, _ in jobs] + [0.0] * (E - n_jobs))
        dt = self.control_dt
        timeout = cfg.physics.episode_timeout
        lc = cfg.labels
        n_hold = max(1, int(round(lc.settle_hold / dt)))
        n_wait = int(round(lc.settle_wait_max / dt))
        n_max = min(int(math.ceil(timeout / dt)) + 2, T + int(delays.max()) + n_wait + 2)
        mon = Monitor(cfg, E, self.arm.q_min, self.arm.q_max, dt, self.physics_dt, self.body_classes)
        obj0 = np.asarray(self.ic["objects"][k_t]["pose"])
        done = np.zeros(E, bool)
        done[n_jobs:] = True
        plan_done_tick = np.full(E, -1)
        calm = np.zeros(E, int)
        finals = [None] * E
        logs = [[] for _ in range(E)]
        ar = np.arange(E)
        tdec = max(1, cfg.collect.traj_decimation)
        cam = "fixed_camera" if "fixed_camera" in self.cams else next(iter(self.cams), None)
        vdec = max(1, int(round(1.0 / (cfg.collect.video_hz * dt))))
        vids = [[] for _ in range(E)]
        t_wall = time.time()
        for k in range(n_max):
            idx = np.clip(k - delays, 0, T - 1)
            started = k >= delays
            q_cmd, ap_cmd, ph = Q[ar, idx], AP[ar, idx], PH[ar, idx]
            qd_cmd = QD[ar, idx] * started[:, None]
            self.set_targets(q_cmd, qd_cmd, ap_cmd)
            self._physics_steps()
            t = (k + 1) * dt
            st = self.read_state(k_t)
            plan_done = (k - delays) >= (L - 1)
            newly = plan_done & (plan_done_tick < 0)
            plan_done_tick[newly] = k
            R_obj = quat_to_mat_batch(st["obj_pose"][:, 3:])
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * dims / 2.0
            bottom = (st["obj_pose"][:, None, :3] + np.einsum("eij,cj->eci", R_obj, corners))[:, :, 2].min(1)
            T_tcp = st["T_tcp"]
            rel = np.einsum("eji,ej->ei", T_tcp[:, :3, :3], st["obj_pose"][:, :3] - T_tcp[:, :3, 3])
            r = {
                "active": ~done, "phase": ph, "q": st["q"], "q_cmd": q_cmd, "qd": st["qd"],
                "obj_pos": st["obj_pose"][:, :3], "obj_vel": st["obj_vel"][:, :3],
                "link_force": st["link_force"], "pad_other_force": st["pad_other_force"], "contact_names": self.contact_names,
                "obj_obstacle_force": st["obj_obstacle_force"], "obj_grasped": st["pad_obj_force"] > cfg.monitor.grasp_min_force,
                "obj_bottom_z": bottom, "obj_env_force": st["obj_env_force"],
                "close_done": ph >= PHASE_ID["lift"], "aperture": st["aperture"], "grasp_width": grasp_w,
                "pad_obj_force": st["pad_obj_force"], "obj_in_tcp": rel, "tcp_z": T_tcp[:, 2, 3],
                "obj_disp": np.linalg.norm(st["obj_pose"][:, :2] - obj0[None, :2], axis=1),
                "tcp_pos": T_tcp[:, :3, 3], "ref_tcp_pos": TCP[ar, idx], "plan_done": plan_done,
            }
            mon.step(t, r)
            slow = (np.linalg.norm(st["obj_vel"][:, :3], axis=1) < lc.settle_lin) & (np.linalg.norm(st["obj_vel"][:, 3:], axis=1) < lc.settle_ang)
            calm = np.where(plan_done & slow, calm + 1, 0)
            if record_traj and k % tdec == 0:
                for e in np.flatnonzero(~done):
                    logs[e].append(self._traj_row(t, e, st, q_cmd, ap_cmd, ph, T_tcp))
            if video and cam and k % vdec == 0:
                fr = self._grab_frames(cam)
                for e in np.flatnonzero(~done):
                    vids[e].append((t, fr[e]))
            for e in np.flatnonzero(~done):
                aborted = mon.state[e].aborted
                settled = plan_done[e] and calm[e] >= n_hold
                waited = plan_done[e] and (k - plan_done_tick[e]) >= n_wait
                out_of_time = t >= timeout - 1e-9
                if aborted or settled or waited or out_of_time:
                    finals[e] = self._final_eval(e, st, mon, settled, t, bool(plan_done[e]), dims)
                    if record_traj:
                        logs[e].append(self._traj_row(t, e, st, q_cmd, ap_cmd, ph, T_tcp))
                    done[e] = True
            if done.all():
                break
        wall = time.time() - t_wall
        outcome = self._grab_frames(cam) if cam and cfg.collect.outcome_frames else None
        out = []
        for e, (c, rz) in enumerate(jobs):
            if finals[e] is None:  # safety net: loop ended without a decision
                finals[e] = self._final_eval(e, st, mon, False, t, bool(plan_done[e]), dims)
            summ = mon.summary(e)
            labels = assemble_labels(summ, finals[e], finals[e]["plan_completed"], lc)
            ctrl = {"type": "joint impedance (implicit PD, PhysX drive) + velocity feed-forward, 120 Hz references",
                    "stiffness": (self.nominal_stiffness * np.asarray(rz.servo_scale)).tolist(),
                    "damping": (self.nominal_damping * np.asarray(rz.servo_scale)).tolist(),
                    "command_delay_ticks": int(rz.command_delay), "gripper_stiffness": cfg.gripper.drive_stiffness,
                    "gripper_damping": cfg.gripper.drive_damping, "gripper_torque_limit": self.grip_torque,
                    "gripper_speed": cfg.gripper.speed}
            traj = _stack_log(logs[e]) if record_traj else {}
            ann = {}
            if outcome is not None:
                ann["outcome_rgb"] = outcome[e]
            if vids[e]:
                ann["video_t"] = np.asarray([v[0] for v in vids[e]], np.float32)
                ann["video_rgb"] = np.stack([v[1] for v in vids[e]])
            out.append({"cid": c.cid, "rid": rz.rid, "realization": dataclasses.asdict(rz), "controller": ctrl, "labels": labels,
                        "monitor": summ, "final": finals[e], "traj": traj, "annotation": ann, "env": e,
                        "batch_wall_s": wall, "batch_ticks": k + 1})
        return out

    def _traj_row(self, t, e, st, q_cmd, ap_cmd, ph, T_tcp):
        return {"t": t, "q": st["q"][e], "q_cmd": q_cmd[e], "qd": st["qd"][e], "aperture": st["aperture"][e],
                "aperture_cmd": ap_cmd[e], "phase": ph[e], "obj_pose": st["obj_pose"][e], "tcp_pos": T_tcp[e, :3, 3],
                "pad_obj_force": st["pad_obj_force"][e], "link_force_max": st["link_force"][e].max() if st["link_force"].size else 0.0}

    def _final_eval(self, e, st, mon, settled, t, plan_completed, dims):
        cfg, lc = self.cfg, self.cfg.labels
        pose = st["obj_pose"][e]
        inside, supported, det = placement_check(pose[:3], pose[3:], dims, self.spec.tray_xy, cfg.layout, lc)
        rel_z = mon.state[e].release_tcp_z
        tcp_z = float(st["T_tcp"][e, 2, 3])
        released = bool(st["pad_obj_force_max"][e] < 0.1 * cfg.monitor.grasp_min_force and st["aperture"][e] >= lc.open_aperture_min)
        withdrawn = bool(rel_z is not None and tcp_z - rel_z >= lc.withdraw_min)
        return {
            "inside": bool(inside), "supported": bool(supported), "settled": bool(settled), "released": released,
            "withdrawn": withdrawn, "within_time": bool(t <= cfg.physics.episode_timeout + 1e-9), "plan_completed": plan_completed,
            "time": float(t), "object_pose": pose.tolist(), "placement": det, "tcp_z": tcp_z, "release_tcp_z": rel_z,
            "aperture": float(st["aperture"][e]), "pad_force_max": float(st["pad_obj_force_max"][e]),
        }

    # ================================================================== dataset record
    def record_episode(self, cands, pool, results, feats, feats_oracle, feat_metas, og_id=None, views=None,
                       candidate_set="regenerated", timing=None):
        """Dataset record for one observation group. views: camera names visible to the policy in this group."""
        cfg = self.cfg
        spec = self.spec
        K = cfg.candidates.max_candidates
        F = len(FEATURE_NAMES)
        kn = np.zeros((K, cfg.candidates.knots, 15), np.float32)
        dur = np.zeros(K, np.float32)
        X = np.zeros((K, F), np.float32)
        Xo = np.zeros((K, F), np.float32)
        items = []
        by_c = {}
        for r in results:
            by_c.setdefault(r["cid"], []).append(r)
        for i, c in enumerate(cands):
            kn[i], dur[i] = c.knots, c.duration
            X[i], Xo[i] = feats[i], feats_oracle[i]
            ref = c.reference
            items.append({"cid": c.cid, "summary": c.summary(), "feature_meta": feat_metas[i], "ref_t": ref.t,
                          "ref_q": ref.q.astype(np.float32), "ref_qd": ref.qd.astype(np.float32),
                          "ref_aperture": ref.aperture.astype(np.float32), "ref_phase": ref.phase.astype(np.int8),
                          "rollouts": sorted(by_c.get(c.cid, []), key=lambda r: r["rid"])})
        ann = {k: v for k, v in self.obs["annotation"].items()}
        ann_json = {"spec": json.loads(spec.to_json()), "initial_condition": self.ic, "ic_validation": self.ic_report,
                    "settle": self.settle_info, "home": {"q": self.home_q.tolist(), **self.home_info},
                    "hidden_families": sorted(set(spec.families) & {"object_friction", "pad_friction", "servo", "command_delay", "object"}),
                    "provisional_parameters": PROVISIONAL}
        cams = list(self.cams)
        views = cams if views is None else list(views)
        hidden = [c for c in cams if c not in views]
        pol = {k: v for k, v in self.obs["policy"].items() if not any(k.startswith(c + "_") for c in hidden)}
        calib = {k: v for k, v in self.obs["calibration"].items() if k not in hidden}
        return {
            "scene_id": spec.scene_id, "spec_json": spec.to_json(), "split": spec.split, "seed": spec.seed,
            "og_id": og_id or f"{self.camera_mode}_{self.percep.source}", "camera_mode": self.camera_mode,
            "perception": self.percep.to_dict(), "pool_summary": pool, "calibration": calib, "views": views,
            "candidate_set": candidate_set, "timing": timing or {},
            "goal": {"tray_xy": list(spec.tray_xy), "tray_interior": list(cfg.layout.tray_interior),
                     "target_hint_xy": list(spec.target_hint_xy)},
            "policy": pol, "annotation_json": ann_json, "annotation_arrays": {k: np.asarray(v) for k, v in ann.items()},
            "feature_names": FEATURE_NAMES,
            "candidates": {"valid_mask": validity_mask(cands, K), "knots": kn, "duration": dur, "features": X,
                           "features_oracle": Xo, "items": items},
        }

    def close(self):
        try:
            self.sim.clear_instance()
        except Exception:
            pass


def _stable_hash(s):
    import hashlib

    return int(hashlib.sha256(str(s).encode()).hexdigest()[:8], 16)


def _stack_log(rows):
    if not rows:
        return {}
    out = {}
    for k in rows[0]:
        out[k] = np.asarray([r[k] for r in rows], dtype=np.float32 if k != "phase" else np.int8)
    return out


def _downsample_rgb(img, w, h):
    H, W = img.shape[:2]
    if (H, W) == (h, w):
        return img
    if H % h == 0 and W % w == 0:
        fy, fx = H // h, W // w
        return img.reshape(h, fy, w, fx, -1).mean((1, 3)).round().astype(np.uint8)
    ys = (np.arange(h) * H / h).astype(int)
    xs = (np.arange(w) * W / w).astype(int)
    return img[ys][:, xs]


def _downsample_depth(depth, w, h):
    """Block average over valid pixels; a block is valid when at least 3/4 of its pixels are."""
    H, W = depth.shape
    if (H, W) == (h, w):
        v = np.isfinite(depth)
        return np.where(v, depth, np.nan), v
    if H % h == 0 and W % w == 0:
        fy, fx = H // h, W // w
        blk = depth.reshape(h, fy, w, fx)
        v = np.isfinite(blk)
        cnt = v.sum((1, 3))
        s = np.where(v, blk, 0.0).sum((1, 3))
        ok = cnt >= 0.75 * fy * fx
        return np.where(ok, s / np.maximum(cnt, 1), np.nan), ok
    ys = (np.arange(h) * H / h).astype(int)
    xs = (np.arange(w) * W / w).astype(int)
    d = depth[ys][:, xs]
    return d, np.isfinite(d)
