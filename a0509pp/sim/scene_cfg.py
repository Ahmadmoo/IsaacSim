"""Isaac Lab scene configuration for the A0509 + 2F-85 pick-and-place cell (import after the app is launched)."""

from __future__ import annotations

import math

import numpy as np

from ..geometry import look_at_ros, mat_to_quat


def focal_length_mm(hfov_deg, horizontal_aperture=20.955):
    return horizontal_aperture / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def object_park_positions(layout, dims_list, row_y=0.25):
    """Parked pool objects rest on the floor under the table top (out of reach and out of camera view)."""
    x0 = layout.table_center[0] - layout.table_size[0] / 2.0 + 0.15
    return [(x0 + 0.12 * k, layout.table_center[1] + row_y, layout.floor_z + d[2] / 2.0 + 0.001) for k, d in enumerate(dims_list)]


def obstacle_park_positions(layout, dims_list):
    return object_park_positions(layout, dims_list, row_y=-0.25)


def fixed_camera_pose(ccfg):
    R = look_at_ros(np.array(ccfg.fixed_pos, float), np.array(ccfg.fixed_look_at, float), up=(0.0, 0.0, 1.0))
    return tuple(float(x) for x in ccfg.fixed_pos), tuple(float(x) for x in mat_to_quat(R))


def wrist_camera_pose(ccfg):
    R = look_at_ros(np.array(ccfg.wrist_pos, float), np.array(ccfg.wrist_look_at, float), up=(1.0, 0.0, 0.0))
    return tuple(float(x) for x in ccfg.wrist_pos), tuple(float(x) for x in mat_to_quat(R))


_HEX = []


def hex_prism_cfg():
    """Spawner config for an upright hexagonal prism (convex-hull collider); flats face +-x, across-flats = width."""
    if _HEX:
        return _HEX[0]
    from dataclasses import MISSING

    from isaaclab.sim.spawners.shapes.shapes import _spawn_geom_from_prim_type
    from isaaclab.sim.spawners.shapes.shapes_cfg import ShapeCfg
    from isaaclab.sim.utils import clone, get_current_stage
    from isaaclab.utils import configclass
    from pxr import Gf, UsdPhysics

    def convex_hull(path, stage=None):
        UsdPhysics.MeshCollisionAPI.Apply(stage.GetPrimAtPath(path)).CreateApproximationAttr().Set("convexHull")

    @clone
    def spawn_hex_prism(prim_path, cfg, translation=None, orientation=None, **kwargs):
        from ..geometry import shape_vertices

        stage = get_current_stage()
        v = shape_vertices("hex_prism", (cfg.across_flats, cfg.across_flats * 2.0 / math.sqrt(3.0), cfg.height))
        idx = [i for j in range(6) for i in (j, (j + 1) % 6, 6 + (j + 1) % 6, 6 + j)] + [5, 4, 3, 2, 1, 0] + list(range(6, 12))
        attributes = {"points": [Gf.Vec3f(*map(float, p)) for p in v], "faceVertexCounts": [4] * 6 + [6, 6],
                      "faceVertexIndices": idx, "doubleSided": True}
        _spawn_geom_from_prim_type(prim_path, cfg, "Mesh", attributes, translation, orientation, stage=stage,
                                   geometry_schema_func=convex_hull)
        return stage.GetPrimAtPath(prim_path)

    @configclass
    class HexPrismCfg(ShapeCfg):
        func: object = spawn_hex_prism
        across_flats: float = MISSING
        height: float = MISSING

    _HEX.append(HexPrismCfg)
    return HexPrismCfg


def _joint_pos_dict(cfg, home_q, mimic):
    d = {n: float(v) for n, v in zip(cfg.robot.arm_joint_names, home_q)}
    d[cfg.gripper.finger_joint] = 0.0
    for j in cfg.gripper.passive_joints:
        d[j] = 0.0
    return d


def build_scene_cfg(cfg, manifest, home_q, num_envs, grip_torque, camera_mode=None):
    """Returns (InteractiveSceneCfg, info dict). The scene holds, per environment: robot, table, tray parts,
    a pool of target objects (one active, the rest parked), a pool of kinematic obstacles, cameras and
    contact sensors."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg, ContactSensorCfg
    from isaaclab.sim.schemas import MassPropertiesCfg
    from isaaclab_physx.sim.schemas import (
        PhysxArticulationRootPropertiesCfg,
        PhysxCollisionPropertiesCfg,
        PhysxRigidBodyPropertiesCfg,
    )
    from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

    R, G, L, P, C = cfg.robot, cfg.gripper, cfg.layout, cfg.physics, cfg.camera
    mode = camera_mode or C.mode
    use_cam_hw = bool(R.wrist_camera_hardware)
    usd = manifest["robot_usd"] if use_cam_hw else manifest.get("robot_usd_no_camera", manifest["robot_usd"])
    paths = manifest["prim_paths"] if use_cam_hw else manifest.get("prim_paths_no_camera", manifest["prim_paths"])
    if mode in ("wrist", "both") and not use_cam_hw:
        raise ValueError("camera.mode needs the wrist camera but robot.wrist_camera_hardware is false")

    scene = InteractiveSceneCfg(num_envs=num_envs, env_spacing=cfg.env_spacing, replicate_physics=True)

    def mat(sf, df):
        return PhysxRigidBodyMaterialCfg(static_friction=sf, dynamic_friction=df, restitution=P.restitution,
                                         friction_combine_mode=P.combine_mode, restitution_combine_mode=P.combine_mode)

    # ---------------------------------------------------------------- robot
    arm = R.arm_joint_names
    actuators = {
        "arm": ImplicitActuatorCfg(
            joint_names_expr=list(arm),
            stiffness={n: float(v) for n, v in zip(arm, R.servo_stiffness)},
            damping={n: float(v) for n, v in zip(arm, R.servo_damping)},
            armature={n: float(v) for n, v in zip(arm, R.servo_armature)},
            joint_effort_limit={n: float(v) for n, v in zip(arm, R.effort_limit)},
            joint_velocity_limit={n: float(v) for n, v in zip(arm, R.v_rated)},
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[G.finger_joint],
            stiffness=float(G.drive_stiffness),
            damping=float(G.drive_damping),
            joint_effort_limit=float(grip_torque),
            joint_velocity_limit=5.0,
        ),
        "passive": ImplicitActuatorCfg(
            joint_names_expr=list(G.passive_joints),
            stiffness=0.0,
            damping=0.0,
            joint_effort_limit=100.0,
            joint_velocity_limit=20.0,
        ),
    }
    scene.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd,
            activate_contact_sensors=True,
            rigid_props=PhysxRigidBodyPropertiesCfg(disable_gravity=not R.robot_gravity, max_depenetration_velocity=1.0),
            articulation_props=PhysxArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=P.position_iterations,
                solver_velocity_iteration_count=P.velocity_iterations,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0),
                                                   joint_pos=_joint_pos_dict(cfg, home_q, manifest.get("gripper_mimic", {}))),
        actuators=actuators,
        soft_joint_pos_limit_factor=1.0,
    )

    # ---------------------------------------------------------------- fixtures (static colliders)
    # Per-env floor tile (no Nucleus asset needed); tiles abut at the env spacing.
    scene.floor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Floor",
        spawn=sim_utils.CuboidCfg(
            size=(cfg.env_spacing, cfg.env_spacing, 0.05),
            collision_props=PhysxCollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=mat(P.table_static_friction, P.table_dynamic_friction),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.30, 0.32), roughness=0.9),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(L.table_center[0], L.table_center[1], L.floor_z - 0.025)),
    )
    scene.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=tuple(L.table_size),
            collision_props=PhysxCollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=mat(P.table_static_friction, P.table_dynamic_friction),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.50, 0.44), roughness=0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(L.table_center)),
    )
    from ..collision import tray_boxes

    for b in tray_boxes(L, L.tray_center_xy, 0.0):
        setattr(scene, b.name, AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/" + b.name.capitalize(),
            spawn=sim_utils.CuboidCfg(
                size=tuple(float(x) for x in 2.0 * b.half),
                collision_props=PhysxCollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                physics_material=mat(P.table_static_friction, P.table_dynamic_friction),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.65), roughness=0.6),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(float(x) for x in b.center)),
        ))

    # ---------------------------------------------------------------- target object pool (dynamic)
    obj_names = []
    shapes = list(L.object_pool_shapes or [])
    shapes = shapes if len(shapes) == len(L.object_pool_dims) else ["box"] * len(L.object_pool_dims)
    for k, (dims, pos) in enumerate(zip(L.object_pool_dims, object_park_positions(L, L.object_pool_dims))):
        name = f"object_{k}"
        obj_names.append(name)
        common = dict(
            rigid_props=PhysxRigidBodyPropertiesCfg(
                linear_damping=P.object_linear_damping, angular_damping=0.05, max_depenetration_velocity=0.5,
                solver_position_iteration_count=P.position_iterations, solver_velocity_iteration_count=P.velocity_iterations,
            ),
            mass_props=MassPropertiesCfg(mass=float(L.target_default_mass)),
            collision_props=PhysxCollisionPropertiesCfg(contact_offset=P.object_contact_offset, rest_offset=P.object_rest_offset),
            physics_material=mat(P.object_static_friction, P.object_dynamic_friction),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.20, 0.15), roughness=0.5),
            activate_contact_sensors=True,
        )
        if shapes[k] == "cylinder":
            spawn = sim_utils.CylinderCfg(radius=float(dims[0]) / 2.0, height=float(dims[2]), axis="Z", **common)
        elif shapes[k] == "hex_prism":
            spawn = hex_prism_cfg()(across_flats=float(dims[0]), height=float(dims[2]), **common)
        else:
            spawn = sim_utils.CuboidCfg(size=tuple(float(x) for x in dims), **common)
        setattr(scene, name, RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Object_{k}",
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
        ))

    # ---------------------------------------------------------------- obstacle pool (kinematic)
    obs_names = []
    for j, (dims, pos) in enumerate(zip(L.obstacle_pool_dims, obstacle_park_positions(L, L.obstacle_pool_dims))):
        name = f"obstacle_{j}"
        obs_names.append(name)
        setattr(scene, name, RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{j}",
            spawn=sim_utils.CuboidCfg(
                size=tuple(float(x) for x in dims),
                rigid_props=PhysxRigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=PhysxCollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                physics_material=mat(P.table_static_friction, P.table_dynamic_friction),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.48), roughness=0.7),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
        ))

    # ---------------------------------------------------------------- lights (global)
    scene.dome_light = AssetBaseCfg(prim_path="/World/DomeLight",
                                    spawn=sim_utils.DomeLightCfg(intensity=1200.0, color=(0.95, 0.95, 0.95)))
    scene.key_light = AssetBaseCfg(prim_path="/World/KeyLight",
                                   spawn=sim_utils.DistantLightCfg(intensity=2500.0, angle=1.0, color=(1.0, 0.98, 0.95)),
                                   init_state=AssetBaseCfg.InitialStateCfg(rot=(0.2706, 0.2706, 0.0, 0.9239)))

    # ---------------------------------------------------------------- cameras
    f_mm = focal_length_mm(C.hfov_deg)
    pin = dict(focal_length=f_mm, horizontal_aperture=20.955, clipping_range=tuple(C.clip))
    cams = []
    if mode in ("fixed", "both"):
        pos, rot = fixed_camera_pose(C)
        scene.fixed_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/FixedCamera",
            update_period=0.0,
            width=C.width, height=C.height,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(**pin),
            offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="ros"),
        )
        cams.append("fixed_camera")
    if mode in ("wrist", "both"):
        pos, rot = wrist_camera_pose(C)
        scene.wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + paths["gripper_base"] + "/wrist_camera",
            update_period=0.0,
            width=C.width, height=C.height,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(**pin),
            offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="ros"),
        )
        cams.append("wrist_camera")

    # ---------------------------------------------------------------- contact sensors
    H = P.control_decimation
    scene.robot_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0.0, history_length=H)
    obj_filters = [f"{{ENV_REGEX_NS}}/Object_{k}" for k in range(len(obj_names))]
    pad_sensor_names = []
    for link in G.contact_links:
        name = f"contact_{link}"
        pad_sensor_names.append(name)
        setattr(scene, name, ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + paths["gripper_bodies"][link],
            update_period=0.0, history_length=H,
            filter_prim_paths_expr=list(obj_filters),
            track_friction_forces=True,
            max_contact_data_count_per_prim=16,
        ))
    obj_sensor_names = []
    obj_partner = ["{ENV_REGEX_NS}/Robot/" + paths["gripper_bodies"][l] for l in G.contact_links]
    obj_partner += [f"{{ENV_REGEX_NS}}/Obstacle_{j}" for j in range(len(obs_names))]
    for k in range(len(obj_names)):
        name = f"object_{k}_contacts"
        obj_sensor_names.append(name)
        setattr(scene, name, ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Object_{k}",
            update_period=0.0, history_length=H,
            filter_prim_paths_expr=list(obj_partner),
        ))

    info = {
        "objects": obj_names, "obstacles": obs_names, "cameras": cams, "pad_sensors": pad_sensor_names,
        "object_sensors": obj_sensor_names, "object_partners": list(G.contact_links) + [f"obstacle_{j}" for j in range(len(obs_names))],
        "robot_usd": usd, "prim_paths": paths, "camera_mode": mode, "focal_length_mm": f_mm,
    }
    return scene, info
