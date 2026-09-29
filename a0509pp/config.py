"""Task configuration. Units: m, s, kg, rad, N, N*m. Quaternions (x, y, z, w).

Values marked in PROVISIONAL are experimental starting points from the brief, not identified hardware data.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from dataclasses import dataclass, field

DATASET_VERSION = "a0509pp-v1.0"

PROVISIONAL = [
    "robot.servo_stiffness", "robot.servo_damping", "robot.effort_limit",
    "robot.adapter_thickness", "robot.adapter_mass", "robot.wrist_camera_mount", "robot.wrist_camera_mass",
    "gripper.drive_stiffness", "gripper.drive_damping", "gripper.max_torque", "gripper.pad_*",
    "physics.*", "camera.*", "randomization.*", "monitor.*",
]


@dataclass
class RobotCfg:
    arm_joint_names: list = field(default_factory=lambda: [f"joint_{i}" for i in range(1, 7)])
    # URDF base link is renamed by prepare_assets to avoid clashing with the gripper's base_link.
    arm_base_link: str = "a0509_base"
    flange_link: str = "link_6"
    q_min: list = field(default_factory=lambda: [-6.2832, -6.2832, -2.7925, -6.2832, -6.2832, -6.2832])
    q_max: list = field(default_factory=lambda: [6.2832, 6.2832, 2.7925, 6.2832, 6.2832, 6.2832])
    v_rated: list = field(default_factory=lambda: [math.pi, math.pi, math.pi, 2 * math.pi, 2 * math.pi, 2 * math.pi])
    # URDF <limit effort>; unverified as motor limits. Torque features are flagged provisional.
    effort_limit: list = field(default_factory=lambda: [194.0, 194.0, 163.0, 50.0, 50.0, 50.0])
    effort_limit_verified: bool = False
    # Operating caps used for time scaling.
    joint_speed_cap: float = 0.5
    joint_accel_cap: float = 1.0
    tcp_lin_speed_cap: float = 0.15
    tcp_ang_speed_cap: float = 0.5
    # Implicit PD servo (position + velocity feedforward), N*m/rad and N*m*s/rad.
    servo_stiffness: list = field(default_factory=lambda: [4000.0, 4000.0, 2500.0, 800.0, 800.0, 400.0])
    servo_damping: list = field(default_factory=lambda: [250.0, 250.0, 150.0, 40.0, 40.0, 20.0])
    servo_armature: list = field(default_factory=lambda: [0.1, 0.1, 0.05, 0.02, 0.02, 0.01])
    # Robot links ignore gravity: ideal gravity compensation of the simulated servo. Payload still loads the arm.
    robot_gravity: bool = False
    # Flange adapter, cylinder coaxial with the flange.
    adapter_thickness: float = 0.010
    adapter_radius: float = 0.0375
    adapter_mass: float = 0.060
    # Gripper rotation about the flange z axis.
    gripper_yaw_on_flange: float = 0.0
    # Wrist camera housing + bracket, rigidly attached to the gripper base.
    wrist_camera_hardware: bool = True
    wrist_camera_mass: float = 0.060
    wrist_bracket_mass: float = 0.030
    wrist_camera_housing: list = field(default_factory=lambda: [0.023, 0.042, 0.042])
    # Home posture; None -> solved by IK for home_tcp_pos with the approach axis down.
    home_q: list | None = None
    home_tcp_pos: list = field(default_factory=lambda: [0.40, 0.0, 0.35])


@dataclass
class GripperCfg:
    finger_joint: str = "finger_joint"
    passive_joints: list = field(default_factory=lambda: [
        "right_outer_knuckle_joint", "left_inner_finger_joint", "right_inner_finger_joint",
        "left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint"])
    finger_links: list = field(default_factory=lambda: [
        "left_outer_knuckle", "right_outer_knuckle", "left_outer_finger", "right_outer_finger",
        "left_inner_finger", "right_inner_finger", "left_inner_knuckle", "right_inner_knuckle",
        "left_fingertip", "right_fingertip"])
    pad_links: list = field(default_factory=lambda: ["left_fingertip", "right_fingertip"])
    # Finger bodies with target-object contact filtering: contact with the target is allowed, anything else is not.
    contact_links: list = field(default_factory=lambda: [
        "left_outer_knuckle", "right_outer_knuckle", "left_outer_finger", "right_outer_finger",
        "left_inner_finger", "right_inner_finger", "left_inner_knuckle", "right_inner_knuckle",
        "left_fingertip", "right_fingertip"])
    base_link: str = "base_link"
    stroke: float = 0.085
    open_aperture: float = 0.085
    joint_upper: float = math.radians(47.0)
    # Linkage geometry from Robotiq payloads (parallel_grip variant), gripper base frame.
    knuckle_pivot: list = field(default_factory=lambda: [0.0, 0.0306, 0.05466])
    finger_pivot: list = field(default_factory=lambda: [0.0, 0.06776, 0.09809])
    # Pad geometry at finger_joint = 0; overwritten by calibrate_gripper.py.
    pad_inner_y_open: float = 0.0425
    pad_center_z_open: float = 0.13372
    pad_half_height: float = 0.011
    pad_half_width: float = 0.011
    pad_thickness: float = 0.016
    # TCP: fixed frame, origin at the pinch centre at this aperture.
    tcp_reference_aperture: float = 0.050
    speed: float = 0.050
    grip_force: float = 40.0
    # finger_joint drive in N*m/rad: Robotiq's recommended 50 N*m/deg and 3 N*m*s/deg (isaacsim_assets guide, 3.1). The
    # force limit (max_torque) sets the grip force; a stiff drive keeps it saturated while squeezing.
    drive_stiffness: float = 2864.8
    drive_damping: float = 171.9
    # Joint torque limit; None -> derived from grip_force and the mean lever arm over 30-60 mm.
    max_torque: float | None = None
    close_margin: float = 0.010
    settle_time: float = 0.30
    calibration_file: str = ""


@dataclass
class LayoutCfg:
    table_size: list = field(default_factory=lambda: [1.20, 0.90, 0.05])
    table_center: list = field(default_factory=lambda: [0.45, 0.0, -0.025])
    floor_z: float = -0.75
    target_default_dims: list = field(default_factory=lambda: [0.040, 0.040, 0.050])
    target_default_mass: float = 0.10
    target_default_xy: list = field(default_factory=lambda: [0.45, -0.15])
    target_x_range: list = field(default_factory=lambda: [0.35, 0.60])
    target_y_range: list = field(default_factory=lambda: [-0.25, 0.05])
    # Object pool: one prim per entry per env. Episode selects the active one; others are parked.
    object_pool_dims: list = field(default_factory=lambda: [[0.040, 0.040, 0.050]])
    tray_interior: list = field(default_factory=lambda: [0.22, 0.18])
    tray_rim_height: float = 0.03
    tray_wall: float = 0.005
    tray_floor: float = 0.005
    tray_center_xy: list = field(default_factory=lambda: [0.45, 0.23])
    obstacle_pool_dims: list = field(default_factory=lambda: [
        [0.05, 0.05, 0.10], [0.08, 0.06, 0.15], [0.12, 0.05, 0.20],
        [0.06, 0.10, 0.08], [0.10, 0.10, 0.05], [0.07, 0.12, 0.12]])
    num_obstacles: list = field(default_factory=lambda: [0, 3])
    obstacle_x_range: list = field(default_factory=lambda: [0.30, 0.70])
    obstacle_y_range: list = field(default_factory=lambda: [-0.30, 0.20])
    robot_keepout_radius: float = 0.20
    spawn_gap: float = 0.03
    park_xy: list = field(default_factory=lambda: [-0.60, 0.0])


@dataclass
class PhysicsCfg:
    dt: float = 1.0 / 240.0
    control_decimation: int = 2
    camera_decimation: int = 8
    gravity: list = field(default_factory=lambda: [0.0, 0.0, -9.81])
    solver_type: int = 1
    position_iterations: int = 16
    velocity_iterations: int = 4
    restitution: float = 0.0
    table_static_friction: float = 0.6
    table_dynamic_friction: float = 0.5
    object_static_friction: float = 0.6
    object_dynamic_friction: float = 0.5
    pad_static_friction: float = 0.9
    pad_dynamic_friction: float = 0.7
    combine_mode: str = "average"
    object_contact_offset: float = 0.002
    object_rest_offset: float = 0.0005
    object_linear_damping: float = 0.1
    object_speculative_ccd: bool = True
    enable_stabilization: bool = False
    enhanced_determinism: bool = False
    default_static_friction: float = 0.5
    default_dynamic_friction: float = 0.5
    episode_timeout: float = 20.0
    settle_min: float = 0.5
    settle_max: float = 3.0
    settle_lin_speed: float = 0.005
    settle_ang_speed: float = 0.05
    settle_hold: float = 0.1


@dataclass
class CameraCfg:
    mode: str = "fixed"
    width: int = 640
    height: int = 480
    net_width: int = 320
    net_height: int = 240
    hfov_deg: float = 70.0
    clip: list = field(default_factory=lambda: [0.05, 2.0])
    fixed_pos: list = field(default_factory=lambda: [0.55, -0.80, 0.85])
    fixed_look_at: list = field(default_factory=lambda: [0.45, 0.0, 0.10])
    # Wrist camera optical centre in the gripper base frame and the point it looks at.
    wrist_pos: list = field(default_factory=lambda: [0.060, 0.0, 0.070])
    wrist_look_at: list = field(default_factory=lambda: [0.0, 0.0, 0.25])
    history: int = 4
    depth_kind: str = "optical_axis_z"
    warmup_renders: int = 12
    rerenders_on_reset: int = 3


@dataclass
class PerceptionCfg:
    mode: str = "camera"
    voxel: float = 0.004
    table_band: float = 0.010
    min_height: float = 0.015
    min_cluster_points: int = 60
    robot_margin: float = 0.015
    tray_margin: float = 0.008
    top_band: float = 0.006
    target_hint_sigma: float = 0.010
    planning_buffer: float = 0.010
    measurement_allowance: float = 0.005
    nominal_object_mass: float = 0.15


@dataclass
class CandidateCfg:
    max_candidates: int = 8
    pool_limit: int = 64
    knots: int = 32
    pregrasp_height: float = 0.10
    lift_height_min: float = 0.10
    obstacle_clearance: float = 0.06
    withdraw_height: float = 0.10
    place_drop_gap: float = 0.008
    grasp_height_options: list = field(default_factory=lambda: ["mid", "high"])
    min_pad_table_clearance: float = 0.004
    min_open_clearance: float = 0.005
    routes: list = field(default_factory=lambda: ["direct", "high", "left", "right"])
    route_high_extra: float = 0.12
    route_lateral_offset: float = 0.16
    cartesian_step: float = 0.005
    joint_step: float = 0.02
    ik_tol_pos: float = 1e-5
    ik_tol_rot: float = 1e-5
    max_joint_jump: float = 0.15
    max_duration: float = 19.0
    min_clearance_env: float = 0.0
    min_clearance_self: float = 0.0
    dedupe_joint_tol: float = 0.05
    wrist_sym_dedupe: bool = True


@dataclass
class FeatureCfg:
    task_scale: list = field(default_factory=lambda: [0.15, 0.15, 0.15, 0.5, 0.5, 0.5])
    sample_dt: float = 1.0 / 30.0
    refine_clearance: float = 0.03
    refine_factor: int = 4
    capability_samples: int = 24
    capability_dt: float = 1.0 / 120.0


@dataclass
class MonitorCfg:
    tracking_error: float = 0.05
    tracking_time: float = 0.10
    contact_force: float = 1.0
    contact_steps: int = 3
    impulse_force: float = 30.0
    joint_limit_margin: float = 1e-3
    object_loss_dist: float = 0.03
    object_loss_time: float = 0.10
    grasp_min_force: float = 0.5
    missed_grasp_margin: float = 0.008
    no_progress_time: float = 2.0
    abort_on_task_failure: bool = False


@dataclass
class LabelCfg:
    footprint_margin: float = 0.005
    support_tol: float = 0.006
    settle_lin: float = 0.02
    settle_ang: float = 0.10
    settle_hold: float = 0.5
    settle_wait_max: float = 2.0
    withdraw_min: float = 0.08
    open_aperture_min: float = 0.06


@dataclass
class RandomizationCfg:
    enabled: list = field(default_factory=list)
    object_width: list = field(default_factory=lambda: [0.030, 0.060])
    object_height: list = field(default_factory=lambda: [0.030, 0.080])
    object_mass: list = field(default_factory=lambda: [0.05, 0.25])
    object_static_friction: list = field(default_factory=lambda: [0.3, 0.9])
    pad_static_friction: list = field(default_factory=lambda: [0.5, 1.1])
    dynamic_ratio: float = 0.8
    friction_quantum: float = 0.01
    light_scale: list = field(default_factory=lambda: [0.7, 1.3])
    depth_sigma: list = field(default_factory=lambda: [0.0, 0.003])
    depth_dropout: list = field(default_factory=lambda: [0.0, 0.03])
    calib_trans: float = 0.003
    calib_rot_deg: float = 0.5
    fixed_cam_trans: float = 0.020
    fixed_cam_rot_deg: float = 5.0
    joint_noise: list = field(default_factory=lambda: [0.0, 0.002])
    command_delay: list = field(default_factory=lambda: [0, 2])
    camera_delay: list = field(default_factory=lambda: [0, 1])
    servo_scale: list = field(default_factory=lambda: [0.9, 1.1])


@dataclass
class CollectCfg:
    num_scenes: int = 100
    realizations: int = 3
    seed: int = 0
    out_dir: str = "data/pilot"
    split: list = field(default_factory=lambda: [0.7, 0.1, 0.1, 0.1])
    video_fraction: float = 0.02
    save_diag_frames: bool = True
    traj_decimation: int = 4
    outcome_frames: bool = True
    video_hz: float = 5.0
    camera_modes: list = field(default_factory=lambda: ["fixed"])
    perception_sources: list = field(default_factory=lambda: ["camera"])


@dataclass
class TaskCfg:
    robot: RobotCfg = field(default_factory=RobotCfg)
    gripper: GripperCfg = field(default_factory=GripperCfg)
    layout: LayoutCfg = field(default_factory=LayoutCfg)
    physics: PhysicsCfg = field(default_factory=PhysicsCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    perception: PerceptionCfg = field(default_factory=PerceptionCfg)
    candidates: CandidateCfg = field(default_factory=CandidateCfg)
    features: FeatureCfg = field(default_factory=FeatureCfg)
    monitor: MonitorCfg = field(default_factory=MonitorCfg)
    labels: LabelCfg = field(default_factory=LabelCfg)
    randomization: RandomizationCfg = field(default_factory=RandomizationCfg)
    collect: CollectCfg = field(default_factory=CollectCfg)
    asset_manifest: str = "assets/generated/asset_manifest.json"
    num_envs: int = 16
    env_spacing: float = 3.0
    device: str = "cuda:0"

    @property
    def control_dt(self):
        return self.physics.dt * self.physics.control_decimation

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), indent=1, sort_keys=True)


def _merge(obj, d):
    for k, v in d.items():
        if not hasattr(obj, k):
            raise KeyError(f"unknown config key: {type(obj).__name__}.{k}")
        cur = getattr(obj, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(obj, k, v)
    return obj


def load_cfg(path: str | None = None, overrides: list[str] | None = None) -> TaskCfg:
    """Load defaults, then a JSON file, then 'section.key=value' overrides (value parsed as JSON)."""
    cfg = TaskCfg()
    if path:
        with open(path) as f:
            _merge(cfg, json.load(f))
    for item in overrides or []:
        key, val = item.split("=", 1)
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            pass
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = getattr(node, p)
        if not hasattr(node, parts[-1]):
            raise KeyError(f"unknown config key: {key}")
        setattr(node, parts[-1], val)
    return cfg


def clone_cfg(cfg: TaskCfg) -> TaskCfg:
    return copy.deepcopy(cfg)
