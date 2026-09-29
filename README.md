# A0509 + Robotiq 2F-85 pick-and-place simulation (Isaac Sim 6.1 / Isaac Lab 3.0 EA)

Camera-based pick-and-place with counterfactual candidate rollouts, built to the specification
`a0509_2f85_simulation_specification.md`. The Isaac Lab calls were checked against the
`v3.0.0-EA` source. They have not been run on a GPU yet. Plan to fix small API details on the first run
(section 8 lists where to look).

---

## 1. Requirements

| Item | Version |
|---|---|
| OS / GPU | Ubuntu 22.04 or 24.04, RTX GPU |
| Isaac Sim | 6.1.0 |
| Isaac Lab | tag `v3.0.0-EA`, Python 3.12 environment |
| Robot description | `github.com/DoosanRobotics/doosan-robot2` (tested at commit `6c5f3ba`) |
| Gripper asset | `github.com/robotiq/isaacsim_assets` (tested at commit `3ddc2b9`), **needs Git LFS** |

```bash
git clone https://github.com/DoosanRobotics/doosan-robot2 ~/src/doosan-robot2
git clone https://github.com/robotiq/isaacsim_assets ~/src/isaacsim_assets
cd ~/src/isaacsim_assets && git lfs install && git lfs pull
```

Run every script with the Isaac Lab Python (activate its environment, or use `./isaaclab.sh -p <script>`).
No extra packages are needed; the Isaac Lab environment already has numpy, scipy, h5py, torch and warp.

Isaac Lab 3.0 command line: it runs headless by default. Use `--viz kit` to open the GUI, and `--device cuda:0` to choose a
GPU. `--headless` and `--enable_cameras` no longer exist; the scripts ask Isaac Lab for camera rendering themselves
whenever the camera mode is not `none`. Config values can be changed from any script with
`--set section.key=value`, for example `--set physics.dt=0.0020833 robot.servo_stiffness=[3000,3000,2000,600,600,300]`.

---

## 2. Run order

| Step | Command | What it does | Pass criterion |
|---|---|---|---|
| M0 | `python scripts/prepare_assets.py --doosan ~/src/doosan-robot2 --robotiq ~/src/isaacsim_assets` | URDF → USD, mount gripper, adapter and wrist-camera housing, write `assets/generated/asset_manifest.json` | manifest written |
| — | `python scripts/check_asset_bundle.py` | Check that all USD references resolve inside the generated bundle | no missing or external dependencies |
| M0 | `python scripts/calibrate_gripper.py` | Measure pad separation vs `finger_joint`; set TCP and home pose | aperture map monotonic, pads parallel |
| M1 | `python scripts/check_import.py` | Joints, limits, gains, masses, FK vs model, drift, self-contact, mimic coupling, camera images | all checks PASS |
| — | `python scripts/tune_servo.py --scales 0.5 1 2` | Step responses and tracking error on planned references | worst tracking error < 0.025 rad |
| M2 | `python scripts/scripted_pick.py --trials 100` | 100 scripted picks, fixed camera, default block | ≥ 90 / 100 |
| M3 | `python scripts/check_sensors.py --camera-mode both` | Depth geometry, perception error, frame timing, contact forces, forbidden-contact detection, grip force | all checks PASS |
| M4 | `python scripts/collect.py --config configs/pilot.json` | 100 scenes × ≤ 8 candidates × 3 realizations | label quality review |
| — | `python scripts/check_repeatability.py --mode same` / `--mode replay --dataset data/pilot` | Repeat and replay from saved initial conditions | label agreement |
| — | `python scripts/check_repeatability.py --mode save ...` twice, then `--mode compare` | 1/240 s vs 1/480 s | small outcome change |
| — | `python scripts/benchmark.py --num-envs-list 1 8 16 32` | Throughput per env count | pick the collection size |
| M5 | `python scripts/train_baselines.py --data data/pilot` | Analytic score and small MLP; Brier, NLL, ECE, reliability, selection, bootstrap by scene | baselines report |

Reports go to `outputs/<step>/` and the dataset folder. Add `--viz kit` to any Isaac script to watch it.

`prepare_assets.py` copies the Robotiq checkout (without `.git`) into `assets/generated/vendor/robotiq`, then builds the
arm and combined robot USDs under `assets/generated/usd`. It checks that every USD dependency is inside the generated
directory and that the authored references are relative. Generate the assets once; subsequent runs use the manifest and
do not need the Doosan or Robotiq checkouts. After `calibrate_gripper.py`, keep or copy the entire `assets/generated/`
directory, including `vendor/`, `usd/`, `models/`, and `gripper_calibration.json`. After moving it, run
`python scripts/check_asset_bundle.py --manifest /path/to/generated/asset_manifest.json`. Pass the same `--manifest`
path to other scripts when it is not at the default location. The generated directory is ignored by Git; store the
validated bundle separately if you need to share it across machines.

Offline tools (no Isaac, no GPU):

```bash
python scripts/selftest_offline.py                        # data pipeline on a kinematic stand-in + env.py on Isaac mocks
python scripts/train_baselines.py --data <dataset>       # baselines
python scripts/fit_arm_spheres.py --doosan ~/src/doosan-robot2   # refit arm collision spheres (needs trimesh, python-fcl)
```

---

## 3. Package layout

| Path | Role |
|---|---|
| `a0509pp/config.py` | All settings (dataclasses), JSON loading and `--set` overrides; list of provisional values |
| `a0509pp/kinematics.py` | A0509 FK, Jacobian, analytic + refined IK (8 branches), RNEA dynamics |
| `a0509pp/gripper_model.py` | 2F-85 aperture ↔ `finger_joint` map, TCP, tool inertia, collision spheres |
| `a0509pp/collision.py` | Sphere-vs-box clearance and self-distance model used by the planner and features |
| `a0509pp/scene_spec.py` | Episode spec, scene sampler with validity checks, realizations, base-scene split |
| `a0509pp/perception.py` | Depth → target and obstacle boxes (policy-visible), plus the labelled privileged oracle |
| `a0509pp/candidates.py` | Grasp hypotheses × IK branches × routes → up to 8 distinct plans, rejection reasons |
| `a0509pp/trajectory.py` | Time scaling under joint and TCP caps, 120 Hz references, 32-knot network input |
| `a0509pp/features.py` | Section 9 features with provenance (nominal and oracle models) |
| `a0509pp/monitor.py`, `labels.py` | Runtime monitor events and section 11 labels |
| `a0509pp/recording.py` | HDF5 writer/reader, `index.csv`, decision-group loader |
| `a0509pp/selectors.py` | Analytic and MLP selectors, calibration metrics, bootstrap by scene |
| `a0509pp/sim/asset_builder.py` | USD assembly of arm + adapter + gripper + camera housing |
| `a0509pp/sim/scene_cfg.py` | Isaac Lab scene: robot, table, tray, object/obstacle pools, cameras, contact sensors |
| `a0509pp/sim/env.py` | `PickPlaceEnv`: the section 15 API |
| `tests/fake_env.py` | Kinematic stand-in used by the offline self-test |
| `tests/mock_isaac.py` | Numpy mocks of the Isaac Lab objects: runs the Isaac-facing code of `env.py` offline (indices, shapes, sensor readout) |
| `configs/*.json` | Pilot, 1000-scene pilot, fast debug, and one distribution-shift config |

### Section 15 API (`PickPlaceEnv`)

| Method | Behaviour |
|---|---|
| `reset(spec)` | Lay out the scene in all envs, settle (≥ 0.5 s and until still), capture 4 camera frames at 30 Hz, capture and broadcast the initial condition, return observations |
| `observe()` | `policy` (RGB, depth, invalid mask, K, reported extrinsics, noisy q/qdot history, gripper readout, goal) and separate `annotation` |
| `perceive(source, views)` | Camera estimate from the chosen views, or `privileged` (labelled) |
| `propose_candidates()` | Candidates, validity, rejection reasons, pool summary |
| `evaluate_physics(cands, model)` | Features under `nominal` (perceived scene) or `oracle` (true scene) |
| `step(q_ref, qd_ref, aperture)` | One 120 Hz tick (2 physics steps); returns state and contact readings |
| `run_candidate(s)` | Rollouts with monitor and labels, batched over envs |
| `capture_initial_condition()` / `restore_initial_condition(record)` | State, targets, RNG and controller notes; restore validates poses and joints |
| `record_episode()` | Record for the HDF5 writer |

---

## 4. How the simulation is set up

1. **Robot.** The official A0509 URDF is converted with the Isaac Lab URDF converter (fixed base, self-collision on, convex
   hulls). The URDF `base_link` is renamed `a0509_base` because the gripper also has a `base_link`. The Robotiq
   `Physx_parallel_grip` configuration is referenced under `link_6` through a 10 mm flange adapter and a fixed joint. The
   gripper's own articulation root and world joint are removed, as the Robotiq guide requires. The adapter, gripper base,
   wrist-camera housing (60 g) and bracket (30 g) share one rigid body with combined mass and inertia. Gripper bodies do not
   collide with each other.
2. **Servo.** The arm uses PhysX implicit PD drives with position targets and velocity feed-forward, fed with 120 Hz
   references. Robot links ignore gravity (ideal gravity compensation). The held object still loads the arm. Gains and URDF
   effort limits are provisional.
3. **Gripper.** Commands are apertures in metres. They map to `finger_joint` through the measured table from
   `calibrate_gripper.py`. The passive joints follow through PhysX mimic constraints. The grip force comes from the
   `finger_joint` torque limit: τ = 2 · F · lever arm ≈ 4.3 N·m for 40 N, not 40 N·m. `check_sensors.py` prints the measured
   force and the torque needed for exactly 40 N. The drive is stiff (Robotiq's 50 N·m/deg, 3 N·m·s/deg), so the torque
   limit, not the stiffness, sets the force. Closing ramps at 50 mm/s to 10 mm below the estimated width; contact stops the
   fingers.
4. **TCP.** Fixed frame on the gripper base, origin at the pad pinch centre at 50 mm aperture, +z toward the object,
   +y along the closing axis. It is written into the manifest and every dataset record.
5. **Physics.** 1/240 s, TGS, 16/4 iterations, restitution 0. Friction is set per material with `average` combine:
   table/object 0.6/0.5, pads 0.9/0.7. The block uses 2 mm contact offset, 0.5 mm rest offset and speculative CCD, as in
   the Robotiq guide.
6. **Cameras.** Modes `fixed`, `wrist`, `both`. 640 × 480 renders, 70° HFOV, 0.05–2.0 m clipping. The policy gets
   320 × 240 RGB, depth (optical-axis z, float32, NaN where invalid) and a validity mask. It gets 4 frames at 30 Hz with
   timestamps, the noisy joint readout at each frame, and a q̇ estimate. Reported extrinsics include the calibration error;
   the true poses go to the annotation group. The wrist-camera housing stays mounted in every mode, so a view comparison
   changes information only. Set `robot.wrist_camera_hardware=false` for a hardware comparison.
7. **Contacts.** Sensors report every physics step. The robot sensor covers all links. Every finger link has its own
   sensor filtered against the objects, so contact with the target is allowed and anything else is not. Each block has a
   sensor filtered against the finger links and obstacles. The remainder is table/tray contact. Forbidden contact: > 1 N
   for 3 consecutive steps, or > 30 N once.
8. **Counterfactual rollouts.** All envs hold the same base scene. After settling, the state of env 0 is copied to every
   env. Each env then runs one (candidate, realization) job. Hidden physics per realization: block mass (inertia rescaled),
   block and pad friction, servo gain scale, command delay. Batches restart from the same initial condition.
9. **Monitor and labels.** Tracking error > 0.05 rad for > 0.1 s, joint limit, forbidden contact, obstacle hit, drag on
   the table, no progress (TCP stalls while the reference moves), and plan timeout abort the rollout. Task events (missed
   grasp, object lost, disturbed) are logged without aborting, so `y_exec` stays observable. `y_task` needs the block inside
   the tray footprint (all corners, 5 mm margin), on the tray floor, still for 0.5 s, released, gripper open, TCP withdrawn
   ≥ 8 cm, and completion within 20 s. Non-finite or exploding states are simulator errors and mask both labels.

---

## 5. Dataset format

`<out>/shard_XXXXX.h5` + `<out>/index.csv` (one row per rollout) + `<out>/manifest.json` (config, asset revisions, versions).

```
scenes/<scene_id>                    attrs: spec (JSON), split, seed
  obs/<og_id>                        attrs: camera_mode, views, candidate_set, perception, pool_summary, calibration, goal, timing
    policy/                          <cam>_rgb (4,240,320,3) u8, <cam>_depth f32 (NaN invalid), <cam>_depth_valid, <cam>_K,
                                     <cam>_T_world_cam (reported), <cam>_frame_t, q, qd, q_history, gripper_aperture, ...
    annotation/                      attrs: hidden_state (spec, initial condition, settle, home); true poses, true extrinsics
    candidates/                      valid_mask (8,), knots (8,32,15), duration (8,), features (8,76), features_oracle (8,76)
      c<i>/                          attrs: summary (grasp, branch, route, clearance), feature_meta (provenance)
        ref_t, ref_q, ref_qd, ref_aperture, ref_phase          full 120 Hz execution reference
        rollouts/r<k>/               attrs: realization, controller, labels, monitor (events), final
          t, q, q_cmd, qd, aperture, aperture_cmd, phase, obj_pose, tcp_pos, ...   executed trajectory (30 Hz)
          annotation/outcome_rgb, video_rgb (diagnostic subset)
```

Observation-group ids look like `fixed_camera` / `wrist_privileged` / `both_camera_shared-both`. With
`--candidates shared`, one plan set is recorded under every view (selection only). With `regenerated`, each view plans
on its own (full system). Splits are by base scene (70/10/10/10). All views and realizations of a scene stay together.

---

## 6. Randomization families

Enable with `--families ...` or in the config: `object` (size from the size pool), `object_shape` (box, upright
cylinder, upright hexagonal prism; list in `randomization.object_shapes`), `object_color`, `object_mass` (hidden),
`obstacles` (0–3 boxes),
`object_friction`, `pad_friction`, `servo` (±10 %), `command_delay` (0–2 ticks), `lighting`, `depth` (σ ≤ 3 mm,
≤ 3 % dropout), `calibration` (≤ 3 mm, 0.5°), `fixed_camera` (≤ 20 mm, 5°), `joint_noise` (σ ≤ 0.002 rad),
`camera_delay` (0–1 frame). Start clean, then add one family at a time. `configs/shift_heavy_lowfriction.json` is an
example held-out shift (0.25–0.40 kg, μ 0.15–0.30). The block pose is always sampled unless `--fixed-pose` is given.
To fix a sensor effect at one level instead of switching it off, give it a one-point range, e.g.
`--set randomization.depth_sigma=[0.002,0.002]`.

---

## 7. Provisional values and known limits

1. **Provisional:** servo gains, URDF effort limits (unverified, torque features flagged), adapter and camera masses,
   gripper torque for 40 N (calibrate in M3), camera placement, noise ranges, monitor thresholds. The list is in
   `config.PROVISIONAL` and in every dataset manifest.
2. **Time budget.** Under the specified caps (0.5 rad/s, 1 rad/s², 0.15 m/s) a full pick-and-place takes 13–19 s. The
   19 s planning limit leaves room for settling within 20 s. Most candidates therefore use the front / elbow-up / wrist+
   IK branch. Candidates differ mainly by grasp yaw, grasp height and route.
3. **Collision model.** The planner and features use conservative spheres (about 1–3 cm). Clearance features are
   estimates, not exact distances. Refit with `fit_arm_spheres.py` if you change the URDF.
4. **Restore.** Restoring writes joint, body and target states through the PhysX tensor API. Solver caches are not
   restored, so repeats are close but not bit-exact. Measure it with `check_repeatability.py`.
   `physics.enhanced_determinism=true` tightens it.
5. **Object colour randomization** edits the USD shader input. If a renderer build ignores it at run time, blocks stay red.
   Nothing else depends on it.
6. `Physx_parallel_grip` models parallel pinching only. Evaluate `Physx_compliant` before any claim about encompassing
   grasps.
7. **Not yet run on a GPU.** The Isaac calls were checked against the Isaac Lab `v3.0.0-EA` source (tag commit
   `ae37b02`) and the Robotiq `isaacsim_assets` layout of 2026-09-28, and `env.py` runs offline on numpy mocks of the
   Isaac Lab objects (`tests/mock_isaac.py`). Physics behaviour, rendering and the USD assembly are only tested by M0–M3
   on your PC, so run them in order.

---

## 8. First-run troubleshooting

| Symptom | Where to look |
|---|---|
| Converter output path or layout differs | `a0509pp/sim/asset_builder.py`: `build_combined` finds `link_6` and the joints by name and prints what it found |
| "Robotiq meshes are Git LFS pointer stubs" | run `git lfs pull` in the Robotiq repo |
| Contact sensor cannot find bodies | `scene_cfg.py`: sensor prim paths come from `prim_paths` in the manifest |
| Joint target / writer API errors | `env.py`: `set_targets`, `write_robot_state`, `_set_servo` (Isaac Lab 3.0 keyword-only writers) |
| Pad friction warning at start | `env._init_materials`; pads keep the USD-bound 0.9/0.7 |
| Camera images black or empty | the startup log should load `isaaclab.python.headless.rendering.kit` (`isaaclab.python.rendering.kit` with `--viz kit`); the scripts set `enable_cameras` before `AppLauncher` |
| Gripper does not squeeze 40 N | M3 prints the torque to use: `--set gripper.max_torque=<value>` |
| Tracking errors in M2 | `tune_servo.py`, then set `robot.servo_stiffness` / `servo_damping` |
