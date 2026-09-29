"""GPU-free end-to-end test: scene sampling, mock cameras, perception, planning, features, the real control / monitor /
label loop on a kinematic stand-in for Isaac (tests/fake_env.py), view-subset records with shared candidates, HDF5
writing and reading, replay from a record, the baselines, and the Isaac-facing code of a0509pp/sim/env.py on numpy
mocks of the Isaac Lab objects (tests/mock_isaac.py: sensor indexing, writers, contact naming, records).

    python scripts/selftest_offline.py
"""

import argparse
import os
import shutil
import sys
import time
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from a0509pp.config import load_cfg  # noqa: E402
from a0509pp.features import FEATURE_NAMES  # noqa: E402
from a0509pp.recording import DatasetWriter, load_decision_groups, read_attr, read_index  # noqa: E402
from a0509pp.scene_spec import EpisodeSpec, Realization, make_object_pool  # noqa: E402
from a0509pp.selectors import AnalyticSelector, MLPSelector, evaluate, rollout_arrays  # noqa: E402
from a0509pp.trajectory import Reference  # noqa: E402
from tests.fake_env import FakeEnv  # noqa: E402
from tests.mock_isaac import build_mock_env  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scenes", type=int, default=3)
parser.add_argument("--envs", type=int, default=4)
parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "selftest_data"))
args = parser.parse_args()

cfg = load_cfg(None, ["camera.width=320", "camera.height=240", "camera.net_width=160", "camera.net_height=120",
                      "camera.history=2", "collect.realizations=2", "collect.traj_decimation=8"])
cfg.randomization.enabled = ["object", "object_color", "object_mass", "obstacles", "object_friction", "pad_friction", "depth", "joint_noise", "camera_delay"]
cfg.layout.object_pool_dims = make_object_pool(cfg, k=4, seed=0)
cfg.collect.split = [0.4, 0.2, 0.2, 0.2]
shutil.rmtree(args.out, ignore_errors=True)
env = FakeEnv(cfg, num_envs=args.envs, camera_mode="both")
writer = DatasetWriter(args.out, shard_size=4, manifest={"dataset_version": "selftest", "feature_names": FEATURE_NAMES})
views = {"fixed": ["fixed_camera"], "wrist": ["wrist_camera"], "both": ["fixed_camera", "wrist_camera"]}
checks = []
t0 = time.time()
n_roll = n_task = n_exec = 0
for i in range(args.scenes):
    spec = env.sample_spec(i, 3, cfg.randomization.enabled, False, cfg.collect.realizations)
    env.reset(spec)
    percep = env.perceive("camera", views["both"])
    if percep.target is not None:
        checks.append(("perception within 1 cm", np.linalg.norm(np.asarray(percep.target.center[:2]) - np.asarray(spec.target.xy)) < 0.01))
    cands, pool = env.propose_candidates(percep)
    feats, _, metas = env.evaluate_physics(cands, "nominal")
    feats_o, _, _ = env.evaluate_physics(cands, "oracle")
    res = env.run_candidates(cands, spec.realizations, video=(i == 0))
    for v, cams in views.items():
        writer.write_scene(env.record_episode(cands, pool, res, feats, feats_o, metas, f"{v}_camera_shared-both", views=cams,
                                              candidate_set="shared:both", timing={"planning_s": pool["plan_wall_s"]}))
    n_roll += len(res)
    n_task += sum(r["labels"]["y_task"] for r in res)
    n_exec += sum(r["labels"]["y_exec"] for r in res)
    fails = sorted({r["labels"]["failure_type"] for r in res if r["labels"]["failure_type"]})
    print(f"{spec.scene_id} {spec.split:5s}: {len(cands)} candidates, {len(res)} rollouts, task {sum(r['labels']['y_task'] for r in res)}, "
          f"exec {sum(r['labels']['y_exec'] for r in res)} {fails}")
writer.close()
checks.append(("some rollouts ran", n_roll > 0))
checks.append(("some task successes", n_task > 0))

rows = read_index(args.out)
checks.append(("index rows = rollouts x views", sum(r["valid"] for r in rows) == 3 * n_roll))
groups = load_decision_groups(args.out, og_filter="fixed_camera")
checks.append(("decision groups load", len(groups) > 0 and groups[0]["X"].shape[1] == len(FEATURE_NAMES)))

r0 = next(r for r in rows if r["valid"] == 1 and r["og_id"].startswith("fixed"))
with h5py.File(os.path.join(args.out, r0["file"]), "r") as f:
    sg = f[f"scenes/{r0['scene_id']}"]
    g = sg[f"obs/{r0['og_id']}"]
    pol = list(g["policy"])
    checks.append(("fixed-view record hides the wrist stream", not any(k.startswith("wrist_camera") for k in pol)
                   and "fixed_camera_depth" in pol and "qd" in pol and "q_history" in pol))
    checks.append(("depth stored lossless (float32)", g["policy"]["fixed_camera_depth"].dtype == np.float32))
    checks.append(("outcome frame stored", "outcome_rgb" in g["candidates"][f"c{r0['cid']}"]["rollouts"][f"r{r0['rid']}"]["annotation"]))
    cg = g["candidates"][f"c{r0['cid']}"]
    rg = cg["rollouts"][f"r{r0['rid']}"]
    spec = EpisodeSpec.from_json(read_attr(sg, "spec"))
    ic = read_attr(g["annotation"], "hidden_state")["initial_condition"]
    checks.append(("initial condition carries RNG and controller state", "rng" in ic and "controller" in ic))
    summ = read_attr(cg, "summary")
    ref = Reference(t=cg["ref_t"][()], q=cg["ref_q"][()].astype(float), qd=cg["ref_qd"][()].astype(float),
                    aperture=cg["ref_aperture"][()].astype(float), phase=cg["ref_phase"][()].astype(int))
    rz, stored = Realization(**read_attr(rg, "realization")), read_attr(rg, "labels")
env.reset(spec)
env.restore_initial_condition(ic)
env.ic = ic
cand = types.SimpleNamespace(cid=summ["cid"], reference=ref, duration=ref.duration, grasp=types.SimpleNamespace(width=summ["grasp"]["width"]))
a = env.run_candidates([cand], [rz])[0]
b = env.run_candidates([cand], [rz])[0]
checks.append(("repeated rollouts from the initial condition agree", a["labels"] == b["labels"] and np.allclose(a["traj"]["obj_pose"], b["traj"]["obj_pose"])))
checks.append(("replay from the HDF5 record reproduces the labels", a["labels"]["y_task"] == stored["y_task"] and a["labels"]["y_exec"] == stored["y_exec"]))

X, y, _ = rollout_arrays(groups)
if len(np.unique(y)) > 1:
    for m in (AnalyticSelector().fit(X, y, val_groups=groups), MLPSelector(epochs=30).fit(X, y)):
        ev = evaluate(m, groups)
        print(f"{m.name}: Brier {ev['brier']:.3f}, NLL {ev['nll']:.3f}, chosen {ev.get('chosen_success', float('nan')):.3f}, "
              f"oracle {ev.get('oracle_success', float('nan')):.3f}, coverage {ev.get('coverage', float('nan')):.2f}")
        checks.append((f"{m.name} baseline trains", np.isfinite(ev["brier"])))

# The Isaac-facing half of a0509pp/sim/env.py on numpy mocks of the Isaac Lab objects (plumbing, not physics).
menv = build_mock_env(cfg, num_envs=3, camera_mode="both")
spec_m = menv.sample_spec(0, 5, cfg.randomization.enabled + ["servo", "command_delay"], False, 2)
menv.reset(spec_m)
checks.append(("mock Isaac: initial condition broadcast validates", menv.ic_report["ok"]))
k, n_pad, D = menv.k_target, len(menv.pad_cs), lambda name: menv.scene[name].data
b3, i_l = menv.cs_names.index("link_3"), cfg.gripper.contact_links.index("left_fingertip")
D("robot_contacts").net_normal_forces_w_history._a[1, 0, b3] = [0, 0, 5]
D("contact_left_fingertip").net_normal_forces_w_history._a[0, 0, 0] = [0, 0, 2]
for side, f in (("left", [3, 0, 0]), ("right", [-4, 0, 0])):
    D(f"contact_{side}_fingertip").net_normal_forces_w_history._a[2, 0, 0] = f
    D(f"contact_{side}_fingertip").normal_force_matrix_w_history._a[2, 0, 0, k] = f
D("contact_right_fingertip").friction_force_matrix_w_history._a[2, 0, 0, k] = [0, 0, 1.5]
D(f"object_{k}_contacts").normal_force_matrix_w_history._a[1, 0, 0, n_pad + 1] = [0, 1, 0]
D(f"object_{k}_contacts").net_normal_forces_w_history._a[1, 0, 0] = [0, 1, 7]
st = menv.read_state(k)
H = st["link_force"].shape[1]
checks.append(("mock Isaac: contact sensor readout indexing", st["link_force"][1, H - 1, b3] == 5 and st["link_force"].sum() == 5
               and st["pad_other_force"][0, H - 1, i_l] == 2 and st["pad_other_force"].sum() == 2
               and np.allclose(st["pad_obj_force"], [0, 0, 3]) and np.allclose(st["pad_obj_force_max"], [0, 0, 4])
               and np.allclose(st["pad_obj_friction"], [0, 0, 1.5]) and st["obj_obstacle_force"][1, H - 1, 1] == 1
               and st["obj_obstacle_force"].sum() == 1 and np.isclose(st["obj_env_force"][1, H - 1], 7) and st["obj_env_force"].sum() == 7))
D("robot_contacts").net_normal_forces_w_history._a[1, :, b3] = [0, 0, 5]  # sustained contacts fill every history sample
D("contact_left_fingertip").net_normal_forces_w_history._a[0, :, 0] = [0, 0, 2]
cands_m, pool_m = menv.propose_candidates(menv.perceive("privileged"))
f_m, _, meta_m = menv.evaluate_physics(cands_m)
res_m = menv.run_candidates(cands_m[:3], spec_m.realizations[:1], video=True)
evs = [r["monitor"]["events"][0] if r["monitor"]["events"] else {} for r in res_m]
checks.append(("mock Isaac: forbidden contacts named from the sensors", evs[0].get("kind") == "forbidden_contact" and "left_fingertip" in evs[0]["detail"]
               and evs[1].get("kind") == "forbidden_contact" and "link_3" in evs[1]["detail"] and evs[2].get("kind") != "forbidden_contact"))
rz_m = res_m[-1]["realization"]
checks.append(("mock Isaac: hidden physics written per env", np.isclose(menv.objects[k].mass[2, 0], rz_m["object_mass"])
               and np.isclose(menv.objects[k].root_view.mats[2, 0, 0], rz_m["object_static_friction"])
               and np.isclose(menv.robot.root_view.mats[2, menv._pad_shape_idx[0], 1], rz_m["pad_dynamic_friction"])
               and np.allclose(menv.robot.stiff[2, menv.arm_ids], menv.nominal_stiffness * np.asarray(rz_m["servo_scale"]))))
rec_m = menv.record_episode(cands_m, pool_m, res_m, f_m, f_m, meta_m, "fixed_privileged", views=["fixed_camera"])
checks.append(("mock Isaac: record hides the wrist stream, keeps frames", not any(n.startswith("wrist") for n in rec_m["policy"])
               and rec_m["policy"]["fixed_camera_rgb"].shape == (cfg.camera.history, cfg.camera.net_height, cfg.camera.net_width, 3)
               and "outcome_rgb" in res_m[0]["annotation"] and "video_rgb" in res_m[0]["annotation"]))

print(f"\n{n_roll} rollouts, task rate {n_task / max(n_roll, 1):.2f}, exec rate {n_exec / max(n_roll, 1):.2f}, {time.time() - t0:.0f} s")
for name, ok in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
sys.exit(0 if all(ok for _, ok in checks) else 1)
