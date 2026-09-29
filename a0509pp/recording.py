"""HDF5 dataset writer/reader. Policy-visible data and annotation-only hidden state live in separate groups."""

from __future__ import annotations

import csv
import json
import os
import time

import h5py
import numpy as np

INDEX_FIELDS = ["file", "scene_id", "og_id", "cid", "rid", "split", "valid", "y_task", "y_exec", "task_label_mask",
                "exec_label_mask", "failure_type", "failure_time", "failure_phase", "duration", "route", "branch",
                "camera_mode", "perception_source", "simulator_error"]


def _json(x):
    return json.dumps(x, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))


def _attr(g, name, obj):
    """JSON metadata as an attribute; large payloads (HDF5 attributes are limited to 64 KiB) go to a string dataset."""
    s = obj if isinstance(obj, str) else _json(obj)
    if len(s.encode()) < 60000:
        g.attrs[name] = s
    else:
        if name + "_json" in g:
            del g[name + "_json"]
        g.create_dataset(name + "_json", data=s)


def read_attr(g, name):
    """Inverse of _attr: parsed JSON from the attribute or the '<name>_json' dataset."""
    if name in g.attrs:
        v = g.attrs[name]
    elif name + "_json" in g:
        v = g[name + "_json"][()]
    else:
        return None
    return json.loads(v.decode() if isinstance(v, bytes) else v)


def _put(g, name, arr, compress=True):
    arr = np.asarray(arr)
    if name in g:
        del g[name]
    kw = {"compression": "gzip", "compression_opts": 4} if compress and arr.size > 256 else {}
    return g.create_dataset(name, data=arr, **kw)


class DatasetWriter:
    def __init__(self, out_dir, shard_size=50, manifest=None):
        os.makedirs(out_dir, exist_ok=True)
        self.dir = out_dir
        self.shard_size = shard_size
        self.manifest = manifest or {}
        self.n_in_shard = 0
        self.shard = 0
        while os.path.exists(self._path(self.shard)):
            self.shard += 1
        self.f = None
        self.index_path = os.path.join(out_dir, "index.csv")
        new = not os.path.exists(self.index_path)
        self.index = open(self.index_path, "a", newline="")
        self.iw = csv.DictWriter(self.index, fieldnames=INDEX_FIELDS)
        if new:
            self.iw.writeheader()
        with open(os.path.join(out_dir, "manifest.json"), "w") as fm:
            json.dump(self.manifest, fm, indent=1, default=str)

    def _path(self, k):
        return os.path.join(self.dir, f"shard_{k:05d}.h5")

    def _file(self):
        if self.f is None or self.n_in_shard >= self.shard_size:
            if self.f is not None:
                self.f.close()
                self.shard += 1
            self.f = h5py.File(self._path(self.shard), "a")
            self.f.attrs["dataset_version"] = self.manifest.get("dataset_version", "")
            _attr(self.f, "manifest", self.manifest)
            self.f.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.n_in_shard = 0
        return self.f

    def write_scene(self, rec):
        """rec: dict produced by PickPlaceEnv.record_episode()."""
        f = self._file()
        sg = f.require_group(f"scenes/{rec['scene_id']}")
        _attr(sg, "spec", rec["spec_json"])
        sg.attrs["split"] = rec["split"]
        sg.attrs["seed"] = rec["seed"]
        og = sg.require_group(f"obs/{rec['og_id']}")
        og.attrs["camera_mode"] = rec["camera_mode"]
        og.attrs["views"] = _json(rec.get("views", []))
        og.attrs["candidate_set"] = rec.get("candidate_set", "regenerated")
        _attr(og, "timing", rec.get("timing", {}))
        _attr(og, "perception", rec["perception"])
        og.attrs["perception_source"] = rec["perception"].get("source", "")
        _attr(og, "pool_summary", rec["pool_summary"])
        _attr(og, "calibration", rec["calibration"])
        _attr(og, "goal", rec["goal"])
        pol = og.require_group("policy")
        for k, v in rec["policy"].items():
            _put(pol, k, v)
        ann = og.require_group("annotation")
        _attr(ann, "hidden_state", rec["annotation_json"])
        for k, v in rec.get("annotation_arrays", {}).items():
            _put(ann, k, v)
        cg = og.require_group("candidates")
        _attr(cg, "feature_names", list(rec["feature_names"]))
        for k in ("valid_mask", "knots", "duration", "features", "features_oracle"):
            if k in rec["candidates"]:
                _put(cg, k, rec["candidates"][k], compress=False)
        for c in rec["candidates"]["items"]:
            g = cg.require_group(f"c{c['cid']}")
            _attr(g, "summary", c["summary"])
            _attr(g, "feature_meta", c["feature_meta"])
            for k in ("ref_t", "ref_q", "ref_qd", "ref_aperture", "ref_phase"):
                _put(g, k, c[k])
            for ro in c.get("rollouts", []):
                rg = g.require_group(f"rollouts/r{ro['rid']}")
                for key in ("realization", "controller", "labels", "monitor", "final"):
                    _attr(rg, key, ro[key])
                for k, v in ro["traj"].items():
                    _put(rg, k, v)
                ag = rg.require_group("annotation")
                for k, v in ro.get("annotation", {}).items():
                    _put(ag, k, v)
                lab = ro["labels"]
                self.iw.writerow({
                    "file": os.path.basename(f.filename), "scene_id": rec["scene_id"], "og_id": rec["og_id"], "cid": c["cid"],
                    "rid": ro["rid"], "split": rec["split"], "valid": 1, "y_task": lab["y_task"], "y_exec": lab["y_exec"],
                    "task_label_mask": int(lab["task_label_mask"]), "exec_label_mask": int(lab["exec_label_mask"]),
                    "failure_type": lab["failure_type"], "failure_time": lab["failure_time"], "failure_phase": lab["failure_phase"],
                    "duration": c["summary"]["duration"], "route": c["summary"]["route"], "branch": c["summary"]["branch_name"],
                    "camera_mode": rec["camera_mode"], "perception_source": rec["perception"].get("source", ""),
                    "simulator_error": int(lab["simulator_error"]),
                })
        if not rec["candidates"]["items"]:
            self.iw.writerow({"file": os.path.basename(f.filename), "scene_id": rec["scene_id"], "og_id": rec["og_id"], "cid": -1,
                              "rid": -1, "split": rec["split"], "valid": 0, "y_task": -1, "y_exec": -1, "task_label_mask": 0,
                              "exec_label_mask": 0, "failure_type": "no_candidate", "camera_mode": rec["camera_mode"],
                              "perception_source": rec["perception"].get("source", "")})
        self.index.flush()
        f.flush()
        self.n_in_shard += 1

    def close(self):
        if self.f is not None:
            self.f.close()
        self.index.close()


def read_index(out_dir):
    with open(os.path.join(out_dir, "index.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("cid", "rid", "valid", "y_task", "y_exec", "task_label_mask", "exec_label_mask", "failure_phase", "simulator_error"):
            r[k] = int(r[k]) if r.get(k, "") not in ("", None) else -1
        for k in ("failure_time", "duration"):
            r[k] = float(r[k]) if r.get(k, "") not in ("", None) else -1.0
    return rows


def load_decision_groups(out_dir, split=None, feature_key="features", og_filter=None):
    """Per observation group: candidate features (K, F), valid mask (K,), knots, planned durations, and per-candidate
    lists of labels over realizations (masked labels left out) with the matching failure types.
    og_filter: keep observation groups whose id contains this string (e.g. 'fixed_camera')."""
    rows = [r for r in read_index(out_dir) if r["valid"] == 1 and (split is None or r["split"] == split)
            and (og_filter is None or og_filter in r["og_id"])]
    groups = {}
    for r in rows:
        groups.setdefault((r["file"], r["scene_id"], r["og_id"]), []).append(r)
    out = []
    for (fname, sid, ogid), rs in groups.items():
        with h5py.File(os.path.join(out_dir, fname), "r") as f:
            og = f[f"scenes/{sid}/obs/{ogid}"]
            cg = og["candidates"]
            X = cg[feature_key][()]
            mask = cg["valid_mask"][()]
            knots = cg["knots"][()]
            dur = cg["duration"][()]
            names = read_attr(cg, "feature_names")
            cset = og.attrs.get("candidate_set", "regenerated")
        K = len(mask)
        yt, ye, ft = ([[] for _ in range(K)] for _ in range(3))
        for r in sorted(rs, key=lambda r: (r["cid"], r["rid"])):
            if r["task_label_mask"]:
                yt[r["cid"]].append(r["y_task"])
                ft[r["cid"]].append(r.get("failure_type", "") or "")
            if r["exec_label_mask"]:
                ye[r["cid"]].append(r["y_exec"])
        out.append({"file": fname, "scene_id": sid, "og_id": ogid, "split": rs[0]["split"], "X": X, "mask": mask, "knots": knots,
                    "duration": dur, "feature_names": names, "y_task": yt, "y_exec": ye, "failure": ft, "candidate_set": cset})
    return out
