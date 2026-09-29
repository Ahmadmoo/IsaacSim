"""Minimal numpy depth renderer (boxes + spheres) for GPU-free tests of perception and planning."""

from __future__ import annotations

import numpy as np


def intrinsics(width, height, hfov_deg):
    fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]])


def render_depth(K, T_world_cam, width, height, boxes=(), spheres=None, max_depth=10.0):
    """Returns (depth optical-z, id map). boxes: list of (center, R, half, id); spheres: (C, r, id)."""
    v, u = np.mgrid[0:height, 0:width]
    d_cam = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1).reshape(-1, 3)
    d_w = d_cam @ T_world_cam[:3, :3].T
    o = T_world_cam[:3, 3]
    t_best = np.full(len(d_w), np.inf)
    ids = np.full(len(d_w), -1, int)
    for c, R, h, bid in boxes:
        oo = R.T @ (o - c)
        dd = d_w @ R
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-h - oo) / dd
            t2 = (h - oo) / dd
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin > 0)
        better = hit & (tmin < t_best)
        t_best[better] = tmin[better]
        ids[better] = bid
    if spheres is not None:
        C, r, sid = spheres
        for ci, ri in zip(C, r):
            oc = o - ci
            b = d_w @ oc
            a = np.einsum("ij,ij->i", d_w, d_w)
            cc = oc @ oc - ri * ri
            disc = b * b - a * cc
            ok = disc >= 0
            t = np.full(len(d_w), np.inf)
            t[ok] = (-b[ok] - np.sqrt(disc[ok])) / a[ok]
            better = ok & (t > 0) & (t < t_best)
            t_best[better] = t[better]
            ids[better] = sid
    depth = t_best.reshape(height, width)
    depth[~np.isfinite(depth) | (depth > max_depth)] = np.inf
    return depth, ids.reshape(height, width)
