"""Outcome definitions and label assembly (section 11)."""

from __future__ import annotations

import numpy as np

from .geometry import quat_to_mat, solid_corners
from .monitor import EXEC_FAILURES, SIM_ERRORS, TASK_FAILURES


def placement_check(obj_pos, obj_quat, dims, tray_xy, layout, lcfg, shape="box"):
    """Footprint inside the tray interior with margin (true geometry, all hull vertices) and supported by the floor."""
    R = quat_to_mat(obj_quat)
    corners = solid_corners(np.asarray(obj_pos), R, dims, shape)
    ix, iy = layout.tray_interior
    m = lcfg.footprint_margin
    inside = bool(np.all(np.abs(corners[:, 0] - tray_xy[0]) <= ix / 2.0 - m) and np.all(np.abs(corners[:, 1] - tray_xy[1]) <= iy / 2.0 - m))
    floor_top = layout.tray_floor
    zmin = float(corners[:, 2].min())
    supported = abs(zmin - floor_top) <= lcfg.support_tol
    return inside, supported, {"corner_margin_x": float(ix / 2.0 - np.abs(corners[:, 0] - tray_xy[0]).max()),
                               "corner_margin_y": float(iy / 2.0 - np.abs(corners[:, 1] - tray_xy[1]).max()),
                               "bottom_z": zmin}


def assemble_labels(mon_summary, final, plan_completed, lcfg, abort_on_task_failure=False):
    """final: dict from the environment's end-of-rollout evaluation (see env.run_candidates)."""
    ev = mon_summary["events"]
    kinds = [e["kind"] for e in ev]
    sim_error = mon_summary["sim_error"]
    exec_viol = any(k in EXEC_FAILURES for k in kinds)
    task_fail_events = [k for k in kinds if k in TASK_FAILURES and k != "object_disturbed"]
    success = (
        plan_completed and not exec_viol and not sim_error and not task_fail_events
        and final.get("inside", False) and final.get("supported", False) and final.get("settled", False)
        and final.get("released", False) and final.get("withdrawn", False) and final.get("within_time", False)
    )
    failure_type = ""
    if not success:
        first = next((e for e in ev if e["kind"] in EXEC_FAILURES | TASK_FAILURES | SIM_ERRORS and e["kind"] != "object_disturbed"), None)
        if first is not None:
            failure_type = first["kind"]
        elif not plan_completed:
            failure_type = "incomplete"
        else:
            for key, name in (("settled", "not_settled"), ("inside", "placement_outside"), ("supported", "placement_outside"),
                              ("released", "not_released"), ("withdrawn", "not_withdrawn"), ("within_time", "timeout")):
                if not final.get(key, False):
                    failure_type = name
                    break
    first_fail = next((e for e in ev if e["kind"] == failure_type), None)
    censored = (not plan_completed) and mon_summary["aborted"] and mon_summary["abort_reason"] in TASK_FAILURES
    return {
        "y_task": int(success),
        "y_exec": int(plan_completed and not exec_viol and not sim_error),
        "task_label_mask": bool(not sim_error),
        "exec_label_mask": bool(not sim_error and (exec_viol or not censored)),
        "failure_type": failure_type,
        "failure_time": float(first_fail["time"]) if first_fail else -1.0,
        "failure_phase": int(first_fail["phase"]) if first_fail else -1,
        "simulator_error": bool(sim_error),
    }
