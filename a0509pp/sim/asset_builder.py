"""Build the combined A0509 + flange adapter + Robotiq 2F-85 (+ wrist camera housing) articulation USD.

URDF preprocessing is plain Python. The USD steps need Isaac Sim's pxr and the Isaac Lab URDF converter,
so call them only after the Kit app is running (scripts/prepare_assets.py).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

GRIPPER_BODIES = [
    "base_link", "left_outer_knuckle", "right_outer_knuckle", "left_outer_finger", "right_outer_finger",
    "left_inner_finger", "right_inner_finger", "left_inner_knuckle", "right_inner_knuckle", "left_fingertip", "right_fingertip",
]
ROBOTIQ_CFG_REL = os.path.join("grippers", "Robotiq_2F_85", "configuration", "Robotiq_2F_85_config_physics_parallel_grip.usda")


def git_rev(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def prepare_urdf(doosan_repo, out_dir, base_name="a0509_base"):
    """Copy a0509.urdf with absolute mesh paths, the arm base renamed, and the massless 'base' frame removed."""
    desc = os.path.join(doosan_repo, "dsr_description2")
    src = os.path.join(desc, "urdf", "a0509.urdf")
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} not found; pass the doosan-robot2 checkout with --doosan")
    tree = ET.parse(src)
    root = tree.getroot()
    for j in list(root.findall("joint")):
        if j.get("name") == "base_link-base":
            root.remove(j)
    for l in list(root.findall("link")):
        if l.get("name") == "base":
            root.remove(l)
        elif l.get("name") == "base_link":
            l.set("name", base_name)
    for j in root.findall("joint"):
        for tag in ("parent", "child"):
            e = j.find(tag)
            if e is not None and e.get("link") == "base_link":
                e.set("link", base_name)
    for m in root.iter("mesh"):
        fn = m.get("filename")
        if fn.startswith("package://dsr_description2/"):
            fn = os.path.join(desc, fn[len("package://dsr_description2/"):])
        if not os.path.exists(fn):
            raise FileNotFoundError(fn)
        m.set("filename", os.path.abspath(fn))
    os.makedirs(out_dir, exist_ok=True)
    root.set("name", "a0509_arm")
    out = os.path.join(out_dir, "a0509_arm.urdf")
    tree.write(out, xml_declaration=True, encoding="utf-8")
    return out, src


def check_robotiq_lfs(robotiq_repo):
    asset = os.path.join(robotiq_repo, "grippers", "Robotiq_2F_85")
    if not os.path.isfile(os.path.join(robotiq_repo, ROBOTIQ_CFG_REL)):
        raise FileNotFoundError(f"Robotiq asset missing: {os.path.join(robotiq_repo, ROBOTIQ_CFG_REL)}")
    bad = []
    for directory, _, files in os.walk(asset):
        for f in files:
            p = os.path.join(directory, f)
            with open(p, "rb") as fh:
                if fh.read(64).startswith(b"version https://git-lfs"):
                    bad.append(os.path.relpath(p, robotiq_repo))
    if bad:
        raise RuntimeError(
            f"Robotiq meshes are Git LFS pointer stubs ({', '.join(bad)}). Run 'git lfs install && git lfs pull' in {robotiq_repo}."
        )


def bundle_robotiq(robotiq_repo, out_dir):
    """Copy the upstream asset tree so its relative USD references survive moving the generated bundle."""
    source = os.path.realpath(robotiq_repo)
    bundled = os.path.realpath(os.path.join(out_dir, "vendor", "robotiq"))
    if os.path.commonpath((source, bundled)) == source:
        raise ValueError("the generated asset directory cannot be inside the Robotiq checkout")
    check_robotiq_lfs(source)
    if os.path.islink(bundled):
        raise ValueError(f"refusing to replace a symlink: {bundled}")
    if os.path.exists(bundled):
        shutil.rmtree(bundled)
    shutil.copytree(source, bundled, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    return bundled


def validate_asset_bundle(out_dir, usd_paths):
    """Fail if a generated USD refers to a missing file or anything outside the bundle."""
    from pxr import Sdf, UsdUtils

    root = os.path.realpath(out_dir)
    report = {}
    for usd in usd_paths:
        usd = os.path.realpath(usd)
        if not os.path.isfile(usd):
            raise FileNotFoundError(usd)
        if os.path.commonpath((root, usd)) != root:
            raise RuntimeError(f"USD is outside the asset bundle: {usd}")
        authored_external = []

        def check_authored_path(layer, dependency):
            # The callback sees asset-valued attributes (for example textures) as well as
            # composition arcs. Resolved files alone cannot reveal an absolute path
            # authored to a file *inside* this bundle, which would break after a move.
            path = dependency.assetPath
            if path and path != usd and (os.path.isabs(path) or "://" in path):
                authored_external.append(path)
            return dependency

        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(usd), check_authored_path)
        # Bare MDL names (OmniPBR.mdl, OmniSurface.mdl, ...) are Kit core materials found on the MDL search path.
        unresolved = [p for p in unresolved if not (str(p).endswith(".mdl") and "/" not in str(p) and "\\" not in str(p))]
        if unresolved:
            raise RuntimeError(f"unresolved dependencies in {usd}: {list(unresolved)}")
        if authored_external:
            raise RuntimeError(f"non-portable authored USD paths in {usd}: {sorted(set(authored_external))}")
        dependencies = [layer.realPath for layer in layers] + list(assets)
        outside = [str(p) for p in dependencies if not os.path.isabs(str(p)) or
                   os.path.commonpath((root, os.path.realpath(str(p)))) != root]
        if outside:
            raise RuntimeError(f"dependencies outside the asset bundle in {usd}: {outside}")
        absolute_refs = []
        for layer in layers:
            sublayers, references, payloads = UsdUtils.ExtractExternalReferences(layer.realPath)
            absolute_refs += [p for p in sublayers + references + payloads if os.path.isabs(p) or "://" in p]
        if absolute_refs:
            raise RuntimeError(f"non-portable authored USD paths in {usd}: {sorted(set(absolute_refs))}")
        report[os.path.relpath(usd, root)] = {"layers": len(layers), "assets": len(assets)}
    return report


def convert_arm(urdf_path, usd_dir, force=True, collision_type="Convex Hull", stiffness=None, damping=None):
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=usd_dir,
        fix_base=True,
        merge_fixed_joints=True,
        self_collision=True,
        collision_type=collision_type,
        force_usd_conversion=force,
        physics_variant="physx",
        robot_type="Manipulator",
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=stiffness, damping=damping),
        ),
    )
    return UrdfConverter(cfg).usd_path


def _find(stage, root_path, name, api=None):
    from pxr import Usd

    hits = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if prim.GetName() == name and (api is None or prim.HasAPI(api)):
            hits.append(prim)
    return hits


def _set_pose_ops(prim, pos, quat):
    """Translate + orient ops whose precision matches any xformOp already authored on the prim (quatf vs quatd)."""
    from pxr import Gf, Sdf, UsdGeom

    def precision(name, float_type):
        a = prim.GetAttribute(name)
        return UsdGeom.XformOp.PrecisionFloat if a and a.GetTypeName() == float_type else UsdGeom.XformOp.PrecisionDouble

    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    pt = precision("xformOp:translate", Sdf.ValueTypeNames.Float3)
    po = precision("xformOp:orient", Sdf.ValueTypeNames.Quatf)
    xf.AddTranslateOp(pt).Set((Gf.Vec3f if pt == UsdGeom.XformOp.PrecisionFloat else Gf.Vec3d)(*pos))
    q = (Gf.Quatf if po == UsdGeom.XformOp.PrecisionFloat else Gf.Quatd)(quat.GetReal(), *quat.GetImaginary())
    xf.AddOrientOp(po).Set(q)


def _rel(path, root):
    s = str(path)
    return s[len(root) + 1:] if s.startswith(root + "/") else s


def inertia_combine(parts):
    """parts: list of (mass, com(3), inertia 3x3 about com) in one frame -> (mass, com, I about com)."""
    M = sum(p[0] for p in parts)
    C = sum(p[0] * np.asarray(p[1]) for p in parts) / M
    I = np.zeros((3, 3))
    for m, c, Ic in parts:
        d = np.asarray(c) - C
        I += np.asarray(Ic) + m * (d @ d * np.eye(3) - np.outer(d, d))
    return M, C, I


def build_combined(arm_usd, robotiq_repo, out_path, rcfg, gcfg, ccfg, with_camera=True):
    """Author the combined articulation. Returns a dict of prim paths relative to the robot root prim."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    robotiq_usd = os.path.join(robotiq_repo, ROBOTIQ_CFG_REL)
    if not os.path.exists(robotiq_usd):
        raise FileNotFoundError(robotiq_usd)
    if os.path.exists(out_path):
        os.remove(out_path)
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root_path = "/A0509_2F85"
    root = stage.DefinePrim(root_path, "Xform")
    stage.SetDefaultPrim(root)
    root.GetReferences().AddReference(os.path.relpath(arm_usd, os.path.dirname(out_path)))

    roots = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1:
        raise RuntimeError(f"expected one articulation root in the converted arm, found {[str(p.GetPath()) for p in roots]}")
    link6 = _find(stage, root_path, rcfg.flange_link, UsdPhysics.RigidBodyAPI)
    if len(link6) != 1:
        raise RuntimeError(f"could not find a unique rigid body '{rcfg.flange_link}'")
    link6 = link6[0]
    if link6.IsInstanceProxy():
        raise RuntimeError("link_6 is an instance proxy; cannot parent the gripper under it")
    arm_joints = {p.GetName(): str(p.GetPath()) for p in Usd.PrimRange(root) if p.IsA(UsdPhysics.RevoluteJoint)}
    missing = [j for j in rcfg.arm_joint_names if j not in arm_joints]
    if missing:
        raise RuntimeError(f"converted arm lacks joints {missing}; found {sorted(arm_joints)}")

    t_a = rcfg.adapter_thickness
    yaw = rcfg.gripper_yaw_on_flange
    qyaw = Gf.Quatd(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    grip = stage.DefinePrim(link6.GetPath().AppendChild("Robotiq_2F_85"), "Xform")
    grip.GetReferences().AddReference(os.path.relpath(robotiq_usd, os.path.dirname(out_path)))
    _set_pose_ops(grip, (0.0, 0.0, t_a), qyaw)

    inner = stage.GetPrimAtPath(grip.GetPath().AppendChild("Robotiq_2F_85"))
    if not inner.IsValid():
        raise RuntimeError("Robotiq asset layout changed: expected <ref>/Robotiq_2F_85")
    inner.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    if "PhysxArticulationAPI" in inner.GetAppliedSchemas():
        inner.RemoveAppliedSchema("PhysxArticulationAPI")
    rj = stage.GetPrimAtPath(inner.GetPath().AppendChild("root_joint"))
    if rj.IsValid():
        rj.SetActive(False)
    base = stage.GetPrimAtPath(inner.GetPath().AppendChild("base_link"))
    if not base.IsValid() or not base.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("Robotiq base_link rigid body not found")

    mount = UsdPhysics.FixedJoint.Define(stage, link6.GetPath().AppendChild("gripper_mount_joint"))
    mount.CreateBody0Rel().SetTargets([link6.GetPath()])
    mount.CreateBody1Rel().SetTargets([base.GetPath()])
    mount.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, t_a))
    mount.CreateLocalRot0Attr().Set(Gf.Quatf(float(qyaw.GetReal()), 0.0, 0.0, float(qyaw.GetImaginary()[2])))
    mount.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    mount.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    def solid(path, kind, pos, size, color):
        if kind == "cylinder":
            g = UsdGeom.Cylinder.Define(stage, path)
            g.CreateAxisAttr("Z")
            g.CreateHeightAttr(float(size[1]))
            g.CreateRadiusAttr(float(size[0]))
        else:
            g = UsdGeom.Cube.Define(stage, path)
            g.CreateSizeAttr(1.0)
        gx = UsdGeom.Xformable(g)
        gx.AddTranslateOp().Set(Gf.Vec3d(*pos))
        if kind != "cylinder":
            gx.AddScaleOp().Set(Gf.Vec3f(*size))
        g.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
        return g

    parts = [(0.77744, (0.0, 0.0, 0.035508), np.eye(3) * 0.001)]
    m_a, r_a = rcfg.adapter_mass, rcfg.adapter_radius
    solid(base.GetPath().AppendChild("flange_adapter"), "cylinder", (0.0, 0.0, -t_a / 2), (r_a, t_a), (0.35, 0.35, 0.38))
    parts.append((m_a, (0.0, 0.0, -t_a / 2), np.diag([m_a * (3 * r_a**2 + t_a**2) / 12] * 2 + [m_a * r_a**2 / 2])))
    cam = None
    if with_camera:
        hx, hy, hz = rcfg.wrist_camera_housing
        cp = tuple(ccfg.wrist_pos)
        solid(base.GetPath().AppendChild("wrist_camera_housing"), "cube", cp, (hx, hy, hz), (0.1, 0.1, 0.12))
        bx0 = 0.0375
        bx1 = cp[0] - hx / 2
        bpos = ((bx0 + bx1) / 2, 0.0, cp[2])
        solid(base.GetPath().AppendChild("wrist_camera_bracket"), "cube", bpos, (max(bx1 - bx0, 0.004), 0.030, 0.030), (0.3, 0.3, 0.3))
        mc, mb = rcfg.wrist_camera_mass, rcfg.wrist_bracket_mass
        parts.append((mc, cp, np.diag([mc * (hy**2 + hz**2), mc * (hx**2 + hz**2), mc * (hx**2 + hy**2)]) / 12))
        parts.append((mb, bpos, np.eye(3) * mb * 0.03**2 / 6))
        cam = {"housing": _rel(base.GetPath().AppendChild("wrist_camera_housing"), root_path)}
    M, C, I = inertia_combine(parts)
    w, V = np.linalg.eigh(I)
    if np.linalg.det(V) < 0:
        V[:, 0] *= -1
    from ..geometry import mat_to_quat

    qa = mat_to_quat(V)
    mass = UsdPhysics.MassAPI.Apply(base)
    mass.CreateMassAttr().Set(float(M))
    mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(*[float(x) for x in C]))
    mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*[float(x) for x in w]))
    mass.CreatePrincipalAxesAttr().Set(Gf.Quatf(float(qa[3]), float(qa[0]), float(qa[1]), float(qa[2])))

    bodies = {}
    for name in GRIPPER_BODIES:
        p = stage.GetPrimAtPath(inner.GetPath().AppendChild(name))
        if not p.IsValid():
            raise RuntimeError(f"gripper body '{name}' not found")
        bodies[name] = p
    for i, a in enumerate(GRIPPER_BODIES):
        api = UsdPhysics.FilteredPairsAPI.Apply(bodies[a])
        rel = api.CreateFilteredPairsRel()
        for b in GRIPPER_BODIES[i + 1:]:
            rel.AddTarget(bodies[b].GetPath())

    finger_joints = {p.GetName(): str(p.GetPath()) for p in Usd.PrimRange(inner) if p.IsA(UsdPhysics.RevoluteJoint)}
    stage.GetRootLayer().customLayerData = {"generator": "a0509pp.sim.asset_builder", "with_camera": bool(with_camera)}
    stage.Save()

    arm_bodies = {p.GetName(): _rel(p.GetPath(), root_path) for p in Usd.PrimRange(root)
                  if p.HasAPI(UsdPhysics.RigidBodyAPI) and not str(p.GetPath()).startswith(str(grip.GetPath()))}
    return {
        "root_prim": root_path,
        "articulation_root": _rel(roots[0].GetPath(), root_path),
        "link6": _rel(link6.GetPath(), root_path),
        "gripper_ref": _rel(grip.GetPath(), root_path),
        "gripper_base": _rel(base.GetPath(), root_path),
        "gripper_bodies": {n: _rel(p.GetPath(), root_path) for n, p in bodies.items()},
        "arm_bodies": arm_bodies,
        "arm_joints": {k: _rel(v, root_path) for k, v in arm_joints.items()},
        "gripper_joints": {k: _rel(v, root_path) for k, v in finger_joints.items()},
        "gripper_base_mass": {"mass": float(M), "com": C.tolist(), "inertia_diag": w.tolist(), "principal_axes_xyzw": qa.tolist()},
        "wrist_camera": cam,
    }


def measure_gripper_geometry(usd_path, rel_paths):
    """Axis-aligned bounds of each gripper body in the gripper base frame at finger_joint = 0 (authored pose)."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim().GetPath()
    base = stage.GetPrimAtPath(root.AppendPath(rel_paths["gripper_base"]))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    xc = UsdGeom.XformCache()
    T_base = np.array(xc.GetLocalToWorldTransform(base)).T
    Tb_inv = np.linalg.inv(T_base)
    out = {}
    for name, rel in rel_paths["gripper_bodies"].items():
        prim = stage.GetPrimAtPath(root.AppendPath(rel))
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
        corners = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        loc = (Tb_inv @ corners.T).T[:, :3]
        out[name] = {"min": loc.min(0).tolist(), "max": loc.max(0).tolist()}
    pads = {}
    if "left_fingertip" in out and "right_fingertip" in out:
        L, R = out["left_fingertip"], out["right_fingertip"]
        inner_y = (abs(L["max"][1]) + abs(R["min"][1])) / 2.0
        zc = (L["min"][2] + L["max"][2] + R["min"][2] + R["max"][2]) / 4.0
        pads = {"pad_inner_y_open": inner_y, "pad_center_z_open": zc,
                "pad_half_height": (L["max"][2] - L["min"][2]) / 2.0, "pad_half_width": (L["max"][0] - L["min"][0]) / 2.0,
                "pad_thickness": L["max"][1] - L["min"][1]}
    palm_top = out.get("base_link", {}).get("max", [0, 0, 0.075])[2]
    return {"bodies": out, "pads": pads, "palm_top_z": palm_top,
            "note": "AABB of each body's geometry in the gripper base frame; pad values from fingertip AABBs"}


def read_mimic(usd_path, joint_rel_paths):
    """PhysX mimic couplings of the gripper joints: q_mimic = -gearing * q_reference - offset."""
    from pxr import Usd

    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim().GetPath()
    out = {}
    for name, rel in joint_rel_paths.items():
        prim = stage.GetPrimAtPath(root.AppendPath(rel))
        if not prim.IsValid():
            continue
        listed = prim.GetMetadata("apiSchemas")
        for schema in set(prim.GetAppliedSchemas()) | set(listed.ApplyOperations([]) if listed else []):
            if not schema.startswith("PhysxMimicJointAPI"):
                continue
            inst = schema.split(":", 1)[1] if ":" in schema else ""
            ns = f"physxMimicJoint:{inst}:" if inst else "physxMimicJoint:"
            g = prim.GetAttribute(ns + "gearing").Get()
            off_attr = prim.GetAttribute(ns + "offset")
            off = off_attr.Get() if off_attr and off_attr.IsValid() else 0.0
            targets = prim.GetRelationship(ns + "referenceJoint").GetTargets()
            out[name] = {"reference": targets[0].name if targets else "", "gearing": float(g if g is not None else -1.0),
                         "offset": float(off or 0.0), "axis": inst}
    return out


def bind_pad_material(usd_path, rel_paths, pad_links, static_friction, dynamic_friction, restitution=0.0):
    """Bind a physics material (average combine) to the pad bodies; runtime values are set per env by the env."""
    from pxr import Sdf, Usd, UsdPhysics, UsdShade

    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim().GetPath()
    mat_path = root.AppendPath("PhysicsMaterials/pad_material")
    mat = UsdShade.Material.Define(stage, mat_path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr().Set(float(static_friction))
    api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    api.CreateRestitutionAttr().Set(float(restitution))
    prim = mat.GetPrim()
    prim.AddAppliedSchema("PhysxMaterialAPI")
    prim.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("average")
    prim.CreateAttribute("physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token).Set("average")
    for link in pad_links:
        body = stage.GetPrimAtPath(root.AppendPath(rel_paths["gripper_bodies"][link]))
        binding = UsdShade.MaterialBindingAPI.Apply(body)
        binding.Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")
    stage.Save()
    return _rel(mat_path, str(root))


def copy_models(out_dir):
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "models")
    for f in ("a0509_kinematics.json", "a0509_spheres.json"):
        shutil.copy(os.path.join(here, f), os.path.join(out_dir, f))
    return {k: os.path.join(out_dir, f) for k, f in (("kinematics_json", "a0509_kinematics.json"), ("arm_spheres_json", "a0509_spheres.json"))}


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
