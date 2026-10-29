"""Build an mjlab-ready MJCF for the AgiBot X2-Ultra from its URDF.

The MuJoCo URDF importer welds the root link to the world (no floating base) and
leaves geoms unnamed, so we build the MjSpec by hand from the URDF tree:
  * pelvis gets a freejoint (floating base);
  * joint order == URDF tree order == X2_JOINT_NAMES (asserted on output);
  * every real link keeps its mesh as a visual geom (group 2, no contact);
  * every real link's collision mesh becomes a geom named ``<link>_collision``
    (group 3, contact) so mjlab's CollisionCfg(".*_collision") selection works;
  * feet additionally get a thin box sole primitive per foot
    (``<side>_foot_sole_collision``) for stable planar contact, since a convex
    hull of the foot mesh can be non-planar at the sole edges;
  * a standing keyframe is computed by dropping the model so the lowest sole
    point touches z=0 at the default pose.

Usage (PT): .venv/bin/python assets/x2/build_x2_mjcf.py \
    --urdf /data/liuminghao/assets/soma-retargeter/robotics/agibot/x2/desc/agibot_x2ultra.urdf \
    --out assets/x2/agibot_x2ultra.xml
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

X2_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "head_yaw_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
)
_ROOT = "pelvis"
# Contact plate approximation, not a manufacturer-calibrated sole model.
SOLE_CONTACT_BAND_M = 0.012
SOLE_THICKNESS_M = 0.012
# default pose: slight knee bend + ankle compensation, arms at sides (G1-like)
HOME_QPOS: dict[str, float] = {
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    ".*_elbow_joint": -0.6,
}


def rpy_to_quat_wxyz(rpy) -> tuple[float, float, float, float]:
    from scipy.spatial.transform import Rotation
    q = Rotation.from_euler("xyz", rpy).as_quat()  # xyzw
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def origin_of(el) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    o = el.find("origin")
    xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0")).split()]) if o is not None else np.zeros(3)
    rpy = tuple(float(v) for v in (o.get("rpy", "0 0 0")).split()) if o is not None else (0.0, 0.0, 0.0)
    return xyz, rpy_to_quat_wxyz(rpy)


def mesh_vertices_in_body(mj_model, mj_data, geom_id: int) -> np.ndarray:
    """Undo compiled mesh centering/orientation before using body-local dimensions."""
    mesh_id = mj_model.geom_dataid[geom_id]
    start, count = mj_model.mesh_vertadr[mesh_id], mj_model.mesh_vertnum[mesh_id]
    vertices = mj_model.mesh_vert[start:start + count].astype(np.float64)
    body_id = mj_model.geom_bodyid[geom_id]
    world = vertices @ mj_data.geom_xmat[geom_id].reshape(3, 3).T + mj_data.geom_xpos[geom_id]
    return (world - mj_data.xpos[body_id]) @ mj_data.xmat[body_id].reshape(3, 3)


def sole_plate_parameters(vertices: np.ndarray) -> tuple[list[float], list[float]]:
    """Fit a 12 mm plate to the lowest 12 mm support band, preserving its bottom."""
    bottom = float(vertices[:, 2].min())
    support = vertices[vertices[:, 2] <= bottom + SOLE_CONTACT_BAND_M]
    lower, upper = support[:, :2].min(axis=0), support[:, :2].max(axis=0)
    half_xy, center_xy = (upper - lower) / 2, (upper + lower) / 2
    if len(support) < 3 or np.any(half_xy <= 0):
        raise ValueError("Foot visual mesh has no nondegenerate plantar support band")
    size = [float(half_xy[0]), float(half_xy[1]), SOLE_THICKNESS_M / 2]
    position = [float(center_xy[0]), float(center_xy[1]), bottom + SOLE_THICKNESS_M / 2]
    return size, position


def validate_joint_ranges(mj_model, source_joints) -> None:
    """A radian XML header cannot reveal an earlier MjSpec degree conversion."""
    for joint in source_joints:
        if joint.get("type") != "revolute":
            continue
        joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint.get("name"))
        limit = joint.find("limit")
        expected = [float(limit.get("lower")), float(limit.get("upper"))]
        if not np.allclose(mj_model.jnt_range[joint_id], expected, atol=1e-10, rtol=0):
            raise ValueError(f"URDF radian range changed during compile: {joint.get('name')}")


def validate_home_qpos(mj_model, qpos: np.ndarray) -> None:
    for joint_id in range(1, mj_model.njnt):
        coordinate = qpos[mj_model.jnt_qposadr[joint_id]]
        lower, upper = mj_model.jnt_range[joint_id]
        if not lower <= coordinate <= upper:
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            raise ValueError(f"Home pose outside joint limits: {name}: {coordinate}")


def build(urdf_path: Path, meshdir: str) -> mujoco.MjSpec:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    joints = root.findall("joint")

    child_of = {}   # parent link -> (joint, child link)
    for j in joints:
        p, c = j.find("parent").get("link"), j.find("child").get("link")
        child_of.setdefault(p, []).append((j, c))

    spec = mujoco.MjSpec()
    # MjSpec defaults to degree-valued angle fields; URDF limits are radians.
    spec.compiler.degree = False
    spec.compiler.meshdir = meshdir
    spec.compiler.autolimits = True

    # ---- meshes (both visual and collision reference the same STL files) ----
    mesh_files = sorted({m.get("filename") for m in root.iter("mesh")})
    for f in mesh_files:
        name = Path(f).name
        spec.add_mesh(name=Path(f).stem, file=f)

    def add_geoms(body, link, is_foot: bool) -> None:
        seen: dict[str, int] = {}
        for vis in link.findall("visual"):
            m = vis.find("geometry").find("mesh")
            if m is not None:
                xyz, quat = origin_of(vis)
                base = f"{link.get('name')}_visual"
                seen[base] = seen.get(base, 0) + 1
                nm = base if seen[base] == 1 else f"{base}{seen[base]}"
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_MESH, meshname=Path(m.get("filename")).stem,
                    name=nm, pos=xyz, quat=quat,
                    group=2, contype=0, conaffinity=0, mass=0.0,
                )
        if is_foot:
            # feet: single flat box replacing the (non-planar convex-hull) foot mesh,
            # sized from the mesh bbox in a post-pass once compiled (see _fit_feet).
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, name=f"{link.get('name')}_sole_collision",
                          size=[0.09, 0.04, 0.012], pos=[0.0, 0.0, -0.044], group=3, mass=0.0)
        else:
            for coll in link.findall("collision"):
                m = coll.find("geometry").find("mesh")
                if m is not None:
                    xyz, quat = origin_of(coll)
                    base = f"{link.get('name')}_collision"
                    seen[base] = seen.get(base, 0) + 1
                    nm = base if seen[base] == 1 else f"{base}{seen[base]}"
                    body.add_geom(
                        type=mujoco.mjtGeom.mjGEOM_MESH, meshname=Path(m.get("filename")).stem,
                        name=nm, pos=xyz, quat=quat,
                        group=3, mass=0.0,
                    )

    def add_body(parent_spec, link_name: str, joint) -> None:
        link = links[link_name]
        xyz, quat = origin_of(joint)
        b = parent_spec.add_body(name=link_name, pos=xyz, quat=quat)
        inert = link.find("inertial")
        if inert is not None:
            ipos, iquat = origin_of(inert)
            i = inert.find("inertia")
            b.ipos = ipos
            b.iquat = iquat
            b.mass = float(inert.find("mass").get("value"))
            b.fullinertia = [float(i.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
        if joint.get("type") == "revolute":
            axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()]) if joint.find("axis") is not None else np.array([1.0, 0, 0])
            lim = joint.find("limit")
            jnt = b.add_joint(name=joint.get("name"), type=mujoco.mjtJoint.mjJNT_HINGE, axis=axis)
            if lim is not None:
                jnt.range = (float(lim.get("lower")), float(lim.get("upper")))
        is_foot = link_name.endswith("ankle_roll_link")
        add_geoms(b, link, is_foot)
        for j, c in child_of.get(link_name, []):
            add_body(b, c, j)

    # root body: pelvis with freejoint
    pelvis = spec.worldbody.add_body(name=_ROOT, pos=[0, 0, 0.9])
    link = links[_ROOT]
    inert = link.find("inertial")
    ipos, iquat = origin_of(inert)
    i = inert.find("inertia")
    pelvis.ipos = ipos
    pelvis.iquat = iquat
    pelvis.mass = float(inert.find("mass").get("value"))
    pelvis.fullinertia = [float(i.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
    pelvis.add_freejoint()
    add_geoms(pelvis, link, False)
    for j, c in child_of[_ROOT]:
        if j.get("type") == "revolute":
            add_body(pelvis, c, j)

    # ---- keyframe: home pose dropped onto the floor ----
    model = spec.compile()
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)]
    assert tuple(names) == X2_JOINT_NAMES, f"joint order mismatch: {names}"
    validate_joint_ranges(model, joints)

    qpos = np.zeros(model.nq)
    qpos[3] = 1.0
    import re
    for jid in range(1, model.njnt):
        nm = names[jid - 1]
        for pat, val in HOME_QPOS.items():
            if re.fullmatch(pat, nm):
                qpos[model.jnt_qposadr[jid]] = val
                break
    validate_home_qpos(model, qpos)

    def _lowest(model, data) -> float:
        best = np.inf
        for g in range(model.ngeom):
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                vertical_extent = np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g]
                best = min(best, float(data.geom_xpos[g][2] - vertical_extent))
            elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                did = model.geom_dataid[g]
                vadr, vnum = model.mesh_vertadr[did], model.mesh_vertnum[did]
                verts = model.mesh_vert[vadr:vadr + vnum].astype(np.float64) @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                best = min(best, float(verts[:, 2].min()))
        return best

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qpos[2] = 1.0
    mujoco.mj_forward(model, data)
    data.qpos[2] -= _lowest(model, data)
    mujoco.mj_forward(model, data)

    # --- fit sole boxes to the foot meshes (use the still-present ankle visual geom) ---
    for side in ("left", "right"):
        gid_vis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_ankle_roll_link_visual")
        if gid_vis < 0:
            continue
        vertices = mesh_vertices_in_body(model, data, gid_vis)
        size, position = sole_plate_parameters(vertices)
        for body_spec in spec.bodies:
            for gs in body_spec.geoms:
                if gs.name == f"{side}_ankle_roll_link_sole_collision":
                    gs.size = size
                    gs.pos = position
        print(f"[fit] {side} body-frame sole: size={size} pos={position}")
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qpos[2] = 1.0
    mujoco.mj_forward(model, data)
    data.qpos[2] -= _lowest(model, data)
    mujoco.mj_forward(model, data)

    # Preserve the structural-neighbor policy for overlapping motor housings.
    # Direct-parent excludes are redundant with default filterparent; the
    # grandparent assemblies are not. This is not a vendor-validated SRDF matrix.
    parents = {j.find("child").get("link"): j.find("parent").get("link") for j in root.findall("joint") if j.get("type") == "revolute"}
    mesh_links = {l.get("name") for l in root.findall("link") if l.find("collision") is not None}
    excluded = set()
    for child, parent in parents.items():
        if child in mesh_links and parent in mesh_links:
            spec.add_exclude(bodyname1=parent, bodyname2=child)
            excluded.add((parent, child))
        gp = parents.get(parent)
        if gp and child in mesh_links and gp in mesh_links:
            spec.add_exclude(bodyname1=gp, bodyname2=child)
            excluded.add((gp, child))
    print(f"[build] excluded {len(excluded)} adjacent self-collision pairs")

    key = spec.add_key(name="home")
    key.qpos = data.qpos.tolist()
    print(f"[build] nq={model.nq} nv={model.nv} standing pelvis z={data.qpos[2]:.4f}")
    return spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "agibot_x2ultra.xml")
    ap.add_argument("--meshdir", type=str, default=None)
    a = ap.parse_args()
    meshdir = a.meshdir or str(a.urdf.parent.resolve())
    spec = build(a.urdf, meshdir)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    xml = spec.to_xml()
    a.out.write_text(xml)
    print(f"[build] wrote {a.out}")


if __name__ == "__main__":
    main()
