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
# default pose: slight knee bend + ankle compensation, arms at sides (G1-like)
HOME_QPOS: dict[str, float] = {
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    ".*_elbow_joint": 0.6,
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


def build(urdf_path: Path, meshdir: str) -> mujoco.MjSpec:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    joints = root.findall("joint")

    child_of = {}   # parent link -> (joint, child link)
    j_of_child = {}  # child link -> joint
    for j in joints:
        p, c = j.find("parent").get("link"), j.find("child").get("link")
        child_of.setdefault(p, []).append((j, c))
        j_of_child[c] = j

    spec = mujoco.MjSpec()
    spec.compiler.meshdir = meshdir
    # Serialize angle-valued fields (joint ranges) as RADIANS. With the default
    # degree=True, MjSpec.to_xml() divides ranges by 180/pi, silently pinning
    # every joint to ~1/57 of its true range (knee 2.41 rad -> 0.042 rad).
    spec.compiler.degree = False
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

    qpos = np.zeros(model.nq)
    import re
    for jid in range(1, model.njnt):
        nm = names[jid - 1]
        for pat, val in HOME_QPOS.items():
            if re.fullmatch(pat, nm):
                qpos[model.jnt_qposadr[jid]] = val
                break
    def _lowest(model, data) -> float:
        best = np.inf
        for g in range(model.ngeom):
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                best = min(best, float(data.geom_xpos[g][2] - model.geom_size[g][2]))
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
        did = model.geom_dataid[gid_vis]
        vadr, vnum = model.mesh_vertadr[did], model.mesh_vertnum[did]
        v = model.mesh_vert[vadr:vadr + vnum].astype(np.float64)
        dx = (v[:, 0].max() - v[:, 0].min()) / 2; dy = (v[:, 1].max() - v[:, 1].min()) / 2
        zmin = v[:, 2].min(); zmax = v[:, 2].max()
        cx = (v[:, 0].max() + v[:, 0].min()) / 2; cy = (v[:, 1].max() + v[:, 1].min()) / 2
        thick = max(0.012, (zmax - zmin) * 0.25)
        # spec objects are pre-compile references: find the geom spec by name
        for body_spec in spec.bodies:
            for gs in body_spec.geoms:
                if gs.name == f"{side}_ankle_roll_link_sole_collision":
                    gs.size = [float(dx), float(dy), float(thick / 2)]
                    gs.pos = [float(cx), float(cy), float(zmin + thick / 2)]
        print(f"[fit] {side} sole: dx={dx:.3f} dy={dy:.3f} zmin={zmin:.3f} cx={cx:.3f} cy={cy:.3f}")
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qpos[2] = 1.0
    mujoco.mj_forward(model, data)
    data.qpos[2] -= _lowest(model, data)
    mujoco.mj_forward(model, data)

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
