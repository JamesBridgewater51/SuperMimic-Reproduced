"""AgiBot X2-Ultra constants (mirrors mjlab's g1_constants structure).

Assumptions beyond the supplied X2 URDF (no calibrated actuator model supplied):
  * rotor inertia / gear trains are unknown for X2; reflected-inertia (armature)
    values reuse the G1 motor classes by joint role, so stiffness/damping follow
    the same 10 Hz / damping-ratio-2 design rule as G1;
  * effort/velocity limits are the X2 URDF's own per-joint values (authoritative);
  * collision: full-collision set with feet (sole boxes) at condim 3.
"""
from pathlib import Path
from xml.etree import ElementTree

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import reflected_inertia_from_two_stage_planetary
from mjlab.utils.spec_config import CollisionCfg

X2_XML: Path = Path(__file__).parents[3] / "assets" / "x2" / "agibot_x2ultra_physics_corrected.xml"

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz, same design rule as G1
DAMPING_RATIO = 2.0

# --- armature assumptions: G1 motor classes by joint role ---
_ARM_7520_22 = reflected_inertia_from_two_stage_planetary(
    (0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5)
)
_ARM_7520_14 = reflected_inertia_from_two_stage_planetary(
    (0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + (48 / 22))
)
_ARM_5020 = reflected_inertia_from_two_stage_planetary(
    (0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + (46 / 18), 1 + (56 / 16))
)
_ARM_4010 = reflected_inertia_from_two_stage_planetary((0.068e-4, 0.0, 0.0), (1, 5, 5))


def _act(target_expr, armature, effort, velocity):
    # NOTE: mjlab's BuiltinPositionActuatorCfg has no velocity_limit field (G1's
    # constants likewise never set one on the cfg); the URDF velocity limits are
    # recorded here for provenance only.
    return BuiltinPositionActuatorCfg(
        target_names_expr=target_expr,
        stiffness=armature * NATURAL_FREQ**2,
        damping=2.0 * DAMPING_RATIO * armature * NATURAL_FREQ,
        effort_limit=effort,
        armature=armature,
    )


X2_ACTUATORS = (
    _act((".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"), _ARM_7520_22, 120.0, 11.94),
    _act((".*_hip_yaw_joint", "waist_yaw_joint"), _ARM_7520_14, 120.0, 11.94),
    _act((".*_ankle_pitch_joint",), _ARM_5020 * 2, 60.0, 13.61),
    _act((".*_ankle_roll_joint",), _ARM_5020, 36.0, 14.66),
    _act(("waist_pitch_joint", "waist_roll_joint"), _ARM_5020, 36.0, 13.09),
    _act(("head_yaw_joint", "head_pitch_joint"), _ARM_4010, 2.3, 10.47),
    _act((".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"), _ARM_7520_14, 60.0, 13.61),
    _act((".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_yaw_joint"), _ARM_5020, 36.0, 14.66),
    _act((".*_wrist_pitch_joint", ".*_wrist_roll_joint"), _ARM_4010, 6.0, 15.71),
)

def _home_root_position() -> tuple[float, float, float]:
    """Keep the initial height bound to the generated contact-geometry keyframe."""
    key = ElementTree.parse(X2_XML).getroot().find("./keyframe/key[@name='home']")
    if key is None or key.get("qpos") is None:
        raise ValueError(f"Missing generated home qpos: {X2_XML}")
    return tuple(float(value) for value in key.get("qpos").split()[:3])


HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=_home_root_position(),
    joint_pos={
        ".*_hip_pitch_joint": -0.1,
        ".*_knee_joint": 0.3,
        ".*_ankle_pitch_joint": -0.2,
        ".*_shoulder_pitch_joint": 0.2,
        ".*_elbow_joint": -0.6,
        "left_shoulder_roll_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
    },
    joint_vel={".*": 0.0},
)

X2_FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype=1,
    conaffinity=1,
    condim={r"^(left|right)_ankle_roll_link_sole_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_ankle_roll_link_sole_collision$": 1, ".*": 0},
    friction={r"^(left|right)_ankle_roll_link_sole_collision$": (0.6,)},
)

X2_ARTICULATION = EntityArticulationInfoCfg(
    actuators=X2_ACTUATORS,
    soft_joint_pos_limit_factor=0.9,
)


def get_spec():
    import mujoco

    spec = mujoco.MjSpec.from_file(str(X2_XML))
    # EntityCfg supplies init_state; do not attach a second home/reserved key.
    for key in list(spec.keys):
        spec.delete(key)
    spec.nkey = 0
    return spec


def get_x2_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(X2_FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=X2_ARTICULATION,
    )


X2_ACTION_SCALE: dict[str, float] = {}
for a in X2_ACTUATORS:
    e = a.effort_limit
    s = a.stiffness
    assert e is not None and s is not None
    for n in a.target_names_expr:
        X2_ACTION_SCALE[n] = 0.25 * e / s


# ------------------------------------------------------------------ attribution variants

def get_x2v_foot3_robot_cfg() -> EntityCfg:
    """V1: 3-box split foot (heel/mid/toe) instead of one sole box.

    Tests the contact-richness suspect: G1's 7-capsule feet provide far richer
    contact than X2's single flat plate, and X2's failures concentrate in
    kneeling/deep-pose families that stress the support surface.
    """
    import mujoco
    spec = mujoco.MjSpec.from_file(str(X2_XML))
    # split each sole box into three along x: [-0.053,-0.018],[-0.018,0.018],[0.018,0.053] (half-lengths ~0.0177/0.018/0.0177)
    for body in spec.bodies:
        for g in body.geoms:
            if g.name and g.name.endswith("_sole_collision"):
                side = "left" if g.name.startswith("left") else "right"
                cx = g.pos[0]
                g.size = [0.0177, g.size[1], g.size[2]]
                g.pos = [cx - 0.0353, g.pos[1], g.pos[2]]
                for off, tag in ((0.0, "mid"), (0.0354, "toe")):
                    ng = body.add_geom(
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        name=f"{side}_ankle_roll_link_sole_{tag}_collision",
                        size=[0.0177, g.size[1], g.size[2]],
                        pos=[cx + off, g.pos[1], g.pos[2]],
                        group=3, mass=0.0,
                    )
    cfg = get_x2_robot_cfg()
    cfg.spec_fn = lambda: spec
    return cfg


def get_x2v_act_robot_cfg() -> EntityCfg:
    """V2: uniform G1-class (5020) armature on every joint.

    Tests the actuator-assumption suspect: X2 armatures were assumed from G1
    motor classes by role; this variant removes the role mapping entirely.
    """
    import dataclasses

    acts = tuple(
        dataclasses.replace(a, armature=_ARM_5020, stiffness=_ARM_5020 * NATURAL_FREQ**2,
                            damping=2.0 * DAMPING_RATIO * _ARM_5020 * NATURAL_FREQ)
        for a in X2_ACTUATORS
    )
    art = dataclasses.replace(X2_ARTICULATION, actuators=acts)
    cfg = get_x2_robot_cfg()
    cfg.articulation = art
    return cfg


def get_x2v_headlock_robot_cfg() -> EntityCfg:
    """V3: head joints effectively locked (tiny effort, high stiffness)."""
    import dataclasses

    acts = tuple(
        dataclasses.replace(a, effort_limit=0.05) if set(a.target_names_expr) & {"head_yaw_joint", "head_pitch_joint"} else a
        for a in X2_ACTUATORS
    )
    art = dataclasses.replace(X2_ARTICULATION, actuators=acts)
    cfg = get_x2_robot_cfg()
    cfg.articulation = art
    return cfg
