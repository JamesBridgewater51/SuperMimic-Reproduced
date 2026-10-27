"""AgiBot X2-Ultra constants (mirrors mjlab's g1_constants structure).

Assumptions (recorded deviations from an official X2 model, which does not exist):
  * rotor inertia / gear trains are unknown for X2; reflected-inertia (armature)
    values reuse the G1 motor classes by joint role, so stiffness/damping follow
    the same 10 Hz / damping-ratio-2 design rule as G1;
  * effort/velocity limits are the X2 URDF's own per-joint values (authoritative);
  * collision: full-collision set with feet (sole boxes) at condim 3.
"""
from pathlib import Path

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import reflected_inertia_from_two_stage_planetary
from mjlab.utils.spec_config import CollisionCfg

X2_XML: Path = Path(__file__).parents[3] / "assets" / "x2" / "agibot_x2ultra.xml"

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

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.7016),
    joint_pos={
        ".*_hip_pitch_joint": -0.1,
        ".*_knee_joint": 0.3,
        ".*_ankle_pitch_joint": -0.2,
        ".*_shoulder_pitch_joint": 0.2,
        ".*_elbow_joint": 0.6,
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

    return mujoco.MjSpec.from_file(str(X2_XML))


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
