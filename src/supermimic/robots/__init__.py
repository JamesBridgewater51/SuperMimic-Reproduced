"""Robot registry: per-robot joint order, model loader, mjlab cfg, and dataset knobs.

``g1`` keeps the original behaviour exactly; ``x2`` adds the AgiBot X2-Ultra.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import mujoco


@dataclass(frozen=True)
class RobotSpec:
    name: str
    joint_names: tuple[str, ...]
    xml_path: Callable[[], str]
    get_robot_cfg: Callable
    action_scale: dict[str, float]
    foot_geom_regex: str
    stand_z: tuple[float, float]  # pelvis-height gate band for the feasibility filter


def _g1_spec() -> RobotSpec:
    from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg

    from supermimic.data.motion_convert import G1_JOINT_NAMES, g1_xml_path

    return RobotSpec(
        name="g1",
        joint_names=G1_JOINT_NAMES,
        xml_path=lambda: str(g1_xml_path()),
        get_robot_cfg=get_g1_robot_cfg,
        action_scale=dict(G1_ACTION_SCALE),
        foot_geom_regex=r"^(left|right)_foot[1-7]_collision$",
        stand_z=(0.60, 0.85),
    )


def _x2_spec() -> RobotSpec:
    from supermimic.assets.x2_constants import X2_ACTION_SCALE, X2_XML, get_x2_robot_cfg

    from supermimic.assets.x2_constants import (  # noqa: F401  (joint order lives with the model)
        X2_XML as _XML,
    )
    from supermimic.robots.x2_joints import X2_JOINT_NAMES

    return RobotSpec(
        name="x2",
        joint_names=X2_JOINT_NAMES,
        xml_path=lambda: str(X2_XML),
        get_robot_cfg=get_x2_robot_cfg,
        action_scale=dict(X2_ACTION_SCALE),
        foot_geom_regex=r"^(left|right)_ankle_roll_link_sole_collision$",
        stand_z=(0.70, 0.97),
    )


_SPECS = {
    "g1": _g1_spec,
    "x2": _x2_spec,
    "x2v_foot3": _x2_spec,
    "x2v_act": _x2_spec,
    "x2v_headlock": _x2_spec,
}


def _variant_spec(base: str, name: str) -> RobotSpec:
    from supermimic.assets import x2_constants

    fn = {
        "x2v_foot3": x2_constants.get_x2v_foot3_robot_cfg,
        "x2v_act": x2_constants.get_x2v_act_robot_cfg,
        "x2v_headlock": x2_constants.get_x2v_headlock_robot_cfg,
    }[name]
    spec = _x2_spec()
    import dataclasses

    return dataclasses.replace(spec, name=name, get_robot_cfg=fn)


def get_robot(name: str) -> RobotSpec:
    if name in ("x2v_foot3", "x2v_act", "x2v_headlock"):
        return _variant_spec("x2", name)
    if name not in _SPECS:
        raise ValueError(f"unknown robot {name!r}; available: {sorted(_SPECS)}")
    return _SPECS[name]()


def load_robot_model(name: str) -> mujoco.MjModel:
    """Standalone CPU MjModel with joints in canonical order (free joint first)."""
    spec = get_robot(name)
    model = mujoco.MjModel.from_xml_path(spec.xml_path())
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE, "first joint must be the free joint"
    assert tuple(names[1:]) == spec.joint_names, f"{name}: joint order mismatch: {names[1:]}"
    assert model.nq == 7 + len(spec.joint_names) and model.nv == 6 + len(spec.joint_names)
    return model
