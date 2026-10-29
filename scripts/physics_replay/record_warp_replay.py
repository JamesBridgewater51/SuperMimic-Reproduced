"""Record genuine free-base MuJoCo Warp position-servo replay of fixed X2 references."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import warp as wp


@wp.kernel
def collect_nonfinite(qpos: wp.array2d(dtype=wp.float32), qvel: wp.array2d(dtype=wp.float32),
                      qacc: wp.array2d(dtype=wp.float32), failures: wp.array(dtype=wp.int32)):
    index = wp.tid()
    if index < qpos.shape[1] and not wp.isfinite(qpos[0, index]):
        wp.atomic_or(failures, 0, 1)
    if index < qvel.shape[1] and not wp.isfinite(qvel[0, index]):
        wp.atomic_or(failures, 0, 2)
    if index < qacc.shape[1] and not wp.isfinite(qacc[0, index]):
        wp.atomic_or(failures, 0, 4)


@wp.kernel
def collect_overflow(overflow: wp.array(dtype=wp.int32), failures: wp.array(dtype=wp.int32)):
    if overflow[0] != 0:
        wp.atomic_or(failures, 0, 8)


MOTIONS = (
    "flip_360_002__A304", "flip_180_R_001__A416", "flip_090_003__A304",
    "side_roll_R_003__A304", "fire_in_the_hole_180_R_002__A473",
    "fire_in_the_hole_360_R_003__A472", "broad_jump_002__A360",
    "stand_up_lying_R_002__A473", "crossed_legs_R_001__A428",
    "dance_hiphop_shuffle_square_R_fast_002__A318",
    "dance_basic_chaines_180_R_001__A306",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, allow_nan=False), flush=True)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-xml", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=Path('/data/liuminghao/assets/soma-retargeter/robotics/agibot/x2/desc/agibot_x2ultra.urdf'))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--only", choices=MOTIONS)
    parser.add_argument("--source-root", type=Path, default=Path("/data/liuminghao/datasets/bones-seed/converted/agibot-x2ultra/per-sequence-optimized"))
    parser.add_argument("--nconmax", type=int, default=1024)
    parser.add_argument("--njmax", type=int, default=4096)
    return parser.parse_args()


def robot_spec_without_keys(path):
    import mujoco
    spec = mujoco.MjSpec.from_file(str(path))
    for key in list(spec.keys):
        spec.delete(key)
    spec.nkey = 0
    return spec


def compile_scene(args):
    import mujoco
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sim import MujocoCfg
    from mjlab.terrains import TerrainEntityCfg
    from supermimic.assets.x2_constants import get_x2_robot_cfg

    robot = get_x2_robot_cfg()
    robot.spec_fn = lambda: robot_spec_without_keys(args.base_xml)
    scene = Scene(SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1, entities={"robot": robot}), device="cpu")
    from types import SimpleNamespace
    options = MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20, tolerance=1e-6)
    options.apply(SimpleNamespace(opt=scene.spec.option))
    for geom in scene.spec.geoms:
        if "sole" in geom.name and geom.name.endswith("_collision"):
            geom.friction = [1.0, geom.friction[1], geom.friction[2]]
    model = scene.compile()
    options.apply(model)
    return scene, model


def servo_contract(model):
    import mujoco
    assert model.nq == 38 and model.nv == 37 and model.nu == 31
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
    assert np.allclose(model.actuator_gear[:, 0], 1) and np.allclose(model.actuator_gear[:, 1:], 0)
    assert np.allclose(model.actuator_biasprm[:, 1], -model.actuator_gainprm[:, 0])
    assert np.all(model.actuator_gainprm[:, 0] > 0) and np.all(model.actuator_biasprm[:, 2] < 0)
    assert set(model.actuator_trnid[:, 0]) == set(range(1, 32))
    assert np.all(model.actuator_forcelimited) and np.all(model.actuator_forcerange[:, 1] > 0)
    assert np.array_equal(model.jnt_qposadr[1:], np.arange(7, 38))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in model.actuator_trnid[:, 0]]
    return {"names": names, "kp": model.actuator_gainprm[:, 0].tolist(), "kd": (-model.actuator_biasprm[:, 2]).tolist(),
            "gear": model.actuator_gear.tolist(), "force_limited": model.actuator_forcelimited.tolist(),
            "force_range": model.actuator_forcerange.tolist(), "ctrl_limited": model.actuator_ctrllimited.tolist(),
            "ctrl_range": model.actuator_ctrlrange.tolist(), "joint_damping": model.dof_damping.tolist(),
            "armature": model.dof_armature.tolist()}


def verify_joint_ranges(args, model):
    import mujoco
    limits = {}
    for joint in ET.parse(args.urdf).getroot().findall('joint'):
        if joint.get('type') == 'revolute':
            limit = joint.find('limit')
            limits[joint.get('name')] = [float(limit.get('lower')), float(limit.get('upper'))]
    assert len(limits) == 31
    for index in range(1, model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index).removeprefix('robot/')
        assert np.allclose(model.jnt_range[index], limits[name], rtol=5e-6, atol=1e-8), (name, model.jnt_range[index], limits[name])
    return {"urdf": str(args.urdf), "urdf_sha256": digest(args.urdf), "joint_ranges_radians": limits}


def save_model(args, scene, model):
    import mujoco
    directory = args.output / "model"
    directory.mkdir(parents=True, exist_ok=True)
    scene.write(directory / "source-scene")
    exported = mujoco.MjModel.from_xml_path(str(directory / "source-scene/scene.xml"))
    assert exported.nq == model.nq and exported.nu == model.nu
    roundtrip = {}
    for field in ("body_mass", "body_inertia", "actuator_gainprm", "actuator_biasprm", "geom_friction"):
        expected, actual = getattr(model, field), getattr(exported, field)
        roundtrip[field] = float(np.max(np.abs(actual-expected)))
        assert np.allclose(actual, expected, rtol=5e-6, atol=1e-8), (field, roundtrip[field])
    mujoco.mj_saveModel(model, str(directory / "compiled-scene.mjb"))
    contract = servo_contract(model)
    ranges = verify_joint_ranges(args, model)
    record = {"base_xml": str(args.base_xml), "base_xml_sha256": digest(args.base_xml),
              "compiled_xml": str(directory / "source-scene/scene.xml"),
              "compiled_xml_sha256": digest(directory / "source-scene/scene.xml"),
              "compiled_mjb_sha256": digest(directory / "compiled-scene.mjb"), "nq": model.nq,
              "nv": model.nv, "nu": model.nu, "nbody": model.nbody, "ngeom": model.ngeom,
              "nkey": model.nkey, "initial_key_qpos": model.key_qpos.tolist(),
              "nexclude": model.nexclude, "mass_kg": float(model.body_mass.sum()), "servo": contract,
              "joint_range_validation": ranges, "xml_roundtrip_max_abs_errors": roundtrip,
              "xml_precision_note": "MjSpec serializes roughly six significant digits; exact compiled numeric model is the MJB",
              "timestep": model.opt.timestep, "decimation": 4, "control_fps": 50,
              "integrator": int(model.opt.integrator), "solver": int(model.opt.solver),
              "solver_tolerance": model.opt.tolerance,
              "tolerance_note": "CPU and Warp use 1e-6; Warp clamps the training CPU request 1e-8 to its supported minimum",
              "warn_overflow_suppressed": False,
              "iterations": model.opt.iterations, "ls_iterations": model.opt.ls_iterations,
              "gravity": model.opt.gravity.tolist(), "disableflags": int(model.opt.disableflags),
              "nconmax_requested": args.nconmax, "njmax_requested": args.njmax}
    atomic_json(directory / "numeric-model-summary.json", record)
    return record


@dataclass
class Reference:
    name: str
    qpos: np.ndarray
    qvel: np.ndarray
    targets: np.ndarray
    source: Path
    source_sha256: str


def reference(name, args, model):
    import mujoco
    from scipy.spatial.transform import Rotation
    from supermimic.data.motion_convert import load_csv, resample_motion
    from supermimic.robots import get_robot
    names = get_robot("x2").joint_names
    path = args.source_root / (name + ".csv")
    motion = resample_motion(load_csv(path, names), 120.0, 50.0)
    qpos = np.zeros((motion.num_frames, model.nq)); qvel = np.zeros((motion.num_frames, model.nv))
    qpos[:, :3] = motion.root_pos; qpos[:, 3:7] = motion.root_quat
    qvel[:, :3] = motion.root_lin_vel_w
    qvel[:, 3:6] = Rotation.from_quat(motion.root_quat[:, [1, 2, 3, 0]]).inv().apply(motion.root_ang_vel_w)
    for j in range(1, model.njnt):
        label = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j).removeprefix("robot/")
        index = names.index(label)
        qpos[:, model.jnt_qposadr[j]] = motion.joint_pos[:, index]
        qvel[:, model.jnt_dofadr[j]] = motion.joint_vel[:, index]
    targets = qpos[:, model.jnt_qposadr[model.actuator_trnid[:, 0]]]
    assert np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(targets).all()
    return Reference(name, qpos, qvel, targets, path, digest(path))


def initial_cpu_state(model, ref):
    import mujoco
    data = mujoco.MjData(model)
    data.qpos[:] = ref.qpos[0]; data.qvel[:] = ref.qvel[0]; data.ctrl[:] = ref.targets[0]
    mujoco.mj_forward(model, data)
    contacts = [{"geom1": int(data.contact[i].geom1), "geom2": int(data.contact[i].geom2),
                 "distance_m": float(data.contact[i].dist)} for i in range(data.ncon)]
    return data, contacts


def snapshot(data):
    return {name: getattr(data, name).numpy()[0].copy() for name in
            ("qpos", "qvel", "qacc", "actuator_force", "qfrc_constraint", "xpos", "nefc", "time", "overflow")}


def contact_summary(data, model):
    count = int(data.nacon.numpy()[0])
    if count == 0:
        return {"count": 0, "self": 0, "ground": 0, "min_distance_m": 0.0}
    pairs = data.contact.geom.numpy()[:min(count, data.naconmax)]
    bodies = model.geom_bodyid[pairs]
    distances = data.contact.dist.numpy()[:len(pairs)]
    return {"count": count, "self": int(np.all(bodies > 0, axis=1).sum()),
            "ground": int(np.any(bodies == 0, axis=1).sum()), "min_distance_m": float(distances.min())}


def state_finite(state):
    finite = all(np.isfinite(state[name]).all() for name in ("qpos", "qvel", "qacc", "actuator_force", "qfrc_constraint", "xpos"))
    return finite and abs(np.linalg.norm(state['qpos'][3:7])-1.0) < 1e-3


def array_records(states, contacts, ref):
    count = len(states)
    arrays = {name: np.asarray([s[name] for s in states]) for name in states[0]}
    arrays.update(time_s=arrays['time'].copy(), force_sample_time_s=np.maximum(arrays['time']-.005, 0),
                  reference_qpos=ref.qpos[:count], reference_qvel=ref.qvel[:count],
                  command_joint_targets=ref.targets[:max(0, count-1)],
                  contact_count=np.asarray([c['count'] for c in contacts]),
                  self_contact_count=np.asarray([c['self'] for c in contacts]),
                  ground_contact_count=np.asarray([c['ground'] for c in contacts]),
                  minimum_contact_distance=np.asarray([c['min_distance_m'] for c in contacts]))
    return arrays


def tracking_summary(arrays, ref, model):
    joint_address = model.jnt_qposadr[1:]
    joint_error = arrays['qpos'][:, joint_address] - arrays['reference_qpos'][:, joint_address]
    root_error = arrays['qpos'][:, :3] - arrays['reference_qpos'][:, :3]
    limits = np.max(np.abs(model.actuator_forcerange), axis=1)
    saturation = np.abs(arrays['actuator_force']) >= limits[None, :] * .98
    return {"recorded_frames": len(arrays['qpos']), "expected_frames": len(ref.qpos),
            "joint_rmse_rad": float(np.sqrt(np.mean(joint_error**2))),
            "root_position_rmse_m": float(np.sqrt(np.mean(root_error**2))),
            "minimum_body_origin_z_m": float(arrays['xpos'][:, 1:, 2].min()),
            "maximum_abs_qvel": float(np.abs(arrays['qvel']).max()),
            "maximum_contacts": int(arrays['contact_count'].max()),
            "maximum_constraints": int(arrays['nefc'].max()),
            "actuator_force_saturation_fraction": float(saturation.mean())}


@contextmanager
def suspended_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled: gc.enable()


def physics_graph(context, data):
    failures = wp.zeros(1, dtype=wp.int32)
    with suspended_gc(), wp.ScopedCapture() as capture:
        for _ in range(4):
            context['mw'].step(context['warp_model'], data)
            wp.launch(collect_nonfinite, context['model'].nq, inputs=[data.qpos, data.qvel, data.qacc, failures])
            wp.launch(collect_overflow, 1, inputs=[data.overflow, failures])
        context['mw'].kinematics(context['warp_model'], data)
    return capture.graph, failures


def record_clip(context, ref):
    wp, mw, model = context['wp'], context['mw'], context['model']
    cpu, initial_contacts = initial_cpu_state(model, ref)
    data = mw.put_data(model, cpu, nworld=1, nconmax=context['args'].nconmax, nccdmax=1024, njmax=context['args'].njmax)
    mw.forward(context['warp_model'], data); wp.synchronize()
    states = [snapshot(data)]; contacts = [contact_summary(data, model)]
    assert state_finite(states[0]), "Nonfinite initialization"
    status = "COMPLETE"; failure = None
    emit("PHYSICS_CLIP_INITIALIZED", name=ref.name, initial_contacts=len(initial_contacts), root_free=True)
    graph, failure_flags = physics_graph(context, data)
    for frame in range(len(ref.qpos)-1):
        data.ctrl.assign(ref.targets[frame:frame+1].astype(np.float32))
        wp.capture_launch(graph); wp.synchronize()
        flags = int(failure_flags.numpy()[0])
        current = snapshot(data)
        if flags or not state_finite(current):
            status = "ENGINE_OVERFLOW" if flags & 8 else "NONFINITE_STATE"
            failure = {"frame": frame+1, "flags": flags, "overflow": data.overflow.numpy().tolist()}
            break
        states.append(current); contacts.append(contact_summary(data, model))
        if frame == 0 or (frame+1) % 100 == 0:
            emit("REAL_WARP_DYNAMICS_PROGRESS", name=ref.name, recorded=len(states), total=len(ref.qpos), root_z=float(current['qpos'][2]))
    captured = CapturedClip(ref, states, contacts, initial_contacts, status, failure)
    return persist_clip(context, captured)


@dataclass
class CapturedClip:
    ref: Reference
    states: list
    contacts: list
    initial_contacts: list
    status: str
    failure: dict | None


def persist_clip(context, captured):
    ref, states, contacts = captured.ref, captured.states, captured.contacts
    initial_contacts, status, failure = captured.initial_contacts, captured.status, captured.failure
    args, model = context['args'], context['model']
    arrays = array_records(states, contacts, ref)
    directory = args.output / 'states'; directory.mkdir(parents=True, exist_ok=True)
    path = directory / (ref.name + '.npz')
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path, **arrays, fps=np.asarray([50.]), joint_names=np.asarray(context['joint_names']),
                        body_names=np.asarray(context['body_names']), actuator_joint_names=np.asarray(context['actuator_joint_names']),
                        valid=np.ones(len(states), dtype=bool), quaternion_order=np.asarray(['wxyz']), engine=np.asarray(['mujoco_warp']))
    record = {"status": status, "failure": failure, "name": ref.name, "source_csv": str(ref.source),
              "source_csv_sha256": ref.source_sha256, "state_file": str(path), "state_sha256": digest(path),
              "model_sha256": context['model_sha256'], "engine": "mujoco_warp", "physics_backend": "mujoco_warp", "physics_dt": .005,
              "control_dt": .02, "substeps": 4, "initialization_only": True, "root_actuation": False,
              "root_actuated": False, "qpos_resets": 1,
              "qpos_overwrites_after_initialization": 0, "autoresets": 0, "domain_randomization": False,
              "body_pose_sample": "Post-integration pure kinematics refresh; qpos/qvel unchanged",
              "force_and_contact_sample": "Last executed physics substep, nominal time_s-0.005 except initial frame",
              "control_semantics": "zero-order-hold reference[k] over [k*dt,(k+1)*dt]; post-state compared to reference[k+1]",
              "initial_contacts": initial_contacts, "metrics": tracking_summary(arrays, ref, model)}
    atomic_json(args.output / 'receipts' / (ref.name+'.json'), record)
    emit('PHYSICS_CLIP_RECORDED', name=ref.name, status=status, **record['metrics'])
    return record


def main():
    args = arguments(); args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo / 'src'))
    import mujoco
    import supermimic
    assert Path(supermimic.__file__).resolve().is_relative_to(args.repo.resolve()), "Wrong source checkout imported"
    base_sha256 = digest(args.base_xml)
    if args.expected_model_sha256:
        assert base_sha256 == args.expected_model_sha256, "Audited model changed before compilation"
    scene, model = compile_scene(args)
    assert digest(args.base_xml) == base_sha256, "Model changed during compilation"
    verify_joint_ranges(args, model)
    description = save_model(args, scene, model)
    emit('COMPILED_ACTUATED_SCENE_READY', nq=model.nq, nv=model.nv, nu=model.nu, model_sha256=description['compiled_mjb_sha256'])
    if args.prepare_only:return
    import warp as wp
    import mujoco_warp as mw
    wp.init(); wp.set_device(args.device)
    warp_model = mw.put_model(model)
    warp_model.opt.warn_overflow = 1
    atomic_json(args.output / 'model/gpu-model-summary.json', {"exclude_signature": warp_model.exclude_signature.numpy().tolist(),
        "geom_contype": warp_model.geom_contype.numpy().tolist(), "geom_conaffinity": warp_model.geom_conaffinity.numpy().tolist(),
        "compiled_cpu_model_sha256": description['compiled_mjb_sha256'], "warn_overflow": int(warp_model.opt.warn_overflow)})
    context = {'args': args, 'wp': wp, 'mw': mw, 'model': model, 'warp_model': warp_model,
               'model_sha256': description['compiled_mjb_sha256'],
               'joint_names': [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1,model.njnt)],
               'body_names': [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, j) or 'world' for j in range(model.nbody)],
               'actuator_joint_names': [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in model.actuator_trnid[:,0]]}
    atomic_json(args.output / 'runtime.json', {"pid": os.getpid(), "engine": "mujoco_warp", "device": args.device,
                "versions": {p: importlib.metadata.version(p) for p in ('mujoco','mujoco-warp','warp-lang','mjlab')},
                "model": description, "started_epoch": time.time()})
    emit('REAL_MUJOCO_WARP_REPLAY_START', pid=os.getpid(), device=args.device, gravity=model.opt.gravity.tolist())
    records = [record_clip(context, reference(name, args, model)) for name in ((args.only,) if args.only else MOTIONS)]
    atomic_json(args.output / 'replay-summary.json', {"recorded":len(records), "complete":sum(r['status']=='COMPLETE' for r in records),"records":records})
    if any(r['status']!='COMPLETE' for r in records):raise SystemExit(2)


if __name__ == '__main__':
    main()
