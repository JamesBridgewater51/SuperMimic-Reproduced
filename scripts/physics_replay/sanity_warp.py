"""Bounded gravity-off joint mobility and CPU/Warp parity on the final actuated MJB."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import mujoco
import mujoco_warp as mw
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_warp_replay import atomic_json, digest, physics_graph, servo_contract


def initial_state(model):
    data = mujoco.MjData(model)
    assert model.nkey == 1
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[2] += 1.0
    data.qvel[:] = 0
    for j in range(1, model.njnt):
        assert model.jnt_range[j, 0]-1e-6 <= data.qpos[model.jnt_qposadr[j]] <= model.jnt_range[j, 1]+1e-6
    data.ctrl[:] = data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]]
    mujoco.mj_forward(model, data)
    return data


def targets(model, data):
    command = data.ctrl.copy()
    changed = []
    for name, delta in [('robot/left_wrist_roll_joint', .15), ('robot/left_knee_joint', .10)]:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator = int(np.flatnonzero(model.actuator_trnid[:, 0] == joint)[0])
        desired = data.qpos[model.jnt_qposadr[joint]] + delta
        assert model.jnt_range[joint, 0] < desired < model.jnt_range[joint, 1]
        command[actuator] = desired
        changed.append({'joint': name, 'qpos_index': int(model.jnt_qposadr[joint]), 'actuator_index': actuator,
                        'initial_rad': float(data.qpos[model.jnt_qposadr[joint]]), 'desired_rad': float(desired)})
    return command, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_binary_path(str(args.model))
    servo_contract(model)
    model.opt.gravity[:] = 0
    cpu = initial_state(model)
    command, changed = targets(model, cpu)
    initial = cpu.qpos.copy()
    wp.init(); wp.set_device('cuda:0')
    gpu_model = mw.put_model(model); gpu_model.opt.warn_overflow = 1
    gpu = mw.put_data(model, cpu, nworld=1, nconmax=1024, nccdmax=1024, njmax=4096)
    context = {'wp': wp, 'mw': mw, 'model': model, 'warp_model': gpu_model}
    print(json.dumps({'event': 'REAL_WARP_MOBILITY_START', 'pid': os.getpid(), 'model_sha256': digest(args.model),
                      'gravity': model.opt.gravity.tolist(), 'initialization_z_lift_m': 1.0, 'targets': changed}), flush=True)
    graph, failures = physics_graph(context, gpu)
    cpu_positions, gpu_positions, torques, contacts = [], [], [], []
    for frame in range(25):
        cpu.ctrl[:] = command
        gpu.ctrl.assign(command[None].astype(np.float32))
        for _ in range(4): mujoco.mj_step(model, cpu)
        wp.capture_launch(graph); wp.synchronize()
        actual = gpu.qpos.numpy()[0].copy()
        assert np.isfinite(actual).all() and not failures.numpy().any()
        cpu_positions.append(cpu.qpos.copy()); gpu_positions.append(actual)
        torques.append(gpu.actuator_force.numpy()[0].copy()); contacts.append(int(gpu.nacon.numpy()[0]))
        if frame in (0, 24):print(json.dumps({'event': 'REAL_WARP_MOBILITY_PROGRESS', 'frame': frame+1,
            'time_s': float(gpu.time.numpy()[0]), 'joint_positions_rad': {x['joint']: float(actual[x['qpos_index']]) for x in changed}}), flush=True)
    cpu_positions, gpu_positions = np.asarray(cpu_positions), np.asarray(gpu_positions)
    max_difference = float(np.max(np.abs(cpu_positions-gpu_positions)))
    for changed_joint in changed:
        slot = changed_joint['qpos_index']
        changed_joint['actual_final_rad'] = float(gpu_positions[-1, slot])
        changed_joint['actual_delta_rad'] = float(gpu_positions[-1, slot] - initial[slot])
        changed_joint['peak_abs_torque_nm'] = float(np.abs(np.asarray(torques)[:, changed_joint['actuator_index']]).max())
        assert changed_joint['actual_delta_rad'] > .02, changed_joint
    assert max_difference < .001, max_difference
    np.savez_compressed(args.output/'mobility-trace.npz', actual_qpos=gpu_positions, cpu_qpos=cpu_positions,
                        torques=np.asarray(torques), contacts=np.asarray(contacts), targets=command)
    report = {'status': 'PASS', 'engine': 'mujoco_warp', 'model_sha256': digest(args.model), 'frames': 25,
              'physics_dt': .005, 'decimation': 4, 'gravity_off': True, 'diagnostic_z_lift_m': 1.0,
              'root_actuated': False, 'qpos_resets': 1, 'post_init_qpos_overwrites': 0,
              'joint_response': changed, 'cpu_warp_max_qpos_component_difference': max_difference,
              'maximum_contacts': max(contacts), 'overflow_flags': failures.numpy().tolist()}
    atomic_json(args.output/'mobility-cpu-warp.json', report)
    print(json.dumps({'event': 'WARP_MOBILITY_AND_CPU_PARITY_PASS', **report}), flush=True)


if __name__ == '__main__':main()
