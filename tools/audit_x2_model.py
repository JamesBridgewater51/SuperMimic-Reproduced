"""Read-only CPU checks of the isolated X2 model and its generated source geometry."""
from pathlib import Path
import hashlib
import itertools
import json
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path('/data/liuminghao')
PROJECT = ROOT / 'supermimic/repo-x2-physics-audit'
ART = ROOT / 'artifacts/supermimic/x2/mjcf-physics-audit'
URDF = ROOT / 'assets/soma-retargeter/robotics/agibot/x2/desc/agibot_x2ultra.urdf'
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT / 'assets/x2'))
from supermimic.robots import load_robot_model, get_robot
from build_x2_mjcf import mesh_vertices_in_body, sole_plate_parameters


def contacts(mj_model, qpos):
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = qpos
    mujoco.mj_forward(mj_model, mj_data)
    pairs = []
    for contact in mj_data.contact[:mj_data.ncon]:
        names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                 for g in (contact.geom1, contact.geom2)]
        pairs.append({'geoms': names, 'distance_m': float(contact.dist)})
    return mj_data, pairs


def urdf_origin(element):
    origin = element.find('origin')
    if origin is None:
        return np.zeros(3), np.eye(3)
    xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
    rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
    return xyz, Rotation.from_euler('xyz', rpy).as_matrix()


def urdf_poses(tree, joint_values, root_position, root_rotation):
    poses = {'pelvis': (root_position, root_rotation)}
    pending = list(tree.findall('joint'))
    while pending:
        ready = [j for j in pending if j.find('parent').get('link') in poses]
        if not ready:
            raise ValueError('Disconnected URDF joint tree')
        for joint in ready:
            parent = joint.find('parent').get('link')
            p, R = poses[parent]
            offset, fixed_R = urdf_origin(joint)
            axis_R = np.eye(3)
            if joint.get('type') == 'revolute':
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                axis_R = Rotation.from_rotvec(axis / np.linalg.norm(axis) * joint_values[joint.get('name')]).as_matrix()
            poses[joint.find('child').get('link')] = (p + R @ offset, R @ fixed_R @ axis_R)
            pending.remove(joint)
    return poses


def fk_and_mass(mj_model, tree):
    rows = []
    names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, mj_model.njnt)]
    for index, euler in enumerate(([0, 0, 0], [.12, -.15, .3], [-.2, .18, -.4])):
        qpos = mj_model.key_qpos[0].copy()
        qpos[:3] = [.07, -.12, .9]
        xyzw = Rotation.from_euler('xyz', euler).as_quat()
        qpos[3:7] = xyzw[[3, 0, 1, 2]]
        if index:
            qpos[7:] += np.sin(np.arange(31) + index) * .15
        mj_data, _ = contacts(mj_model, qpos)
        values = dict(zip(names, qpos[7:], strict=True))
        expected = urdf_poses(tree, values, qpos[:3], Rotation.from_quat(qpos[[4, 5, 6, 3]]).as_matrix())
        position_errors, rotation_errors = [], []
        for body in range(1, mj_model.nbody):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body)
            position, rotation = expected[name]
            position_errors.append(np.linalg.norm(position - mj_data.xpos[body]))
            rotation_errors.append(Rotation.from_matrix(rotation.T @ mj_data.xmat[body].reshape(3, 3)).magnitude())
        matrix = np.empty((mj_model.nv, mj_model.nv))
        mujoco.mj_fullM(mj_model, mj_data, matrix)
        rows.append({'max_position_error_m': float(max(position_errors)),
                     'max_rotation_error_rad': float(max(rotation_errors)),
                     'mass_matrix_min_eigenvalue': float(np.linalg.eigvalsh(matrix).min()),
                     'qacc_finite': bool(np.isfinite(mj_data.qacc).all())})
    return rows


def sole_geometry(mj_model, mj_data):
    rows = []
    for side in ('left', 'right'):
        name = side + '_ankle_roll_link_visual'
        geom = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        vertices = mesh_vertices_in_body(mj_model, mj_data, geom)
        size, position = sole_plate_parameters(vertices)
        sole = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, side + '_ankle_roll_link_sole_collision')
        actual_size, actual_pos = mj_model.geom_size[sole], mj_model.geom_pos[sole]
        assert np.allclose(size, actual_size, atol=1e-6)
        assert np.allclose(position, actual_pos, atol=1e-6)
        rows.append({'side': side, 'visual_body_frame_min_m': vertices.min(0).tolist(),
                     'visual_body_frame_max_m': vertices.max(0).tolist(), 'sole_half_sizes_m': actual_size.tolist(),
                     'sole_body_frame_pos_m': actual_pos.tolist(),
                     'sole_bottom_error_m': float(abs(actual_pos[2] - actual_size[2] - vertices[:, 2].min()))})
    return rows


def parent_filter_probe(explicit):
    pair = '<contact><pair geom1="ga" geom2="gb"/></contact>' if explicit else ''
    xml = '<mujoco><worldbody><body name="a" pos="0 0 1"><freejoint/><geom name="ga" type="sphere" size=".1"/><body name="b" pos=".05 0 0"><joint type="hinge"/><geom name="gb" type="sphere" size=".1"/></body></body></worldbody>' + pair + '</mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    return len(contacts(model, model.qpos0)[1])


def main():
    model = load_robot_model('x2')
    tree = ET.parse(URDF).getroot()
    baseline_path = ART / 'baseline/agibot_x2ultra.xml'
    baseline = mujoco.MjModel.from_xml_path(str(baseline_path))
    original_xml = ET.fromstring(baseline_path.read_text())
    contact_node = original_xml.find('contact')
    for exclude in list(contact_node.findall('exclude')):
        contact_node.remove(exclude)
    no_exclusions = mujoco.MjModel.from_xml_string(ET.tostring(original_xml, encoding='unicode'))
    base_data, current_pairs = contacts(model, model.key_qpos[0])
    _, legacy_pairs = contacts(no_exclusions, no_exclusions.key_qpos[0])
    sweeps = []
    for joint in range(1, model.njnt):
        address = model.jnt_qposadr[joint]
        for fraction in (-.6, -.3, .3, .6):
            qpos = model.key_qpos[0].copy()
            lo, hi = model.jnt_range[joint]
            qpos[address] = np.clip(qpos[address] + fraction * (hi-lo)/2, lo, hi)
            data, pairs = contacts(model, qpos)
            if pairs:
                sweeps.append({'joint': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint),
                               'qpos_rad': float(qpos[address]), 'contacts': pairs})
            assert np.isfinite(data.qacc).all()
    body_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)}
    omitted = [link.get('name') for link in tree.findall('link') if link.get('name') not in body_names]
    unchanged = {key: bool(np.array_equal(getattr(model, key), getattr(baseline, key)))
                 for key in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'jnt_axis', 'jnt_pos')}
    angles = [j for j in tree.findall('joint') if j.get('type') == 'revolute']
    ranges = [{'joint': j.get('name'), 'error_rad': float(np.max(np.abs(model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.get('name'))] - [float(j.find('limit').get('lower')), float(j.find('limit').get('upper'))])))} for j in angles]
    parser_message = ''
    try:
        mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size="1" fricton="1"/></worldbody></mujoco>')
    except ValueError as error:
        parser_message = str(error)
    assert max(row['error_rad'] for row in ranges) < 1e-5
    assert np.all(model.key_qpos[0, 7:] >= model.jnt_range[1:, 0]) and np.all(model.key_qpos[0, 7:] <= model.jnt_range[1:, 1])
    report = {'status': 'CPU_MODEL_AUDIT_PASS', 'model_sha256': hashlib.sha256(Path(get_robot('x2').xml_path()).read_bytes()).hexdigest(),
              'home_self_contacts': current_pairs, 'unexcluded_legacy_home_contacts': legacy_pairs,
              'parent_filter_default_contacts': parent_filter_probe(False), 'explicit_geom_pair_parent_contacts': parent_filter_probe(True),
              'joint_sweeps': 124, 'sweeps_with_preserved_contacts': sweeps, 'fk_and_mass': fk_and_mass(model, tree),
              'soles': sole_geometry(model, base_data), 'body_joint_inertial_unchanged': unchanged,
              'urdf_joint_range_errors': ranges, 'root_fixed_frames_omitted_by_original_builder': omitted,
              'urdf_inertial_nonzero_rpy': [l.get('name') for l in tree.findall('link') if l.find('inertial/origin') is not None and np.any(np.fromstring(l.find('inertial/origin').get('rpy', '0 0 0'), sep=' ') != 0)],
              'urdf_nonunit_mesh_scales': [m.get('scale') for m in tree.iter('mesh') if not np.allclose(np.fromstring(m.get('scale', '1 1 1'), sep=' '), 1)],
              'unknown_attribute_error': parser_message, 'actuator_parameters_changed': False,
              'hardware_rotor_inertias_and_gains_verified': False}
    assert all(unchanged.values()) and not current_pairs and parser_message
    assert report['parent_filter_default_contacts'] == 0 and report['explicit_geom_pair_parent_contacts'] > 0
    assert all(r['max_position_error_m'] < 1e-5 and r['max_rotation_error_rad'] < 1e-5 and r['mass_matrix_min_eigenvalue'] > 0 for r in report['fk_and_mass'])
    output = ART / 'tests/cpu-model-radians-audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({'status': report['status'], 'legacy_contact_count': len(legacy_pairs),
                      'corrected_home_contact_count': len(current_pairs), 'preserved_contact_sweep_cases': len(sweeps),
                      'max_urdf_fk_position_error_m': max(r['max_position_error_m'] for r in report['fk_and_mass']),
                      'omitted_fixed_frames': omitted, 'output': str(output)}))


if __name__ == '__main__':
    main()
