"""Production regressions for the X2 degree, home-sign and compiled-mesh frame defects."""
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT / 'assets/x2'))
from build_x2_mjcf import build, mesh_vertices_in_body, sole_plate_parameters, validate_home_qpos, validate_joint_ranges
from supermimic.robots import load_robot_model

URDF = Path('/data/liuminghao/assets/soma-retargeter/robotics/agibot/x2/desc/agibot_x2ultra.urdf')
BASELINE = Path('/data/liuminghao/artifacts/supermimic/x2/mjcf-physics-audit/baseline/agibot_x2ultra.xml')


def raw_stl_vertices(path):
    payload = path.read_bytes()
    triangles = int.from_bytes(payload[80:84], 'little')
    if len(payload) != 84 + triangles * 50:
        raise ValueError('Expected the known binary STL fixture')
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attributes', '<u2')])
    return np.frombuffer(payload, dtype=dtype, offset=84)['vertices'].reshape(-1, 3).astype(float)


class X2PhysicsRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ET.parse(URDF).getroot()
        cls.model = load_robot_model('x2')
        cls.baseline = mujoco.MjModel.from_xml_path(str(BASELINE))
        cls.generated_spec = build(URDF, str(URDF.parent))
        cls.generated = cls.generated_spec.compile()

    def test_urdf_radian_ranges_survive_programmatic_compile_and_serialization(self):
        self.assertFalse(self.generated_spec.compiler.degree)
        validate_joint_ranges(self.generated, self.tree.findall('joint'))
        with self.assertRaisesRegex(ValueError, 'radian range changed'):
            validate_joint_ranges(self.baseline, self.tree.findall('joint'))
        for joint in self.tree.findall('joint'):
            if joint.get('type') != 'revolute':
                continue
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint.get('name'))
            limit = joint.find('limit')
            np.testing.assert_allclose(self.model.jnt_range[jid], [float(limit.get('lower')), float(limit.get('upper'))], atol=1e-5, rtol=0)
            self.assertTrue(self.model.jnt_limited[jid])

    def test_x2_home_is_feasible_and_g1_positive_elbow_default_is_rejected(self):
        validate_home_qpos(self.model, self.model.key_qpos[0])
        for side in ('left', 'right'):
            with self.subTest(side=side):
                qpos = self.model.key_qpos[0].copy()
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, side + '_elbow_joint')
                qpos[self.model.jnt_qposadr[jid]] = .6
                with self.assertRaisesRegex(ValueError, 'Home pose outside joint limits'):
                    validate_home_qpos(self.model, qpos)

    def test_compiled_foot_vertices_match_independent_raw_urdf_visual_geometry(self):
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.model.key_qpos[0]
        mujoco.mj_forward(self.model, data)
        for side in ('left', 'right'):
            with self.subTest(side=side):
                link = self.tree.find("link[@name='" + side + "_ankle_roll_link']")
                visual = link.find('visual')
                raw = raw_stl_vertices(URDF.parent / visual.find('geometry/mesh').get('filename'))
                origin = visual.find('origin')
                xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
                expected = raw @ Rotation.from_euler('xyz', rpy).as_matrix().T + xyz
                geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, side + '_ankle_roll_link_visual')
                actual = mesh_vertices_in_body(self.model, data, geom)
                np.testing.assert_allclose(actual.min(0), expected.min(0), atol=2e-6)
                np.testing.assert_allclose(actual.max(0), expected.max(0), atol=2e-6)
                size, pos = sole_plate_parameters(actual)
                self.assertLess(size[2] * 2, np.ptp(expected[:, 2]) / 4)
                self.assertAlmostEqual(pos[2] - size[2], expected[:, 2].min(), delta=2e-6)

    def test_body_local_foot_geometry_is_invariant_to_robot_world_pose(self):
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.model.key_qpos[0]
        mujoco.mj_forward(self.model, data)
        geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'left_ankle_roll_link_visual')
        expected = mesh_vertices_in_body(self.model, data, geom)
        data.qpos[:3] += [.2, -.3, .4]
        data.qpos[3:7] = Rotation.from_euler('xyz', [.2, -.3, .4]).as_quat()[[3, 0, 1, 2]]
        data.qpos[7] += .1
        mujoco.mj_forward(self.model, data)
        np.testing.assert_allclose(mesh_vertices_in_body(self.model, data, geom), expected, atol=1e-12)

    def test_collision_plate_correction_does_not_change_urdf_body_inertials_or_axes(self):
        for field in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'jnt_axis', 'jnt_pos'):
            with self.subTest(field=field):
                np.testing.assert_array_equal(getattr(self.model, field), getattr(self.baseline, field))


if __name__ == '__main__':
    unittest.main()
