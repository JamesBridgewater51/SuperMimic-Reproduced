"""Capacity and sticky-overflow regressions for the corrected X2 physics profile."""
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.envs.warp_safety import raise_on_warp_overflow, validate_x2_constraint_capacity


class X2WarpSafetyTests(unittest.TestCase):
    def test_old_constraint_capacity_is_rejected_for_observed_contact_rich_motion(self):
        with self.assertRaisesRegex(ValueError, 'measured 276 constraint rows'):
            validate_x2_constraint_capacity(250)
        validate_x2_constraint_capacity(512)
        for worlds in (1, 4096):
            with self.subTest(worlds=worlds):
                cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[], robot='x2', num_envs=worlds))
                validate_x2_constraint_capacity(cfg.sim.njmax)
                self.assertGreater(cfg.sim.njmax, 276)
                if worlds == 1:
                    self.assertGreater(cfg.sim.nconmax, 69)

    def test_overflow_is_fatal_and_preserves_world_flags_for_diagnosis(self):
        flags = np.array([0, 4, 0], dtype=np.int32)
        original = flags.copy()
        with self.assertRaisesRegex(RuntimeError, r'worlds=\[1\].*bitmasks=\[4\]'):
            raise_on_warp_overflow(flags)
        np.testing.assert_array_equal(flags, original)
        raise_on_warp_overflow(np.zeros(3, dtype=np.int32))


if __name__ == '__main__':
    unittest.main()
