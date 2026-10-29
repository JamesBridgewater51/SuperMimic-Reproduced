"""X2 Warp buffer profile and fail-fast checks; no silent overflow continuation."""
import numpy as np
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sim import SimulationCfg

# Measured corrected-model stand-up replay: 69 contacts, 276 constraint rows.
# Contacts are a shared pool; the single-world maximum is not a pooled-overflow claim.
X2_TESTED_PEAK_EFC = 276
X2_NJMAX = 512
X2_NCONMAX = 35
X2_SINGLE_WORLD_NCONMAX = 128


def validate_x2_constraint_capacity(njmax: int) -> None:
    if not isinstance(njmax, int) or njmax < X2_TESTED_PEAK_EFC:
        raise ValueError(f"X2 njmax={njmax} cannot contain measured {X2_TESTED_PEAK_EFC} constraint rows per world")


def raise_on_warp_overflow(flags: np.ndarray) -> None:
    """Overflow is undefined physics, not an episode failure that can be reset away."""
    bad_worlds = np.flatnonzero(flags)
    if bad_worlds.size:
        first = bad_worlds[:8]
        masks = np.asarray(flags).reshape(-1)[first].tolist()
        raise RuntimeError(f"MuJoCo Warp buffer overflow: worlds={first.tolist()}, bitmasks={masks}; discard this transition and increase the relevant capacity")


class OverflowVisibleSimulationCfg(SimulationCfg):
    def apply_wp_opt(self, wp_opt) -> None:
        super().apply_wp_opt(wp_opt)
        wp_opt.warn_overflow = 1


class OverflowCheckedRlEnv(ManagerBasedRlEnv):
    def assert_physics_buffers(self) -> None:
        # One small device-to-host flag read per control step is deliberate safety overhead.
        raise_on_warp_overflow(self.sim.wp_data.overflow.numpy())

    def _reset_idx(self, env_ids=None) -> None:
        # Auto-reset clears per-world overflow flags; inspect them first.
        self.assert_physics_buffers()
        super()._reset_idx(env_ids)

    def step(self, action):
        transition = super().step(action)
        self.assert_physics_buffers()
        return transition
