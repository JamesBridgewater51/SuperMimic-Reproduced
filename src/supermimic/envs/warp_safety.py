"""X2 Warp buffer profile and fail-fast checks; no silent overflow continuation.

Overflow bits are split into two classes (mujoco_warp 3.16 solver semantics):
- fatal buffer/collision overflows mean physics data was truncated: raise;
- solver-budget bits (ITERATIONS | LS_ITERATIONS) mean the Newton solver or its
  line search returned the best iterate after exhausting opt.iterations /
  opt.ls_iterations without meeting tolerance -- a solution-quality condition
  that CPU MuJoCo also produces silently at the same budget. These are counted
  and reported periodically, never raised, so the physics budget stays
  comparable with the G1 baseline while remaining visible.
"""
import numpy as np
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sim import SimulationCfg

# Measured corrected-model stand-up replay: 69 contacts, 276 constraint rows.
# Contacts are a shared pool; the single-world maximum is not a pooled-overflow claim.
X2_TESTED_PEAK_EFC = 276
X2_TESTED_PEAK_CONTACTS = 69
X2_NJMAX = 512
X2_NCONMAX = 128  # was 35 (broken near-rigid model); corrected model measured 69/world.
X2_SINGLE_WORLD_NCONMAX = 128
QUALITY_REPORT_EVERY = 1_000

try:  # pinned mujoco_warp 3.16; keep a numeric fallback if the private path moves
    from mujoco_warp._src.types import OverflowType as _OverflowType

    SOLVER_BUDGET_MASK = int(_OverflowType.ITERATIONS) | int(_OverflowType.LS_ITERATIONS)
except Exception:
    SOLVER_BUDGET_MASK = (1 << 9) | (1 << 10)
FATAL_OVERFLOW_MASK = ~SOLVER_BUDGET_MASK


def validate_x2_constraint_capacity(njmax: int) -> None:
    if not isinstance(njmax, int) or njmax < X2_TESTED_PEAK_EFC:
        raise ValueError(f"X2 njmax={njmax} cannot contain measured {X2_TESTED_PEAK_EFC} constraint rows per world")


def raise_on_warp_overflow(flags: np.ndarray) -> None:
    """Fatal overflow is undefined physics, not an episode failure that can be reset away."""
    bad_worlds = np.flatnonzero(np.asarray(flags) & FATAL_OVERFLOW_MASK)
    if bad_worlds.size:
        first = bad_worlds[:8]
        masks = np.asarray(flags).reshape(-1)[first].tolist()
        raise RuntimeError(f"MuJoCo Warp fatal buffer overflow: worlds={first.tolist()}, bitmasks={masks}; discard this transition and increase the relevant capacity")


class OverflowVisibleSimulationCfg(SimulationCfg):
    def apply_wp_opt(self, wp_opt) -> None:
        super().apply_wp_opt(wp_opt)
        wp_opt.warn_overflow = 1


class OverflowCheckedRlEnv(ManagerBasedRlEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._budget_checks = 0
        self._budget_world_steps = 0

    def assert_physics_buffers(self) -> None:
        # One small device-to-host flag read per control step is deliberate safety overhead.
        flags = self.sim.wp_data.overflow.numpy()
        raise_on_warp_overflow(flags)
        self._budget_checks += 1
        self._budget_world_steps += int(np.count_nonzero(flags & SOLVER_BUDGET_MASK))
        if self._budget_checks >= QUALITY_REPORT_EVERY:
            total = self._budget_checks * flags.size
            share = self._budget_world_steps / total if total else 0.0
            print(f"[warp-safety] solver budget exhausted in {self._budget_world_steps}/{total} "
                  f"world-steps ({share:.2%}) over last {self._budget_checks} checks", flush=True)
            self._budget_checks = 0
            self._budget_world_steps = 0

    def _reset_idx(self, env_ids=None) -> None:
        # Auto-reset clears per-world overflow flags; inspect them first.
        self.assert_physics_buffers()
        super()._reset_idx(env_ids)

    def step(self, action):
        transition = super().step(action)
        self.assert_physics_buffers()
        return transition
