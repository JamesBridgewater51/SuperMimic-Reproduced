#!/usr/bin/env python3
"""Never-stop 1-GPU dev supervisor for corrected-model X2 SuperMimic.

Inherits the cluster continuation supervisor's contract, adapted to a local process:
- LR horizon per leg: max(H0, ceil(iter / decay)) so the cosine never rebounds;
- effectively unbounded loop cap via total_iterations=1e9 (lr_horizon carries decay);
- checkpoint health gate (wm/policy/critic all finite) before every resume;
- STOP sentinel honored; single-instance flock; bounded failure budget; backoff.
"""
import fcntl
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ISO = Path("/data/liuminghao/supermimic/repo-x2-physics-audit")
VENV = Path("/data/liuminghao/supermimic/repo-x2/.venv")
RUN = "x2_fixed_dev_neverstop"
LIB = "/data/liuminghao/datasets/motion-libraries/supermimic-x2/bones_seed_qd12_compact.pt"
LOGROOT = Path("/data/liuminghao/logs/supermimic/x2")
CKPTROOT = Path("/data/liuminghao/ckpts/supermimic/x2")
CKPT_DIR = CKPTROOT / RUN
SUP = LOGROOT / (RUN + "_supervisor")
MODEL_SHA = "4b5ef94ee6e8b073aed50dd2c5ecb1eb27edf32c5d3a5bac50179f7908d60197"
H0 = 400_000
DECAY = 250_000 / 320_000
TOTAL = 1_000_000_000
MAX_HOURS = 1_000_000_000
CKPT_INTERVAL = 4_000
MIN_LEG_SECONDS = 300
FAILS_MAX = 3
GPU_WAIT_MIB = 60_000


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    with (SUP / "supervisor.log").open("a") as stream:
        stream.write(f"{now()} {message}\n")


def save_state(**fields) -> None:
    state = {"run": RUN, "library": LIB, "model_base_sha256": MODEL_SHA, "updated_utc": now(), **fields}
    temporary = SUP / ".state.tmp"
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(SUP / "state.json")


def preflight() -> bool:
    code = (
        "import supermimic, hashlib\n"
        "from pathlib import Path\n"
        f"assert {str(ISO)!r} in supermimic.__file__, supermimic.__file__\n"
        "from supermimic.assets.x2_constants import X2_XML\n"
        "assert hashlib.sha256(Path(X2_XML).read_bytes()).hexdigest() == " + repr(MODEL_SHA) + "\n"
        "import mujoco\n"
        "m = mujoco.MjModel.from_xml_path(str(X2_XML))\n"
        "names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]\n"
        "assert abs(m.jnt_range[names.index('left_knee_joint')][1] - 2.408554) < 1e-4\n"
        "from supermimic.envs.tracking_env_cfg import make_env\n"
        "assert 'OverflowCheckedRlEnv' in __import__('inspect').getsource(make_env)\n"
        "print('PREFLIGHT_OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env=child_env(0), cwd=str(ISO), timeout=600)
    log(f"preflight rc={result.returncode} out={result.stdout.strip()[-200:]} err={result.stderr.strip()[-300:]}")
    return result.returncode == 0


def child_env(horizon: int) -> dict:
    return dict(os.environ,
                PYTHONPATH=str(ISO / "src"), PYTHONNOUSERSITE="1",
                CUDA_VISIBLE_DEVICES="0", WORLD_SIZE="1", OMP_NUM_THREADS="16",
                TORCHINDUCTOR_CACHE_DIR="/data/liuminghao/supermimic/inductor-cache")


def latest_ckpt() -> Path | None:
    numbered = sorted(CKPT_DIR.glob("ckpt_0*.pt"))
    if numbered:
        return numbered[-1]
    fallback = CKPT_DIR / "ckpt_latest.pt"
    return fallback if fallback.exists() else None


def ckpt_iteration(ckpt: Path) -> int:
    match = re.fullmatch(r"ckpt_(\d+)\.pt", ckpt.name)
    if match:
        return int(match.group(1))
    import torch
    return int(torch.load(ckpt, map_location="cpu", weights_only=False).get("iteration", -1))


def ckpt_healthy(ckpt: Path) -> bool:
    import torch
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    return all(torch.isfinite(tensor).all() for key in ("wm", "policy", "critic") for tensor in data[key].values())


def horizon_for(iteration: int) -> int:
    return max(H0, math.ceil(iteration / DECAY))


def gpu_clear() -> bool:
    try:
        used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0])
    except Exception:
        return True
    return used < GPU_WAIT_MIB


def stop_requested() -> bool:
    return (SUP / "STOP").exists()


def train_command(ckpt: Path | None, horizon: int) -> list[str]:
    command = [str(VENV / "bin/torchrun"), "--standalone", "--nproc_per_node=1", str(VENV / "bin/sm-train"),
               "--run-name", RUN, "--env.robot", "x2",
               "--data.library", LIB, "--env.compact-library"]
    if ckpt is not None:
        command += ["--resume", str(ckpt)]
    command += ["--schedule.total-iterations", str(TOTAL), "--schedule.max-hours", str(MAX_HOURS),
                "--schedule.lr-horizon", str(horizon), "--schedule.checkpoint-interval", str(CKPT_INTERVAL),
                "--log-root", str(LOGROOT), "--ckpt-root", str(CKPTROOT)]
    return command


def run_leg(index: int, ckpt: Path | None, horizon: int) -> tuple[int, float]:
    command = train_command(ckpt, horizon)
    log(f"leg {index} start resume={ckpt} horizon={horizon} lr_total={TOTAL}")
    (SUP / f"leg_{index:03d}.cmd").write_text(" ".join(command))
    started = time.time()
    with (SUP / f"leg_{index:03d}.log").open("ab") as stream:
        code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                              env=child_env(horizon), cwd=str(ISO)).returncode
    return code, time.time() - started


def main() -> int:
    SUP.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with (SUP / "supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another supervisor holds the lock", file=sys.stderr)
            return 1
        (SUP / "process.json").write_text(json.dumps(
            {"pid": os.getpid(), "started_utc": now(), "run": RUN, "library": LIB,
             "model_base_sha256": MODEL_SHA, "lr_horizon_base": H0, "decay_fraction": DECAY,
             "total_iterations": TOTAL}, indent=2))
        if not preflight():
            save_state(status="HALT_PREFLIGHT")
            return 2
        fails = 0
        for leg in range(10_000):
            if stop_requested():
                save_state(status="STOPPED_BY_SENTINEL", leg=leg)
                log("STOP sentinel found; exiting")
                return 0
            ckpt = latest_ckpt()
            if ckpt is not None and not ckpt_healthy(ckpt):
                save_state(status="HALT_UNHEALTHY_CHECKPOINT", leg=leg, ckpt=str(ckpt))
                log(f"checkpoint not finite: {ckpt}; halting for diagnosis")
                return 3
            iteration = ckpt_iteration(ckpt) if ckpt is not None else 0
            horizon = horizon_for(iteration)
            while not gpu_clear():
                log("GPU busy; waiting 60s")
                time.sleep(60)
                if stop_requested():
                    save_state(status="STOPPED_BY_SENTINEL", leg=leg)
                    return 0
            code, duration = run_leg(leg, ckpt, horizon)
            fails = fails + 1 if duration < MIN_LEG_SECONDS else 0
            save_state(status="RUNNING" if fails < FAILS_MAX else "HALT_FAIL_BUDGET", leg=leg + 1,
                       last_exit_code=code, last_leg_seconds=round(duration, 1), consecutive_short_fails=fails,
                       last_ckpt=str(ckpt) if ckpt else None, last_horizon=horizon)
            log(f"leg {leg} exit={code} duration={duration:.0f}s fails={fails}")
            if fails >= FAILS_MAX:
                return 4
            if stop_requested():
                save_state(status="STOPPED_BY_SENTINEL", leg=leg + 1)
                return 0
            time.sleep(min(60 * 2 ** fails, 600))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
