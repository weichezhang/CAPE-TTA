#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from cape_tta.libero_oracle import (
    branch_candidates,
    check_branch_determinism,
    goal_predicate_progress,
)


def candidate_chunks(horizon: int, magnitude: float) -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    zero = np.zeros((horizon, 7), dtype=np.float32)
    chunks.append(zero.copy())
    for dim in range(3):
        for sign in (-1.0, 1.0):
            x = zero.copy()
            x[:, dim] = sign * magnitude
            chunks.append(x)
    for grip in (-1.0, 1.0):
        x = zero.copy()
        x[:, -1] = grip
        chunks.append(x)
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init-state-id", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--magnitude", type=float, default=0.15)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--output", default="results/libero_oracle_smoke.json")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    # LIBERO's repository uses a nested namespace layout:
    #   <repo>/libero/libero/__init__.py
    # Its internal modules import "libero.libero", so putting <repo>/libero
    # on sys.path incorrectly shadows the outer namespace with the inner package.
    repo_root = os.environ.get("LIBERO_REPO_ROOT", "/tmp/LIBERO")
    shadow_path = os.path.join(repo_root, "libero")
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != os.path.abspath(shadow_path)]
    if os.path.isdir(repo_root) and repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path

    suite_cls = benchmark.get_benchmark_dict()[args.suite]
    suite = suite_cls()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    init_state = init_states[args.init_state_id]
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=128,
        camera_widths=128,
    )
    env.seed(0)
    env.reset()
    env.set_init_state(init_state)

    chunks = candidate_chunks(args.horizon, args.magnitude)
    det = check_branch_determinism(env, chunks[0], repeats=args.repeats, state_atol=1e-7)
    p0 = goal_predicate_progress(env)
    results = branch_candidates(env, chunks)
    env.close()

    payload = {
        "kind": "LIBERO Oracle-CAPE infrastructure smoke test",
        "scientific_result": False,
        "suite": args.suite,
        "task_id": args.task_id,
        "task_language": task.language,
        "init_state_id": args.init_state_id,
        "initial_progress": p0,
        "determinism": det.to_dict(),
        "candidate_progress_gains": [r.progress_gain for r in results],
        "candidate_success": [r.success for r in results],
        "best_gain": max(r.progress_gain for r in results),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not det.passed:
        raise SystemExit("Branch determinism failed; do not report oracle results.")


if __name__ == "__main__":
    main()
