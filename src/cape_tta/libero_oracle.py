"""LIBERO-only oracle utilities for validating CAPE's counterfactual premise.

This is an evaluation oracle, not the deployable CAPE-TTA method.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass
class BranchResult:
    initial_progress: float
    final_progress: float
    progress_gain: float
    success: bool


@dataclass
class DeterminismResult:
    repeats: int
    max_final_state_linf: float
    max_final_progress_abs: float
    success_consistent: bool
    passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _unwrap(env):
    cur = env
    seen = set()
    while hasattr(cur, "env") and id(cur) not in seen:
        seen.add(id(cur))
        nxt = getattr(cur, "env")
        if nxt is cur:
            break
        cur = nxt
    return cur


def goal_predicate_progress(env) -> float:
    base = _unwrap(env)
    problem = getattr(base, "parsed_problem", None)
    if problem is None or "goal_state" not in problem:
        raise AttributeError("LIBERO base environment does not expose parsed_problem['goal_state']")
    goals = list(problem["goal_state"])
    if not goals:
        return float(bool(base._check_success()))
    values = [bool(base._eval_predicate(state)) for state in goals]
    return float(np.mean(values))



def soft_task_potential(
    env,
    *,
    reach_scale: float = 0.12,
    goal_scale: float = 0.20,
    reach_weight: float = 0.30,
) -> float:
    """Simulation-only continuous task potential for short counterfactual branches.

    Official LIBERO reward remains sparse. This diagnostic potential uses privileged
    simulator geometry and BDDL goal semantics; it must be reported separately from
    benchmark success and must not be described as an official LIBERO reward.
    """
    base = _unwrap(env)
    problem = getattr(base, "parsed_problem", None)
    if problem is None or "goal_state" not in problem:
        raise AttributeError("LIBERO environment does not expose goal_state")

    scores = []
    for goal in list(problem["goal_state"]):
        if bool(base._eval_predicate(goal)):
            scores.append(1.0)
            continue

        pred = str(goal[0]).lower()
        if pred in {"on", "in", "stack"} and len(goal) >= 3:
            obj_name, target_name = goal[1], goal[2]
            states = getattr(base, "object_states_dict", {})
            if obj_name not in states or target_name not in states:
                scores.append(0.0)
                continue

            obj_pos = np.asarray(states[obj_name].get_geom_state()["pos"], dtype=float)
            target_pos = np.asarray(states[target_name].get_geom_state()["pos"], dtype=float)
            goal_dist = float(np.linalg.norm(obj_pos - target_pos))
            goal_score = float(np.exp(-goal_dist / goal_scale))

            reach_score = 0.0
            try:
                obs = base._get_observations(force_update=True)
                eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
                reach_dist = float(np.linalg.norm(eef - obj_pos))
                reach_score = float(np.exp(-reach_dist / reach_scale))
            except Exception:
                pass

            scores.append(reach_weight * reach_score + (1.0 - reach_weight) * goal_score)
        else:
            # Articulated predicates (open/close/turn-on/off) remain binary until
            # a task-specific joint-normalized potential is added and validated.
            scores.append(0.0)

    return float(np.mean(scores)) if scores else float(bool(base._check_success()))

def snapshot_sim_state(env) -> np.ndarray:
    if hasattr(env, "get_sim_state"):
        return np.asarray(env.get_sim_state()).copy()
    return np.asarray(env.sim.get_state().flatten()).copy()


def restore_sim_state(env, state: np.ndarray):
    state = np.asarray(state)
    if hasattr(env, "regenerate_obs_from_state"):
        return env.regenerate_obs_from_state(state)
    if hasattr(env, "set_init_state"):
        try:
            return env.set_init_state(state)
        except Exception:
            pass
    env.sim.reset()
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    base = _unwrap(env)
    if hasattr(base, "_post_process"):
        base._post_process()
    if hasattr(base, "_update_observables"):
        base._update_observables(force=True)
    if hasattr(base, "_get_observations"):
        return base._get_observations()
    return None


def _roll_chunk(env, action_chunk: Iterable[np.ndarray]) -> tuple[BranchResult, np.ndarray]:
    p0 = goal_predicate_progress(env)
    base = _unwrap(env)
    success = bool(base._check_success())
    for action in action_chunk:
        _, _, done, _ = env.step(np.asarray(action, dtype=float).tolist())
        success = bool(done)
        if success:
            break
    p1 = goal_predicate_progress(env)
    return BranchResult(p0, p1, p1 - p0, success), snapshot_sim_state(env)


def evaluate_action_chunk(env, state: np.ndarray, action_chunk: Iterable[np.ndarray]) -> BranchResult:
    restore_sim_state(env, state)
    result, _ = _roll_chunk(env, action_chunk)
    return result


def branch_candidates(env, action_chunks: Iterable[Iterable[np.ndarray]]) -> list[BranchResult]:
    state = snapshot_sim_state(env)
    results: list[BranchResult] = []
    try:
        for chunk in action_chunks:
            results.append(evaluate_action_chunk(env, state, chunk))
    finally:
        restore_sim_state(env, state)
    return results


def check_branch_determinism(
    env,
    action_chunk: Iterable[np.ndarray],
    *,
    repeats: int = 3,
    state_atol: float = 1e-8,
    progress_atol: float = 1e-12,
) -> DeterminismResult:
    if repeats < 2:
        raise ValueError("repeats must be >= 2")
    state0 = snapshot_sim_state(env)
    finals = []
    progresses = []
    successes = []
    try:
        for _ in range(repeats):
            restore_sim_state(env, state0)
            result, final_state = _roll_chunk(env, action_chunk)
            finals.append(final_state)
            progresses.append(result.final_progress)
            successes.append(result.success)
    finally:
        restore_sim_state(env, state0)

    ref = finals[0]
    max_state = max(float(np.max(np.abs(x - ref))) for x in finals[1:])
    p_ref = progresses[0]
    max_prog = max(abs(float(x) - float(p_ref)) for x in progresses[1:])
    succ_consistent = len(set(bool(x) for x in successes)) == 1
    passed = bool(max_state <= state_atol and max_prog <= progress_atol and succ_consistent)
    return DeterminismResult(repeats, max_state, max_prog, succ_consistent, passed)


def replay_from_initial_state(
    env_factory,
    init_state: np.ndarray,
    prefix_actions: Iterable[np.ndarray],
    action_chunk: Iterable[np.ndarray],
) -> tuple[BranchResult, np.ndarray]:
    """Evaluate one candidate in a fresh LIBERO environment.

    Recreating the environment and replaying the factual prefix restores both
    MuJoCo state and robosuite controller state. This is the preferred oracle
    mechanism; flattened MuJoCo state alone is insufficient for exact branching
    under stateful low-level controllers such as OSC.
    """
    env = env_factory()
    try:
        env.reset()
        env.set_init_state(np.asarray(init_state))
        for action in prefix_actions:
            _, _, done, _ = env.step(np.asarray(action, dtype=float).tolist())
            if bool(done):
                break
        result, final_state = _roll_chunk(env, action_chunk)
        return result, final_state
    finally:
        env.close()


def fresh_replay_branch_candidates(
    env_factory,
    init_state: np.ndarray,
    prefix_actions: Iterable[np.ndarray],
    action_chunks: Iterable[Iterable[np.ndarray]],
) -> list[BranchResult]:
    """Branch by reconstructing each candidate from init state + factual prefix."""
    prefix = [np.asarray(a, dtype=float).copy() for a in prefix_actions]
    results: list[BranchResult] = []
    for chunk in action_chunks:
        result, _ = replay_from_initial_state(
            env_factory, init_state, prefix, chunk
        )
        results.append(result)
    return results


def check_fresh_replay_determinism(
    env_factory,
    init_state: np.ndarray,
    prefix_actions: Iterable[np.ndarray],
    action_chunk: Iterable[np.ndarray],
    *,
    repeats: int = 3,
    state_atol: float = 1e-7,
    progress_atol: float = 1e-12,
) -> DeterminismResult:
    """Check determinism using independent fresh environments.

    This intentionally avoids in-place simulator restoration, because MuJoCo
    flattened state does not include robosuite controller memory.
    """
    if repeats < 2:
        raise ValueError("repeats must be >= 2")
    prefix = [np.asarray(a, dtype=float).copy() for a in prefix_actions]
    chunk = [np.asarray(a, dtype=float).copy() for a in action_chunk]
    finals = []
    progresses = []
    successes = []
    for _ in range(repeats):
        result, final_state = replay_from_initial_state(
            env_factory, init_state, prefix, chunk
        )
        finals.append(final_state)
        progresses.append(result.final_progress)
        successes.append(result.success)

    ref = finals[0]
    max_state = max(float(np.max(np.abs(x - ref))) for x in finals[1:])
    p_ref = progresses[0]
    max_prog = max(abs(float(x) - float(p_ref)) for x in progresses[1:])
    succ_consistent = len(set(bool(x) for x in successes)) == 1
    passed = bool(max_state <= state_atol and max_prog <= progress_atol and succ_consistent)
    return DeterminismResult(repeats, max_state, max_prog, succ_consistent, passed)
