#!/usr/bin/env python3
"""OpenVLA -> LIBERO fresh-replay Oracle-CAPE pilot.

This is a pilot, not a paper benchmark. It uses one real OpenVLA checkpoint to
generate legal action-token candidates at a shared LIBERO state, then evaluates
each first-action intervention in a fresh environment followed by frozen,
deterministic OpenVLA continuation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from cape_tta.libero_oracle import goal_predicate_progress, soft_task_potential
from cape_tta.openvla_bridge import (
    generate_action,
    load_openvla,
    prepare_inputs,
    to_libero_action,
)


DUMMY_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


def fix_libero_path() -> None:
    root = os.environ.get("LIBERO_REPO_ROOT", "/tmp/LIBERO")
    shadow = os.path.join(root, "libero")
    sys.path[:] = [
        p for p in sys.path
        if os.path.abspath(p or ".") != os.path.abspath(shadow)
    ]
    if os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)


def load_init_state(suite, task, init_id, get_libero_path):
    import torch
    path = os.path.join(
        get_libero_path("init_states"),
        task.problem_folder,
        task.init_states_file,
    )
    try:
        states = torch.load(path, weights_only=False)
    except TypeError:
        states = torch.load(path)
    return states[init_id]


def settle(env, init_state, steps: int):
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(steps):
        obs, _, done, _ = env.step(DUMMY_ACTION.tolist())
        if done:
            break
    return obs


def evaluate_branch(
    env_factory,
    init_state,
    settle_steps,
    first_action,
    horizon,
    handle,
    task_language,
):
    env = env_factory()
    try:
        obs = settle(env, init_state, settle_steps)
        p0_sparse = goal_predicate_progress(env)
        p0_soft = soft_task_potential(env)
        actions = [np.asarray(first_action, dtype=np.float32).tolist()]

        obs, _, done, _ = env.step(actions[0])
        for _ in range(max(0, horizon - 1)):
            if done:
                break
            raw, _, _ = generate_action(
                handle,
                obs["agentview_image"],
                task_language,
                do_sample=False,
                restrict_to_action_tokens=False,
            )
            act = to_libero_action(raw)
            actions.append(act.tolist())
            obs, _, done, _ = env.step(act.tolist())

        p1_sparse = goal_predicate_progress(env)
        p1_soft = soft_task_potential(env)
        final_state = np.asarray(env.get_sim_state()).copy()
        return {
            "sparse_progress_start": p0_sparse,
            "sparse_progress_end": p1_sparse,
            "sparse_gain": p1_sparse - p0_sparse,
            "soft_progress_start": p0_soft,
            "soft_progress_end": p1_soft,
            "soft_gain": p1_soft - p0_soft,
            "success": bool(done),
            "actions_executed": actions,
            "final_state": final_state,
        }
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="openvla/openvla-7b-finetuned-libero-spatial")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init-state-id", type=int, default=0)
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--settle-steps", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-flash-attention", action="store_true")
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--input-dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--output", default="results/openvla_oracle_pilot.json")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    fix_libero_path()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    init_state = load_init_state(suite, task, args.init_state_id, get_libero_path)
    bddl = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )

    def env_factory():
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=256,
            camera_widths=256,
        )
        env.seed(0)
        return env

    handle = load_openvla(
        args.checkpoint,
        args.suite,
        use_flash_attention=not args.no_flash_attention,
        load_in_8bit=args.load_in_8bit,
        input_dtype=args.input_dtype,
    )

    probe = env_factory()
    try:
        obs = settle(probe, init_state, args.settle_steps)
        raw_image = np.asarray(obs["agentview_image"]).copy()
    finally:
        probe.close()

    # Validate our generation wrapper against the checkpoint's official
    # predict_action path before sampling alternatives.
    custom_base, custom_lp, custom_tokens = generate_action(
        handle,
        raw_image,
        task.language,
        do_sample=False,
        restrict_to_action_tokens=False,
    )
    official_inputs = prepare_inputs(handle, raw_image, task.language)
    official_base = handle.model.predict_action(
        **official_inputs,
        unnorm_key=handle.unnorm_key,
        do_sample=False,
    )
    parity_error = float(np.max(np.abs(custom_base - official_base)))
    if parity_error > 1e-6:
        raise RuntimeError(f"OpenVLA wrapper parity failed: max_abs={parity_error}")

    candidates = [{
        "kind": "greedy_base",
        "raw_action": custom_base,
        "env_action": to_libero_action(custom_base),
        "logprob": custom_lp,
        "tokens": custom_tokens,
    }]

    for k in range(1, args.candidates):
        raw, lp, toks = generate_action(
            handle,
            raw_image,
            task.language,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed + k,
            restrict_to_action_tokens=True,
        )
        candidates.append({
            "kind": "sampled",
            "raw_action": raw,
            "env_action": to_libero_action(raw),
            "logprob": lp,
            "tokens": toks,
        })

    branch_results = []
    for cand in candidates:
        branch_results.append(
            evaluate_branch(
                env_factory,
                init_state,
                args.settle_steps,
                cand["env_action"],
                args.horizon,
                handle,
                task.language,
            )
        )

    # Exact fresh-replay determinism check for the greedy baseline branch.
    base_repeat = evaluate_branch(
        env_factory,
        init_state,
        args.settle_steps,
        candidates[0]["env_action"],
        args.horizon,
        handle,
        task.language,
    )
    state_linf = float(np.max(np.abs(
        branch_results[0]["final_state"] - base_repeat["final_state"]
    )))
    soft_repeat_error = abs(
        branch_results[0]["soft_progress_end"] - base_repeat["soft_progress_end"]
    )

    soft_gains = np.array([x["soft_gain"] for x in branch_results], dtype=float)
    oracle_idx = int(np.argmax(soft_gains))
    payload = {
        "kind": "OpenVLA Oracle-CAPE pilot",
        "scientific_result": False,
        "checkpoint": args.checkpoint,
        "cape_commit": os.environ.get("CAPE_COMMIT"),
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "suite": args.suite,
        "task_id": args.task_id,
        "task_language": task.language,
        "init_state_id": args.init_state_id,
        "candidate_count": args.candidates,
        "horizon": args.horizon,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "load_in_8bit": args.load_in_8bit,
        "input_dtype": args.input_dtype,
        "unnorm_key": handle.unnorm_key,
        "wrapper_parity_max_abs": parity_error,
        "fresh_replay_determinism": {
            "final_state_linf": state_linf,
            "soft_progress_abs": soft_repeat_error,
            "success_consistent": branch_results[0]["success"] == base_repeat["success"],
        },
        "base_soft_gain": float(soft_gains[0]),
        "oracle_best_index": oracle_idx,
        "oracle_best_soft_gain": float(soft_gains[oracle_idx]),
        "oracle_margin_over_base": float(soft_gains[oracle_idx] - soft_gains[0]),
        "candidates": [],
    }

    for cand, result in zip(candidates, branch_results):
        payload["candidates"].append({
            "kind": cand["kind"],
            "raw_action": np.asarray(cand["raw_action"]).tolist(),
            "env_action": np.asarray(cand["env_action"]).tolist(),
            "logprob": cand["logprob"],
            "tokens": cand["tokens"],
            "sparse_gain": result["sparse_gain"],
            "soft_gain": result["soft_gain"],
            "success": result["success"],
            "steps_executed": len(result["actions_executed"]),
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
