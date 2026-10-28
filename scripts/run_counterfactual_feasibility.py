#!/usr/bin/env python3
"""V5: paired short rollouts over a fixed development grid; no parameter updates."""
import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

from probe_openvla_diversity import GRID, Tee


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--libero-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('results/feasibility'))
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--init-state-ids', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--checkpoint', default='openvla/openvla-7b-finetuned-libero-spatial')
    parser.add_argument('--checkpoint-revision', default='962318cec55ac10993ff0f5f43eda9a270b4c873')
    args = parser.parse_args()
    if args.horizon < 1 or args.task_id < 0 or any(i < 0 for i in args.init_state_ids) or len(set(args.init_state_ids)) != len(args.init_state_ids):
        parser.error('Positive horizon and unique nonnegative initial-state IDs required')
    run_dir = args.output_root.resolve() / ('feasibility-' + dt.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    seeds = [8, 9, 10, 11]
    result = {'status': 'running', 'scientific_result': False, 'scope': 'privileged_development_feasibility',
              'task_id': args.task_id, 'init_state_ids': args.init_state_ids, 'horizon': args.horizon,
              'settle_steps': 10, 'seeds': seeds, 'grid': GRID,
              'eligible_actions_per_setting': 5, 'parameter_updates': 0, 'states': []}

    def save():
        (run_dir / 'feasibility.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')

    with (run_dir / 'run.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                import numpy as np
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError('No CUDA GPU visible')
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                repo = Path(__file__).resolve().parents[1]
                result.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                              checkpoint=args.checkpoint, checkpoint_revision=args.checkpoint_revision)
                for name, directory in [('cape_commit', repo), ('libero_commit', args.libero_root)]:
                    result[name] = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
                result['source_modified'] = bool(subprocess.check_output(['git', '-C', str(repo), 'diff', '--name-only', 'HEAD'], text=True).strip())
                with (run_dir / 'pip-freeze.txt').open('w') as frozen:
                    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=frozen, stderr=subprocess.STDOUT, check=True)
                from cape_tta.runtime_config import write_libero_config
                config = run_dir / 'libero-config'
                write_libero_config(args.libero_root, config / 'config.yaml')
                os.environ.update(LIBERO_REPO_ROOT=str(args.libero_root.resolve()), LIBERO_CONFIG_PATH=str(config),
                                  MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa', TOKENIZERS_PARALLELISM='false')
                from run_openvla_oracle_pilot import fix_libero_path, settle, evaluate_branch
                fix_libero_path()
                from libero.libero import benchmark
                from libero.libero.envs import OffScreenRenderEnv
                from cape_tta.openvla_bridge import generate_action, load_openvla, prepare_inputs, to_libero_action
                from cape_tta.rollout_evidence import summarize_group, replay_summary
                suite = benchmark.get_benchmark_dict()['libero_spatial']()
                task = suite.get_task(args.task_id)
                initial_states = suite.get_task_init_states(args.task_id)
                if any(i >= len(initial_states) for i in args.init_state_ids):
                    raise ValueError('Initial-state ID out of range')
                bddl = args.libero_root.resolve() / 'libero/libero/bddl_files' / task.problem_folder / task.bddl_file

                def env_factory():
                    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
                    env.seed(0)
                    return env

                print('Loading one model for all settings and initial states', flush=True)
                handle = load_openvla(args.checkpoint, 'libero_spatial', use_flash_attention=False,
                                     load_in_8bit=True, input_dtype='fp16', revision=args.checkpoint_revision)
                for init_id in args.init_state_ids:
                    print(f'BEGIN init={init_id}, horizon={args.horizon}', flush=True)
                    state = {'init_state_id': init_id, 'task_language': task.language, 'actions': {},
                             'branches': {}, 'settings': [], 'status': 'running'}
                    result['states'].append(state); save()
                    probe = env_factory()
                    try:
                        obs = settle(probe, initial_states[init_id], 10)
                        image = obs['agentview_image'].copy()
                    finally:
                        probe.close()
                    raw, lp, tokens = generate_action(handle, image, task.language, do_sample=False, restrict_to_action_tokens=False)
                    with torch.inference_mode():
                        official = handle.model.predict_action(**prepare_inputs(handle, image, task.language),
                                                               unnorm_key=handle.unnorm_key, do_sample=False)
                    parity = float(np.max(np.abs(raw - official)))
                    state['wrapper_parity_max_abs'] = parity
                    if not np.isfinite(parity) or parity > 1e-6:
                        raise RuntimeError(f'Wrapper parity failed: {parity}')
                    action_ids = {}

                    def register(action):
                        env_action = to_libero_action(action).tolist()
                        if not all(np.isfinite(env_action)):
                            raise ValueError('Nonfinite candidate action')
                        key = tuple(env_action)
                        if key not in action_ids:
                            action_id = f'action-{len(action_ids)}'
                            action_ids[key] = action_id
                            state['actions'][action_id] = env_action
                        return action_ids[key]

                    baseline_id = register(raw)
                    state['baseline_action_id'] = baseline_id
                    state['baseline_tokens'] = tokens
                    for setting in GRID:
                        row = {'setting': setting, 'samples': [], 'candidate_action_ids': []}
                        state['settings'].append(row)
                        for seed in seeds:
                            raw, logprob, candidate_tokens = generate_action(handle, image, task.language, do_sample=True,
                                temperature=setting['temperature'], top_p=setting['top_p'], top_k=setting['top_k'],
                                seed=seed, restrict_to_action_tokens=True)
                            action_id = register(raw)
                            row['candidate_action_ids'].append(action_id)
                            row['samples'].append({'seed': seed, 'action_id': action_id, 'tokens': candidate_tokens,
                                                   'processed_score_logprob': logprob})
                        save()
                    print(f'init={init_id}: {len(state["actions"])} unique executed actions across all settings', flush=True)

                    def roll(action_id):
                        np.random.seed(7); torch.manual_seed(7); torch.cuda.manual_seed_all(7)
                        branch = evaluate_branch(env_factory, initial_states[init_id], 10, state['actions'][action_id],
                                                 args.horizon, handle, task.language)
                        branch['final_state'] = np.asarray(branch['final_state']).tolist()
                        return branch

                    # Exact duplicates share a deterministic rollout. Preserve all
                    # draws so that equal sampling budgets remain inspectable.
                    for action_id in state['actions']:
                        print(f'init={init_id}: rollout {action_id}', flush=True)
                        state['branches'][action_id] = roll(action_id)
                        save()
                    print(f'init={init_id}: independent baseline replay', flush=True)
                    repeated = roll(baseline_id)
                    state['baseline_repeat'] = repeated
                    state['fresh_replay_determinism'] = replay_summary(state['branches'][baseline_id], repeated)
                    if not state['fresh_replay_determinism']['passed']:
                        raise RuntimeError('Baseline fresh replay failed; comparisons invalid')
                    for row in state['settings']:
                        row['summary'] = summarize_group(row['candidate_action_ids'], state['branches'], baseline_id)
                        print(json.dumps({'init': init_id, 'setting': row['setting']['name'], **row['summary']}), flush=True)
                    state['status'] = 'state_complete'; save()
                result['status'] = 'feasibility_complete'
            except BaseException as exc:
                result['status'] = 'failed'
                result['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
                traceback.print_exc()
            finally:
                result['elapsed_seconds'] = time.monotonic() - started
                save()
    sums = {str(p.relative_to(run_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in run_dir.rglob('*') if p.is_file()}
    (run_dir / 'sha256.json').write_text(json.dumps(sums, indent=2), encoding='utf-8')
    print('STATUS:', result['status'])
    print('BUNDLE:', shutil.make_archive(str(run_dir), 'zip', run_dir))
    return 0 if result['status'] == 'feasibility_complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
