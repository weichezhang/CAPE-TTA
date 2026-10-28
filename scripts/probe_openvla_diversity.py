#!/usr/bin/env python3
"""Portable CUDA diversity diagnostic. No training or task-effectiveness claims.

Requires the tested Python 3.10/OpenVLA environment and a local LIBERO checkout.
Always writes a ZIP after argument parsing, even if CUDA/dependencies are missing.
"""
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


GRID = [
    {'name': 'original', 'temperature': 0.7, 'top_p': 0.95, 'top_k': None},
    {'name': 'untruncated_07', 'temperature': 0.7, 'top_p': 1.0, 'top_k': 0},
    {'name': 'untruncated_10', 'temperature': 1.0, 'top_p': 1.0, 'top_k': 0},
    {'name': 'untruncated_15', 'temperature': 1.5, 'top_p': 1.0, 'top_k': 0},
]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            if not getattr(stream, 'closed', False):
                stream.write(text); stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            if not getattr(stream, 'closed', False):
                stream.flush()

    def close(self):
        # Loggers can retain this proxy through interpreter shutdown.
        # It owns neither the console nor the run.log file.
        self.flush()

    def isatty(self):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--libero-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('results/diversity'))
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--init-state-ids', type=int, nargs='+', default=[0])
    parser.add_argument('--seeds', type=int, nargs='+', default=[8, 9, 10, 11])
    parser.add_argument('--checkpoint', default='openvla/openvla-7b-finetuned-libero-spatial')
    parser.add_argument('--checkpoint-revision', default='962318cec55ac10993ff0f5f43eda9a270b4c873')
    args = parser.parse_args()
    if args.task_id < 0 or any(i < 0 for i in args.init_state_ids) or len(set(args.seeds)) != len(args.seeds):
        parser.error('IDs must be nonnegative and seeds unique')
    run_dir = args.output_root.resolve() / ('probe-' + dt.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running', 'scientific_result': False, 'scope': 'development_sampling_diagnostic',
              'grid': GRID, 'seeds': args.seeds, 'task_id': args.task_id,
              'init_state_ids': args.init_state_ids, 'states': []}
    started = time.monotonic()

    def save():
        (run_dir / 'diversity.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')

    with (run_dir / 'run.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError('No local CUDA GPU visible to this Python process')
                result['gpu'] = torch.cuda.get_device_name(0)
                result['torch'] = torch.__version__
                result['cuda'] = torch.version.cuda
                repo = Path(__file__).resolve().parents[1]
                result['cape_commit'] = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
                result['source_modified'] = bool(subprocess.check_output(['git', '-C', str(repo), 'diff', '--name-only', 'HEAD'], text=True).strip())
                result['libero_commit'] = subprocess.check_output(['git', '-C', str(args.libero_root), 'rev-parse', 'HEAD'], text=True).strip()
                result['checkpoint'] = args.checkpoint
                result['checkpoint_revision'] = args.checkpoint_revision
                with (run_dir / 'pip-freeze.txt').open('w') as frozen:
                    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=frozen, stderr=subprocess.STDOUT, check=True)
                from cape_tta.runtime_config import write_libero_config
                config_dir = run_dir / 'libero-config'
                write_libero_config(args.libero_root, config_dir / 'config.yaml')
                os.environ.update(LIBERO_REPO_ROOT=str(args.libero_root.resolve()),
                                  LIBERO_CONFIG_PATH=str(config_dir), MUJOCO_GL='osmesa',
                                  PYOPENGL_PLATFORM='osmesa', TOKENIZERS_PARALLELISM='false')
                from run_openvla_oracle_pilot import fix_libero_path, settle
                fix_libero_path()
                from libero.libero import benchmark
                from libero.libero.envs import OffScreenRenderEnv
                from cape_tta.openvla_bridge import generate_action, load_openvla, to_libero_action
                from cape_tta.sampling_diagnostics import candidate_summary
                suite = benchmark.get_benchmark_dict()['libero_spatial']()
                task = suite.get_task(args.task_id)
                initial_states = suite.get_task_init_states(args.task_id)
                if any(i >= len(initial_states) for i in args.init_state_ids):
                    raise ValueError('init-state ID out of range')
                print('Loading checkpoint once for all grid settings', flush=True)
                handle = load_openvla(args.checkpoint, 'libero_spatial', use_flash_attention=False,
                                     load_in_8bit=True, input_dtype='fp16', revision=args.checkpoint_revision)
                for init_id in args.init_state_ids:
                    env = OffScreenRenderEnv(bddl_file_name=str(args.libero_root.resolve() / 'libero/libero/bddl_files' / task.problem_folder / task.bddl_file),
                                             camera_heights=256, camera_widths=256)
                    try:
                        env.seed(0)
                        obs = settle(env, initial_states[init_id], 10)
                        image = obs['agentview_image'].copy()
                    finally:
                        env.close()
                    raw, lp, tokens = generate_action(handle, image, task.language, do_sample=False, restrict_to_action_tokens=False)
                    state = {'init_state_id': init_id, 'task_language': task.language,
                             'baseline': {'tokens': tokens, 'env_action': to_libero_action(raw).tolist()}, 'settings': []}
                    result['states'].append(state); save()
                    for setting in GRID:
                        print(f'init={init_id} setting={setting["name"]}', flush=True)
                        row = {'setting': setting, 'candidates': []}
                        state['settings'].append(row)
                        for seed in args.seeds:
                            diagnostic = {}
                            action, score_lp, token_ids = generate_action(handle, image, task.language, do_sample=True,
                                temperature=setting['temperature'], top_p=setting['top_p'], top_k=setting['top_k'],
                                seed=seed, diagnostics=diagnostic, restrict_to_action_tokens=True)
                            row['candidates'].append({'seed': seed, 'tokens': token_ids,
                                'raw_action': action.tolist(), 'env_action': to_libero_action(action).tolist(),
                                'processed_score_logprob': score_lp, 'distribution': diagnostic})
                            save()
                        row['summary'] = candidate_summary(row['candidates'], tokens, state['baseline']['env_action'])
                        print(json.dumps(row['summary']), flush=True)
                        save()
                    # Same-seed repeat is a reproducibility check, not a demand for diversity.
                    repeat_setting = GRID[0]
                    again, _, repeat_tokens = generate_action(handle, image, task.language, do_sample=True,
                        temperature=repeat_setting['temperature'], top_p=repeat_setting['top_p'], seed=args.seeds[0],
                        restrict_to_action_tokens=True)
                    state['same_seed_tokens_reproduced'] = repeat_tokens == state['settings'][0]['candidates'][0]['tokens']
                    save()
                result['status'] = 'diagnostic_complete'
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
    bundle = shutil.make_archive(str(run_dir), 'zip', run_dir)
    print('STATUS:', result['status']); print('BUNDLE:', bundle)
    return 0 if result['status'] == 'diagnostic_complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
