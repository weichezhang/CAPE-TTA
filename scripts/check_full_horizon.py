#!/usr/bin/env python3
"""V6: full-horizon check of V5-selected interventions on development states."""
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

from probe_openvla_diversity import Tee


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--libero-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('results/full_horizon'))
    parser.add_argument('--init-state-ids', type=int, nargs='+', default=[0])
    args = parser.parse_args()
    if len(set(args.init_state_ids)) != len(args.init_state_ids):
        parser.error('Initial states must be unique')
    repo = Path(__file__).resolve().parents[1]
    selection_file = repo / 'configs/development/v5_selected_interventions.json'
    selected = json.loads(selection_file.read_text())
    choices = {row['init_state_id']: row for row in selected['states']}
    if any(i not in choices for i in args.init_state_ids):
        parser.error('Only V5 development states 0, 1, 2 are available')
    run_dir = args.output_root.resolve() / ('full-' + dt.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running', 'scientific_result': False,
              'scope': 'development_full_horizon_surrogate_check', 'max_steps': 220, 'settle_steps': 10,
              'init_state_ids': args.init_state_ids, 'selection': selected, 'parameter_updates': 0,
              'warning': 'V5-informed privileged intervention, not a deployable policy or held-out test', 'states': []}
    started = time.monotonic()

    def save():
        (run_dir / 'full_horizon.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')

    with (run_dir / 'run.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                import numpy as np
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError('No CUDA GPU visible')
                result.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda)
                result['cape_commit'] = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
                result['libero_commit'] = subprocess.check_output(['git', '-C', str(args.libero_root), 'rev-parse', 'HEAD'], text=True).strip()
                result['source_modified'] = bool(subprocess.check_output(['git', '-C', str(repo), 'diff', '--name-only', 'HEAD'], text=True).strip())
                with (run_dir / 'pip-freeze.txt').open('w') as frozen:
                    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=frozen, stderr=subprocess.STDOUT, check=True)
                from cape_tta.runtime_config import write_libero_config
                config = run_dir / 'libero-config'
                write_libero_config(args.libero_root, config / 'config.yaml')
                os.environ.update(LIBERO_REPO_ROOT=str(args.libero_root.resolve()), LIBERO_CONFIG_PATH=str(config),
                                  MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa', TOKENIZERS_PARALLELISM='false')
                from run_openvla_oracle_pilot import fix_libero_path, settle
                fix_libero_path()
                from libero.libero import benchmark
                from libero.libero.envs import OffScreenRenderEnv
                from cape_tta.openvla_bridge import generate_action, load_openvla, prepare_inputs, to_libero_action
                from cape_tta.full_episode import roll_episode
                suite = benchmark.get_benchmark_dict()[selected['suite']]()
                task = suite.get_task(selected['task_id'])
                initial_states = suite.get_task_init_states(selected['task_id'])
                bddl = args.libero_root.resolve() / 'libero/libero/bddl_files' / task.problem_folder / task.bddl_file

                def env_factory():
                    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
                    env.seed(0)
                    return env

                torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
                handle = load_openvla(selected['checkpoint'], selected['suite'], use_flash_attention=False,
                                     load_in_8bit=True, input_dtype='fp16', revision=selected['checkpoint_revision'])
                result['remote_code_modules'] = []
                for obj in (handle.model, handle.model.config, handle.processor):
                    name = type(obj).__module__
                    source = getattr(sys.modules.get(name), '__file__', None)
                    result['remote_code_modules'].append({'module': name,
                        'sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest() if source and Path(source).is_file() else None})

                def policy(obs):
                    raw, _, _ = generate_action(handle, obs['agentview_image'], task.language,
                                                do_sample=False, restrict_to_action_tokens=False)
                    return to_libero_action(raw).tolist()

                for init_id in args.init_state_ids:
                    record = {'init_state_id': init_id, 'branches': {}, 'status': 'running'}
                    result['states'].append(record); save()
                    probe = env_factory()
                    try:
                        obs = settle(probe, initial_states[init_id], 10)
                        image = obs['agentview_image'].copy()
                    finally:
                        probe.close()
                    raw, _, _ = generate_action(handle, image, task.language, do_sample=False, restrict_to_action_tokens=False)
                    with torch.inference_mode():
                        official = handle.model.predict_action(**prepare_inputs(handle, image, task.language),
                                                               unnorm_key=handle.unnorm_key, do_sample=False)
                    parity = float(np.max(np.abs(raw - official)))
                    baseline_match = float(np.max(np.abs(to_libero_action(raw) - np.asarray(choices[init_id]['baseline_env_action']))))
                    record.update(wrapper_parity_max_abs=parity, v5_baseline_action_max_abs=baseline_match)
                    if not np.isfinite(parity + baseline_match) or parity > 1e-6 or baseline_match > 1e-8:
                        raise RuntimeError('Wrapper parity or V5 baseline reproduction failed')
                    for name, first_action in [('frozen_baseline', choices[init_id]['baseline_env_action']),
                                               ('v5_privileged_first_action', choices[init_id]['selected_env_action'])]:
                        np.random.seed(7); torch.manual_seed(7); torch.cuda.manual_seed_all(7)
                        branch = {'status': 'running', 'trajectory': []}
                        record['branches'][name] = branch; save()

                        def progress(row):
                            branch['trajectory'].append(row)
                            save()
                            if row['step'] % 10 == 0 or row['done']:
                                print(f'init={init_id} {name}: step={row["step"]}/220 done={row["done"]}', flush=True)

                        print(f'BEGIN init={init_id} {name}', flush=True)
                        branch.update(roll_episode(env_factory, initial_states[init_id], first_action, policy,
                                                   max_steps=220, settle_steps=10, on_step=progress))
                        branch['status'] = 'branch_complete'; save()
                    record['status'] = 'state_complete'; save()
                result['status'] = 'full_horizon_complete'
            except BaseException as exc:
                result['status'] = 'failed'
                result['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
                traceback.print_exc()
            finally:
                result['elapsed_seconds'] = time.monotonic() - started
                save()
    sums = {str(p.relative_to(run_dir)): hashlib.sha256(p.read_bytes()).hexdigest() for p in run_dir.rglob('*') if p.is_file()}
    (run_dir / 'sha256.json').write_text(json.dumps(sums, indent=2), encoding='utf-8')
    print('STATUS:', result['status']); print('BUNDLE:', shutil.make_archive(str(run_dir), 'zip', run_dir))
    return 0 if result['status'] == 'full_horizon_complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
