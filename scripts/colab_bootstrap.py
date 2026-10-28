#!/usr/bin/env python3
"""Run the pilot with fail-fast stages and durable, inspectable run artifacts.

This script is executed by the host Colab Python; model code uses isolated 3.10.
"""
import argparse
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

LIBERO_SHA = '8f1084e3132a39270c3a13ebe37270a43ece2a01'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'scope': 'infrastructure_pilot', 'scientific_result': False,
                'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'stage': 'startup'}
    log = (run_dir / 'run.log').open('w', encoding='utf-8')
    started = time.monotonic()

    def record():
        (run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    def run(command, *, env=None):
        print('$ ' + ' '.join(map(str, command)), flush=True)
        log.write('$ ' + ' '.join(map(str, command)) + '\n'); log.flush()
        tail = []
        with subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1, env=env) as proc:
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line); log.flush()
                tail.append(line); tail = tail[-200:]
            status = proc.wait()
        if status:
            raise subprocess.CalledProcessError(status, command, output=''.join(tail))
        return ''.join(tail)

    def stage(name):
        manifest['stage'] = name; record()
        print('\n=== ' + name + ' ===', flush=True)

    try:
        stage('GPU check')
        smi = shutil.which('nvidia-smi')
        if smi is None:
            raise RuntimeError('Choose Runtime → Change runtime type → T4 GPU and reconnect.')
        manifest['gpu'] = run([smi, '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader']).strip()
        manifest['cape_commit'] = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        manifest['host_python'] = sys.version
        stage('System dependencies')
        run(['apt-get', 'update', '-qq'])
        install_env = dict(os.environ, DEBIAN_FRONTEND='noninteractive')
        run(['apt-get', 'install', '-y', '-qq', 'curl', 'git', 'ffmpeg', 'bzip2', 'libosmesa6-dev',
             'libgl1', 'libglx-mesa0', 'libglfw3', 'build-essential'], env=install_env)
        stage('Isolated Python 3.10')
        mamba = Path('/content/cape-tools/bin/micromamba')
        if not mamba.is_file():
            archive = Path('/content/cape-micromamba.tar.bz2')
            run(['curl', '-fLsS', '--retry', '3', 'https://micro.mamba.pm/api/micromamba/linux-64/latest', '-o', archive])
            mamba.parent.mkdir(parents=True, exist_ok=True)
            run(['tar', '-xjf', archive, '-C', mamba.parent.parent, 'bin/micromamba'])
        env_dir = Path('/content/cape-env-v3')
        py = env_dir / 'bin/python'
        if not py.is_file():
            run([mamba, 'create', '-y', '-p', env_dir, '-c', 'conda-forge', 'python=3.10', 'pip'])
        run([py, '-c', "import sys; assert sys.version_info[:2] == (3, 10)"])
        stage('Python dependencies')
        run([py, '-m', 'pip', 'install', 'pip==24.0', 'setuptools==69.5.1', 'wheel==0.43.0'])
        run([py, '-m', 'pip', 'install', '--index-url', 'https://download.pytorch.org/whl/cu121',
             'torch==2.2.0', 'torchvision==0.17.0', 'torchaudio==2.2.0'])
        run([py, '-m', 'pip', 'install', '-r', repo / 'requirements/openvla_colab.txt'])
        stage('LIBERO checkout and configuration')
        libero = Path('/content/LIBERO-' + LIBERO_SHA[:12])
        if not libero.exists():
            run(['git', 'clone', 'https://github.com/Lifelong-Robot-Learning/LIBERO.git', libero])
        actual = subprocess.check_output(['git', '-C', str(libero), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != LIBERO_SHA:
            dirty = subprocess.check_output(['git', '-C', str(libero), 'status', '--porcelain'], text=True)
            if dirty.strip():
                raise RuntimeError('Refusing to change a modified LIBERO checkout: ' + str(libero))
            run(['git', '-C', libero, 'checkout', '--detach', LIBERO_SHA])
        manifest['libero_commit'] = LIBERO_SHA
        run([py, '-m', 'pip', 'install', '--no-build-isolation', '--no-deps', '-e', repo, '-e', libero])
        # Load by path: host Python need not import torch or the package __init__.
        import importlib.util
        spec = importlib.util.spec_from_file_location('runtime_config', repo / 'src/cape_tta/runtime_config.py')
        config = importlib.util.module_from_spec(spec); spec.loader.exec_module(config)
        config_dir = run_dir / 'libero-config'
        paths = config.write_libero_config(libero, config_dir / 'config.yaml')
        manifest['libero_paths'] = paths
        env = dict(os.environ, MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa',
                   LIBERO_CONFIG_PATH=str(config_dir), LIBERO_REPO_ROOT=str(libero),
                   CAPE_COMMIT=manifest['cape_commit'], TOKENIZERS_PARALLELISM='false',
                   TF_CPP_MIN_LOG_LEVEL='2', HF_HOME='/content/cape-hf-cache', PYTHONUNBUFFERED='1')
        # Save complete dependency versions, not just the tail of pip output.
        with (run_dir / 'pip-freeze.txt').open('w') as frozen:
            subprocess.run([py, '-m', 'pip', 'freeze'], stdout=frozen, stderr=subprocess.STDOUT, check=True)
        run([py, '-m', 'pip', 'check'])
        stage('Import and configuration preflight')
        code = ('import os, yaml, torch, mujoco, transformers; '
                'from libero.libero import get_libero_path; '
                'from libero.libero.envs import OffScreenRenderEnv; '
                'assert torch.cuda.is_available(); '
                'print("GPU", torch.cuda.get_device_name(0)); '
                'print("MuJoCo", mujoco.__version__); '
                'print("BDDL", get_libero_path("bddl_files"))')
        run([py, '-u', '-c', code], env=env)
        stage('OpenVLA Oracle pilot')
        output = run_dir / 'openvla_oracle_pilot_colab.json'
        run([py, '-u', repo / 'scripts/run_openvla_oracle_pilot.py', '--checkpoint',
             'openvla/openvla-7b-finetuned-libero-spatial', '--suite', 'libero_spatial',
             '--task-id', '0', '--init-state-id', '0', '--candidates', '4', '--horizon', '12',
             '--settle-steps', '10', '--temperature', '0.7', '--top-p', '0.95', '--seed', '7',
             '--no-flash-attention', '--load-in-8bit', '--input-dtype', 'fp16', '--output', output], env=env)
        stage('Evidence validation')
        spec = importlib.util.spec_from_file_location('pilot_validation', repo / 'src/cape_tta/pilot_validation.py')
        validator = importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)
        validation = validator.validate_pilot(json.loads(output.read_text()))
        (run_dir / 'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
        if not validation['passed']:
            raise RuntimeError('Pilot evidence validation failed: ' + '; '.join(validation['errors']))
        manifest['status'] = 'pilot_validated'
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = {'type': type(exc).__name__, 'message': str(exc),
                             'command_output_tail': getattr(exc, 'output', None),
                             'traceback': traceback.format_exc()}
        traceback.print_exc()
    finally:
        manifest['elapsed_seconds'] = time.monotonic() - started
        record(); log.close()
        # Manifest is included; only the checksum file itself is excluded.
        sums = {str(p.relative_to(run_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in run_dir.rglob('*') if p.is_file() and p.name != 'sha256.json'}
        (run_dir / 'sha256.json').write_text(json.dumps(sums, indent=2), encoding='utf-8')
        bundle = shutil.make_archive(str(run_dir), 'zip', run_dir)
        print('RUN_STATUS:', manifest['status'], flush=True)
        print('BUNDLE:', bundle, flush=True)
    return 0 if manifest['status'] == 'pilot_validated' else 1


if __name__ == '__main__':
    raise SystemExit(main())
