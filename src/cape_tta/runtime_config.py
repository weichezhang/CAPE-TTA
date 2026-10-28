"""Dependency-light LIBERO config generation and path validation."""
from pathlib import Path


def write_libero_config(root, target):
    root = Path(root).resolve()
    benchmark = root / 'libero' / 'libero'
    paths = {
        'benchmark_root': benchmark,
        'bddl_files': benchmark / 'bddl_files',
        'init_states': benchmark / 'init_files',
        'datasets': root / 'libero' / 'datasets',
        'assets': benchmark / 'assets',
    }
    missing = [str(paths[k]) for k in ('benchmark_root', 'bddl_files', 'init_states', 'assets') if not paths[k].is_dir()]
    if missing:
        raise FileNotFoundError('Missing LIBERO directories: ' + ', '.join(missing))
    paths['datasets'].mkdir(parents=True, exist_ok=True)
    # JSON-quoted YAML scalars also work when paths contain spaces or colons.
    import json
    text = ''.join(f'{key}: {json.dumps(str(value))}\n' for key, value in paths.items())
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')
    return {key: str(value) for key, value in paths.items()}
