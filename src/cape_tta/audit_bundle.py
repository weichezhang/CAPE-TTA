"""Restore only a checksummed V8 case directory from an exported launcher ZIP."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


def restore_case_bundle(archive, target, *, task_id, init_state_id):
    target=Path(target)
    if target.exists():raise ValueError('Refuse to overwrite an existing case')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive entries')
        if sum(x.file_size for x in z.infolist())>100*1024*1024:raise ValueError('Bundle exceeds 100 MiB limit')
        for info in z.infolist():
            path=PurePosixPath(info.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in info.filename or ((info.external_attr>>16)&0o170000)==0o120000:
                raise ValueError('Unsafe archive path or symbolic link')
        sums=json.loads(z.read('case/sha256.json'))
        actual={n[len('case/'):] for n in names if n.startswith('case/') and not n.endswith('/') and n!='case/sha256.json'}
        if actual!=set(sums):raise ValueError('Manifest does not cover every case file')
        for name,digest in sums.items():
            if not isinstance(digest,str) or hashlib.sha256(z.read('case/'+name)).hexdigest()!=digest:
                raise ValueError('Checksum mismatch')
        state=json.loads(z.read('case/audit.json'));cfg=state['config']
        if cfg.get('protocol')!='v8_validation_v1' or cfg.get('task_id')!=task_id or cfg.get('init_state_id')!=init_state_id:
            raise ValueError('Wrong protocol, task or initial state')
        target.parent.mkdir(parents=True,exist_ok=True)
        stage=Path(tempfile.mkdtemp(prefix='restore-',dir=target.parent))
        try:
            for name in [*sums,'sha256.json']:
                dest=stage/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read('case/'+name))
            stage.rename(target)
        finally:
            if stage.exists():shutil.rmtree(stage)
    return state['status']
