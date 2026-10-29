"""Host-side V8 orchestration helpers; experiment source remains pinned separately."""
import hashlib,importlib.util,io,json,os
from pathlib import Path,PurePosixPath
import shutil,tempfile,time,zipfile

CASES=[(1,0),(1,1),(2,0),(2,1)]
EXPERIMENT_SHA='998ffd17f9ad344088b8e98b53bbe8da5ff9626c'


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def checkpoint(case,target):
    """Write a self-consistent case ZIP, even while the experiment log is open."""
    case=Path(case);target=Path(target)
    if not (case/'audit.json').is_file():return False
    audit=(case/'audit.json').read_bytes();json.loads(audit)
    payload={'audit.json':audit}
    for rel in ['run.log','pip-freeze.txt','libero-config/config.yaml']:
        p=case/rel
        if p.is_file():payload[rel]=p.read_bytes()
    payload['sha256.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in payload.items()},indent=2).encode()
    target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix('.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in payload.items():z.writestr('case/'+n,b)
    temp.replace(target)
    return True


def restore(archive,root,experiment_source):
    """Accept old one-case ZIP, a checkpoint ZIP, or the new four-case ZIP."""
    root=Path(root);helper=load_module('restore_v8',Path(experiment_source)/'src/cape_tta/audit_bundle.py')
    restored=[]
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or sum(i.file_size for i in z.infolist())>200*1024*1024:raise ValueError('Invalid or oversized archive')
        for i in z.infolist():
            p=PurePosixPath(i.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in i.filename or ((i.external_attr>>16)&0o170000)==0o120000:raise ValueError('Unsafe ZIP entry')
        targets=[('',1,0)] if 'case/audit.json' in names else [(f'task-{t}-init-{s}/',t,s) for t,s in CASES if f'task-{t}-init-{s}/case/audit.json' in names]
        if not targets:raise ValueError('This ZIP contains no V8 case; do not use V3-V7')
        for prefix,t,s in targets:
            target=root/f'task-{t}-init-{s}'/'case'
            if target.exists():
                # Refuse to discard newer local work or silently ignore an ambiguous restore.
                raise ValueError(f'{target} already exists; use the existing runtime directly without restore')
            with tempfile.TemporaryDirectory() as tmp:
                filtered=Path(tmp)/'case.zip'
                with zipfile.ZipFile(filtered,'w',zipfile.ZIP_DEFLATED) as sub:
                    for n in names:
                        if n.startswith(prefix+'case/') and not n.endswith('/'):sub.writestr(n[len(prefix):],z.read(n))
                status=helper.restore_case_bundle(filtered,target,task_id=t,init_state_id=s)
            restored.append(dict(task_id=t,init_state_id=s,status=status))
    return restored


def export_all(root,destination):
    root=Path(root);destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.with_suffix('.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for t,s in CASES:
            case=root/f'task-{t}-init-{s}'/'case'
            if (case/'audit.json').exists():
                with tempfile.TemporaryDirectory() as d:
                    cp=Path(d)/'case.zip';checkpoint(case,cp)
                    with zipfile.ZipFile(cp) as cz:
                        for n in cz.namelist():z.writestr(f'task-{t}-init-{s}/'+n,cz.read(n))
        for name in ['all_status.json','all_launcher.log','validation_summary.json']:
            if (root/name).is_file():z.writestr(name,(root/name).read_bytes())
    temp.replace(destination);return str(destination)


def complete_case(case,t,s,source):
    from math import isfinite
    d=json.loads((Path(case)/'audit.json').read_text())
    if d['status']!='validation_case_complete':return None
    cfg=d['config']
    if (cfg['protocol'],cfg['task_id'],cfg['init_state_id'])!=('v8_validation_v1',t,s):raise ValueError('Wrong case configuration')
    if d['runtime']['source_sha']!=EXPERIMENT_SHA:raise ValueError('Wrong experiment source')
    audit=load_module('ranking_v8',Path(source)/'src/cape_tta/ranking_audit.py')
    branches=d['branches'];base=d['baseline_id']
    if any(k not in branches or branches[k]['status']!='complete' for k in d['actions']):raise ValueError('Incomplete candidate pool')
    if not audit.replay_equal(branches[base],branches['baseline_repeat']):raise ValueError('Baseline replay differs')
    for b in branches.values():
        rows=b['trajectory']
        if len(rows)!=b['steps_executed'] or [x['step'] for x in rows]!=list(range(1,len(rows)+1)):raise ValueError('Invalid trajectory numbering')
        if any(x['done'] for x in rows[:-1]) or bool(rows[-1]['done'])!=b['success']:raise ValueError('Invalid termination')
        if not all(isfinite(v) for x in rows for v in x['action']):raise ValueError('Nonfinite command')
    metrics=[dict(setting=g['name'],horizon=h,**audit.pool_summary([x['action_id'] for x in g['draws']],branches,base,h)) for g in d['groups'] for h in cfg['horizons']]
    if metrics!=d['summaries']:raise ValueError('Stored summaries differ from recomputation')
    return dict(task_id=t,init_state_id=s,unique_actions=len(d['actions']),metrics=metrics)
