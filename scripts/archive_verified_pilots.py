#!/usr/bin/env python3
"""Re-audit user-supplied REAL bundles; preserve raw JSON and provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from cape_tta.budget_pilot import atomic_json

SOURCES = {
    'v3': ('run-20261028-045620-7e26d300.zip','openvla_oracle_pilot_colab.json'),
    'v4': ('diagnostic-20261028-052518-1db6bcfd.zip','diversity.json'),
    'v5': ('feasibility-20261028-054219-6bd651bb.zip','feasibility.json'),
    'v6-init0': ('full-check-20261028-115543-36875ca2.zip','full_horizon.json'),
    'v6-init12': ('full-check-12-20261028-121920-0e860805.zip','full_horizon.json'),
    'v8-task1-init0': ('task-1-init-0.zip','audit.json'),
}


def main():
    p=argparse.ArgumentParser();p.add_argument('--uploads',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); report=dict(scope='historical_real_results_reaudit_not_new_GPU_experiments',bundles=[])
    for label,(name,target) in SOURCES.items():
        path=args.uploads/name
        with zipfile.ZipFile(path) as z:
            checks=0
            for manifest in [n for n in z.namelist() if n.endswith('sha256.json')]:
                parent=str(Path(manifest).parent)
                for relative,expected in json.loads(z.read(manifest)).items():
                    member=str(Path(parent)/relative)
                    if hashlib.sha256(z.read(member)).hexdigest()!=expected: raise ValueError('Hash mismatch: '+member)
                    checks+=1
            member=next(n for n in z.namelist() if Path(n).name==target)
            raw=z.read(member);folder=args.output/label;folder.mkdir(parents=True,exist_ok=True)
            (folder/target).write_bytes(raw)
            d=json.loads(raw)
            record=dict(label=label,original_archive=name,archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                raw_member=member,raw_sha256=hashlib.sha256(raw).hexdigest(),verified_manifest_entries=checks,
                source_commit=d.get('cape_commit',d.get('runtime',{}).get('source_sha')),status=d.get('status'),
                raw_result_path=str(Path(label)/target),parameter_adaptation_executed=False)
            if label=='v5':
                record['states']=[dict(init_state_id=s['init_state_id'],task_id=d['task_id'],
                    replay=s['fresh_replay_determinism'],settings=[dict(name=g['setting']['name'],summary=g['summary']) for g in s['settings']]) for s in d['states']]
            if label=='v8-task1-init0':
                record['branches']={k:dict(success=v['success'],steps_executed=v['steps_executed'],status=v['status']) for k,v in d['branches'].items()}
                record['replay_passed']=d['replay_passed'];record['candidate_evaluation_complete']=False
            atomic_json(folder/'provenance.json',record);report['bundles'].append(record)
    atomic_json(args.output/'reaudit.json',report)
    print(json.dumps(dict(bundles=len(report['bundles']),verified_entries=sum(r['verified_manifest_entries'] for r in report['bundles']),status='passed')))


if __name__=='__main__':main()
