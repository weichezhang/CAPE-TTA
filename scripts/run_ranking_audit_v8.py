#!/usr/bin/env python3
"""V8 validation pilot: full outcomes of ALL proposals, bounded branch batches."""
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys, traceback
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--libero-root',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--task-id',type=int,choices=[1,2],default=1)
    p.add_argument('--init-state-id',type=int,choices=[0,1],default=0)
    p.add_argument('--max-new-branches',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.max_new_branches<1:p.error('At least one branch per invocation')
    repo=Path(__file__).resolve().parents[1];out=args.run_dir.resolve()
    if out.exists() and not args.resume:p.error('Existing run directory requires --resume')
    if args.resume and not (out/'audit.json').is_file():p.error('Missing audit.json')
    out.mkdir(parents=True,exist_ok=True)
    config=dict(protocol='v8_validation_v1',suite='libero_spatial',task_id=args.task_id,init_state_id=args.init_state_id,
                seeds=[8,9,10,11],horizons=[12,24,48],budget=220,settle_steps=10,
                checkpoint='openvla/openvla-7b-finetuned-libero-spatial',
                revision='962318cec55ac10993ff0f5f43eda9a270b4c873',
                settings=[dict(name='original',temperature=.7,top_p=.95,top_k=50),dict(name='untruncated_15',temperature=1.5,top_p=1.,top_k=0)])
    result=json.loads((out/'audit.json').read_text()) if args.resume else dict(config=config,status='running',actions={},groups=[],branches={},attempts=[])
    if result['config']!=config:p.error('Resume configuration differs')
    def save():
        temp=out/'audit.tmp';temp.write_text(json.dumps(result,indent=2,allow_nan=False));temp.replace(out/'audit.json')
    started=time.monotonic();attempt={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_branches':0};result['attempts'].append(attempt)
    from probe_openvla_diversity import Tee
    import contextlib
    exitcode=1
    with (out/'run.log').open('a') as log, contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
        try:
            import numpy as np
            import torch
            if not torch.cuda.is_available():raise RuntimeError('No CUDA GPU visible; no experiment executed')
            from cape_tta.runtime_config import write_libero_config
            write_libero_config(args.libero_root,out/'libero-config/config.yaml')
            os.environ.update(LIBERO_REPO_ROOT=str(args.libero_root.resolve()),LIBERO_CONFIG_PATH=str(out/'libero-config'),MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa')
            from run_openvla_oracle_pilot import fix_libero_path
            fix_libero_path()
            from libero.libero import benchmark
            from libero.libero.envs import OffScreenRenderEnv
            from cape_tta.openvla_bridge import load_openvla,generate_action,to_libero_action,prepare_inputs
            from cape_tta.libero_oracle import soft_task_potential
            from cape_tta.ranking_audit import pool_summary,replay_equal
            source_sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
            if subprocess.check_output(['git','-C',str(repo),'diff','--name-only','HEAD'],text=True).strip():raise RuntimeError('Tracked source modified')
            handle=load_openvla(config['checkpoint'],config['suite'],revision=config['revision'],use_flash_attention=False,load_in_8bit=True,input_dtype='fp16')
            modules={}
            for obj in [handle.model,handle.model.config,handle.processor]:
                name=type(obj).__module__;path=Path(sys.modules[name].__file__);modules[name]=hashlib.sha256(path.read_bytes()).hexdigest()
            frozen=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
            runtime=dict(source_sha=source_sha,libero_sha=subprocess.check_output(['git','-C',str(args.libero_root),'rev-parse','HEAD'],text=True).strip(),
                         torch=torch.__version__,gpu=torch.cuda.get_device_name(0),remote_modules=modules,pip_freeze_sha256=hashlib.sha256(frozen.encode()).hexdigest())
            if 'runtime' in result and runtime!=result['runtime']:raise RuntimeError('Runtime/source differs; start a separate run rather than silently resume')
            result['runtime']=runtime;(out/'pip-freeze.txt').write_text(frozen)
            if runtime['libero_sha']!='8f1084e3132a39270c3a13ebe37270a43ece2a01':raise RuntimeError('Unexpected LIBERO revision')
            torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
            suite=benchmark.get_benchmark_dict()[config['suite']]();task=suite.get_task(args.task_id);initial=suite.get_task_init_states(args.task_id)[args.init_state_id]
            bddl=args.libero_root.resolve()/'libero/libero/bddl_files'/task.problem_folder/task.bddl_file
            def factory():
                e=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256);e.seed(0);return e
            def reset(e):
                e.reset();obs=e.set_init_state(initial)
                for _ in range(10):
                    obs,_,done,_=e.step([0,0,0,0,0,0,-1])
                    if done:raise RuntimeError('Success during settling invalidates comparison')
                return obs
            def policy(obs):
                raw,_,_=generate_action(handle,obs['agentview_image'],task.language,do_sample=False,restrict_to_action_tokens=False)
                return to_libero_action(raw).tolist()
            # Regenerate the proposal pool on resume; verify exact identity before reusing outcomes.
            e=factory()
            try:image=reset(e)['agentview_image'].copy()
            finally:e.close()
            actions={};groups=[]
            def register(raw):
                a=to_libero_action(raw).tolist()
                if not all(np.isfinite(a)):raise ValueError('Nonfinite action')
                for key,v in actions.items():
                    if a==v:return key
                key=f'action-{len(actions)}';actions[key]=a;return key
            raw,_,_=generate_action(handle,image,task.language,do_sample=False,restrict_to_action_tokens=False);baseline=register(raw)
            with torch.inference_mode():official=handle.model.predict_action(**prepare_inputs(handle,image,task.language),unnorm_key=handle.unnorm_key,do_sample=False)
            parity=float(np.max(np.abs(raw-official)))
            if not np.isfinite(parity) or parity>1e-6:raise RuntimeError('Decoder parity failed')
            for setting in config['settings']:
                row=dict(name=setting['name'],draws=[])
                for seed in config['seeds']:
                    raw,_,tokens=generate_action(handle,image,task.language,do_sample=True,temperature=setting['temperature'],top_p=setting['top_p'],top_k=setting['top_k'],seed=seed)
                    row['draws'].append(dict(seed=seed,action_id=register(raw),tokens=tokens))
                groups.append(row)
            if result['actions'] and (actions!=result['actions'] or groups!=result['groups']):raise RuntimeError('Regenerated proposal pool differs')
            result.update(actions=actions,groups=groups,baseline_id=baseline,wrapper_parity=parity);save()
            def rollout(first,label):
                np.random.seed(7);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
                env=factory();trajectory=[];potentials={}
                try:
                    obs=reset(env);phi0=float(soft_task_potential(env));done=False
                    for step in range(1,config['budget']+1):
                        action=first if step==1 else policy(obs)
                        obs,_,done,_=env.step(action)
                        trajectory.append(dict(step=step,action=list(action),done=bool(done)))
                        if step in config['horizons'] or done or step==config['budget']:
                            phi=float(soft_task_potential(env))
                            if not np.isfinite(phi):raise RuntimeError('Nonfinite potential')
                            if step in config['horizons']:potentials[str(step)]=phi
                        # A partial branch is saved for diagnostics; it must be replayed from the beginning after interruption.
                        result['partial_branch']=dict(label=label,trajectory=trajectory);save()
                        if step%10==0 or done:print(label,step,done,flush=True)
                        if done:break
                    if done:
                        for h in config['horizons']:
                            if h>step:potentials[str(h)]=phi # Explicit absorbing successful terminal state.
                    return dict(status='complete',success=bool(done),steps_executed=step,trajectory=trajectory,
                                initial_potential=phi0,horizon_potential=potentials,terminal_potential=phi)
                finally:env.close()
            # Baseline, independent replay, then remaining distinct candidates.
            queue=[(baseline,actions[baseline]),('baseline_repeat',actions[baseline])]+[(k,v) for k,v in actions.items() if k!=baseline]
            for label,action in queue:
                if label in result['branches']:continue
                if attempt['new_branches']>=args.max_new_branches:break
                result['status']='running';save()
                result['branches'][label]=rollout(action,label);attempt['new_branches']+=1;result.pop('partial_branch',None);save()
                if baseline in result['branches'] and 'baseline_repeat' in result['branches']:
                    if not replay_equal(result['branches'][baseline],result['branches']['baseline_repeat']):raise RuntimeError('Baseline trajectory/potential replay failed')
                    result['replay_passed']=True
            if baseline in result['branches'] and 'baseline_repeat' in result['branches']:
                if not replay_equal(result['branches'][baseline],result['branches']['baseline_repeat']):raise RuntimeError('Baseline replay failed, including on resume')
                result['replay_passed']=True
            for b in result['branches'].values():
                if baseline in result['branches'] and abs(b['initial_potential']-result['branches'][baseline]['initial_potential'])>1e-8:raise RuntimeError('Unmatched initial potential')
            if all(label in result['branches'] for label,_ in queue):
                result['summaries']=[dict(setting=g['name'],horizon=h,**pool_summary([x['action_id'] for x in g['draws']],result['branches'],baseline,h)) for g in groups for h in config['horizons']]
                result['status']='validation_case_complete'
            else:result['status']='paused_branch_budget'
            exitcode=0
        except BaseException:
            result['status']='failed';attempt['error']=traceback.format_exc();traceback.print_exc()
        finally:
            attempt['elapsed_seconds']=time.monotonic()-started;save()
    sums={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='sha256.json'}
    (out/'sha256.json').write_text(json.dumps(sums,indent=2));print('STATUS',result['status'],'BUNDLE',shutil.make_archive(str(out),'zip',out),flush=True)
    return exitcode

if __name__=='__main__':raise SystemExit(main())
