#!/usr/bin/env python3
"""V9: one model load, factual action cache, optional actual short-horizon LoRA."""
import argparse
import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from cape_tta.budget_pilot import atomic_json, digest, load_sealed, seal

CHECKPOINT = 'openvla/openvla-7b-finetuned-libero-spatial'
REVISION = '962318cec55ac10993ff0f5f43eda9a270b4c873'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--libero-root', type=Path, required=True)
    p.add_argument('--bank-root', type=Path, required=True)
    p.add_argument('--adaptation', choices=['none', 'candidate-self-distill'], default='candidate-self-distill')
    args = p.parse_args()
    root = args.bank_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    (root/'banks').mkdir(exist_ok=True); (root/'observations').mkdir(exist_ok=True)
    status = dict(status='running', scope='v9_low_compute_pilot', experiments_completed=[], failure=None)
    started = time.monotonic()
    from probe_openvla_diversity import Tee
    with (root/'gpu.log').open('a') as log, contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
        try:
            import numpy as np
            import torch
            from cape_tta.runtime_config import write_libero_config
            repo = Path(__file__).resolve().parents[1]
            config = json.loads((repo/'configs/pilot/v9_minimal.json').read_text())
            write_libero_config(args.libero_root, root/'libero-config/config.yaml')
            os.environ.update(LIBERO_REPO_ROOT=str(args.libero_root.resolve()), LIBERO_CONFIG_PATH=str(root/'libero-config'),
                              MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa', TOKENIZERS_PARALLELISM='false')
            from run_openvla_oracle_pilot import fix_libero_path, settle
            fix_libero_path()
            from libero.libero import benchmark
            from libero.libero.envs import OffScreenRenderEnv
            from cape_tta.libero_oracle import soft_task_potential, goal_predicate_progress, snapshot_sim_state
            from cape_tta.openvla_bridge import load_openvla, generate_action, to_libero_action, prepare_inputs
            if not torch.cuda.is_available(): raise RuntimeError('Free CUDA runtime required; no paid fallback')
            torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
            torch.manual_seed(config['seed']); np.random.seed(config['seed'])
            def git(path): return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'], text=True).strip()
            if subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip():
                raise RuntimeError('Refusing modified experiment source')
            frozen = subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True)
            (root/'pip-freeze.txt').write_text(frozen)
            runtime = dict(source_commit=git(repo), libero_commit=git(args.libero_root), seed=config['seed'],
                checkpoint=CHECKPOINT, checkpoint_revision=REVISION, quantization='bitsandbytes_8bit', input_dtype='fp16',
                GPU=torch.cuda.get_device_name(0), torch=torch.__version__, CUDA=torch.version.cuda, python=sys.version,
                pip_freeze_sha256=hashlib.sha256(frozen.encode()).hexdigest(), protocol=config)
            identity = root/'runtime.json'
            if identity.exists() and json.loads(identity.read_text()) != runtime:
                raise RuntimeError('Runtime/protocol fingerprint changed; use a separate run directory')
            atomic_json(identity, runtime)
            needed = []
            for t in config['task_ids']:
                for s in config['init_state_ids']:
                    key = f'spatial-task-{t}-init-{s}-seed-{config["seed"]}'
                    path = root/'banks'/f'{key}.json'
                    old = load_sealed(path) if path.exists() else None
                    if old is not None and old['status'] != 'bank_complete': raise RuntimeError('Incomplete bank requires inspection')
                    if old is not None and args.adaptation != 'none' and old.get('adaptation',{}).get('status') == 'failed':
                        raise RuntimeError('Previous adapter failure retained; inspect before retrying or expanding')
                    if old is None or (args.adaptation != 'none' and 'adaptation' not in old): needed.append((t,s,key,path))
            if not needed:
                status['status']='requested_banks_already_complete'; return 0
            handle = load_openvla(CHECKPOINT, 'libero_spatial', revision=REVISION,
                                 load_in_8bit=True, use_flash_attention=False, input_dtype='fp16')
            code_hashes = {}
            for obj in (type(handle.model), type(handle.processor), type(handle.model.config)):
                f = inspect.getfile(obj); code_hashes[obj.__name__] = digest(f)
            for path in (root/'banks').glob('*.json'):
                if path.name.endswith('.sha256.json'): continue
                if load_sealed(path)['model_code_sha256'] != code_hashes:
                    raise RuntimeError('Model remote code changed; cached bank cannot be mixed')
            suite = benchmark.get_benchmark_dict()['libero_spatial']()
            def environment(t):
                task = suite.get_task(t)
                bddl = args.libero_root/'libero/libero/bddl_files'/task.problem_folder/task.bddl_file
                e = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256); e.seed(config['environment_seed']); return e
            def generate(image, language, **kwargs):
                with torch.autocast('cuda', dtype=torch.float16):
                    raw, lp, tokens = generate_action(handle, image, language, **kwargs)
                if len(tokens)!=7 or not all(handle.model.vocab_size-256 <= x < handle.model.vocab_size for x in tokens):
                    raise ValueError('Non-action token emitted')
                action=to_libero_action(raw).tolist()
                if not np.isfinite(action).all(): raise ValueError('Nonfinite action')
                return dict(action=action, tokens=tokens, processed_score_logprob=lp)
            def policy(image, language): return generate(image, language, do_sample=False, restrict_to_action_tokens=False)
            def continuation(env, obs, language, h, first=None):
                actions=[]; trace=[]
                for i in range(h):
                    a=first if i==0 and first is not None else policy(obs['agentview_image'], language)['action']
                    obs, _, done, _=env.step(a); actions.append(a)
                    trace.append(dict(step=i+1, success=bool(done), soft_potential=soft_task_potential(env), sparse_progress=goal_predicate_progress(env)))
                    if done: break
                return actions, trace, snapshot_sim_state(env).tolist()
            # Finish and seal ALL base/candidate banks before attempting any gradients.
            for t,s,key,path in needed:
                if path.exists(): continue
                task=suite.get_task(t); init=np.asarray(suite.get_task_init_states(t)[s]); env=environment(t)
                episode_started=time.monotonic()
                try:
                    obs=settle(env, init, config['settle_steps'])
                    prefix=[[0.,0.,0.,0.,0.,0.,-1.]]*config['settle_steps']
                    prior_prefix=None; prefix_source=None
                    prior_path=repo/'results/verified/v8-task1-init0/audit.json'
                    if (t,s)==(1,0) and prior_path.exists():
                        prior=json.loads(prior_path.read_text())
                        provenance=json.loads((prior_path.parent/'provenance.json').read_text())
                        if digest(prior_path)!=provenance['raw_sha256']: raise ValueError('Historical prefix checksum mismatch')
                        if (prior['replay_passed'] and prior['runtime']['libero_sha']==runtime['libero_commit']
                            and prior['config']['checkpoint']==CHECKPOINT and prior['config']['revision']==REVISION
                            and prior['runtime']['torch']==runtime['torch'] and prior['runtime']['gpu']==runtime['GPU']
                            and set(prior['runtime']['remote_modules'].values())==set(code_hashes.values())):
                            prior_prefix=prior['branches'][prior['baseline_id']]['trajectory'][:config['factual_prefix_steps']]
                            if len(prior_prefix)!=config['factual_prefix_steps'] or any(x['done'] for x in prior_prefix):
                                raise ValueError('Historical factual prefix unavailable')
                            prefix_source=dict(source='verified_v8_recorded_factual_prefix',raw_sha256=digest(prior_path),
                                               source_commit=prior['runtime']['source_sha'],model_calls_saved=len(prior_prefix))
                    for step in range(config['factual_prefix_steps']):
                        a=prior_prefix[step]['action'] if prior_prefix is not None else policy(obs['agentview_image'],task.language)['action']
                        prefix.append(a)
                        obs,_,done,_=env.step(a)
                        if done: raise RuntimeError('Episode solved before declared intervention; do not replace it silently')
                    image=obs['agentview_image'].copy(); image_path=root/'observations'/f'{key}.npy'; np.save(image_path,image)
                    baseline=policy(image,task.language)
                    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
                        official=handle.model.predict_action(**prepare_inputs(handle,image,task.language),unnorm_key=handle.unnorm_key,do_sample=False)
                    error=float(np.max(np.abs(to_libero_action(official)-np.asarray(baseline['action']))))
                    if error>1e-6 or not np.isfinite(error): raise RuntimeError('Official wrapper parity failed')
                    candidates=[]
                    # Four draws from ONE seeded generator stream, not four reported experimental seeds.
                    torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
                    for k in range(config['K']):
                        c=generate(image,task.language,do_sample=True,temperature=config['temperature'],top_p=1.,top_k=0,restrict_to_action_tokens=True)
                        c['candidate_index']=k; candidates.append(c)
                    state=snapshot_sim_state(env).tolist()
                    acts,trace,final=continuation(env,obs,task.language,config['H'],baseline['action'])
                    differences=[float(np.linalg.norm(np.array(c['action'])-baseline['action'])) for c in candidates]
                    features=[baseline['processed_score_logprob'], float(np.mean([c['processed_score_logprob'] for c in candidates])),
                              float(np.std([c['processed_score_logprob'] for c in candidates])),float(np.mean(differences)),float(np.max(differences)),
                              len(set(tuple(c['action']) for c in candidates))/config['K']]
                    bank=dict(status='bank_complete', episode_id=key, task_id=t, init_state_id=s, task_language=task.language,
                        intervention_step=config['factual_prefix_steps'], horizon=config['H'], runtime=runtime, model_code_sha256=code_hashes,
                        bddl_relative=str(Path(task.problem_folder)/task.bddl_file), environment_seed=config['environment_seed'],
                        initial_state=init.tolist(), prefix_actions=prefix, intervention_sim_state=state, observation_sha256=digest(image_path),
                        prefix_cache_provenance=prefix_source,
                        wrapper_parity_max_abs=error, baseline=baseline, candidates=candidates, features=features,
                        frozen_actions=acts, frozen_trace=trace, frozen_final_state=final, base_generation_seconds=time.monotonic()-episode_started)
                    atomic_json(path,bank);seal(path); print('BANK_COMPLETE',key,flush=True)
                finally: env.close()
            if args.adaptation != 'none':
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
                torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
                base=prepare_model_for_kbit_training(handle.model,use_gradient_checkpointing=True)
                model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],lora_dropout=0.,bias='none'))
                handle.model=model
                params={n:v for n,v in model.named_parameters() if v.requires_grad}
                if not params or any('lora_' not in n for n in params): raise RuntimeError('Unexpected trainable parameters')
                snapshot={n:v.detach().cpu().clone() for n,v in params.items()}
                def restore():
                    with torch.no_grad():
                        for n,v in params.items(): v.copy_(snapshot[n].to(v.device))
                    model.eval(); model.config.use_cache=True
                for t,s,key,path in needed:
                    bank=load_sealed(path)
                    if 'adaptation' in bank:
                        if bank['adaptation']['status']!='complete': raise RuntimeError('Previous adapter failure retained; inspect before retry')
                        continue
                    update_started=time.monotonic(); restore()
                    try:
                        image_path=root/'observations'/f'{key}.npy'
                        if digest(image_path)!=bank['observation_sha256']: raise ValueError('Observation cache mismatch')
                        image=np.load(image_path); language=bank['task_language']
                        wrapped_base=policy(image,language)['action']
                        if wrapped_base!=bank['baseline']['action']: raise RuntimeError('PEFT preparation changed base action; stop')
                        # k-bit training preparation may cast nonquantized layers.
                        # Verify the WHOLE frozen continuation before interpreting
                        # any difference as a parameter-adaptation effect.
                        control_env=environment(t)
                        try:
                            control_env.reset(); control_obs=control_env.set_init_state(np.asarray(bank['initial_state']))
                            for a in bank['prefix_actions']: control_obs,_,_,_=control_env.step(a)
                            control_actions,control_trace,control_final=continuation(control_env,control_obs,language,config['H'],wrapped_base)
                        finally: control_env.close()
                        if control_actions!=bank['frozen_actions'] or control_trace!=bank['frozen_trace'] or control_final!=bank['frozen_final_state']:
                            raise RuntimeError('PEFT preparation altered frozen continuation; adaptation comparison invalid')
                        # Target chosen WITHOUT simulator outcomes or Oracle ranking.
                        target=max(bank['candidates'],key=lambda c:(c['processed_score_logprob'],-c['candidate_index']))
                        inputs=prepare_inputs(handle,image,language); ids=inputs['input_ids']
                        if ids[0,-1].item()!=29871: ids=torch.cat([ids,ids.new_tensor([[29871]])],dim=1)
                        tokens=ids.new_tensor([target['tokens']]); full=torch.cat([ids,tokens],dim=1)
                        labels=torch.full_like(full,-100); labels[:,-7:]=tokens
                        inputs=dict(inputs,input_ids=full,attention_mask=torch.ones_like(full),labels=labels)
                        optimizer=torch.optim.AdamW(params.values(),lr=1e-5,weight_decay=0)
                        scaler=torch.cuda.amp.GradScaler(); model.train(); model.config.use_cache=False
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda',dtype=torch.float16): loss=model(**inputs).loss
                        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                        norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True)
                        scaler.step(optimizer); scaler.update(); model.eval();model.config.use_cache=True
                        training_loss=float(loss.detach())
                        delta=max(float((v.detach().cpu()-snapshot[n]).abs().max()) for n,v in params.items())
                        if delta<=0 or not np.isfinite(delta): raise RuntimeError('No actual parameter update')
                        adapter=root/'adapters'/key; model.save_pretrained(adapter)
                        del inputs,full,labels,tokens,loss,optimizer; torch.cuda.empty_cache()
                        env=environment(t)
                        try:
                            env.reset();obs=env.set_init_state(np.asarray(bank['initial_state']))
                            for a in bank['prefix_actions']:
                                obs,_,done,_=env.step(a)
                                if done: raise RuntimeError('Terminal factual prefix')
                            if snapshot_sim_state(env).tolist()!=bank['intervention_sim_state']: raise RuntimeError('Adapted branch initial state differs')
                            acts,trace,final=continuation(env,obs,language,config['H'])
                        finally: env.close()
                        restore()
                        rollback=policy(image,language)['action']
                        if rollback!=bank['baseline']['action'] or any(not torch.equal(v.detach().cpu(),snapshot[n]) for n,v in params.items()):
                            raise RuntimeError('Adapter rollback mismatch')
                        bank['adaptation']=dict(status='complete',method='candidate_proposal_likelihood_self_distillation',
                            privileged_target=False,target_candidate_index=target['candidate_index'],optimizer_steps=1,rank=2,lr=1e-5,
                            training_loss=training_loss,
                            parameter_delta_max_abs=delta,rollback_verified=True,zero_adapter_frozen_continuation_verified=True,gradient_norm=float(norm),
                            actions=acts,trace=trace,final_state=final,seconds=time.monotonic()-update_started,
                            adapter_file_sha256={str(f.relative_to(adapter)):digest(f) for f in adapter.rglob('*') if f.is_file()})
                    except BaseException:
                        bank['adaptation']=dict(status='failed',traceback=traceback.format_exc()); raise
                    finally:
                        atomic_json(path,bank);seal(path)
                    print('ADAPTATION_COMPLETE',key,flush=True)
            status.update(status='gpu_banks_complete',experiments_completed=[x[2] for x in needed])
        except BaseException:
            status.update(status='failed',failure=traceback.format_exc()); traceback.print_exc()
        finally:
            status['elapsed_seconds']=time.monotonic()-started
            atomic_json(root/'gpu_status.json',status);seal(root/'gpu_status.json')
    return 0 if status['status']=='gpu_banks_complete' else 1


if __name__=='__main__': raise SystemExit(main())
