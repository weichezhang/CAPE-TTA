#!/usr/bin/env python3
"""Development-only actual LoRA update and matched episode checks; no calibration claim."""
import argparse, contextlib, hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--libero-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--init-state-ids', type=int, nargs='+', default=[0])
    p.add_argument('--optimizer-steps', type=int, default=1)
    args = p.parse_args()
    if args.optimizer_steps not in (1, 2, 4) or not set(args.init_state_ids) <= {0, 1, 2}:
        p.error('Development states 0,1,2 and optimizer steps 1,2,4 only')
    repo = Path(__file__).resolve().parents[1]
    out = args.output_root.resolve(); out.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', scope='privileged_target_development_lora', states=[],
                  calibrated=False, held_out=False, optimizer_steps=args.optimizer_steps)
    def save():
        (out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    from probe_openvla_diversity import Tee
    with (out/'run.log').open('w') as log, contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
        try:
            import numpy as np
            import torch
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from cape_tta.runtime_config import write_libero_config
            write_libero_config(args.libero_root, out/'libero-config/config.yaml')
            os.environ.update(LIBERO_REPO_ROOT=str(args.libero_root.resolve()), LIBERO_CONFIG_PATH=str(out/'libero-config'), MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa')
            from run_openvla_oracle_pilot import fix_libero_path, settle
            fix_libero_path()
            from libero.libero import benchmark
            from libero.libero.envs import OffScreenRenderEnv
            from cape_tta.openvla_bridge import load_openvla, generate_action, prepare_inputs, to_libero_action
            from cape_tta.full_episode import roll_episode
            from cape_tta.selective_update import episode_utility
            selected = json.loads((repo/'configs/development/v5_selected_interventions.json').read_text())
            targets = json.loads((repo/'configs/development/v7_target_tokens.json').read_text())
            if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
            result.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                          source_sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip())
            (out/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
            torch.manual_seed(7); np.random.seed(7)
            handle = load_openvla(selected['checkpoint'], selected['suite'], revision=selected['checkpoint_revision'], use_flash_attention=False, load_in_8bit=True, input_dtype='fp16')
            base = handle.model
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
            wrapped = get_peft_model(base, LoraConfig(r=4,lora_alpha=8,target_modules=['q_proj','v_proj'],lora_dropout=0.0,bias='none'))
            # Generic PEFT wrapper retains the custom multimodal forward/generate interface.
            handle.model = wrapped
            params = {n:v for n,v in wrapped.named_parameters() if v.requires_grad}
            if not params or any('lora_' not in n for n in params): raise RuntimeError('Unexpected trainable parameter set')
            snapshot = {n:v.detach().cpu().clone() for n,v in params.items()}
            result['trainable_parameters'] = {n:list(v.shape) for n,v in params.items()}
            def frozen_digest():
                digest=hashlib.sha256()
                for n,v in wrapped.named_parameters():
                    if not v.requires_grad:
                        digest.update(n.encode()); digest.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
                return digest.hexdigest()
            before_hash = frozen_digest(); result['frozen_parameter_sha256_before']=before_hash
            def restore():
                with torch.no_grad():
                    for n,v in params.items(): v.copy_(snapshot[n].to(v.device))
                wrapped.eval()
            suite = benchmark.get_benchmark_dict()[selected['suite']](); task=suite.get_task(0)
            initials=suite.get_task_init_states(0)
            bddl=args.libero_root.resolve()/'libero/libero/bddl_files'/task.problem_folder/task.bddl_file
            def factory():
                env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256);env.seed(0);return env
            def policy(obs):
                raw,_,_=generate_action(handle,obs['agentview_image'],task.language,do_sample=False,restrict_to_action_tokens=False)
                return to_libero_action(raw).tolist()
            for state_id in args.init_state_ids:
                restore(); env=factory()
                try: image=settle(env,initials[state_id],10)['agentview_image'].copy()
                finally: env.close()
                obs={'agentview_image':image}; baseline_first=policy(obs)
                row=dict(init_state_id=state_id, branches={}, losses=[]);result['states'].append(row);save()
                def episode(name, first):
                    torch.manual_seed(7);np.random.seed(7);trajectory=[]
                    def progress(x):
                        trajectory.append(x)
                        if x['step']%10==0 or x['done']: print(state_id,name,x['step'],x['done'],flush=True)
                    started=time.monotonic()
                    r=roll_episode(factory,initials[state_id],first,policy,on_step=progress)
                    r.update(trajectory=trajectory,seconds=time.monotonic()-started)
                    row['branches'][name]=r;save();return r
                baseline=episode('frozen',baseline_first)
                inputs=prepare_inputs(handle,image,task.language)
                ids=inputs['input_ids']
                if ids[0,-1].item()!=29871: ids=torch.cat([ids,ids.new_tensor([[29871]])],dim=1)
                tokens=ids.new_tensor([targets[str(state_id)]['tokens']])
                full=torch.cat([ids,tokens],dim=1)
                labels=torch.full_like(full,-100);labels[:,-tokens.shape[1]:]=tokens
                train_inputs=dict(inputs);train_inputs.update(input_ids=full,attention_mask=torch.ones_like(full),labels=labels)
                optimizer=torch.optim.AdamW(params.values(),lr=1e-5,weight_decay=0)
                wrapped.train();wrapped.config.use_cache=False
                # FP16 vision inputs, FP32 adapters, dynamic loss scaling on T4.
                scaler=torch.cuda.amp.GradScaler()
                for step in range(args.optimizer_steps):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.float16): loss=wrapped(**train_inputs).loss
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                    scaler.scale(loss).backward();scaler.unscale_(optimizer)
                    norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.0,error_if_nonfinite=True)
                    scaler.step(optimizer);scaler.update();row['losses'].append(float(loss.detach()))
                    print('UPDATE',state_id,step,float(loss.detach()),'grad_norm',float(norm),flush=True)
                wrapped.eval();wrapped.config.use_cache=True
                delta=max(float((v.detach().cpu()-snapshot[n]).abs().max()) for n,v in params.items())
                if not np.isfinite(delta) or delta<=0: raise RuntimeError('No real adapter parameter change')
                row['adapter_delta_max_abs']=delta
                wrapped.save_pretrained(out/f'adapter-state-{state_id}')
                changed_first=policy(obs);updated=episode('ungated_lora',changed_first)
                row['update_margin']=episode_utility(updated['success'],updated['steps_executed'])-episode_utility(baseline['success'],baseline['steps_executed'])
                restore();rollback_first=policy(obs)
                if not np.allclose(rollback_first,baseline_first,atol=1e-6,rtol=0): raise RuntimeError('Rollback action mismatch')
                if any(not torch.equal(v.detach().cpu(),snapshot[n]) for n,v in params.items()): raise RuntimeError('Rollback parameter mismatch')
                row['rollback_action_error']=float(np.max(np.abs(np.array(rollback_first)-baseline_first)))
                episode('rollback_replay',rollback_first)
                row['rollback_episode_matches']=row['branches']['rollback_replay']['trajectory']==baseline['trajectory']
                if not row['rollback_episode_matches']: raise RuntimeError('Rollback trajectory differs')
                save()
            after_hash=frozen_digest();result['frozen_parameter_sha256_after']=after_hash
            if after_hash!=before_hash: raise RuntimeError('Frozen base parameters changed')
            result['peak_cuda_bytes']=torch.cuda.max_memory_allocated();result['status']='development_update_complete'
        except BaseException:
            result.update(status='failed',traceback=traceback.format_exc());traceback.print_exc()
        finally: save()
    sums={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file()}
    (out/'sha256.json').write_text(json.dumps(sums,indent=2))
    print('BUNDLE',shutil.make_archive(str(out),'zip',out),flush=True)
    return 0 if result['status']=='development_update_complete' else 1

if __name__=='__main__': raise SystemExit(main())
