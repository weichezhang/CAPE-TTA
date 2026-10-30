#!/usr/bin/env python3
"""CPU fresh replay of immutable action banks; imports no VLA bridge."""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from cape_tta.budget_pilot import atomic_json, digest, evaluate_benefits, load_sealed, seal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank-root', type=Path, required=True)
    p.add_argument('--libero-root', type=Path, required=True)
    p.add_argument('--alpha', type=float, default=.1)
    args = p.parse_args()
    root = args.bank_root.resolve(); out = root / 'cpu'; out.mkdir(exist_ok=True)
    report = dict(status='running', scope='cached_action_consequence_and_optional_short_horizon_parameter_benefit', episodes=[])
    started = time.monotonic()
    try:
        # No checkpoint download, model load, or GPU allocation in this process.
        os.environ.update(CUDA_VISIBLE_DEVICES='', MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa')
        from cape_tta.runtime_config import write_libero_config
        write_libero_config(args.libero_root, out / 'libero-config/config.yaml')
        os.environ.update(LIBERO_CONFIG_PATH=str(out/'libero-config'), LIBERO_REPO_ROOT=str(args.libero_root))
        sys.path.insert(0, str(args.libero_root.resolve()))
        from libero.libero.envs import OffScreenRenderEnv
        from cape_tta.libero_oracle import soft_task_potential, goal_predicate_progress, snapshot_sim_state
        for path in sorted((root / 'banks').glob('*.json')):
            if path.name.endswith('.sha256.json'): continue
            bank = load_sealed(path)
            if bank['status'] != 'bank_complete': continue
            actual_commit = subprocess.check_output(['git','-C',str(args.libero_root),'rev-parse','HEAD'],text=True).strip()
            if actual_commit != bank['runtime']['libero_commit']: raise ValueError('LIBERO version changed')
            key = bank['episode_id']; cached = out / (key + '.json')
            if cached.exists():
                row = load_sealed(cached)
                if row['bank_sha256'] != digest(path): raise ValueError('Cached evaluation bank changed')
                report['episodes'].append(row); continue
            bddl = args.libero_root / 'libero/libero/bddl_files' / bank['bddl_relative']
            def branch(actions):
                env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
                env.seed(bank['environment_seed'])
                try:
                    env.reset(); env.set_init_state(np.asarray(bank['initial_state']))
                    for action in bank['prefix_actions']:
                        _, _, done, _ = env.step(action)
                        if done: raise ValueError('Prefix already terminal')
                    initial = snapshot_sim_state(env)
                    if not np.array_equal(initial, np.asarray(bank['intervention_sim_state'])):
                        raise ValueError('CPU/GPU factual prefix state mismatch')
                    p0 = soft_task_potential(env); sparse0 = goal_predicate_progress(env)
                    trace = []
                    for i, action in enumerate(actions):
                        a = np.asarray(action, dtype=float)
                        if a.shape != (7,) or not np.isfinite(a).all(): raise ValueError('Invalid cached action')
                        _, _, done, _ = env.step(a.tolist())
                        trace.append(dict(step=i+1, success=bool(done), soft_potential=soft_task_potential(env), sparse_progress=goal_predicate_progress(env)))
                        if done: break
                    return dict(success=bool(trace[-1]['success']), steps_executed=len(trace), soft_start=p0,
                                soft_end=trace[-1]['soft_potential'], soft_gain=trace[-1]['soft_potential']-p0,
                                sparse_gain=trace[-1]['sparse_progress']-sparse0, trace=trace,
                                initial_state=initial.tolist(), final_state=snapshot_sim_state(env).tolist())
                finally: env.close()
            frozen = branch(bank['frozen_actions']); repeat = branch(bank['frozen_actions'])
            if frozen != repeat: raise ValueError('Fresh baseline replay differs')
            # Source GPU trajectory was closed-loop; replay must reproduce it exactly.
            if frozen['trace'] != bank['frozen_trace'] or frozen['final_state'] != bank['frozen_final_state']:
                raise ValueError('CPU replay differs from source GPU factual rollout')
            row = dict(episode_id=key, task_id=bank['task_id'], init_state_id=bank['init_state_id'],
                       intervention_step=bank['intervention_step'], bank_sha256=digest(path),
                       state_deviation=0., fresh_replay_passed=True, frozen_outcome=frozen,
                       action_consequences=[], parameter_adaptation=None)
            # Shared factual continuation: explicitly open-loop action consequence.
            dedup = {tuple(bank['frozen_actions'][0]): frozen}
            for candidate in bank['candidates']:
                k = tuple(candidate['action'])
                if k not in dedup: dedup[k] = branch([candidate['action']] + bank['frozen_actions'][1:])
                r = dedup[k]
                row['action_consequences'].append(dict(oracle_kind='action_consequence', outcome=r,
                    realized_benefit=r['soft_gain']-frozen['soft_gain'], candidate=candidate,
                    continuation='shared_factual_open_loop_actions', effective_horizon=len(bank['frozen_actions']), full_adaptation_benefit=False))
            row['oracle_best_soft_margin'] = max([0.] + [r['realized_benefit'] for r in row['action_consequences']])
            if bank.get('adaptation', {}).get('status') == 'complete':
                meta = bank['adaptation']
                if meta['parameter_delta_max_abs'] <= 0 or not meta['rollback_verified'] or not meta['zero_adapter_frozen_continuation_verified']:
                    raise ValueError('Missing actual parameter-update/rollback evidence')
                adapted = branch(meta['actions'])
                if adapted['trace'] != meta['trace'] or adapted['final_state'] != meta['final_state']:
                    raise ValueError('Adapted GPU/CPU trajectory mismatch')
                row['parameter_adaptation'] = dict(oracle_kind='parameter_adaptation', episode_id=key,
                    task_id=bank['task_id'], init_state_id=bank['init_state_id'], intervention_step=bank['intervention_step'],
                    frozen_outcome=frozen, adapted_outcome=adapted, features=bank['features'],
                    realized_benefit=adapted['soft_gain']-frozen['soft_gain'], predicted_benefit=None,
                    candidate_log_probabilities=[c['processed_score_logprob'] for c in bank['candidates']],
                    parameter_update=meta, runtime=bank['runtime'], horizon=bank['horizon'],
                    estimand='closed_loop_parameter_update_benefit_over_short_horizon_not_full_episode_success')
            atomic_json(cached, row); seal(cached)
            report['episodes'].append(row)
            print('CPU_EPISODE_COMPLETE', key, flush=True)
        margins = [c['realized_benefit'] for e in report['episodes'] for c in e['action_consequences']]
        report['action_diagnostic'] = dict(count=len(margins), positive=sum(x > 1e-5 for x in margins),
            negative=sum(x < -1e-5 for x in margins), near_zero=sum(abs(x) <= 1e-5 for x in margins),
            scope='Open-loop diagnostic; duplicates retained; candidates are not independent episodes')
        rows = [e['parameter_adaptation'] for e in report['episodes'] if e['parameter_adaptation'] is not None]
        if len(rows) >= 6:
            report['benefit_gate'] = evaluate_benefits(rows, alpha=args.alpha)
            for e in report['episodes']:
                for pred in report['benefit_gate']['per_episode']:
                    if e['episode_id'] == pred['episode_id']: e['parameter_adaptation'].update(pred)
        else:
            report['benefit_gate'] = dict(status='NO_GO_FOR_EXPANSION', reason='Fewer than six independently replay-verified parameter-adaptation episodes', available=len(rows))
        report['status'] = 'cpu_evaluation_complete'
    except BaseException:
        report.update(status='failed', traceback=traceback.format_exc()); traceback.print_exc()
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        atomic_json(out/'report.json', report); seal(out/'report.json')
        gate=report.get('benefit_gate',{})
        text = ('# V9 pilot go/no-go\n\n'
                f"Execution: {report['status']}. Replay-verified episodes: {len(report['episodes'])}.\n\n"
                f"Decision: {gate.get('status','NO_GO_FOR_EXPANSION')}.\n\n"
                'Only replay-verified parameter-update trajectories enter the benefit predictor. '
                'Candidate consequences are reported separately. All benefits here concern a short-horizon privileged soft potential, '
                'not full-episode benchmark success. GO_EXPLORATORY is permission to collect more evidence, not a significance or acceptance claim.\n\n'
                'A null quantile with q_is_infinite=true means the finite-sample quantile is +infinity: '
                'LCB=-infinity and all interventions are rejected. Null HAR means no accepted interventions. '
                'Marginal split-conformal coverage requires exchangeable calibration/test episodes; no conditional HAR bound is claimed.\n\n'
                'Inspect report.json for per-episode outcomes, coverage, acceptance, HIP, HAR, predictor error and GO checks. '
                'Do not expand suites while execution fails or the decision is NO_GO_FOR_EXPANSION. '
                'First inspect metric support, effective horizon, candidate diversity, update/rollback checks, '
                'label signs and predictor features. With only six episodes, collect more independent calibration episodes '
                'before expecting a finite alpha=0.1 gate; do not inflate n with candidate draws or frames.\n')
        (out/'go_no_go.md').write_text(text)
    return 0 if report['status'] == 'cpu_evaluation_complete' else 1


if __name__ == '__main__': raise SystemExit(main())
