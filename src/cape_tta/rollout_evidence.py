"""Summaries for privileged short-rollout development experiments."""
import math


def summarize_group(candidate_ids, branches, baseline_id):
    if not candidate_ids or baseline_id not in branches:
        raise ValueError('Missing candidate or baseline evidence')
    samples = [branches[key] for key in candidate_ids]
    baseline = branches[baseline_id]
    for branch in samples + [baseline]:
        if not math.isfinite(branch['soft_gain']) or not math.isfinite(branch['sparse_gain']):
            raise ValueError('Nonfinite rollout gain')
        if type(branch['success']) is not bool:
            raise ValueError('Success evidence must be boolean')
    # Baseline is explicitly eligible. Nonnegative oracle margins are guaranteed
    # by that construction, not statistical evidence of improvement.
    eligible = [baseline_id] + candidate_ids
    best = max(eligible, key=lambda key: branches[key]['soft_gain'])
    differences = [b['soft_gain'] - baseline['soft_gain'] for b in samples]
    return {
        'sample_count': len(candidate_ids),
        'eligible_count_including_baseline': len(eligible),
        'unique_sampled_actions': len(set(candidate_ids)),
        'unique_eligible_actions': len(set(eligible)),
        'baseline_soft_gain': baseline['soft_gain'],
        'oracle_best_action_id': best,
        'oracle_soft_margin': branches[best]['soft_gain'] - baseline['soft_gain'],
        'mean_sampled_soft_margin': sum(differences) / len(differences),
        'sampled_strictly_better_count': sum(d > 1e-9 for d in differences),
        'sampled_strictly_worse_count': sum(d < -1e-9 for d in differences),
        'baseline_success_within_horizon': baseline['success'],
        'any_candidate_success_within_horizon': any(branches[key]['success'] for key in eligible),
        'scope': 'privileged first-action intervention with frozen continuation; not a deployable method',
    }


def replay_summary(first, repeated, state_atol=1e-7, progress_atol=1e-12):
    left, right = first['final_state'], repeated['final_state']
    if not left or len(left) != len(right):
        raise ValueError('Missing or mismatched replay states')
    if not all(math.isfinite(float(v)) for v in list(left) + list(right)):
        raise ValueError('Nonfinite replay state')
    state_error = max(abs(float(a) - float(b)) for a, b in zip(left, right))
    progress_error = abs(first['soft_progress_end'] - repeated['soft_progress_end'])
    finite = math.isfinite(state_error) and math.isfinite(progress_error)
    consistent = first['success'] == repeated['success']
    return {'passed': bool(finite and state_error <= state_atol and progress_error <= progress_atol and consistent),
            'final_state_linf': state_error, 'soft_progress_abs': progress_error,
            'success_consistent': consistent}
