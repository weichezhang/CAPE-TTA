"""Finite-pool oracle diagnostics; descriptive until independently evaluated."""
from math import isfinite


def utility(branch, budget=220):
    steps=branch['steps_executed']
    if not 1 <= steps <= budget: raise ValueError('Invalid executed steps')
    return float(branch['success']) - 0.1*steps/budget


def pool_summary(ids, branches, baseline_id, horizon=12, budget=220):
    """Keep nominal multiplicity. Ties favor baseline then proposal order."""
    eligible=[baseline_id]+list(ids)
    if any(i not in branches for i in eligible): raise ValueError('Incomplete pool')
    scores=[branches[i]['horizon_potential'][str(horizon)] for i in eligible]
    if not all(isfinite(x) for x in scores): raise ValueError('Nonfinite score')
    chosen=eligible[max(range(len(eligible)),key=lambda j:scores[j])]
    successes=[float(branches[i]['success']) for i in eligible]
    values=[utility(branches[i],budget) for i in eligible]
    chosen_success=float(branches[chosen]['success'])
    return dict(selected_action_id=chosen, unique_actions=len(set(eligible)), nominal_entries=len(eligible),
                selected_success=chosen_success, frozen_success=float(branches[baseline_id]['success']),
                uniform_pool_expected_success=sum(successes)/len(successes),
                uniform_draw_expected_success=sum(successes[1:])/len(ids) if ids else None,
                success_margin_vs_uniform_pool=chosen_success-sum(successes)/len(successes),
                selected_utility=utility(branches[chosen],budget),
                uniform_pool_expected_utility=sum(values)/len(values),
                full_horizon_pool_best_utility=max(values),
                utility_regret=max(values)-utility(branches[chosen],budget),
                warning='Finite-pool privileged labels, not deployment performance or independent draw samples')


def replay_equal(a,b,tolerance=1e-8):
    if a['success']!=b['success'] or a['steps_executed']!=b['steps_executed']: return False
    if len(a['trajectory'])!=len(b['trajectory']):return False
    for x,y in zip(a['trajectory'],b['trajectory']):
        if x['done']!=y['done'] or len(x['action'])!=len(y['action']):return False
        if any(abs(u-v)>tolerance for u,v in zip(x['action'],y['action'])):return False
    return all(abs(a['horizon_potential'][k]-b['horizon_potential'][k])<=tolerance for k in a['horizon_potential'])
