"""Small dependency-free summaries; no assertion of task improvement."""
import math


def probability_summary(probabilities):
    values = [float(p) for p in probabilities]
    if not values or any(not math.isfinite(p) or p < 0 for p in values):
        raise ValueError('Expected finite nonnegative probabilities')
    total = sum(values)
    if not math.isclose(total, 1., rel_tol=1e-4, abs_tol=1e-6):
        raise ValueError('Probabilities must sum to one')
    positive = [p for p in values if p > 0]
    entropy = -sum(p * math.log(p) for p in positive)
    return {'support_size': len(positive), 'max_probability': max(values),
            'entropy_nats': entropy, 'effective_support': math.exp(entropy)}


def candidate_summary(candidates, baseline_tokens, baseline_action):
    if not candidates:
        raise ValueError('No candidates')
    token_keys = [tuple(c['tokens']) for c in candidates]
    action_keys = [tuple(c['env_action']) for c in candidates]
    distances = [max(abs(float(a) - float(b)) for a, b in zip(c['env_action'], baseline_action))
                 for c in candidates]
    return {'candidate_count': len(candidates),
            'unique_token_candidates': len(set(token_keys)),
            'unique_executed_actions': len(set(action_keys)),
            'baseline_token_matches': sum(key == tuple(baseline_tokens) for key in token_keys),
            'max_action_linf_from_baseline': max(distances),
            'scope': 'first-action diversity only; no rollout benefit measured'}
