"""Validate pilot evidence. Passing does not establish method effectiveness."""
import math


def validate_pilot(data):
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    require(data.get('scientific_result') is False, 'pilot must retain scientific_result=false')
    require(data.get('kind') == 'OpenVLA Oracle-CAPE pilot', 'unexpected result kind')
    for key in ('cape_commit', 'checkpoint_revision'):
        value = data.get(key, '')
        require(isinstance(value, str) and len(value) == 40 and all(c in '0123456789abcdef' for c in value), key + ' must be a full commit SHA')
    parity = data.get('wrapper_parity_max_abs')
    require(finite(parity) and 0 <= parity <= 1e-6, 'wrapper parity failed or missing')
    replay = data.get('fresh_replay_determinism', {})
    require(replay.get('passed') is True, 'fresh replay did not pass')
    for name, limit in (('final_state_linf', 1e-7), ('soft_progress_abs', 1e-12)):
        value = replay.get(name)
        require(finite(value) and 0 <= value <= limit, 'invalid replay ' + name)
    require(replay.get('success_consistent') is True, 'replay success mismatch')
    candidates = data.get('candidates', [])
    require(isinstance(candidates, list) and len(candidates) >= 1, 'no candidates')
    if not isinstance(candidates, list) or not candidates:
        return {'passed': False, 'errors': errors}
    require(type(data.get('candidate_count')) is int and data['candidate_count'] == len(candidates), 'candidate count mismatch')
    require(candidates[0].get('kind') == 'greedy_base', 'candidate zero must be greedy baseline')
    gains = []
    for index, candidate in enumerate(candidates):
        for key in ('raw_action', 'env_action'):
            values = candidate.get(key, [])
            require(isinstance(values, list) and len(values) == 7 and all(finite(v) for v in values), f'candidate {index}: invalid {key}')
        require(type(candidate.get('success')) is bool, f'candidate {index}: missing success')
        require(finite(candidate.get('logprob')), f'candidate {index}: invalid log probability')
        require(type(candidate.get('steps_executed')) is int and 1 <= candidate['steps_executed'] <= data.get('horizon', 0), f'candidate {index}: invalid rollout length')
        tokens = candidate.get('tokens', [])
        require(isinstance(tokens, list) and len(tokens) == 7 and all(type(t) is int for t in tokens), f'candidate {index}: invalid tokens')
        gains.append(candidate.get('soft_gain'))
    if all(finite(g) for g in gains):
        best = data.get('oracle_best_index')
        require(type(best) is int and 0 <= best < len(gains), 'invalid oracle index')
        expected = max(gains)
        for key, value in (('base_soft_gain', gains[0]), ('oracle_best_soft_gain', expected), ('oracle_margin_over_base', expected - gains[0])):
            actual = data.get(key)
            require(finite(actual) and math.isclose(actual, value, rel_tol=1e-7, abs_tol=1e-9), key + ' inconsistent')
        if type(best) is int and 0 <= best < len(gains):
            require(math.isclose(gains[best], expected, abs_tol=1e-9), 'oracle index is not maximal')
    else:
        errors.append('nonfinite or missing soft gain')
    return {'passed': not errors, 'errors': errors,
            'unique_token_candidates': len({tuple(c.get('tokens', [])) for c in candidates}),
            'scope': 'single-state infrastructure pilot; not a paper benchmark'}
