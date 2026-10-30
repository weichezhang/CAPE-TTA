"""CPU-only pilot analysis. No simulator outcomes are synthesized here."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(path):
    path = Path(path)
    atomic_json(str(path) + '.sha256.json', {'sha256': digest(path)})


def load_sealed(path):
    path = Path(path)
    if json.loads(Path(str(path) + '.sha256.json').read_text())['sha256'] != digest(path):
        raise ValueError('Checksum mismatch: ' + str(path))
    return json.loads(path.read_text())


def conformal_q(predictions, benefits, alpha):
    """Upper quantile of f(X)-m; m >= f(X)-q is the one-sided target."""
    if not 0 < alpha < 1 or len(predictions) != len(benefits):
        raise ValueError('Invalid calibration arguments')
    scores = sorted(float(p) - float(m) for p, m in zip(predictions, benefits))
    if not all(math.isfinite(x) for x in scores):
        raise ValueError('Nonfinite calibration score')
    k = math.ceil((len(scores) + 1) * (1 - alpha))
    return scores[k - 1] if 1 <= k <= len(scores) else math.inf


def episode_split(rows, seed=7):
    """One predeclared intervention per episode; no repeated-frame inflation."""
    ids = [r['episode_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Exactly one intervention per episode required in this pilot')
    if len(ids) < 6:
        raise ValueError('At least six independent episodes required')
    ids = sorted(ids); np.random.RandomState(seed).shuffle(ids)
    a = max(2, int(len(ids) * .4)); b = max(2, int(len(ids) * .3))
    return dict(train=ids[:a], calibration=ids[a:a+b], test=ids[a+b:])


def evaluate_benefits(rows, alpha=.1, seed=7, epsilon=1e-5):
    """Fixed ridge predictor, episode split, and descriptive GO/HOLD decision.

    Deliberate small task/state pools do not establish exchangeability with new
    tasks. Empirical metrics are pilot diagnostics, never a HAR guarantee.
    """
    rows = sorted(rows, key=lambda x: x['episode_id'])
    if any(r['oracle_kind'] != 'parameter_adaptation' for r in rows):
        raise ValueError('Candidate-action labels cannot masquerade as adaptation labels')
    splits = episode_split(rows, seed)
    ix = {r['episode_id']: i for i, r in enumerate(rows)}
    idx = {s: [ix[e] for e in v] for s, v in splits.items()}
    x = np.asarray([r['features'] for r in rows], dtype=float)
    y = np.asarray([r['realized_benefit'] for r in rows], dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite predictor data')
    tr, ca, te = (idx[s] for s in ('train', 'calibration', 'test'))
    mu = x[tr].mean(0); scale = x[tr].std(0); scale[scale < 1e-8] = 1
    z = np.column_stack([np.ones(len(rows)), (x - mu) / scale])
    penalty = np.eye(z.shape[1]); penalty[0, 0] = 0
    w = np.linalg.solve(z[tr].T @ z[tr] + penalty, z[tr].T @ y[tr])
    pred = z @ w
    q = conformal_q(pred[ca], y[ca], alpha)
    lcb = pred[te] - q; accept = lcb > 0
    harm = y[te] < 0
    selected = np.where(accept, y[te], 0.)
    def rate(a): return float(np.mean(a))
    def policy(mask):
        return dict(adaptation_acceptance_rate=rate(mask),
                    mean_realized_benefit=rate(np.where(mask, y[te], 0.)),
                    mean_benefit_among_accepted=float(y[te][mask].mean()) if mask.any() else None,
                    HIP=rate(harm & mask), HAR=rate(harm[mask]) if mask.any() else None,
                    accepted_count=int(mask.sum()), test_count=len(te),
                    mean_success=rate([rows[i]['adapted_outcome' if m else 'frozen_outcome']['success'] for i, m in zip(te, mask)]),
                    mean_soft_progress=rate([rows[i]['adapted_outcome' if m else 'frozen_outcome']['soft_gain'] for i, m in zip(te, mask)]))
    cape = policy(accept); always = policy(np.ones(len(te), dtype=bool)); frozen = policy(np.zeros(len(te), dtype=bool))
    mse = float(np.mean((pred[te] - y[te])**2))
    constant_mse = float(np.mean((y[tr].mean() - y[te])**2))
    checks = dict(positive_and_negative_benefits=bool((y > epsilon).any() and (y < -epsilon).any()),
                  predictor_beats_train_mean=bool(mse < constant_mse),
                  finite_quantile=bool(math.isfinite(q)),
                  rejects_some_harm=bool((harm & ~accept).any()),
                  nontrivial_acceptance=bool(accept.any()),
                  lower_HAR=bool(cape['HAR'] is not None and always['HAR'] is not None and cape['HAR'] < always['HAR']),
                  no_observed_success_or_progress_drop=bool(cape['mean_success'] >= frozen['mean_success'] and cape['mean_soft_progress'] >= frozen['mean_soft_progress'] - epsilon))
    return dict(status='GO_EXPLORATORY' if all(checks.values()) else 'NO_GO_FOR_EXPANSION', checks=checks,
                split=splits, alpha=alpha, q=q if math.isfinite(q) else None, q_is_infinite=not math.isfinite(q),
                calibration_count=len(ca), coverage=rate(y[te] >= lcb), predictor_test_mse=mse,
                train_mean_test_mse=constant_mse, CAPE=cape, Always_Adapt=always, Frozen=frozen,
                predictor=dict(type='ridge_l2_1_fixed', mean=mu.tolist(), scale=scale.tolist(), coefficients=w.tolist()),
                per_episode=[dict(episode_id=r['episode_id'], split=next(s for s,v in splits.items() if r['episode_id'] in v),
                                  predicted_benefit=float(pred[i]), realized_benefit=float(y[i]),
                                  LCB=float(pred[i]-q) if math.isfinite(q) else None,
                                  LCB_is_negative_infinity=not math.isfinite(q)) for i,r in enumerate(rows)],
                interpretation='Descriptive pilot; conditional on exchangeability split conformal covers marginally. No HAR bound. No significance claim. Null HAR means no accepted intervention.')
