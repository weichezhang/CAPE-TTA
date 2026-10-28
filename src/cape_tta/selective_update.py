"""Split conformal gate for FIXED update-benefit predictors, not action scores."""
import math


def correction(predictions, observed, alpha=0.05):
    if not 0 < alpha < 1 or len(predictions) != len(observed):
        raise ValueError('Invalid alpha or mismatched calibration records')
    residuals = sorted(float(p) - float(y) for p, y in zip(predictions, observed))
    if not all(math.isfinite(x) for x in residuals):
        raise ValueError('Nonfinite calibration record')
    k = math.ceil((len(residuals) + 1) * (1 - alpha))
    return residuals[k - 1] if k <= len(residuals) else math.inf


def accept(prediction, q):
    if not math.isfinite(prediction) or math.isnan(q):
        raise ValueError('Invalid prediction or correction')
    return prediction - q > 0


def episode_utility(success, steps, budget=220):
    # Success dominates duration. A failed episode is censored, not completed.
    if budget < 1 or not 1 <= steps <= budget:
        raise ValueError('Invalid episode length')
    return float(bool(success)) - 0.1 * steps / budget
