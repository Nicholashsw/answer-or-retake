"""Risk-coverage metrics, threshold rules and intervals."""
import numpy as np
from scipy.stats import beta


def rc_curve(score, correct):
    """Coverage and risk when answering the top-k questions by score, k = 1..n."""
    order = np.argsort(-np.asarray(score), kind="stable")
    err = 1 - np.asarray(correct)[order]
    k = np.arange(1, len(err) + 1)
    return k / len(err), np.cumsum(err) / k


def aurc(score, correct):
    return float(rc_curve(score, correct)[1].mean())


def oracle_aurc(correct):
    err = np.sort(1 - np.asarray(correct))
    k = np.arange(1, len(err) + 1)
    return float((np.cumsum(err) / k).mean())


def threshold_at(score, correct, target, rule="point", delta=0.10):
    """Lowest threshold on a calibration split whose risk stays under the target.

    rule="point": observed risk <= target.
    rule="ucb": one-sided Clopper-Pearson upper bound at 1 - delta <= target (a stricter heuristic).
    Returns inf when no threshold qualifies (answer nothing).
    """
    score, correct = np.asarray(score, dtype=float), np.asarray(correct)
    order = np.argsort(-score, kind="stable")
    s, err = score[order], 1 - correct[order]
    k = np.arange(1, len(s) + 1)
    n_err = np.cumsum(err)
    risk = n_err / k if rule == "point" else beta.ppf(1 - delta, n_err + 1, k - n_err)
    end_of_tie = np.r_[s[1:] != s[:-1], True]
    ok = np.where((risk <= target) & end_of_tie)[0]
    return np.inf if len(ok) == 0 else s[ok.max()]


def coverage_risk(answered, correct):
    answered = np.asarray(answered, dtype=bool)
    if not answered.any():
        return 0.0, float("nan")
    return float(answered.mean()), float((1 - np.asarray(correct)[answered]).mean())


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def paired_bootstrap_aurc(score_a, score_b, correct, reps=1000, seed=0):
    """AURC(a) - AURC(b) with a 95% percentile interval, resampling questions."""
    rng = np.random.default_rng(seed)
    score_a, score_b, correct = map(np.asarray, (score_a, score_b, correct))
    diffs = []
    for _ in range(reps):
        i = rng.integers(0, len(correct), len(correct))
        diffs.append(aurc(score_a[i], correct[i]) - aurc(score_b[i], correct[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return aurc(score_a, correct) - aurc(score_b, correct), lo, hi
