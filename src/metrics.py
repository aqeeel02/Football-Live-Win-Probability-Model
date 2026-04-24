"""Metrics for remaining-goals and win/draw/loss probability forecasts."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score


def ranked_probability_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Ranked Probability Score for ordered outcomes: home win, draw, away win."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    outcome_cdf = np.zeros_like(probs)

    for i, outcome in enumerate(outcomes):
        if outcome == 0:
            outcome_cdf[i] = [1, 1, 1]
        elif outcome == 1:
            outcome_cdf[i] = [0, 1, 1]
        else:
            outcome_cdf[i] = [0, 0, 1]

    pred_cdf = np.cumsum(probs, axis=1)
    return float(np.mean(0.5 * np.sum((pred_cdf[:, :2] - outcome_cdf[:, :2]) ** 2, axis=1)))


def multiclass_brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Multiclass Brier score for home/draw/away probabilities."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    one_hot = np.eye(3)[outcomes]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def multiclass_roc_auc(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """One-vs-rest multiclass ROC AUC."""
    one_hot = np.eye(3)[np.asarray(outcomes, dtype=int)]
    return float(roc_auc_score(one_hot, probs, multi_class="ovr"))


def remaining_goals_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE and MSE for two remaining-goals targets."""
    return {
        "remaining_goals_mae": float(mean_absolute_error(y_true, y_pred)),
        "remaining_goals_mse": float(mean_squared_error(y_true, y_pred)),
    }

def ece(y_true_bin, y_prob, n_bins=10):
    bins   = np.linspace(0, 1, n_bins + 1)
    total  = len(y_prob)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true_bin[mask].mean()
        conf = y_prob[mask].mean()
        ece_val += (mask.sum() / total) * abs(acc - conf)
    return ece_val

def print_ece(probs, outcomes, model_name='Model'):
    """Print ECE for all three outcomes."""
    labels = {0: 'Home win', 1: 'Draw', 2: 'Away win'}
    print(f'{model_name} — Expected Calibration Error (lower is better)')
    print('=' * 48)
    for cls in range(3):
        y_true_bin = (outcomes == cls).astype(int)
        e = ece(y_true_bin, probs[:, cls])
        print(f'  {labels[cls]:<10}: ECE = {e:.4f}')
    print('=' * 48)
