"""Helpers for saving small reproducibility artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def save_split_ids(path, train_ids, val_ids, test_ids) -> None:
    """Save match-level split ids in a Git-safe pickle."""
    artifact = {
        "train_ids": list(train_ids),
        "val_ids": list(val_ids),
        "test_ids": list(test_ids),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(artifact, f)


def save_predictions(path, rows: pd.DataFrame, probs: np.ndarray, y_pred_goals: np.ndarray) -> None:
    """Save row metadata plus model predictions for evaluation/visualization."""
    out = rows.copy()
    out["pred_home_goals_remaining"] = y_pred_goals[:, 0]
    out["pred_away_goals_remaining"] = y_pred_goals[:, 1]
    out["home_win_prob"] = probs[:, 0]
    out["draw_prob"] = probs[:, 1]
    out["away_win_prob"] = probs[:, 2]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(out, f)
