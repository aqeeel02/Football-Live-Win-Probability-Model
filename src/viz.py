"""Visualization helpers for win probability outputs."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from src.metrics import ece
import matplotlib.gridspec as gridspec
from sklearn.calibration import calibration_curve

def plot_combined_win_probability_with_goals(match_id, test_rows, probs, model_name):
    plot_df = test_rows.copy()

    plot_df["home_win_prob"] = probs[:, 0]
    plot_df["draw_prob"] = probs[:, 1]
    plot_df["away_win_prob"] = probs[:, 2]

    m = plot_df[plot_df["match_id"] == match_id].sort_values("possession").copy()

    if m.empty:
        print("Match not found in test_rows.")
        return

    prob_cols = ["home_win_prob", "draw_prob", "away_win_prob"]
    m[prob_cols] = m[prob_cols].clip(0, 1)
    m[prob_cols] = m[prob_cols].div(m[prob_cols].sum(axis=1), axis=0)

    x = np.arange(len(m))

    home = 100 * m["home_win_prob"].values
    draw = 100 * m["draw_prob"].values
    away = 100 * m["away_win_prob"].values

    home_team = m["home_team"].iloc[0]
    away_team = m["away_team"].iloc[0]
    final_home = int(m["final_home_score"].iloc[0])
    final_away = int(m["final_away_score"].iloc[0])

    plt.figure(figsize=(15,5))

    plt.stackplot(
        x,
        home,
        draw,
        away,
        colors=["#f2c766", "#e6e6e6", "#2f5da8"],
        labels=[f"{home_team} win", "Draw", f"{away_team} win"],
        alpha=0.95
    )

    m["prev_home_score"] = m["home_score"].shift(1).fillna(0)
    m["prev_away_score"] = m["away_score"].shift(1).fillna(0)

    goal_rows = m[
        (m["home_score"] > m["prev_home_score"]) |
        (m["away_score"] > m["prev_away_score"])
    ]

    for idx, row in goal_rows.iterrows():
        goal_x = m.index.get_loc(idx)

        if row["home_score"] > row["prev_home_score"]:
            score_label = f'{int(row["home_score"])}-{int(row["away_score"])}'
            color = "#b8860b"
        else:
            score_label = f'{int(row["home_score"])}-{int(row["away_score"])}'
            color = "#123c7c"

        plt.axvline(goal_x, color=color, linestyle="--", linewidth=1.5, alpha=0.85)

        plt.text(
            goal_x,
            103,
            score_label,
            ha="center",
            va="bottom",
            fontsize=9,
            color=color
        )

    plt.ylim(0,110)
    plt.ylabel("Probability")
    plt.xlabel("Possession index")
    plt.yticks([0,20,40,60,80,100], ["0%","20%","40%","60%","80%","100%"])

    plt.title(
        f"{model_name.upper()} Win Probability - "
        f"{home_team} {final_home} - {final_away} {away_team}"
    )

    plt.legend(loc="upper center", ncol=3, frameon=False)
    plt.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(
        f"../results/win_prob_{model_name.lower()}_{match_id}.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.show()

def plot_calibration(probs, outcomes, model_name='Model', n_bins=10):
    """
    Calibration curves and confidence histograms for a 3-class WDL model.
    
    Parameters
    ----------
    probs      : np.ndarray, shape (n_samples, 3) — predicted probabilities
    outcomes   : np.ndarray, shape (n_samples,)   — true outcome labels (0,1,2)
    model_name : str — used in plot title and legend
    n_bins     : int — number of calibration bins
    """
    labels = {0: 'Home win', 1: 'Draw', 2: 'Away win'}
    colors = {0: '#3266ad', 1: '#888780', 2: '#D85A30'}

    fig = plt.figure(figsize=(15,10))
    gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

    for cls in range(3):
        y_true_bin = (outcomes == cls).astype(int)
        y_prob     = probs[:, cls]

        # ── Calibration curve (top row) ────────────────────────────────────
        ax_cal = fig.add_subplot(gs[0, cls])

        fraction_pos, mean_pred = calibration_curve(
            y_true_bin, y_prob, n_bins=n_bins, strategy='uniform'
        )

        ax_cal.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
        ax_cal.plot(mean_pred, fraction_pos, 'o-',
                    color=colors[cls], linewidth=2, markersize=5,
                    label=model_name)
        ax_cal.fill_between([0, 1], [0, 1], [1, 1],
                            alpha=0.04, color='red',   label='Over-confident region')
        ax_cal.fill_between([0, 1], [0, 0], [0, 1],
                            alpha=0.04, color='green', label='Under-confident region')

        ax_cal.set_title(f'Calibration — {labels[cls]}', fontsize=12)
        ax_cal.set_xlabel('Mean predicted probability')
        ax_cal.set_ylabel('Fraction of positives')
        ax_cal.set_xlim(0, 1)
        ax_cal.set_ylim(0, 1)
        ax_cal.legend(fontsize=8, loc='upper left')
        ax_cal.grid(True, alpha=0.25)

        # ── Confidence histogram (bottom row) ──────────────────────────────
        ax_hist = fig.add_subplot(gs[1, cls])

        ax_hist.hist(y_prob[outcomes == cls], bins=20, alpha=0.7,
                     color=colors[cls], label=f'True {labels[cls]}', density=True)
        ax_hist.hist(y_prob[outcomes != cls], bins=20, alpha=0.4,
                     color='gray', label='Other outcomes', density=True)

        ax_hist.set_title(f'Predicted P({labels[cls]}) distribution', fontsize=12)
        ax_hist.set_xlabel('Predicted probability')
        ax_hist.set_ylabel('Density')
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True, alpha=0.25)

    fig.suptitle(f'{model_name} — Win Probability Calibration',
                 fontsize=14, fontweight='500', y=1.01)
    plt.tight_layout()
    plt.show()