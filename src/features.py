"""Feature engineering utilities for possession-level win probability modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson


MAX_GOALS_REMAINING = 8


NUMERIC_COLS = [
    "game_time_pct",
    "goal_diff",
    "time_remaining",
    "home_elo_diff",
    "is_home_possession",
    "home_score",
    "away_score",
    "home_xg_cumulative",
    "away_xg_cumulative",
    "possession_duration",
    "events_count",
    "dribbles_count",
    "dribbles_completed_pct",
    "carries_count",
    "carries_total_length",
    "carries_mean_length",
    "passes_count",
    "passes_completed_pct",
    "passes_total_length",
    "passes_mean_length",
    "shots_count",
    "shots_xg_total",
    "shots_xg_mean",
    "shots_on_target_pct",
]

TARGET_COLS = ["future_home_goals", "future_away_goals"]


def euclidean_dist(start, end) -> float:
    """Return Euclidean distance between two StatsBomb coordinate pairs."""
    try:
        return float(np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2))
    except Exception:
        return np.nan


def poisson_wdl_from_lambdas(
    goal_diff_home,
    lambda_home,
    lambda_away,
    max_goals: int = MAX_GOALS_REMAINING,
):
    """Convert remaining-goal means into home/draw/away probabilities."""
    goal_diff_home = np.asarray(goal_diff_home, dtype=float)
    lambda_home = np.maximum(np.asarray(lambda_home, dtype=float), 1e-6)
    lambda_away = np.maximum(np.asarray(lambda_away, dtype=float), 1e-6)

    home_win_prob = np.zeros_like(lambda_home, dtype=float)
    draw_prob = np.zeros_like(lambda_home, dtype=float)
    away_win_prob = np.zeros_like(lambda_home, dtype=float)

    for h in range(max_goals + 1):
        p_h = poisson.pmf(h, lambda_home)
        for a in range(max_goals + 1):
            prob = p_h * poisson.pmf(a, lambda_away)
            final_diff = goal_diff_home + h - a
            home_win_prob += np.where(final_diff > 0, prob, 0.0)
            draw_prob += np.where(final_diff == 0, prob, 0.0)
            away_win_prob += np.where(final_diff < 0, prob, 0.0)

    total = home_win_prob + draw_prob + away_win_prob
    total = np.where(total > 0, total, 1.0)
    return home_win_prob / total, draw_prob / total, away_win_prob / total


def engineer_features_future_goals(events_df: pd.DataFrame) -> pd.DataFrame:
    """Create possession features and remaining-goals targets.

    The official final score from match metadata is treated as authoritative.
    Matches whose event-derived goal total does not match the official score
    are removed, because they would create incorrect supervised labels.
    """
    df = events_df.sort_values(["match_id", "period", "minute", "second"]).reset_index(drop=True).copy()

    df["is_goal"] = np.where(df.get("shot.outcome.name") == "Goal", 1, 0)
    df["home_goal_event"] = np.where(df["team.name"] == df["home_team"], df["is_goal"], 0)
    df["away_goal_event"] = np.where(df["team.name"] == df["away_team"], df["is_goal"], 0)

    df["home_score"] = df.groupby("match_id")["home_goal_event"].cumsum()
    df["away_score"] = df.groupby("match_id")["away_goal_event"].cumsum()
    df["tracked_final_home_score"] = df.groupby("match_id")["home_score"].transform("max")
    df["tracked_final_away_score"] = df.groupby("match_id")["away_score"].transform("max")

    if {"official_home_score", "official_away_score"}.issubset(df.columns):
        df["final_home_score"] = df["official_home_score"].astype(int)
        df["final_away_score"] = df["official_away_score"].astype(int)

        match_quality = df.groupby("match_id").agg(
            tracked_home=("tracked_final_home_score", "max"),
            tracked_away=("tracked_final_away_score", "max"),
            official_home=("final_home_score", "first"),
            official_away=("final_away_score", "first"),
        )
        valid_match_ids = match_quality.index[
            (match_quality["tracked_home"] == match_quality["official_home"])
            & (match_quality["tracked_away"] == match_quality["official_away"])
        ]
        df = df[df["match_id"].isin(valid_match_ids)].copy()
    else:
        df["final_home_score"] = df["tracked_final_home_score"]
        df["final_away_score"] = df["tracked_final_away_score"]

    df["goal_diff"] = df["home_score"] - df["away_score"]
    df["home_elo_diff"] = df["home_elo"] - df["away_elo"]
    df["game_time_pct"] = np.clip(((df["minute"] * 60 + df["second"]) / (90 * 60)) * 100, 0, 100)
    df["time_remaining"] = np.maximum(0, 90 - (df["minute"] + df["second"] / 60))

    shot_xg = df["shot.statsbomb_xg"].fillna(0) if "shot.statsbomb_xg" in df.columns else 0
    df["home_xg_event"] = np.where(df["team.name"] == df["home_team"], shot_xg, 0)
    df["away_xg_event"] = np.where(df["team.name"] == df["away_team"], shot_xg, 0)
    df["home_xg_cumulative"] = df.groupby("match_id")["home_xg_event"].cumsum()
    df["away_xg_cumulative"] = df.groupby("match_id")["away_xg_event"].cumsum()

    rows = []
    grouped = df.groupby(["match_id", "possession", "possession_team.name"], dropna=False)
    for (match_id, poss_id, team), poss_df in grouped:
        end = poss_df.iloc[-1]
        home_team = end["home_team"]
        away_team = end["away_team"]

        row = {
            "match_id": match_id,
            "possession": poss_id,
            "team": team,
            "home_team": home_team,
            "away_team": away_team,
            "possession_duration": poss_df["duration"].sum(skipna=True),
            "events_count": len(poss_df),
        }

        dribbles = poss_df[poss_df["type.name"] == "Dribble"]
        row["dribbles_count"] = len(dribbles)
        row["dribbles_completed_pct"] = (
            (dribbles["dribble.outcome.name"] == "Complete").mean()
            if len(dribbles) and "dribble.outcome.name" in dribbles.columns
            else 0
        )

        carries = poss_df[poss_df["type.name"] == "Carry"].copy()
        row["carries_count"] = len(carries)
        if len(carries):
            carries["carry_length"] = carries.apply(
                lambda r: euclidean_dist(r.get("location"), r.get("carry.end_location")),
                axis=1,
            )
            row["carries_total_length"] = carries["carry_length"].sum(skipna=True)
            row["carries_mean_length"] = carries["carry_length"].mean(skipna=True)
        else:
            row["carries_total_length"] = 0
            row["carries_mean_length"] = 0

        passes = poss_df[poss_df["type.name"] == "Pass"]
        row["passes_count"] = len(passes)
        if len(passes):
            row["passes_completed_pct"] = passes["pass.outcome.name"].isna().mean() if "pass.outcome.name" in passes else 0
            row["passes_total_length"] = passes["pass.length"].sum(skipna=True) if "pass.length" in passes else 0
            row["passes_mean_length"] = passes["pass.length"].mean(skipna=True) if "pass.length" in passes else 0
        else:
            row["passes_completed_pct"] = 0
            row["passes_total_length"] = 0
            row["passes_mean_length"] = 0

        shots = poss_df[poss_df["type.name"] == "Shot"]
        row["shots_count"] = len(shots)
        if len(shots):
            row["shots_xg_total"] = shots["shot.statsbomb_xg"].sum(skipna=True) if "shot.statsbomb_xg" in shots else 0
            row["shots_xg_mean"] = shots["shot.statsbomb_xg"].mean(skipna=True) if "shot.statsbomb_xg" in shots else 0
            row["shots_on_target_pct"] = (
                shots["shot.outcome.name"].isin(["Goal", "Saved", "Saved To Post"]).mean()
                if "shot.outcome.name" in shots
                else 0
            )
        else:
            row["shots_xg_total"] = 0
            row["shots_xg_mean"] = 0
            row["shots_on_target_pct"] = 0

        row["team_elo"] = end["home_elo"] if team == home_team else end["away_elo"]
        row["opp_elo"] = end["away_elo"] if team == home_team else end["home_elo"]
        row["home_elo_diff"] = end["home_elo_diff"]
        row["goal_diff"] = end["goal_diff"]
        row["game_time_pct"] = end["game_time_pct"]
        row["time_remaining"] = end["time_remaining"]
        row["home_score"] = end["home_score"]
        row["away_score"] = end["away_score"]
        row["final_home_score"] = end["final_home_score"]
        row["final_away_score"] = end["final_away_score"]
        row["home_xg_cumulative"] = end["home_xg_cumulative"]
        row["away_xg_cumulative"] = end["away_xg_cumulative"]
        row["is_home_possession"] = int(team == home_team)
        row["future_home_goals"] = max(0, row["final_home_score"] - row["home_score"])
        row["future_away_goals"] = max(0, row["final_away_score"] - row["away_score"])

        if row["final_home_score"] > row["final_away_score"]:
            row["final_outcome"] = 0
        elif row["final_home_score"] == row["final_away_score"]:
            row["final_outcome"] = 1
        else:
            row["final_outcome"] = 2

        rows.append(row)

    feat_df = pd.DataFrame(rows).sort_values(["match_id", "possession"]).reset_index(drop=True)
    feat_df["oracle_home_win_prob"], feat_df["oracle_draw_prob"], feat_df["oracle_away_win_prob"] = poisson_wdl_from_lambdas(
        feat_df["goal_diff"].values,
        feat_df["future_home_goals"].values,
        feat_df["future_away_goals"].values,
    )
    return feat_df


def validate_targets(possession_features: pd.DataFrame) -> dict[str, float]:
    """Return target consistency checks for remaining-goals labels."""
    df = possession_features.copy()
    calc_final_home = df["home_score"] + df["future_home_goals"]
    calc_final_away = df["away_score"] + df["future_away_goals"]
    return {
        "home_target_consistency": float((calc_final_home == df["final_home_score"]).mean()),
        "away_target_consistency": float((calc_final_away == df["final_away_score"]).mean()),
        "negative_home_targets": int((df["future_home_goals"] < 0).sum()),
        "negative_away_targets": int((df["future_away_goals"] < 0).sum()),
        "matches": int(df["match_id"].nunique()),
    }
