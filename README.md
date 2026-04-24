# Dynamic Soccer Win Probability

This project builds an in-game soccer win probability model from StatsBomb event data and Club Elo ratings. The model works at possession level: at each game state, it predicts the remaining goals for the home and away team, then converts those predicted goal rates into home-win, draw, and away-win probabilities with a Poisson outcome layer.

The target design follows the idea used in Robberechts, Van Haaren, and Davis (KDD 2021): predict future scoring distributions first, then derive win/draw/loss probabilities.

## Results

Current cleaned experiment:

| Metric | LSTM | Random Forest Baseline |
|---|---:|---:|
| Remaining-goals MAE | 0.6522 | 0.7692 |
| Ranked Probability Score | 0.1069 | 0.1382 |
| Multiclass Brier Score | 0.3563 | 0.4503 |

Target validation after cleaning:

| Check | Value |
|---|---:|
| Matches retained | 793 |
| Home target consistency | 1.000000 |
| Away target consistency | 1.000000 |
| Negative home targets | 0 |
| Negative away targets | 0 |

## Project Structure

```text
dl-win-probability/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── features.py
│   ├── artifacts.py
│   ├── metrics.py
│   ├── models.py
│   └── viz.py
├── data/
│   ├── .gitkeep
│   └── split_ids.pkl
├── models/
│   └── scaler.pkl
├── results/
│   └── all_predictions.pkl
└── notebooks/
    ├── 01_data_collection.ipynb
    ├── 02_preprocessing.ipynb
    ├── 03_models.ipynb
    └── 04_evaluation.ipynb
```

## Method

1. Load StatsBomb match and event data.
2. Merge historical Club Elo ratings using backward date matching.
3. Aggregate event data into possession-level features.
4. Build supervised targets:
   - `future_home_goals = final_home_score - current_home_score`
   - `future_away_goals = final_away_score - current_away_score`
5. Validate target consistency against official final scores.
6. Train an LSTM with Poisson loss to predict remaining home and away goals.
7. Convert predicted remaining-goal means into home/draw/away probabilities.
8. Evaluate with remaining-goals MAE, Ranked Probability Score, and multiclass Brier score.

## Data Cleaning Note

Official final scores from match metadata are treated as authoritative. Matches where event-derived goal counts do not match official final scores are excluded before target construction, because inconsistent score reconstruction would produce incorrect supervised labels.

## Baseline

The baseline is a Random Forest regressor trained on simple game-state features:

- match time
- score difference
- time remaining
- Elo difference

It predicts the same remaining-goals targets as the LSTM and uses the same Poisson probability conversion.

## Reproducing

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebooks in order:

1. `notebooks/01_data_collection.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03_models.ipynb`
4. `notebooks/04_evaluation.ipynb`

Raw StatsBomb event files and large Elo CSV files are not committed. Place them locally according to the paths used in the data collection notebook.

## Next Improvements

- Add chronological train/validation/test split for future-season generalization.
- Add Elo ablation: train with and without Elo features.
- Add calibration plots for home/draw/away probabilities.
- Export `results/all_predictions.pkl` from the evaluation notebook.

## Artifact Export

After training/evaluation, save reproducibility artifacts with:

```python
from src.artifacts import save_split_ids, save_predictions

save_split_ids("data/split_ids.pkl", train_ids, val_ids, test_ids)
save_predictions("results/all_predictions.pkl", test_rows, probs, y_pred_goals)
```
