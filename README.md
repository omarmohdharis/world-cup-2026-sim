# World Cup 2026 simulation

Monte Carlo simulation of the 2026 FIFA World Cup, built on
[international football results since 1872](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017).

**Live dashboard:** see the GitHub Pages site for this repo (`docs/`).

## Method

1. **Elo ratings** (`src/elo.py`) — eloratings.net formulation: K = 20-60 by match
   importance, home advantage = 100, goal-difference multiplier, penalty
   shootouts count as draws. Replayed over 49,000+ matches since 1872.
2. **Match model** (`src/train_model.py`) — double Poisson GLM on goals
   (features: Elo gap, home, attack/defense form, rest) with a Dixon-Coles
   low-score correction. Validated on a 2024+ time split against an Elo-only
   logistic baseline.
3. **Tournament simulation** (`src/simulate.py`) — real group draw and played
   results locked in; 10,000 tournament runs; Elo and form update after every
   simulated match; official FIFA knockout bracket (matches 73-104) with
   constraint-matched third-place allocation.

## Reproduce

```bash
pip install pandas numpy scipy scikit-learn statsmodels kagglehub
python src/preprocess.py        # needs data/raw (download via kagglehub)
python src/build_features.py
python src/train_model.py
python src/simulate.py 10000    # also regenerates docs/data.js
```
