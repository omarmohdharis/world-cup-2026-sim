"""
Step 4: Train the match prediction model.

Approach: DOUBLE POISSON GLM with a Dixon-Coles low-score correction.

Each team's goals in a match are modelled as Poisson(lambda), with

    log(lambda) = b0 + b1*(elo_gap/100) + b2*is_home + b3*gf5 + b4*opp_ga5
                  + b5*(rest_days/7)

fit on a long-format frame (two rows per match — one per team). One shared
model produces lambda for both sides; the joint scoreline distribution is the
outer product of the two Poisson pmfs over a 0..12 goal grid, with the
Dixon-Coles tau() adjustment multiplied into the four low-score cells
(0-0, 1-0, 0-1, 1-1) to fix the slight draw underprediction of independent
Poissons:

    tau(0,0) = 1 - lh*la*rho      tau(1,0) = 1 + la*rho
    tau(0,1) = 1 + lh*rho         tau(1,1) = 1 - rho

rho is fit by grid search, minimizing W/D/L log loss on the training years.

Validation: TIME-BASED split (train 2000-2023, test 2024+). Random splits
would leak: a 2025 row's Elo encodes results that are "future" relative to a
2024 test row. Baselines for comparison:
  - constant class frequencies (the floor)
  - multinomial logistic regression on elo_diff alone (the "Elo-only" model)

Outputs models/poisson_model.pkl: {"model": fitted GLM, "rho": float,
"features": [...], "max_goals": int} for the tournament simulator.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS = Path(__file__).resolve().parents[1] / "models"

TRAIN_START = "2000-01-01"
TEST_START = "2024-01-01"
MAX_GOALS = 12
FEATURES = ["elo_gap100", "is_home", "gf5", "opp_ga5", "rest_wk"]


def long_design(matches: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match: each team as the 'attacking' side."""
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        rows.append(
            pd.DataFrame(
                {
                    "match_id": matches.index,
                    "goals": matches[f"{side}_score"],
                    "elo_gap100": (matches[f"{side}_elo"] - matches[f"{opp}_elo"]) / 100,
                    "is_home": ((side == "home") & ~matches.neutral).astype(float),
                    "gf5": matches[f"{side}_gf5"],
                    "opp_ga5": matches[f"{opp}_ga5"],
                    "rest_wk": matches[f"{side}_rest_days"] / 7,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def score_matrix(lh: float, la: float, rho: float) -> np.ndarray:
    """Joint P(home=i, away=j) grid with Dixon-Coles adjustment."""
    grid = np.outer(
        poisson.pmf(np.arange(MAX_GOALS + 1), lh),
        poisson.pmf(np.arange(MAX_GOALS + 1), la),
    )
    grid[0, 0] *= 1 - lh * la * rho
    grid[1, 0] *= 1 + la * rho
    grid[0, 1] *= 1 + lh * rho
    grid[1, 1] *= 1 - rho
    # Large |rho| with high lambdas can push a cell negative; the DC
    # adjustment is only valid where tau > 0, so clamp before renormalizing.
    grid = np.maximum(grid, 0.0)
    return grid / grid.sum()


def wdl_probs(lh: np.ndarray, la: np.ndarray, rho: float) -> np.ndarray:
    """Per-match [P(away_win), P(draw), P(home_win)] from the score grids."""
    out = np.empty((len(lh), 3))
    for i, (h, a) in enumerate(zip(lh, la)):
        g = score_matrix(h, a, rho)
        out[i] = [np.triu(g, 1).sum(), np.trace(g), np.tril(g, -1).sum()]
    return out


def lambdas_for(matches: pd.DataFrame, model) -> tuple[np.ndarray, np.ndarray]:
    X = long_design(matches)
    lam = model.predict(sm.add_constant(X[FEATURES], has_constant="add"))
    n = len(matches)
    return lam.values[:n], lam.values[n:]  # home rows first, away rows second


def fit_rho(lh, la, y, grid=np.arange(-0.30, 0.201, 0.005)) -> float:
    losses = [log_loss(y, wdl_probs(lh, la, r), labels=[0, 1, 2]) for r in grid]
    return float(grid[int(np.argmin(losses))])


def main():
    MODELS.mkdir(exist_ok=True)
    df = pd.read_csv(DATA / "model_data.csv", parse_dates=["date"])
    df = df[df.date >= TRAIN_START].dropna(
        subset=[f"{s}_{c}" for s in ("home", "away") for c in ("gf5", "ga5", "rest_days")]
    )
    train = df[df.date < TEST_START].copy()
    test = df[df.date >= TEST_START].copy()
    # y: 0 = away win, 1 = draw, 2 = home win
    to_y = {"away_win": 0, "draw": 1, "home_win": 2}
    y_train = train.outcome.map(to_y).values
    y_test = test.outcome.map(to_y).values
    print(f"train: {len(train)} matches ({TRAIN_START[:4]}-2023)   test: {len(test)} (2024+)")

    # --- double Poisson GLM ---------------------------------------------
    design = long_design(train)
    glm = sm.GLM(
        design.goals,
        sm.add_constant(design[FEATURES], has_constant="add"),
        family=sm.families.Poisson(),
    ).fit()
    print("\n", glm.summary().tables[1])

    lh_tr, la_tr = lambdas_for(train, glm)
    rho = fit_rho(lh_tr, la_tr, y_train)
    print(f"\nDixon-Coles rho = {rho:.3f}")

    lh_te, la_te = lambdas_for(test, glm)
    p_poisson = wdl_probs(lh_te, la_te, rho)
    p_poisson_raw = wdl_probs(lh_te, la_te, 0.0)

    # --- baselines -------------------------------------------------------
    freq = np.bincount(y_train, minlength=3) / len(y_train)
    p_const = np.tile(freq, (len(test), 1))

    logit = LogisticRegression(max_iter=1000)
    logit.fit(train[["elo_diff"]], y_train)
    p_logit = logit.predict_proba(test[["elo_diff"]])

    # --- evaluation ------------------------------------------------------
    print(f"\n{'model':<34}{'log loss':>10}{'accuracy':>10}{'P(draw) avg':>13}")
    for name, p in [
        ("constant class frequencies", p_const),
        ("multinomial logit (elo_diff only)", p_logit),
        ("double Poisson (rho=0)", p_poisson_raw),
        ("double Poisson + Dixon-Coles", p_poisson),
    ]:
        ll = log_loss(y_test, p, labels=[0, 1, 2])
        acc = (p.argmax(1) == y_test).mean()
        print(f"{name:<34}{ll:>10.4f}{acc:>10.3f}{p[:, 1].mean():>13.3f}")
    print(f"{'actual draw rate (test)':<34}{'':>10}{'':>10}{(y_test == 1).mean():>13.3f}")
    print(f"\navg goals/team — predicted {np.r_[lh_te, la_te].mean():.2f}, "
          f"actual {np.r_[test.home_score, test.away_score].mean():.2f}")

    with open(MODELS / "poisson_model.pkl", "wb") as f:
        pickle.dump(
            {"model": glm, "rho": rho, "features": FEATURES, "max_goals": MAX_GOALS},
            f,
        )
    print(f"\nsaved -> {MODELS / 'poisson_model.pkl'}")


if __name__ == "__main__":
    main()
