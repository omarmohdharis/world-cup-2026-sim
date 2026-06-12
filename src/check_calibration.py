"""Quick sanity check: is the Elo expected score calibrated against reality?"""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "processed"

df = pd.read_csv(OUT / "model_data.csv", parse_dates=["date"])
df = df[df.date >= "1990-01-01"]

# Actual points share for the home team: win=1, draw=0.5, loss=0.
df["w_home"] = df.outcome.map({"home_win": 1.0, "draw": 0.5, "away_win": 0.0})
df["bin"] = pd.cut(df.home_expected, bins=[0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
cal = df.groupby("bin", observed=True).agg(
    predicted=("home_expected", "mean"),
    actual=("w_home", "mean"),
    n=("w_home", "size"),
)
print(cal.round(3).to_string())

print("\nMost recent rows with features:")
cols = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "home_elo", "away_elo", "home_expected",
    "home_form5", "away_form5", "home_rest_days",
]
print(df[cols].tail(3).round(2).to_string(index=False))
