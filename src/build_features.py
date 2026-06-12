"""
Step 3: Feature engineering.

Takes the cleaned matches, replays history through the Elo engine, and adds
per-team rolling form features. Every feature is computed from matches
STRICTLY BEFORE the one being described (shift(1) everywhere) — no leakage
of the match's own result into its features.

Outputs
-------
data/processed/model_data.csv      one row per played match, with features + target
data/processed/elo_current.csv     today's Elo rating for every team
data/processed/wc2026_fixtures.csv fixtures re-saved with current Elo attached

Features per match
------------------
  home_elo, away_elo          pre-match Elo ratings
  elo_diff                    home_elo + HFA*(not neutral) - away_elo
  home_expected               Elo expected score for the home team
  neutral                     neutral venue flag
  k                           match importance (Elo K factor)
  *_form5  / *_form10         mean points (3/1/0) over team's last 5 / 10 games
  *_gf5, *_ga5                mean goals for / against over last 5 games
  *_rest_days                 days since the team's previous match (capped at 60)
  outcome                     target: home_win / draw / away_win
"""

from pathlib import Path

import pandas as pd

import elo

OUT = Path(__file__).resolve().parents[1] / "data" / "processed"


def team_long_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, match): points, goals for/against."""
    home = pd.DataFrame(
        {
            "match_id": matches.index,
            "team": matches.home_team,
            "date": matches.date,
            "gf": matches.home_score,
            "ga": matches.away_score,
        }
    )
    away = pd.DataFrame(
        {
            "match_id": matches.index,
            "team": matches.away_team,
            "date": matches.date,
            "gf": matches.away_score,
            "ga": matches.home_score,
        }
    )
    long = pd.concat([home, away]).sort_values(["team", "date", "match_id"])
    long["points"] = 1.0
    long.loc[long.gf > long.ga, "points"] = 3.0
    long.loc[long.gf < long.ga, "points"] = 0.0
    return long


def rolling_features(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby("team")
    # shift(1): only matches BEFORE the current one feed its features.
    long["form5"] = g.points.transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    long["form10"] = g.points.transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )
    long["gf5"] = g.gf.transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    long["ga5"] = g.ga.transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    long["rest_days"] = g.date.transform(
        lambda s: s.diff().dt.days
    ).clip(upper=60)
    return long


def main():
    matches = pd.read_csv(OUT / "matches.csv", parse_dates=["date"])

    matches, ratings = elo.run_history(matches)
    matches["k"] = matches.tournament.map(elo.k_factor)

    long = rolling_features(team_long_frame(matches))
    feat_cols = ["form5", "form10", "gf5", "ga5", "rest_days"]
    indexed = long.set_index(["match_id", "team"])[feat_cols]
    for side in ("home", "away"):
        side_feats = indexed.loc[
            list(zip(matches.index, matches[f"{side}_team"]))
        ].reset_index(drop=True)
        side_feats.columns = [f"{side}_{c}" for c in feat_cols]
        matches = pd.concat([matches, side_feats.set_index(matches.index)], axis=1)

    matches.to_csv(OUT / "model_data.csv", index=False)

    # Current ratings table (only teams active in the last 4 years).
    last_seen = (
        long.groupby("team").date.max().rename("last_match")
    )
    elo_now = (
        pd.Series(ratings, name="elo")
        .rename_axis("team")
        .to_frame()
        .join(last_seen)
        .sort_values("elo", ascending=False)
    )
    elo_now["active"] = elo_now.last_match >= "2022-06-12"
    elo_now.to_csv(OUT / "elo_current.csv")

    # Attach current Elo to the real WC2026 fixtures for the simulator.
    fixtures = pd.read_csv(OUT / "wc2026_fixtures.csv", parse_dates=["date"])
    fixtures["home_elo"] = fixtures.home_team.map(ratings)
    fixtures["away_elo"] = fixtures.away_team.map(ratings)
    fixtures.to_csv(OUT / "wc2026_fixtures.csv", index=False)

    print(f"model_data.csv : {matches.shape[0]} rows, {matches.shape[1]} cols")
    print("\nTop 15 Elo ratings today:")
    print(
        elo_now[elo_now.active]
        .head(15)
        .elo.round(0)
        .astype(int)
        .to_string()
    )
    missing = fixtures[fixtures.home_elo.isna() | fixtures.away_elo.isna()]
    print(f"\nWC2026 fixtures missing a rating: {len(missing)}")
    if len(missing):
        print(missing[["home_team", "away_team"]].to_string())


if __name__ == "__main__":
    main()
