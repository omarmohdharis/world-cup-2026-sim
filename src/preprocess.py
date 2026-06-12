"""
Step 1: Pre-processing for the 2026 World Cup simulation project.

Reads the raw Kaggle data (martj42/international-football-results) and produces:
  - data/processed/matches.csv         all *played* matches, cleaned
  - data/processed/wc2026_fixtures.csv the real 2026 WC group-stage fixtures (unplayed)

Cleaning steps
--------------
1. Parse dates, sort chronologically.
2. Standardize team names:
   a. Date-aware mapping of former names -> current names (former_names.csv),
      e.g. Zaire -> DR Congo, Dahomey -> Benin. Date-aware matters because a
      former name can collide with a different current team (pre-1954
      "Ireland" is today's Northern Ireland).
   b. Successor mapping for defunct states so a team's rating history carries
      over (the convention used by eloratings.net): Soviet Union -> Russia,
      Yugoslavia / Serbia and Montenegro -> Serbia, Czechoslovakia -> Czech
      Republic. East Germany and Saarland are left as separate defunct teams.
3. Split played matches (score present) from scheduled fixtures (score null).
4. Attach penalty-shootout winners from shootouts.csv. For Elo a match that
   went to penalties counts as a DRAW (the full-time/AET score is tied); the
   shootout winner is kept in its own column for knockout-round logic.
5. Add the regulation-time outcome column used as the prediction target.
"""

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parents[1] / "data" / "processed"

# Defunct-state successors: carry rating history onto the modern team.
SUCCESSORS = {
    "Soviet Union": "Russia",
    "Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "Czechoslovakia": "Czech Republic",
}


def load_raw():
    results = pd.read_csv(RAW / "results.csv", parse_dates=["date"])
    shootouts = pd.read_csv(RAW / "shootouts.csv", parse_dates=["date"])
    former = pd.read_csv(
        RAW / "former_names.csv", parse_dates=["start_date", "end_date"]
    )
    return results, shootouts, former


def standardize_names(df: pd.DataFrame, former: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for row in former.itertuples():
        in_window = (df.date >= row.start_date) & (df.date <= row.end_date)
        for col in ("home_team", "away_team"):
            df.loc[in_window & (df[col] == row.former), col] = row.current
    for col in ("home_team", "away_team"):
        df[col] = df[col].replace(SUCCESSORS)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results, shootouts, former = load_raw()

    results = standardize_names(results, former)
    shootouts = standardize_names(shootouts, former)
    results = results.sort_values("date").reset_index(drop=True)

    played = results.dropna(subset=["home_score", "away_score"]).copy()
    fixtures = results[results.home_score.isna()].copy()

    played[["home_score", "away_score"]] = played[
        ["home_score", "away_score"]
    ].astype(int)

    # Attach shootout winners (knockout matches that ended level).
    shootouts = shootouts.rename(columns={"winner": "shootout_winner"})
    played = played.merge(
        shootouts[["date", "home_team", "away_team", "shootout_winner"]],
        on=["date", "home_team", "away_team"],
        how="left",
    )

    # Regulation/AET outcome — the Elo result and the modelling target.
    played["outcome"] = "draw"
    played.loc[played.home_score > played.away_score, "outcome"] = "home_win"
    played.loc[played.home_score < played.away_score, "outcome"] = "away_win"

    played.to_csv(OUT / "matches.csv", index=False)
    fixtures.to_csv(OUT / "wc2026_fixtures.csv", index=False)

    print(f"played matches : {len(played)}  ({played.date.min().date()} -> {played.date.max().date()})")
    print(f"wc2026 fixtures: {len(fixtures)}")
    print(f"teams          : {pd.concat([played.home_team, played.away_team]).nunique()}")
    print(f"shootouts merged: {played.shootout_winner.notna().sum()}")
    print(played.outcome.value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
