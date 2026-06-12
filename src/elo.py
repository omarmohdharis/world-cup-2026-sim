"""
Step 2: Elo rating engine for international football.

This follows the World Football Elo Ratings system (eloratings.net), the de
facto standard for national teams. One match updates both teams' ratings by

    R_new = R_old + K * G * (W - W_e)

where

  W    actual result: 1 win, 0.5 draw, 0 loss (penalty shootouts = draw,
       because the match itself finished level).

  W_e  expected result from the logistic curve

           W_e = 1 / (1 + 10 ** (-dr / 400))

       dr = (own rating + home advantage if playing at home) - opponent's
       rating (+ their home advantage if they are at home). A 400-point gap
       means the stronger team is expected to score ~0.91; equal ratings
       give 0.5. HOME_ADVANTAGE = 100 Elo points (~64% expected score for
       the home team between equals), applied only when neutral == False.

  K    importance of the match (how much rating can move):
           60  World Cup finals tournament
           50  continental finals (Euro, Copa America, AFCON, Asian Cup,
               Gold Cup, ...) and Confederations Cup
           40  World Cup & continental qualifiers, Nations League
           30  all other tournaments
           20  friendlies

  G    goal-difference multiplier — winning big should move ratings more:
           margin 0 or 1 -> 1
           margin 2      -> 1.5
           margin N >= 3 -> (11 + N) / 8     (3 -> 1.75, 4 -> 1.875, ...)

The system is zero-sum: what the winner gains, the loser drops. Everyone
starts at INITIAL_RATING = 1500 and 150+ years of matches burn ratings in.
"""

import pandas as pd

INITIAL_RATING = 1500.0
HOME_ADVANTAGE = 100.0

K_WORLD_CUP = 60
K_CONTINENTAL_FINAL = 50
K_QUALIFIER = 40
K_OTHER = 30
K_FRIENDLY = 20

CONTINENTAL_FINALS = {
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "CONCACAF Championship",
    "Oceania Nations Cup",
    "Confederations Cup",
}


def k_factor(tournament: str) -> int:
    if tournament == "FIFA World Cup":
        return K_WORLD_CUP
    if tournament in CONTINENTAL_FINALS:
        return K_CONTINENTAL_FINAL
    if "qualification" in tournament or "Nations League" in tournament:
        return K_QUALIFIER
    if tournament == "Friendly":
        return K_FRIENDLY
    return K_OTHER


def expected_score(rating: float, opp_rating: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opp_rating - rating) / 400.0))


def goal_multiplier(margin: int) -> float:
    margin = abs(margin)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11 + margin) / 8


def update(
    home_rating: float,
    away_rating: float,
    home_score: int,
    away_score: int,
    tournament: str,
    neutral: bool,
) -> tuple[float, float]:
    """Return (new_home_rating, new_away_rating) after one match."""
    hfa = 0.0 if neutral else HOME_ADVANTAGE
    w_e_home = expected_score(home_rating + hfa, away_rating)

    if home_score > away_score:
        w_home = 1.0
    elif home_score < away_score:
        w_home = 0.0
    else:
        w_home = 0.5

    delta = (
        k_factor(tournament)
        * goal_multiplier(home_score - away_score)
        * (w_home - w_e_home)
    )
    return home_rating + delta, away_rating - delta


def run_history(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replay all matches chronologically.

    Returns the matches frame with pre-match rating columns added
    (home_elo, away_elo, elo_diff, home_expected) and the final ratings
    dict {team: rating} to seed the tournament simulation.
    """
    ratings: dict[str, float] = {}
    home_elos, away_elos, expecteds = [], [], []

    for m in matches.itertuples():
        rh = ratings.get(m.home_team, INITIAL_RATING)
        ra = ratings.get(m.away_team, INITIAL_RATING)
        hfa = 0.0 if m.neutral else HOME_ADVANTAGE

        home_elos.append(rh)
        away_elos.append(ra)
        expecteds.append(expected_score(rh + hfa, ra))

        ratings[m.home_team], ratings[m.away_team] = update(
            rh, ra, m.home_score, m.away_score, m.tournament, m.neutral
        )

    out = matches.copy()
    out["home_elo"] = home_elos
    out["away_elo"] = away_elos
    out["elo_diff"] = (
        out.home_elo
        + (~out.neutral).astype(float) * HOME_ADVANTAGE
        - out.away_elo
    )
    out["home_expected"] = expecteds
    return out, ratings
