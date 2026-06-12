"""
Step 5: Monte Carlo simulation of the 2026 World Cup.

Usage:  python src/simulate.py [n_sims]      (default 10000)

Per simulation run:
  1. Group stage: start from the REAL standings (Group A's two played
     results are locked in), then sample scores for the 70 remaining real
     fixtures from the double-Poisson + Dixon-Coles model.
  2. After EVERY simulated match, the team's Elo (K=60, eloratings.net
     update) and rolling form (last-5 goals for/against) are updated, so a
     team that storms its group enters the knockouts stronger.
  3. Group ranking: points, goal difference, goals scored, then random
     (head-to-head and fair-play tiebreakers are approximated by the random
     step). Top 2 advance plus the 8 best third-placed teams (ranked by the
     same keys).
  4. Knockouts follow the official FIFA bracket (matches 73-104). Qualified
     third-placed teams are assigned to their bracket slots by constraint
     matching against the allowed-groups list of each slot (FIFA's Annex C
     defines one valid assignment per scenario; we pick a random valid one).
  5. Knockout draws go to extra time/penalties: winner sampled with
     probability = Elo expected score; for ratings the match counts as a
     draw (the eloratings.net shootout convention).

Venue assumptions: group-stage home advantage comes from the real fixture
list (hosts are non-neutral at home). Knockout matches are treated as
neutral, except the USA from the quarterfinals on (all QF/SF/final venues
are in the USA).

Output: results/sim_results.csv with per-team probabilities of reaching
each stage, plus a printed summary.
"""

import json
import pickle
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

import elo

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

GROUPS = {
    "A": ["Mexico", "South Korea", "Czech Republic", "South Africa"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
TEAM_GROUP = {t: g for g, teams in GROUPS.items() for t in teams}

# Official R32 bracket (FIFA matches 73-88). "3?" = third-placed team slot.
R32 = {
    73: ("2A", "2B"), 74: ("1E", "3?"), 75: ("1F", "2C"), 76: ("1C", "2F"),
    77: ("1I", "3?"), 78: ("2E", "2I"), 79: ("1A", "3?"), 80: ("1L", "3?"),
    81: ("1D", "3?"), 82: ("1G", "3?"), 83: ("2K", "2L"), 84: ("1H", "2J"),
    85: ("1B", "3?"), 86: ("1J", "2H"), 87: ("1K", "3?"), 88: ("2D", "2G"),
}
# Which groups' third-placed teams may fill each slot (FIFA Annex C).
SLOT_ALLOWED = {
    74: "ABCDF", 77: "CDFGH", 79: "CEFHI", 80: "EHIJK",
    81: "BEFIJ", 82: "AEHIJ", 85: "EFGIJ", 87: "DEIJL",
}
R16 = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
       93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF = {101: (97, 98), 102: (99, 100)}
ROUND_DATES = {"r32": "2026-06-30", "r16": "2026-07-05", "qf": "2026-07-10",
               "sf": "2026-07-14", "final": "2026-07-19"}
HOSTS_QF_ON = {"United States"}  # all QF/SF/final venues are in the USA

MAX_GOALS = 12
LOG_FACT = np.cumsum(np.r_[0.0, np.log(np.arange(1, MAX_GOALS + 1))])
KS = np.arange(MAX_GOALS + 1)


def poisson_pmf(lam: float) -> np.ndarray:
    return np.exp(KS * np.log(lam) - lam - LOG_FACT)


class Model:
    def __init__(self, path):
        with open(path, "rb") as f:
            art = pickle.load(f)
        params = art["model"].params
        self.beta = np.array([params["const"]] + [params[f] for f in art["features"]])
        self.rho = art["rho"]

    def lambdas(self, s_home, s_away, date, home_is_home):
        def lam(me, opp, is_home):
            rest = min((date - me.last_date).days, 60) / 7
            x = np.array([1.0, (me.elo - opp.elo) / 100, float(is_home),
                          me.gf5(), opp.ga5(), rest])
            return float(np.exp(x @ self.beta))
        return lam(s_home, s_away, home_is_home), lam(s_away, s_home, False)

    def score_grid(self, lh, la):
        grid = np.outer(poisson_pmf(lh), poisson_pmf(la))
        grid[0, 0] *= 1 - lh * la * self.rho
        grid[1, 0] *= 1 + la * self.rho
        grid[0, 1] *= 1 + lh * self.rho
        grid[1, 1] *= 1 - self.rho
        grid = np.maximum(grid, 0.0)
        return grid / grid.sum()


class TeamState:
    __slots__ = ("elo", "recent", "last_date")

    def __init__(self, rating, recent, last_date):
        self.elo = rating
        self.recent = deque(recent, maxlen=5)  # (gf, ga) of last 5 matches
        self.last_date = last_date

    def gf5(self):
        return sum(g for g, _ in self.recent) / len(self.recent)

    def ga5(self):
        return sum(a for _, a in self.recent) / len(self.recent)

    def copy(self):
        return TeamState(self.elo, self.recent, self.last_date)


def seed_states(matches: pd.DataFrame) -> dict[str, TeamState]:
    ratings = elo.run_history(matches)[1]
    states = {}
    for team in TEAM_GROUP:
        mine = matches[(matches.home_team == team) | (matches.away_team == team)]
        last5 = mine.tail(5)
        recent = [
            (r.home_score, r.away_score) if r.home_team == team
            else (r.away_score, r.home_score)
            for r in last5.itertuples()
        ]
        states[team] = TeamState(ratings[team], recent, mine.date.max())
    return states


def play(model, states, home, away, date, neutral, rng, knockout=False):
    """Simulate one match; update Elo + form; return (gh, ga, winner)."""
    sh, sa = states[home], states[away]
    lh, la = model.lambdas(sh, sa, date, home_is_home=not neutral)
    grid = model.score_grid(lh, la)
    flat = rng.choice(grid.size, p=grid.ravel())
    gh, ga = divmod(flat, MAX_GOALS + 1)

    winner = home if gh > ga else away if ga > gh else None
    if knockout and winner is None:
        # extra time / penalties: Elo expected score, no home advantage
        p_home = elo.expected_score(sh.elo, sa.elo)
        winner = home if rng.random() < p_home else away

    sh.elo, sa.elo = elo.update(sh.elo, sa.elo, gh, ga, "FIFA World Cup", neutral)
    sh.recent.append((gh, ga))
    sa.recent.append((ga, gh))
    sh.last_date = sa.last_date = date
    return gh, ga, winner


def rank_group(rows, rng):
    """rows: {team: [pts, gd, gf]} -> teams sorted best first."""
    return sorted(rows, key=lambda t: (*rows[t], rng.random()), reverse=True)


def assign_thirds(qualified_groups, rng):
    """Match qualified third-place groups to bracket slots (backtracking)."""
    slots = sorted(SLOT_ALLOWED, key=lambda s: sum(g in SLOT_ALLOWED[s] for g in qualified_groups))
    assignment = {}

    def backtrack(i, remaining):
        if i == len(slots):
            return True
        slot = slots[i]
        options = [g for g in remaining if g in SLOT_ALLOWED[slot]]
        rng.shuffle(options)
        for g in options:
            assignment[slot] = g
            if backtrack(i + 1, remaining - {g}):
                return True
            del assignment[slot]
        return False

    if not backtrack(0, set(qualified_groups)):  # shouldn't happen (Annex C)
        for slot, g in zip(slots, qualified_groups):
            assignment[slot] = g
    return assignment


def simulate_once(model, base_states, fixtures, played, rng):
    states = {t: s.copy() for t, s in base_states.items()}
    table = {g: {t: [0, 0, 0] for t in teams} for g, teams in GROUPS.items()}

    def record(group, team, gf, ga):
        pts = 3 if gf > ga else 1 if gf == ga else 0
        row = table[group][team]
        row[0] += pts
        row[1] += gf - ga
        row[2] += gf

    for m in played.itertuples():  # real results already on the books
        record(TEAM_GROUP[m.home_team], m.home_team, m.home_score, m.away_score)
        record(TEAM_GROUP[m.away_team], m.away_team, m.away_score, m.home_score)
    for m in fixtures.itertuples():
        gh, ga, _ = play(model, states, m.home_team, m.away_team, m.date, m.neutral, rng)
        record(TEAM_GROUP[m.home_team], m.home_team, gh, ga)
        record(TEAM_GROUP[m.away_team], m.away_team, ga, gh)

    pos = {}  # "1A" -> team
    thirds = {}  # group -> team
    orders = {}  # group -> [teams best to worst]
    for g, rows in table.items():
        order = rank_group(rows, rng)
        orders[g] = order
        pos[f"1{g}"], pos[f"2{g}"] = order[0], order[1]
        thirds[g] = order[2]
    third_rank = sorted(
        thirds, key=lambda g: (*table[g][thirds[g]], rng.random()), reverse=True
    )
    qualified = set(third_rank[:8])
    slot_group = assign_thirds(qualified, rng)
    for slot, g in slot_group.items():
        pos[f"3?{slot}"] = thirds[g]

    reached = {t: "group" for t in TEAM_GROUP}
    for g in GROUPS:
        reached[pos[f"1{g}"]] = reached[pos[f"2{g}"]] = "r32"
    for g in qualified:
        reached[thirds[g]] = "r32"

    def ko_round(pairs, resolve, stage, date):
        winners = {}
        for match_no, (a, b) in pairs.items():
            ta, tb = resolve(match_no, a), resolve(match_no, b)
            host_playing = stage in ("qf", "sf", "final") and bool({ta, tb} & HOSTS_QF_ON)
            if host_playing and tb in HOSTS_QF_ON:
                ta, tb = tb, ta  # host takes the home slot
            _, _, w = play(model, states, ta, tb, pd.Timestamp(date),
                           neutral=not host_playing, rng=rng, knockout=True)
            winners[match_no] = w
        return winners

    w32 = ko_round(R32, lambda n, k: pos[f"3?{n}"] if k == "3?" else pos[k],
                   "r32", ROUND_DATES["r32"])
    for w in w32.values():
        reached[w] = "r16"
    w16 = ko_round(R16, lambda n, k: w32[k], "r16", ROUND_DATES["r16"])
    for w in w16.values():
        reached[w] = "qf"
    wqf = ko_round(QF, lambda n, k: w16[k], "qf", ROUND_DATES["qf"])
    for w in wqf.values():
        reached[w] = "sf"
    wsf = ko_round(SF, lambda n, k: wqf[k], "sf", ROUND_DATES["sf"])
    fa, fb = wsf.values()
    reached[fa] = reached[fb] = "final"
    if fb in HOSTS_QF_ON:
        fa, fb = fb, fa
    _, _, champ = play(model, states, fa, fb, pd.Timestamp(ROUND_DATES["final"]),
                       neutral=fa not in HOSTS_QF_ON, rng=rng, knockout=True)
    reached[champ] = "champion"
    return reached, orders, table, {thirds[g] for g in qualified}


STAGES = ["r32", "r16", "qf", "sf", "final", "champion"]

ISO2 = {
    "Mexico": "mx", "South Korea": "kr", "Czech Republic": "cz", "South Africa": "za",
    "Canada": "ca", "Bosnia and Herzegovina": "ba", "Qatar": "qa", "Switzerland": "ch",
    "Brazil": "br", "Morocco": "ma", "Haiti": "ht", "Scotland": "gb-sct",
    "United States": "us", "Paraguay": "py", "Australia": "au", "Turkey": "tr",
    "Germany": "de", "Curaçao": "cw", "Ivory Coast": "ci", "Ecuador": "ec",
    "Netherlands": "nl", "Japan": "jp", "Sweden": "se", "Tunisia": "tn",
    "Belgium": "be", "Egypt": "eg", "Iran": "ir", "New Zealand": "nz",
    "Spain": "es", "Cape Verde": "cv", "Saudi Arabia": "sa", "Uruguay": "uy",
    "France": "fr", "Senegal": "sn", "Iraq": "iq", "Norway": "no",
    "Argentina": "ar", "Algeria": "dz", "Austria": "at", "Jordan": "jo",
    "Portugal": "pt", "DR Congo": "cd", "Uzbekistan": "uz", "Colombia": "co",
    "England": "gb-eng", "Croatia": "hr", "Ghana": "gh", "Panama": "pa",
}


def win_prob(model, states, ta, tb, date, neutral):
    """P(ta advances past tb), including extra time/penalties on a draw."""
    sa, sb = states[ta], states[tb]
    lh, la = model.lambdas(sa, sb, date, home_is_home=not neutral)
    grid = model.score_grid(lh, la)
    return float(np.tril(grid, -1).sum() + np.trace(grid) * elo.expected_score(sa.elo, sb.elo))


def predicted_bracket(model, base_states, group_order, thirdq):
    """Single most-likely tournament: predicted group standings fill the
    official bracket, then the model's favorite advances at every node.
    Ratings stay frozen at today's values (this is a point estimate, not a
    sample)."""
    states = {t: s.copy() for t, s in base_states.items()}
    pos = {}
    for g, order in group_order.items():
        pos[f"1{g}"], pos[f"2{g}"] = order[0], order[1]
    thirds = {g: group_order[g][2] for g in GROUPS}
    top8 = sorted(GROUPS, key=lambda g: -thirdq[thirds[g]])[:8]
    slot_group = assign_thirds(set(top8), np.random.default_rng(0))
    for slot, g in slot_group.items():
        pos[f"3?{slot}"] = thirds[g]

    winners = {}

    def run(pairs, resolve, stage, date):
        out = []
        for n, (a, b) in pairs.items():
            ta, tb = resolve(n, a), resolve(n, b)
            host = stage in ("qf", "sf", "final") and bool({ta, tb} & HOSTS_QF_ON)
            if host and tb in HOSTS_QF_ON:
                ta, tb = tb, ta
            p = win_prob(model, states, ta, tb, pd.Timestamp(date), neutral=not host)
            winners[n] = ta if p >= 0.5 else tb
            out.append({"match": n, "a": ta, "b": tb, "pA": round(p, 3), "winner": winners[n]})
        return out

    bracket = {
        "r32": run(R32, lambda n, k: pos[f"3?{n}"] if k == "3?" else pos[k], "r32", ROUND_DATES["r32"]),
        "r16": run(R16, lambda n, k: winners[k], "r16", ROUND_DATES["r16"]),
        "qf": run(QF, lambda n, k: winners[k], "qf", ROUND_DATES["qf"]),
        "sf": run(SF, lambda n, k: winners[k], "sf", ROUND_DATES["sf"]),
        "final": run({104: (101, 102)}, lambda n, k: winners[k], "final", ROUND_DATES["final"]),
    }
    bracket["champion"] = bracket["final"][0]["winner"]
    return bracket


def main(n_sims=10000):
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    model = Model(ROOT / "models" / "poisson_model.pkl")
    matches = pd.read_csv(DATA / "matches.csv", parse_dates=["date"])
    fixtures = pd.read_csv(DATA / "wc2026_fixtures.csv", parse_dates=["date"])
    fixtures = fixtures.sort_values("date")
    played = matches[(matches.tournament == "FIFA World Cup") & (matches.date >= "2026-06-01")]
    base_states = seed_states(matches)
    print(f"{len(played)} real results locked in, {len(fixtures)} fixtures to simulate")
    print(f"running {n_sims} tournament simulations...")

    counts = {t: dict.fromkeys(STAGES, 0) for t in TEAM_GROUP}
    pos_counts = {t: [0, 0, 0, 0] for t in TEAM_GROUP}
    thirdq = dict.fromkeys(TEAM_GROUP, 0)
    pts_sum = dict.fromkeys(TEAM_GROUP, 0.0)
    gd_sum = dict.fromkeys(TEAM_GROUP, 0.0)
    for _ in range(n_sims):
        reached, orders, table, third_qualifiers = simulate_once(
            model, base_states, fixtures, played, rng
        )
        for g, order in orders.items():
            for i, t in enumerate(order):
                pos_counts[t][i] += 1
            for t in GROUPS[g]:
                pts_sum[t] += table[g][t][0]
                gd_sum[t] += table[g][t][1]
        for t in third_qualifiers:
            thirdq[t] += 1
        for team, stage in reached.items():
            if stage == "group":
                continue
            for s in STAGES[: STAGES.index(stage) + 1]:
                counts[team][s] += 1

    out = pd.DataFrame(counts).T / n_sims
    out["won_group"] = pd.Series({t: pos_counts[t][0] for t in TEAM_GROUP}) / n_sims
    out.insert(0, "group", pd.Series(TEAM_GROUP))
    out = out.sort_values(["champion", "final", "sf"], ascending=False)
    out.to_csv(RESULTS / "sim_results.csv")

    show = (out[STAGES[::-1] + ["won_group"]] * 100).round(1)
    show.insert(0, "group", out.group)
    print("\n=== probability (%) of reaching each stage — top 20 ===")
    print(show.head(20).to_string())
    print(f"\nsaved -> {RESULTS / 'sim_results.csv'}")

    export_dashboard(model, base_states, played, n_sims, out,
                     pos_counts, thirdq, pts_sum, gd_sum)


def export_dashboard(model, base_states, played, n, out, pos_counts, thirdq, pts_sum, gd_sum):
    group_order = {
        g: sorted(GROUPS[g], key=lambda t: (-pts_sum[t], -pos_counts[t][0]))
        for g in GROUPS
    }
    bracket = predicted_bracket(model, base_states, group_order, thirdq)

    groups_json = {
        g: [
            {
                "team": t,
                "expPts": round(pts_sum[t] / n, 2),
                "expGd": round(gd_sum[t] / n, 1),
                "p1": round(pos_counts[t][0] / n, 3),
                "p2": round(pos_counts[t][1] / n, 3),
                "p3q": round(thirdq[t] / n, 3),
                "advance": round(out.loc[t, "r32"], 3),
            }
            for t in group_order[g]
        ]
        for g in GROUPS
    }
    odds = [
        {"team": t, "champion": round(r.champion, 4), "final": round(r.final, 3),
         "sf": round(r.sf, 3)}
        for t, r in out.iterrows()
    ]
    data = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "nSims": n,
        "codes": ISO2,
        "groups": groups_json,
        "titleOdds": odds,
        "bracket": bracket,
        "played": [
            {"home": m.home_team, "away": m.away_team,
             "score": f"{int(m.home_score)}–{int(m.away_score)}"}
            for m in played.itertuples()
        ],
    }
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    with open(docs / "data.js", "w", encoding="utf-8") as f:
        f.write("const DATA = " + json.dumps(data, ensure_ascii=False) + ";\n")
    print(f"dashboard data -> {docs / 'data.js'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10000)
