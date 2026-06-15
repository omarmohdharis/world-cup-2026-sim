"""
Fetch WC 2026 match odds from The Odds API (the-odds-api.com).
Writes docs/odds.json with normalized home/draw/away implied probabilities
for upcoming fixtures, averaged across all available bookmakers.

Usage: python src/fetch_odds.py
Requires: ODDS_API_KEY environment variable
Free plan: 500 requests/month at https://the-odds-api.com
"""

import json
import os
import urllib.request
from pathlib import Path

API_KEY = os.environ.get("ODDS_API_KEY", "")
SPORT = "soccer_fifa_world_cup"
URL = (
    "https://api.the-odds-api.com/v4/sports/{sport}/odds/"
    "?apiKey={key}&regions=eu,uk&markets=h2h&oddsFormat=decimal"
)
OUT = Path(__file__).resolve().parents[1] / "docs" / "odds.json"

NAME_MAP = {
    "USA": "United States",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Cote d'Ivoire": "Ivory Coast",
    "Ivory Coast": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Czechia": "Czech Republic",
}


def normalize(prices):
    """Decimal odds -> implied probabilities with vig removed."""
    implied = [1.0 / p for p in prices]
    total = sum(implied)
    return [round(x / total, 4) for x in implied]


def fetch():
    if not API_KEY:
        print("ODDS_API_KEY not set — skipping odds fetch")
        return

    url = URL.format(sport=SPORT, key=API_KEY)
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        events = json.loads(resp.read())
        remaining = resp.headers.get("x-requests-remaining", "?")

    print(f"  Odds API: {len(events)} event(s), {remaining} requests remaining this month")

    results = []
    for ev in events:
        raw_home = ev["home_team"]
        raw_away = ev["away_team"]
        home = NAME_MAP.get(raw_home, raw_home)
        away = NAME_MAP.get(raw_away, raw_away)

        h_prices, d_prices, a_prices = [], [], []
        for bm in ev.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt["key"] != "h2h":
                    continue
                by_name = {o["name"]: o["price"] for o in mkt["outcomes"]}
                if raw_home in by_name and "Draw" in by_name and raw_away in by_name:
                    h_prices.append(by_name[raw_home])
                    d_prices.append(by_name["Draw"])
                    a_prices.append(by_name[raw_away])

        if not h_prices:
            continue

        avg = [
            sum(h_prices) / len(h_prices),
            sum(d_prices) / len(d_prices),
            sum(a_prices) / len(a_prices),
        ]
        ph, pd_, pa = normalize(avg)
        results.append({
            "home": home,
            "away": away,
            "date": ev["commence_time"][:10],
            "ph": ph,
            "pd": pd_,
            "pa": pa,
        })

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(results)} odds entries -> {OUT}")


if __name__ == "__main__":
    fetch()
