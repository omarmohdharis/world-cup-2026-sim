"""
Full pipeline update: fetch new WC results → append to matches.csv →
rebuild features → re-run simulation → refresh ticker data.

Usage:  python src/update.py [--sims N]   (default 10 000)

Run this after any group-stage matchday to get fresh probabilities and
an up-to-date ticker.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
SRC = ROOT / "src"

API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "6a132804de7140acac0135ca7675a456")
URL = "https://api.football-data.org/v4/competitions/WC/matches?status=FINISHED"

NAME_MAP = {
    "USA": "United States",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Curaçao": "Curaçao",
    "Czechia": "Czech Republic",
}


def fetch_api():
    req = urllib.request.Request(URL, headers={"X-Auth-Token": API_KEY})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def update_matches(api_data: dict) -> int:
    """Append any new WC 2026 finished matches to matches.csv. Returns count added."""
    matches = pd.read_csv(DATA / "matches.csv", parse_dates=["date"])
    existing_keys = set(
        zip(matches.date.dt.date.astype(str), matches.home_team, matches.away_team)
    )

    new_rows = []
    for m in api_data.get("matches", []):
        if m["status"] != "FINISHED":
            continue
        ft = m["score"]["fullTime"]
        if ft["home"] is None:
            continue

        home = NAME_MAP.get(m["homeTeam"]["name"], m["homeTeam"]["name"])
        away = NAME_MAP.get(m["awayTeam"]["name"], m["awayTeam"]["name"])
        date = m["utcDate"][:10]

        if (date, home, away) in existing_keys:
            continue

        home_score, away_score = int(ft["home"]), int(ft["away"])
        if home_score > away_score:
            outcome = "home_win"
        elif away_score > home_score:
            outcome = "away_win"
        else:
            outcome = "draw"

        new_rows.append({
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
            "tournament": "FIFA World Cup",
            "city": m.get("venue", ""),
            "country": "United States",
            "neutral": True,
            "shootout_winner": None,
            "outcome": outcome,
        })

    if new_rows:
        added = pd.DataFrame(new_rows)
        added["date"] = pd.to_datetime(added["date"])
        updated = pd.concat([matches, added], ignore_index=True).sort_values("date")
        updated.to_csv(DATA / "matches.csv", index=False)
        print(f"  Added {len(new_rows)} new match(es) to matches.csv")
        for r in new_rows:
            print(f"    {r['date']}  {r['home_team']} {r['home_score']}–{r['away_score']} {r['away_team']}")
    else:
        print("  No new matches since last update.")

    return len(new_rows)


def update_ticker(api_data: dict):
    """Write docs/results.json from all finished WC matches (most recent first)."""
    results = []
    for m in api_data.get("matches", []):
        if m["status"] != "FINISHED":
            continue
        ft = m["score"]["fullTime"]
        if ft["home"] is None:
            continue
        home = NAME_MAP.get(m["homeTeam"]["name"], m["homeTeam"]["name"])
        away = NAME_MAP.get(m["awayTeam"]["name"], m["awayTeam"]["name"])
        results.append({
            "home": home,
            "away": away,
            "score": f"{ft['home']}–{ft['away']}",
            "date": m["utcDate"][:10],
        })
    results.sort(key=lambda r: r["date"], reverse=True)
    with open(DOCS / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Ticker updated: {len(results)} results -> docs/results.json")


def run(cmd: list[str], label: str):
    print(f"\n&gt; {label}")
    result = subprocess.run([sys.executable] + cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"  ERROR: {label} failed (exit {result.returncode})")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=10_000)
    args = parser.parse_args()

    print("-" * 52)
    print("WC 2026 -- full pipeline update")
    print("-" * 52)

    print("\n&gt; Fetching results from football-data.org...")
    api_data = fetch_api()
    total = sum(1 for m in api_data.get("matches", []) if m["status"] == "FINISHED")
    print(f"  API returned {total} finished match(es)")

    print("\n&gt; Updating matches.csv...")
    added = update_matches(api_data)

    print("\n&gt; Updating ticker (results.json)...")
    update_ticker(api_data)

    if added > 0:
        run(["src/build_features.py"], "Rebuilding features (Elo + form)")
    else:
        print("\n&gt; Skipping feature rebuild — no new matches")

    run([f"src/simulate.py", str(args.sims)], f"Re-running simulation ({args.sims:,} sims)")

    print("\n" + "-" * 52)
    print("Update complete. Refresh the dashboard to see new probabilities.")


if __name__ == "__main__":
    main()
