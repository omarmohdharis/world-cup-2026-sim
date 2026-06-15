"""
Fetch finished 2026 World Cup results from football-data.org and write
them to docs/results.json for the live ticker.

Usage:  python src/fetch_results.py

Run this whenever you want fresh results — or wire it into a scheduled task.
The output is read by the dashboard ticker independently of the simulation.
"""

import json
import urllib.request
from pathlib import Path

API_KEY = "6a132804de7140acac0135ca7675a456"
URL = "https://api.football-data.org/v4/competitions/WC/matches?status=FINISHED"
OUT = Path(__file__).resolve().parents[1] / "docs" / "results.json"

# football-data.org team names that differ from our internal names.
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


def fetch():
    req = urllib.request.Request(URL, headers={"X-Auth-Token": API_KEY})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    results = []
    for m in data.get("matches", []):
        home = NAME_MAP.get(m["homeTeam"]["name"], m["homeTeam"]["name"])
        away = NAME_MAP.get(m["awayTeam"]["name"], m["awayTeam"]["name"])
        ft = m["score"]["fullTime"]
        if ft["home"] is None:
            continue
        results.append({
            "home": home,
            "away": away,
            "score": f"{ft['home']}–{ft['away']}",
            "date": m["utcDate"][:10],
        })

    # Most recent first.
    results.sort(key=lambda r: r["date"], reverse=True)

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} results -> {OUT}")
    for r in results[:5]:
        print(f"  {r['date']}  {r['home']} {r['score']} {r['away']}")


if __name__ == "__main__":
    fetch()
