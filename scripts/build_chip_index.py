"""Build public/chips.json, the per-country chip index the app samples from.

Reads each country's chips parquet, data config and the dataset README straight
from Source Cooperative.

Usage: python scripts/build_chip_index.py [OUT]
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

import duckdb

BASE = "https://data.source.coop/kerner-lab/fields-of-the-world"
COUNTRIES = [
    "austria", "belgium", "brazil", "cambodia", "corsica", "croatia", "denmark",
    "estonia", "finland", "france", "germany", "india", "kenya", "latvia",
    "lithuania", "luxembourg", "netherlands", "portugal", "rwanda", "slovakia",
    "slovenia", "south_africa", "spain", "sweden", "vietnam",
]


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "ftw-chip-index/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode()


def readme_rows() -> dict[str, dict[str, str]]:
    """Country -> {year, license} from the dataset README's table."""
    rows = {}
    for line in fetch(f"{BASE}/README.md").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 9 or not cells[1][:4].isdigit():
            continue
        license_ = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", cells[8])
        rows[cells[0].lower().replace(" ", "_")] = {"name": cells[0], "year": cells[1], "license": license_}
    return rows


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "public/chips.json")
    meta = readme_rows()
    con = duckdb.connect()
    con.sql("INSTALL httpfs; LOAD httpfs; SET http_retries=10;")
    index = {}
    for c in COUNTRIES:
        cfg = json.loads(fetch(f"{BASE}/{c}/data_config_{c}.json"))
        splits: dict[str, list[str]] = {}
        for aoi_id, split in con.sql(
            f"SELECT aoi_id::VARCHAR, coalesce(split, 'none') FROM read_parquet('{BASE}/{c}/chips_{c}.parquet') ORDER BY aoi_id"
        ).fetchall():
            splits.setdefault(split, []).append(str(aoi_id))
        # Some configs hold one season per grid; the grids share dates in practice.
        s = cfg["seasons"][0] if isinstance(cfg["seasons"], list) else cfg["seasons"]
        index[c] = {
            **meta[c],
            "window_a": [s["window_a"]["start"], s["window_a"]["end"]],
            "window_b": [s["window_b"]["start"], s["window_b"]["end"]],
            "splits": splits,
        }
        print(c, {k: len(v) for k, v in splits.items()}, meta[c]["license"])
    out.write_text(json.dumps({"base": BASE, "countries": index}, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size:,} B)")


if __name__ == "__main__":
    main()
