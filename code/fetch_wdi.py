"""
fetch_wdi.py -- download raw World Development Indicators series and country
metadata for the AF5039 cross-sectional growth regression.

Outputs
-------
data/wdi_raw.csv        long format: iso3, country, indicator, year, value
data/country_meta.csv   iso3, country, region, income_group  (aggregates dropped)

Usage (from repo root):
    python code/fetch_wdi.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.worldbank.org/v2"

# Indicator code -> short name used downstream.
INDICATORS = {
    "NY.GDP.PCAP.KD": "gdppc",       # GDP per capita, constant 2015 US$
    "NE.GDI.TOTL.ZS": "gcf",         # Gross capital formation, % of GDP
    "FP.CPI.TOTL.ZG": "infl",        # Inflation, consumer prices, annual %
    "SL.UEM.TOTL.ZS": "unemp",       # Unemployment, % of total labour force (ILO est.)
    "NE.TRD.GNFS.ZS": "trade",       # Trade, % of GDP
    "SP.POP.GROW": "popgrow",        # Population growth, annual %
}

# 2023 for the regressors, 2024 for the dependent variable; 2023 level is the
# base for computing 2024 growth, so both years are required.
START, END = 2022, 2024

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _get(url: str, params: dict, retries: int = 3):
    """GET with retry/backoff; the WDI API times out intermittently."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=90)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"request failed: {url} -- {exc}") from exc
            time.sleep(2 * attempt)


def fetch_indicator(code: str) -> pd.DataFrame:
    payload = _get(
        f"{BASE}/country/all/indicator/{code}",
        {"date": f"{START}:{END}", "format": "json", "per_page": 20000},
    )
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise RuntimeError(f"no observations returned for {code}")
    return pd.DataFrame(
        {
            "iso3": o["countryiso3code"],
            "country": o["country"]["value"],
            "indicator": code,
            "year": int(o["date"]),
            "value": o["value"],
        }
        for o in payload[1]
    )


def fetch_country_meta() -> pd.DataFrame:
    """Country list with region and income group.

    The World Bank returns regional and income aggregates (World, Euro area,
    ...) in the same endpoint as real economies. Aggregates carry region id
    'NA', which is how they are dropped here.
    """
    payload = _get(f"{BASE}/country", {"format": "json", "per_page": 400})
    rows = [
        {
            "iso3": c["id"],
            "country": c["name"],
            "region": c["region"]["value"],
            "region_id": c["region"]["id"],
            "income_group": c["incomeLevel"]["value"],
        }
        for c in payload[1]
    ]
    meta = pd.DataFrame(rows)
    aggregates = meta["region_id"].str.strip() == "NA"
    print(f"country list: {len(meta)} entries, dropping {aggregates.sum()} aggregates")
    return meta.loc[~aggregates].drop(columns="region_id").reset_index(drop=True)


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)

    frames = []
    for code, short in INDICATORS.items():
        df = fetch_indicator(code)
        print(f"{code:<16} {short:<8} {len(df):>5} obs")
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(DATA / "wdi_raw.csv", index=False)
    print(f"\nwrote data/wdi_raw.csv ({len(raw):,} rows)")

    meta = fetch_country_meta()
    meta.to_csv(DATA / "country_meta.csv", index=False)
    print(f"wrote data/country_meta.csv ({len(meta)} economies)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
