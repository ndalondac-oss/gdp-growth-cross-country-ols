"""
build_panel.py -- turn the raw long WDI extract into the estimation sample.

Construction follows the AF5039 Option 1 (cross-section) specification:

    Y   growth24   real GDP per capita growth in 2024, % 
                   = 100 * (gdppc_2024 / gdppc_2023 - 1), constant 2015 US$
    X1  lngdppc23  natural log of real GDP per capita in 2023
    X2  gcf23      gross capital formation, % of GDP, 2023
    X3  highinc    = 1 if World Bank income group is "High income", else 0

Outputs
-------
data/growth_panel.csv
data/growth_panel.dta   (Stata 14 format, for the .do file)

Usage (from repo root):
    python code/build_panel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SHORT = {
    "NY.GDP.PCAP.KD": "gdppc",
    "NE.GDI.TOTL.ZS": "gcf",
    "FP.CPI.TOTL.ZG": "infl",
    "SL.UEM.TOTL.ZS": "unemp",
    "NE.TRD.GNFS.ZS": "trade",
    "SP.POP.GROW": "popgrow",
}

VARIABLE_LABELS = {
    "growth24": "Real GDP per capita growth, 2024 (%)",
    "lngdppc23": "Log real GDP per capita, 2023 (constant 2015 US$)",
    "gcf23": "Gross capital formation, 2023 (% of GDP)",
    "highinc": "=1 if World Bank high-income economy",
    "infl23": "Inflation, consumer prices, 2023 (annual %)",
    "unemp23": "Unemployment, 2023 (% of labour force)",
    "trade23": "Trade openness, 2023 (% of GDP)",
    "popgrow23": "Population growth, 2023 (annual %)",
    "gdppc23": "Real GDP per capita, 2023 (constant 2015 US$)",
    "gdppc24": "Real GDP per capita, 2024 (constant 2015 US$)",
}


def load_wide() -> pd.DataFrame:
    raw = pd.read_csv(DATA / "wdi_raw.csv")
    raw["short"] = raw["indicator"].map(SHORT)

    # long -> wide, one row per economy, one column per indicator-year
    wide = raw.pivot_table(index="iso3", columns=["short", "year"], values="value")
    wide.columns = [f"{s}{y % 100:02d}" for s, y in wide.columns]
    return wide.reset_index()


def main() -> int:
    wide = load_wide()
    meta = pd.read_csv(DATA / "country_meta.csv")

    # inner join drops the aggregates already filtered out of country_meta
    df = meta.merge(wide, on="iso3", how="inner")

    # --- dependent variable: 2024 growth in real GDP per capita -------------
    df["growth24"] = 100.0 * (df["gdppc24"] / df["gdppc23"] - 1.0)

    # --- regressors ---------------------------------------------------------
    df["lngdppc23"] = np.log(df["gdppc23"])
    df["highinc"] = (df["income_group"] == "High income").astype(int)

    core = ["growth24", "lngdppc23", "gcf23", "highinc"]
    extras = ["infl23", "unemp23", "trade23", "popgrow23", "gdppc23", "gdppc24"]
    keep = ["iso3", "country", "region", "income_group"] + core + extras

    panel = df[keep].copy()

    before = len(panel)
    panel = panel.dropna(subset=core).reset_index(drop=True)
    print(f"economies with metadata:        {before}")
    print(f"complete cases on Y, X1-X3:     {len(panel)}")
    print(f"dropped for missing core vars:  {before - len(panel)}")
    print(f"high-income share:              {panel['highinc'].mean():.1%}")

    if len(panel) < 100:
        raise SystemExit(f"sample is {len(panel)} < 100 required by the brief")

    panel.to_csv(DATA / "growth_panel.csv", index=False)
    panel.to_stata(
        DATA / "growth_panel.dta",
        write_index=False,
        version=114,
        variable_labels={k: v for k, v in VARIABLE_LABELS.items() if k in panel},
    )
    print("\nwrote data/growth_panel.csv and data/growth_panel.dta")
    return 0


if __name__ == "__main__":
    sys.exit(main())
