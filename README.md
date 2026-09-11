# Cross-sectional growth regression, 171 economies

An end-to-end econometrics pipeline in Stata and Python: pull raw World Bank
indicators from the public API, build a clean cross-section, estimate the model
by OLS, test the Gauss-Markov assumptions, and check how far the results hold up
under alternative estimators.

The question is a standard one in growth empirics — does a country's 2024 growth
rate in real GDP per capita depend on how rich it already was, how much it
invests, and whether it is a high-income economy?

## Specification

```
growth24_i = β₀ + β₁·lngdppc23_i + β₂·gcf23_i + β₃·highinc_i + ε_i
```

| Symbol | Variable | Definition | WDI code |
|---|---|---|---|
| Y | `growth24` | Real GDP per capita growth, 2024 (%), constant 2015 US$ | derived from `NY.GDP.PCAP.KD` |
| X₁ | `lngdppc23` | Log real GDP per capita, 2023 — conditional convergence term | `NY.GDP.PCAP.KD` |
| X₂ | `gcf23` | Gross capital formation, 2023 (% of GDP) | `NE.GDI.TOTL.ZS` |
| X₃ | `highinc` | 1 if a World Bank high-income economy, else 0 | country metadata |

Growth is computed from the two income levels rather than taken from the
World Bank's published growth series, so the numerator and denominator are
guaranteed to come from the same vintage of the constant-price series.

The sample is every economy with complete data on all four variables: **171 of
217**. The 78 regional and income aggregates the API returns alongside real
economies (World, Euro area, and so on) are dropped by filtering on the
metadata region code, not by name matching.

## Results

| | OLS | Robust (HC1) | WLS | Trimmed |
|---|---|---|---|---|
| `lngdppc23` | 0.189 (0.335) | 0.189 (0.288) | 0.166 (0.322) | −0.053 (0.236) |
| `gcf23` | **0.140** (0.035) | **0.140** (0.044) | **0.154** (0.033) | **0.112** (0.034) |
| `highinc` | −0.938 (0.995) | −0.938 (0.851) | −0.920 (1.013) | −0.851 (0.756) |
| N | 171 | 171 | 171 | 168 |
| R² | 0.095 | 0.095 | 0.124 | 0.137 |

Standard errors in parentheses; **bold** marks significance at the 1% level.

A percentage point more investment as a share of GDP is associated with roughly
0.11–0.15 percentage points more growth, and that coefficient holds its sign and
magnitude across every specification. Neither the convergence term nor the
income dummy is distinguishable from zero. The model explains about 10% of the
cross-sectional variation, which is unsurprising for a single year: annual
growth rates are dominated by country-specific shocks, and convergence effects
show up over decades rather than over twelve months.

### Diagnostics

| Test | Statistic | p | Conclusion |
|---|---|---|---|
| Breusch-Pagan | 1.73 | 0.631 | No evidence against homoskedasticity |
| White (8 df) | 4.23 | 0.836 | No evidence against homoskedasticity |
| Ramsey RESET (3 df) | 0.31 | 0.815 | No evidence of functional-form misspecification |
| Jarque-Bera | 2966.87 | <0.001 | Normality rejected |
| VIF | max 3.03 | — | No multicollinearity problem |

Normality fails badly, driven by a left tail of sharp contractions (West Bank and
Gaza −24.7%, Sudan −14.7%, Timor-Leste −10.2%). With n = 171 the central limit
theorem covers inference on the slopes, but the outliers are worth addressing
directly, which the trimmed specification does: dropping the three observations
with studentised residuals beyond ±3 raises R² from 0.095 to 0.137 and leaves the
investment coefficient intact.

Two implementation notes.

On the RESET test, the auxiliary regression includes the second, third and fourth
powers of the fitted values, three restrictions in total. This is what Stata's
`estat ovtest` does by default; `statsmodels`' `linear_reset` has to be called
with `power=4` to match it, since `power=2` adds the square alone and tests a
single restriction. The conclusion is the same under either, but only the
three-power version is comparable across the two code paths.

On the White test. Feeding every square and cross product
into the auxiliary regression gives a rank-deficient design whenever a dummy is
present, since `highinc² = highinc`. `statsmodels`' `het_white` does exactly
that and reports degrees of freedom — and therefore a p-value — computed from the
nominal column count. `white_test()` in `code/growth_analysis.py` builds the
auxiliary design, drops exact duplicates and then any column that adds no rank,
and takes the degrees of freedom from what survives.

## Layout

```
code/
  fetch_wdi.py              pull raw indicators + country metadata from the WDI API
  build_panel.py            long → wide, construct Y and X₁–X₃, write .csv and .dta
  growth_analysis.do        Stata: estimation, diagnostics, robustness, figures
  growth_analysis.py        Python replication of the .do file
  growth_analysis.ipynb     annotated notebook walkthrough (outputs included)
data/
  wdi_raw.csv               raw API extract, long format, 2022–24
  country_meta.csv          iso3, country, region, income group (aggregates removed)
  growth_panel.csv          the estimation sample
  growth_panel.dta          same, Stata format, with variable labels
output/
  descriptive_stats.csv     Table 1
  regression_tables.csv     Table 2, all four specifications
  regression_tables.rtf     Table 2, Word-readable
  diagnostics.csv           Table 3
  vif.csv, model_fit.csv    supporting tables
  resid_plot.png            residuals vs fitted, with LOWESS overlay
  qq_plot.png               normal Q-Q of residuals
  resid_hist.png            residual density vs fitted normal
  scatter_convergence.png   growth vs initial income, by income group
  growth_analysis_python.log   full log of the Python run
```

The `output/` files committed here are from the Python run. The Stata path writes
its own tables and log alongside them, under `regression_tables_stata.rtf`,
`descriptive_stats.rtf` and `growth_analysis_stata.log`, so neither path
overwrites the other's results and the two can be compared directly.

## Running it

```bash
pip install -r requirements.txt

python code/fetch_wdi.py        # writes data/wdi_raw.csv, data/country_meta.csv
python code/build_panel.py      # writes data/growth_panel.{csv,dta}
python code/growth_analysis.py  # writes everything under output/
```

Or in Stata, from the repository root, which reproduces the same tables and
figures from the same raw inputs:

```stata
ssc install estout    // once, for esttab
do code/growth_analysis.do
```

Both paths start from `data/wdi_raw.csv`, so the cleaning logic is replicated
rather than shared — which is the point of having both. Output filenames are
suffixed by language (`growth_analysis_python.log`,
`growth_analysis_stata.log`) so the two can be diffed against each other.

One caveat on the diagnostics: `estat hettest, rhs iid`, `estat imtest, white`
and `estat ovtest` are the Stata counterparts of the three tests in Table 3, and
the Python code is written to match their default settings rather than the other
way round. Where a test can be specified more than one way — the RESET powers
above being the clearest case — the Stata default is taken as the reference.

`fetch_wdi.py` hits the live API, so re-running it after a World Bank data
revision will shift the estimates slightly. The committed `wdi_raw.csv` is the
extract the reported results come from; the API reported a last-updated date of
2026-07-13 at the time of the pull.

## Notes

- Tested with Python 3.12 and the versions pinned in `requirements.txt`. The
  Stata code targets Stata 17 and needs `estout` for `esttab`/`estpost`.
- Figures are written headlessly through the `Agg` backend, so the scripts run
  on a server with no display.
- The WDI API drops observations intermittently and times out under load; both
  fetch paths retry with backoff rather than failing the run.

## Data source

World Bank, World Development Indicators, retrieved through the public API at
`api.worldbank.org/v2`. WDI data is published under CC BY 4.0.
