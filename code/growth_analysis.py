"""
growth_analysis.py -- Python replication of growth_analysis.do.

Estimates the AF5039 cross-sectional specification

    growth24_i = b0 + b1*lngdppc23_i + b2*gcf23_i + b3*highinc_i + e_i

by OLS, runs the three diagnostic tests (heteroskedasticity, multicollinearity,
functional-form / omitted variables), and reports three robustness checks
(HC1 robust standard errors, feasible WLS, and an outlier-trimmed subsample).

Outputs (all under output/)
---------------------------
descriptive_stats.csv       Table 1
regression_tables.csv/.rtf  Table 2 (OLS, robust, WLS, trimmed)
diagnostics.csv             Table 3
resid_plot.png              residuals vs fitted
qq_plot.png                 normal Q-Q of residuals
resid_hist.png              residual density vs normal
scatter_convergence.png     growth vs log initial income, by income group
growth_analysis_python.log  full run log (the .do file writes the Stata log)

Usage (from repo root):
    python code/growth_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"

Y = "growth24"
XS = ["lngdppc23", "gcf23", "highinc"]
FORMULA = f"{Y} ~ " + " + ".join(XS)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


class Tee:
    """Mirror stdout to the log file so the console and log always agree."""

    def __init__(self, path: Path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text: str) -> None:
        self.stdout.write(text)
        self.file.write(text)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# Section A: descriptives and OLS
# ---------------------------------------------------------------------------
def descriptives(df: pd.DataFrame) -> pd.DataFrame:
    cols = [Y] + XS
    tab = df[cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    tab.columns = ["N", "Mean", "SD", "Min", "Median", "Max"]
    tab["Skew"] = df[cols].skew()
    tab["Kurtosis"] = df[cols].kurtosis() + 3.0  # report raw, not excess
    return tab.round(3)


def coef_frame(res, label: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"{label}: coef": res.params.round(4),
            f"{label}: se": res.bse.round(4),
            f"{label}: t": res.tvalues.round(3),
            f"{label}: p": res.pvalues.round(4),
        }
    )


# ---------------------------------------------------------------------------
# Section B: diagnostics
# ---------------------------------------------------------------------------
def white_test(res) -> tuple[float, float, int]:
    """White (1980) LM test, built on a full-rank auxiliary design.

    statsmodels' het_white feeds every square and cross product into the
    auxiliary regression. With a dummy regressor that design is rank
    deficient -- highinc**2 is identical to highinc -- which makes the
    reported degrees of freedom, and therefore the p-value, wrong. Squares
    and cross products are constructed here and then screened: exact
    duplicates go first, then any column that adds no rank.
    """
    exog = pd.DataFrame(res.model.exog, columns=res.model.exog_names)
    regressors = [c for c in exog.columns if c != "Intercept"]

    aux = pd.DataFrame({"Intercept": 1.0}, index=exog.index)
    for i, a in enumerate(regressors):
        aux[a] = exog[a]
        for b in regressors[i:]:
            aux[f"{a}:{b}" if a != b else f"{a}^2"] = exog[a] * exog[b]

    # exact duplicates (dummy^2 == dummy, dummy:dummy == dummy)
    aux = aux.T.drop_duplicates().T

    # then drop any remaining linearly dependent column, left to right
    keep: list[str] = []
    for col in aux.columns:
        trial = keep + [col]
        if np.linalg.matrix_rank(aux[trial].to_numpy()) == len(trial):
            keep.append(col)
    aux = aux[keep]

    e2 = res.resid.to_numpy() ** 2
    fit = sm.OLS(e2, aux.to_numpy()).fit()
    dof = aux.shape[1] - 1
    lm = fit.rsquared * len(e2)
    return lm, float(st.chi2.sf(lm, dof)), dof


def diagnostics(res, df: pd.DataFrame) -> pd.DataFrame:
    resid = res.resid
    exog = res.model.exog
    rows = []

    # (1) Heteroskedasticity -- Breusch-Pagan and White
    bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(resid, exog)
    rows.append(("Breusch-Pagan (LM)", bp_lm, bp_p, "Homoskedastic errors"))
    w_lm, w_p, w_dof = white_test(res)
    rows.append((f"White (LM, {w_dof} df)", w_lm, w_p, "Homoskedastic errors"))

    # (2) Multicollinearity -- VIF (reported separately, no p-value)
    # (3) Functional form / omitted variables -- Ramsey RESET
    # power=4 adds yhat^2, yhat^3 and yhat^4 to the auxiliary regression, which
    # is what Stata's `estat ovtest` does. power=2 would add the square alone
    # (1 df) and the two implementations would not be comparable.
    reset = linear_reset(res, power=4, use_f=True)
    rows.append(("Ramsey RESET (F, powers 2-4)", reset.fvalue, reset.pvalue,
                 "Linear functional form correctly specified"))

    # Supporting: normality of residuals
    jb, jb_p, skew, kurt = sm.stats.stattools.jarque_bera(resid)
    rows.append(("Jarque-Bera", jb, jb_p, "Normally distributed errors"))

    return pd.DataFrame(rows, columns=["Test", "Statistic", "p-value", "Null hypothesis"]).round(4)


def vif_table(df: pd.DataFrame) -> pd.DataFrame:
    X = sm.add_constant(df[XS])
    return pd.DataFrame(
        {
            "Variable": X.columns,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        }
    ).round(3)


# ---------------------------------------------------------------------------
# Section C: robustness
# ---------------------------------------------------------------------------
def wls_fit(df: pd.DataFrame, ols_res):
    """Feasible WLS: model log squared residuals on the regressors, then weight
    each observation by the inverse of its fitted error variance."""
    log_e2 = np.log(ols_res.resid**2)
    aux = smf.ols("log_e2 ~ " + " + ".join(XS), data=df.assign(log_e2=log_e2)).fit()
    weights = 1.0 / np.exp(aux.fittedvalues)
    return smf.wls(FORMULA, data=df, weights=weights).fit()


def trimmed_fit(df: pd.DataFrame, ols_res):
    """Drop observations with |studentised residual| > 3 and re-estimate."""
    infl = ols_res.get_influence()
    student = pd.Series(infl.resid_studentized_external, index=df.index)
    keep = student.abs() <= 3
    dropped = df.loc[~keep, ["country", Y]]
    return smf.ols(FORMULA, data=df.loc[keep]).fit(cov_type="HC1"), dropped


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(df: pd.DataFrame, res) -> None:
    fitted, resid = res.fittedvalues, res.resid

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.scatter(fitted, resid, s=16, alpha=0.7, edgecolor="none")
    ax.axhline(0, lw=1, color="black")
    lo = sm.nonparametric.lowess(resid, fitted, frac=0.6)
    ax.plot(lo[:, 0], lo[:, 1], lw=1.2, color="crimson", label="LOWESS")
    ax.set(xlabel="Fitted values", ylabel="Residuals",
           title="Figure 1: Residuals vs fitted values")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "resid_plot.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    sm.qqplot(resid, line="s", ax=ax, markersize=4, alpha=0.7)
    ax.set_title("Figure 2: Normal Q-Q plot of residuals")
    fig.tight_layout()
    fig.savefig(OUT / "qq_plot.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.hist(resid, bins=30, density=True, alpha=0.6, edgecolor="white")
    grid = np.linspace(resid.min(), resid.max(), 300)
    ax.plot(grid, st.norm.pdf(grid, resid.mean(), resid.std(ddof=0)),
            lw=1.4, color="crimson", label="Normal density")
    ax.set(xlabel="Residual", ylabel="Density",
           title="Figure 3: Residual distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "resid_hist.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for flag, label, colour in [(0, "Non-high-income", "#3b6ea5"),
                                (1, "High-income", "#c04a38")]:
        sub = df[df["highinc"] == flag]
        ax.scatter(sub["lngdppc23"], sub[Y], s=18, alpha=0.75,
                   label=label, color=colour, edgecolor="none")
    b = res.params
    xs = np.linspace(df["lngdppc23"].min(), df["lngdppc23"].max(), 50)
    at_mean_gcf = b["Intercept"] + b["gcf23"] * df["gcf23"].mean()
    ax.plot(xs, at_mean_gcf + b["lngdppc23"] * xs, lw=1.3, color="#3b6ea5")
    ax.plot(xs, at_mean_gcf + b["highinc"] + b["lngdppc23"] * xs,
            lw=1.3, color="#c04a38", ls="--")
    ax.set(xlabel="Log real GDP per capita, 2023",
           ylabel="Real GDP per capita growth, 2024 (%)",
           title="Figure 4: Conditional convergence, fitted at mean investment")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "scatter_convergence.png")
    plt.close(fig)

    print("\nfigures written: resid_plot.png, qq_plot.png, resid_hist.png, "
          "scatter_convergence.png")


def write_rtf(table: pd.DataFrame, path: Path, title: str) -> None:
    """Minimal RTF table writer -- opens in Word without Stata or esttab."""
    head = r"{\rtf1\ansi\deff0{\fonttbl{\f0 Calibri;}}\fs20 "
    body = [r"{\b " + title + r"}\par\par "]
    body.append("\t".join([table.index.name or ""] + [str(c) for c in table.columns]) + r"\par ")
    for idx, row in table.iterrows():
        body.append("\t".join([str(idx)] + [str(v) for v in row.values]) + r"\par ")
    path.write_text(head + "".join(body) + "}", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tee = Tee(OUT / "growth_analysis_python.log")
    sys.stdout = tee
    try:
        df = pd.read_csv(DATA / "growth_panel.csv")
        print("AF5039 Econometrics -- cross-sectional growth regression")
        print(f"Python replication run | N = {len(df)} economies")
        print(f"Specification: {FORMULA}")

        rule("SECTION A.1  Descriptive statistics (Table 1)")
        desc = descriptives(df)
        print(desc.to_string())
        desc.to_csv(OUT / "descriptive_stats.csv")

        rule("SECTION A.2  OLS estimates")
        ols = smf.ols(FORMULA, data=df).fit()
        print(ols.summary())

        rule("SECTION B.1  Diagnostic tests (Table 3)")
        diag = diagnostics(ols, df)
        print(diag.to_string(index=False))
        diag.to_csv(OUT / "diagnostics.csv", index=False)

        rule("SECTION B.2  Variance inflation factors")
        vifs = vif_table(df)
        print(vifs.to_string(index=False))
        vifs.to_csv(OUT / "vif.csv", index=False)

        rule("SECTION C.1  Robustness: HC1 robust standard errors")
        robust = smf.ols(FORMULA, data=df).fit(cov_type="HC1")
        print(robust.summary())

        rule("SECTION C.2  Robustness: feasible weighted least squares")
        wls = wls_fit(df, ols)
        print(wls.summary())

        rule("SECTION C.3  Robustness: outlier-trimmed subsample")
        trimmed, dropped = trimmed_fit(df, ols)
        print(f"dropped {len(dropped)} observations with |studentised resid| > 3:")
        print(dropped.to_string(index=False) if len(dropped) else "  (none)")
        print(trimmed.summary())

        rule("COMPARISON  Table 2: coefficients across specifications")
        table = pd.concat(
            [
                coef_frame(ols, "OLS"),
                coef_frame(robust, "Robust"),
                coef_frame(wls, "WLS"),
                coef_frame(trimmed, "Trimmed"),
            ],
            axis=1,
        )
        table.index.name = "Regressor"
        print(table.to_string())
        table.to_csv(OUT / "regression_tables.csv")
        write_rtf(table, OUT / "regression_tables.rtf",
                  "Table 2: OLS and robustness estimates, 2024 growth")

        fit = pd.DataFrame(
            {
                "Model": ["OLS", "Robust (HC1)", "WLS", "Trimmed"],
                "N": [int(m.nobs) for m in (ols, robust, wls, trimmed)],
                "R-squared": [round(m.rsquared, 4) for m in (ols, robust, wls, trimmed)],
                "Adj. R-squared": [round(m.rsquared_adj, 4) for m in (ols, robust, wls, trimmed)],
                "F / Wald": [round(m.fvalue, 3) for m in (ols, robust, wls, trimmed)],
                "p(F)": [round(m.f_pvalue, 4) for m in (ols, robust, wls, trimmed)],
            }
        )
        print()
        print(fit.to_string(index=False))
        fit.to_csv(OUT / "model_fit.csv", index=False)

        make_figures(df, ols)
        print("\nrun complete.")
    finally:
        sys.stdout = tee.stdout
        tee.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
