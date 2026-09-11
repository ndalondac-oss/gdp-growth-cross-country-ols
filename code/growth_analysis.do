*==============================================================================
* growth_analysis.do
*
* AF5039 Econometrics -- cross-sectional determinants of real GDP per capita
* growth in 2024 across 171 economies.
*
*   growth24 = b0 + b1*lngdppc23 + b2*gcf23 + b3*highinc + e
*
* Section A  OLS estimation and descriptive statistics
* Section B  three diagnostic tests: heteroskedasticity, multicollinearity,
*            functional form / omitted variables
* Section C  robustness: HC1 robust standard errors, feasible WLS, and an
*            outlier-trimmed subsample
*
* Inputs   data/wdi_raw.csv, data/country_meta.csv  (see code/fetch_wdi.py)
* Outputs  data/growth_panel.dta
*          output/growth_analysis_stata.log
*          output/regression_tables_stata.rtf, output/descriptive_stats.rtf
*
*          The table filename is suffixed _stata so that running this file and
*          code/growth_analysis.py in either order leaves both sets of results
*          on disk to be compared.
*          output/resid_plot.png, output/qq_plot.png,
*          output/resid_hist.png, output/scatter_convergence.png
*
* Run from the repository root:
*     do code/growth_analysis.do
*
* Requires estout:  ssc install estout
*==============================================================================

clear all
set more off
version 17

capture log close
log using "output/growth_analysis_stata.log", replace text

*------------------------------------------------------------------------------
* 1. Country metadata (region and income group), aggregates already removed
*------------------------------------------------------------------------------
tempfile meta
import delimited "data/country_meta.csv", varnames(1) clear stringcols(_all)
keep iso3 country region income_group
save `meta'

*------------------------------------------------------------------------------
* 2. Raw WDI extract: long -> wide, one row per economy
*------------------------------------------------------------------------------
import delimited "data/wdi_raw.csv", varnames(1) clear

* indicator codes -> short names
gen str8 short = ""
replace short = "gdppc"   if indicator == "NY.GDP.PCAP.KD"
replace short = "gcf"     if indicator == "NE.GDI.TOTL.ZS"
replace short = "infl"    if indicator == "FP.CPI.TOTL.ZG"
replace short = "unemp"   if indicator == "SL.UEM.TOTL.ZS"
replace short = "trade"   if indicator == "NE.TRD.GNFS.ZS"
replace short = "popgrow" if indicator == "SP.POP.GROW"
drop if short == ""

* value is read as a string when the API returns blanks for missing years
capture confirm string variable value
if !_rc destring value, replace force

* a single stub per indicator-year, e.g. gdppc23, gcf23
gen varname = short + string(mod(year, 100), "%02.0f")
keep iso3 varname value
reshape wide value, i(iso3) j(varname) string
rename value* *

merge 1:1 iso3 using `meta', keep(match) nogenerate

*------------------------------------------------------------------------------
* 3. Construct Y, X1, X2, X3
*------------------------------------------------------------------------------
* Y: real GDP per capita growth in 2024 (%), constant 2015 US$
gen double growth24  = 100 * (gdppc24 / gdppc23 - 1)

* X1: log of real GDP per capita in 2023 -- conditional convergence term
gen double lngdppc23 = ln(gdppc23)

* X2: gross capital formation as a share of GDP in 2023
rename gcf23 gcf23_pct
gen double gcf23 = gcf23_pct
drop gcf23_pct

* X3: dummy for World Bank high-income economies
gen byte highinc = (income_group == "High income")

label variable growth24  "Real GDP per capita growth, 2024 (%)"
label variable lngdppc23 "Log real GDP per capita, 2023"
label variable gcf23     "Gross capital formation, 2023 (% of GDP)"
label variable highinc   "=1 if high-income economy"
label define hi 0 "Non-high-income" 1 "High-income"
label values highinc hi

* estimation sample: complete cases on Y and X1-X3
drop if missing(growth24, lngdppc23, gcf23, highinc)
count
display "Estimation sample: " r(N) " economies"

order iso3 country region income_group growth24 lngdppc23 gcf23 highinc
compress
save "data/growth_panel.dta", replace

*==============================================================================
* SECTION A -- Descriptive statistics and OLS
*==============================================================================
summarize growth24 lngdppc23 gcf23 highinc, detail

estpost summarize growth24 lngdppc23 gcf23 highinc, detail
esttab using "output/descriptive_stats.rtf", replace ///
    cells("count mean sd min p50 max skewness kurtosis") ///
    nomtitle nonumber label ///
    title("Table 1: Descriptive statistics")

* pairwise correlations, reported alongside the VIFs in Section B
pwcorr growth24 lngdppc23 gcf23 highinc, star(0.05)

* baseline OLS
regress growth24 lngdppc23 gcf23 highinc
estimates store ols

* F-test that the slope coefficients are jointly zero
test lngdppc23 gcf23 highinc

*==============================================================================
* SECTION B -- Diagnostic tests
*
* Run on the baseline OLS fit, before any robust covariance is applied:
* estat hettest and estat imtest are not available after regress, robust.
*==============================================================================

* (1) Heteroskedasticity: Breusch-Pagan/Cook-Weisberg, then White
estat hettest, rhs iid
estat imtest, white

* (2) Multicollinearity
estat vif

* (3) Functional form / omitted variables: Ramsey RESET on powers of yhat
estat ovtest

* Supporting evidence on normality of the residuals
predict double yhat, xb
predict double resid, residuals
predict double rstud, rstudent
sktest resid
summarize resid, detail

* Figure 1: residuals vs fitted, with a LOWESS overlay
twoway (scatter resid yhat, msize(small) mcolor(%70))                        ///
       (lowess resid yhat, lcolor(cranberry) lwidth(medthick))               ///
       (function y = 0, range(yhat) lcolor(black) lwidth(thin)),             ///
       legend(order(2 "LOWESS") position(1) ring(0) region(lstyle(none)))    ///
       xtitle("Fitted values") ytitle("Residuals")                           ///
       title("Figure 1: Residuals vs fitted values")                         ///
       graphregion(color(white)) name(resid_plot, replace)
graph export "output/resid_plot.png", replace width(1600)

* Figure 2: normal Q-Q plot
qnorm resid, title("Figure 2: Normal Q-Q plot of residuals") ///
    graphregion(color(white)) name(qq_plot, replace)
graph export "output/qq_plot.png", replace width(1400)

* Figure 3: residual density against a fitted normal
histogram resid, normal percent                                              ///
    xtitle("Residual") title("Figure 3: Residual distribution")              ///
    graphregion(color(white)) name(resid_hist, replace)
graph export "output/resid_hist.png", replace width(1600)

* Figure 4: conditional convergence, fitted at mean investment
quietly summarize gcf23
local mgcf = r(mean)
quietly regress growth24 lngdppc23 gcf23 highinc
local b0 = _b[_cons] + _b[gcf23] * `mgcf'
local b1 = _b[lngdppc23]
local b3 = _b[highinc]
twoway (scatter growth24 lngdppc23 if highinc == 0, msize(small) mcolor(navy%70))   ///
       (scatter growth24 lngdppc23 if highinc == 1, msize(small) mcolor(cranberry%70)) ///
       (function y = `b0' + `b1'*x, range(lngdppc23) lcolor(navy))                  ///
       (function y = `b0' + `b3' + `b1'*x, range(lngdppc23)                          ///
            lcolor(cranberry) lpattern(dash)),                                       ///
       legend(order(1 "Non-high-income" 2 "High-income") position(7) ring(0)         ///
              region(lstyle(none)))                                                  ///
       xtitle("Log real GDP per capita, 2023")                                        ///
       ytitle("Real GDP per capita growth, 2024 (%)")                                 ///
       title("Figure 4: Conditional convergence at mean investment")                  ///
       graphregion(color(white)) name(scatter_convergence, replace)
graph export "output/scatter_convergence.png", replace width(1600)

*==============================================================================
* SECTION C -- Robustness checks
*==============================================================================

* C.1 Heteroskedasticity-consistent (HC1) standard errors
regress growth24 lngdppc23 gcf23 highinc, robust
estimates store robust

* C.2 Feasible weighted least squares.
*     Model the log of the squared OLS residuals on the regressors, then weight
*     each observation by the inverse of its fitted error variance.
quietly regress growth24 lngdppc23 gcf23 highinc
predict double e_ols, residuals
gen double loge2 = ln(e_ols^2)
regress loge2 lngdppc23 gcf23 highinc
predict double ghat, xb
gen double wt = 1 / exp(ghat)
regress growth24 lngdppc23 gcf23 highinc [aweight = wt]
estimates store wls

* C.3 Outlier-trimmed subsample: drop |studentised residual| > 3
list country growth24 rstud if abs(rstud) > 3, clean noobs
regress growth24 lngdppc23 gcf23 highinc if abs(rstud) <= 3, robust
estimates store trimmed

* Table 2: all four specifications side by side
esttab ols robust wls trimmed using "output/regression_tables_stata.rtf", replace ///
    b(4) se(4) t(3) star(* 0.10 ** 0.05 *** 0.01)                           ///
    stats(N r2 r2_a F, fmt(0 4 4 3)                                         ///
          labels("Observations" "R-squared" "Adjusted R-squared" "F / Wald")) ///
    mtitles("OLS" "Robust (HC1)" "WLS" "Trimmed") label nogaps               ///
    title("Table 2: OLS and robustness estimates, 2024 growth")

esttab ols robust wls trimmed, b(4) se(4) mtitles("OLS" "Robust" "WLS" "Trimmed")

log close

* end of file
