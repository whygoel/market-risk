# Multi-Asset Risk Validation Suite

**End-to-End Market Risk Modelling, Backtesting, and Regulatory Capital Calculation under Basel III FRTB for an Equally Weighted Portfolio of NIFTY 50, Gold, USDINR, US Treasuries, and Bitcoin**

---

## Executive Summary

This report presents a comprehensive validation of a GARCH-based risk model for a five‑asset equally weighted portfolio spanning the period 5 August 2016 to 4 August 2026. The portfolio consists of:

* **NIFTY 50** (Indian equity)
* **Gold Futures** (Commodity / safe haven)
* **USDINR Spot** (Currency)
* **iShares 20+ Year Treasury Bond ETF (TLT)** (US government bonds)
* **Bitcoin‑USD** (Cryptocurrency)

The analysis proceeds through every stage required by a quantitative risk manager or model validator: data preparation, stationarity tests, volatility modelling (GARCH family selection), Value‑at‑Risk (VaR) and Expected Shortfall (ES) estimation, Extreme Value Theory for catastrophic risk, rigorous backtesting (Kupiec, Christoffersen, McNeil‑Frey), factor decomposition via PCA, dynamic correlation analysis, and finally regulatory capital calculation under the Fundamental Review of the Trading Book (FRTB).

### Key Findings

* Portfolio daily returns are stationary (ADF $p = 0.0000$) and exhibit no linear autocorrelation (Ljung‑Box $p > 0.05$), justifying a constant‑mean model.
* Strong volatility clustering is confirmed (ARCH‑LM $p = 0.0000$; squared‑return Ljung‑Box $p < 10^{-12}$), necessitating GARCH modelling.
* After comparing GARCH(1,1)‑Normal, GARCH(1,1)‑t, EGARCH(1,1,1)‑t, and TGARCH(1,1,1)‑t, the **GARCH(1,1)‑t** model is selected (lowest BIC, all parameters significant). The asymmetry parameters in EGARCH and TGARCH are not significant, indicating no leverage effect in this diversified portfolio.
* Degrees of freedom $\nu = 4.16$ confirm extremely heavy tails, far from normality.
* Conditional VaR (95%) from GARCH‑t is 1.55%, VaR (99%) is 2.74%. EWMA (RiskMetrics) severely underestimates tail risk (99% VaR only 1.55%).
* EVT (POT‑GPD) provides the catastrophic risk estimates: 99.9% VaR ($\theta$‑adj) = 5.70%, ES = 7.67%, highlighting the severe tail risk from crypto and equity components.
* Rolling backtest over the last 250 days: 5 breaches vs 12.5 expected $\rightarrow$ conservative but independent breaches (Christoffersen $p = 0.65$). ES backtest passes ($p = 0.10$).
* Regulatory capital under FRTB (1‑day) = **5.30% of portfolio value**, driven by the average ES over the last year and a backtesting multiplier of 1.6 (Yellow zone).
* PCA reveals two dominant factors: "Risk‑Off" (Gold + Treasury, 24.4%) and "Risk‑On" (NIFTY + Crypto, 23.6%), with USDINR providing independent diversification.
* Rolling correlations show that NIFTY–Crypto correlation spikes to 0.6–0.7 during crises, while Gold's safe‑haven role has eroded in recent years.

The model is deemed **valid** for regulatory and internal risk management, albeit conservative—a preferable outcome for capital adequacy.

---

## 1. Data Preparation & Portfolio Construction

### 1.1 Data Sources and Cleaning
Daily adjusted close prices for five assets are downloaded via `yfinance` for the period 2016‑08‑05 to 2026‑08‑05 (10 years). Tickers: `^NSEI`, `GC=F`, `INR=X`, `TLT`, `BTC-USD`. After renaming columns for readability, rows with any missing values are dropped, resulting in a clean DataFrame of 2,380 business days.

![Raw price time series (2016-2026)](Raw%20price%20time%20series%20%282016%E2%80%912026%29.png)  
*Figure 1: Raw price time series (2016‑2026).*

### 1.2 Log Returns
Daily log returns are computed as:

$$r_t = 100 \times \ln(P_t / P_{t-1})$$

The resulting DataFrame contains 2,379 observations.

Descriptive statistics of asset returns:

| Asset | Mean (%) | Std (%) | Min (%) | Max (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Crypto** | 0.198 | 4.311 | –46.47 | 22.51 |
| **Gold** | 0.047 | 1.084 | –12.07 | 5.91 |
| **USDINR** | 0.015 | 0.390 | –2.31 | 2.64 |
| **Treasury** | –0.010 | 0.958 | –9.01 | 7.25 |
| **NIFTY** | 0.044 | 1.047 | –13.90 | 8.40 |

Crypto dominates in both mean and volatility; Treasury shows a negative mean due to rising yields; USDINR is extremely low volatility.

### 1.3 Equally Weighted Portfolio Return
The portfolio return is the simple arithmetic average of the five asset returns (20% each, rebalanced daily). Series length: 2,379.

**Portfolio Return Statistics**

| Statistic | Value |
| :--- | :--- |
| **Mean** | 0.059% |
| **Std Dev** | 0.976% |
| **Skewness** | –0.863 |
| **Kurtosis** | 10.926 |
| **Min** | –11.40% |
| **Max** | 4.43% |

The portfolio exhibits strong negative skew and massive excess kurtosis—clear evidence of fat tails. A normal distribution would be a poor approximation.

![Portfolio daily returns time series](Portfolio%20daily%20returns%20time%20series.png)  
![Histogram with normal overlay](histogram%20with%20normal%20overlay.png)  
*Figure 2: Portfolio daily returns time series and histogram with normal overlay.*

Cumulative growth of ₹1 invested at the start reached ₹3.64 by the end of 2026 ($3.64\times$), demonstrating attractive long‑term returns, punctuated by sharp drawdowns.

---

## 2. Stationarity & White Noise Diagnostics

### 2.1 Augmented Dickey‑Fuller (ADF) Test
* **Raw Prices** – All five price series fail to reject the unit root null ($p$-values between 0.59 and 0.99), confirming non‑stationarity.
* **Portfolio Returns** – ADF statistic = –26.61, $p$-value = 0.0000, strongly stationary. This justifies modelling returns directly.

### 2.2 Ljung‑Box Tests
* **On returns** (lags 5, 10, 20): $p$-values 0.22, 0.08, 0.14 $\rightarrow$ no significant autocorrelation $\rightarrow$ returns are white noise. A constant mean model is appropriate.
* **On squared returns** (lags 5, 10, 20): $p$-values $< 10^{-12}$ $\rightarrow$ strong ARCH effects $\rightarrow$ volatility clustering is present.

### 2.3 ARCH‑LM Test
Engle's test statistic = 69.53, $p$-value = 0.0000. Confirms the need for a GARCH model.

---

## 3. Volatility Modelling (GARCH Family)

### 3.1 Model Candidates
Four specifications are fitted on the full portfolio return series:

1. **GARCH(1,1) – Normal** (baseline)
2. **GARCH(1,1) – Student's t** (fat tails)
3. **EGARCH(1,1,1) – t** (asymmetric, captures leverage)
4. **TGARCH(1,1,1) – t** (GJR‑GARCH, alternative asymmetry)

All models assume a constant mean $\mu$, justified by the white‑noise property.

### 3.2 Estimation Results

| Model | $\mu$ | $\omega$ | $\alpha$ | $\beta$ | $\gamma$ | $\nu$ | $\alpha+\beta$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GARCH(1,1)-Normal** | 0.0591<br>($p=0.0008$) | 0.0568<br>($p=0.076$) | 0.1057<br>($p=0.039$) | 0.8394<br>($p\approx 0$) | — | — | **0.945** |
| **GARCH(1,1)-t** | 0.0524<br>($p=0.0006$) | 0.0269<br>($p=0.005$) | 0.0723<br>($p<0.0001$) | 0.9049<br>($p\approx 0$) | — | 4.161<br>($p\approx 0$) | **0.977** |
| **EGARCH(1,1,1)-t** | 0.0521 | 0.0108<br>($p=0.080$) | 0.1638<br>($p<0.0001$) | 0.9707<br>($p\approx 0$) | 0.0077<br>($p=0.610$) | 4.132 | — |
| **TGARCH(1,1,1)-t** | 0.0529 | 0.0256<br>($p=0.014$) | 0.0752<br>($p<0.0001$) | 0.9083<br>($p\approx 0$) | -0.0101<br>($p=0.612$) | 4.154 | — |

**Notes:**
* Values in parentheses represent $p$-values.
* $\nu$ represents the degrees of freedom parameter for Student-t innovations.
* $\gamma$ captures asymmetric volatility effects in EGARCH/TGARCH models.

### 3.3 Model Selection

| Model | AIC | BIC |
| :--- | :--- | :--- |
| **GARCH‑n** | 6366.49 | 6389.59 |
| **GARCH‑t** | 6062.58 | **6091.45** |
| **EGARCH‑t** | 6057.52 | 6092.16 |
| **TGARCH‑t** | 6064.29 | 6098.94 |

EGARCH‑t has the lowest AIC, but its extra asymmetry parameter is insignificant; GARCH‑t has the lowest BIC and all parameters are significant. The principle of parsimony, combined with the lack of leverage effect in a diversified portfolio, selects **GARCH(1,1)‑t** as the final model.

### 3.4 Residual Diagnostics

Standardised residuals $z_t = \varepsilon_t / \sigma_t$ from the GARCH‑t model are examined.

* Mean = 0.0041, Std = 0.9906 $\rightarrow$ close to $(0,1)$.
* Ljung‑Box on $z_t$: all $p$-values $> 0.18$ $\rightarrow$ no autocorrelation remaining.
* Ljung‑Box on $z_t^2$: all $p$-values $> 0.23$ $\rightarrow$ no remaining ARCH effects.

![ACF of standardised residuals and squared residuals](ACF%20of%20standardised%20residuals%20and%20squared%20residuals.png)  
*Figure 3: ACF of standardised residuals and squared residuals – no significant spikes.*

**QQ‑plot against $t(4.16)$** – The central quantiles align well with the theoretical distribution; the tails show slight conservatism (theoretical $t$ expects slightly more extreme values than observed). This indicates the model's fat‑tailed assumption is adequate, and any remaining conservatism is acceptable for risk management.

---

## 4. Value‑at‑Risk & Expected Shortfall

### 4.1 Methodologies

* **Historical VaR/ES** – Non‑parametric, based on empirical quantiles.
* **Parametric GARCH‑t** – 1‑step conditional forecast using the fitted model.
* **EWMA (RiskMetrics)** – Exponential weighted volatility ($\lambda=0.94$), normal assumption, zero mean.

### 4.2 Results

| Method | 95% VaR (%) | 99% VaR (%) | 95% ES (%) | 99% ES (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Historical** | 1.4148 | 2.7139 | 2.2616 | 3.8503 |
| **GARCH‑t(1,1)** | 1.5509 | 2.7370 | 2.3337 | 3.7934 |
| **EWMA** | 1.0982 | 1.5532 | 1.3771 | 1.7794 |

The GARCH‑t VaR is slightly higher than historical, reflecting the current elevated volatility. EWMA severely underestimates risk, especially at the 99% level, because it assumes normality.

![VaR comparison bar chart](VaR%20comparison%20bar%20chart.png)  
*Figure 4: VaR comparison bar chart.*

---

## 5. Extreme Value Theory (EVT) – Peaks Over Threshold

### 5.1 Loss Series & Threshold
Define loss $L_t = -r_t$. The threshold $u$ is chosen as the 95th percentile of losses ($= 1.4148\%$), consistent with the historical 95% VaR. This yields 119 exceedances.

![Mean Excess Plot](Mean%20Excess%20Plot.png)  
*Figure 5: Mean Excess Plot – linearity beyond ~1.4% supports the threshold.*

### 5.2 GPD Fit
Exceedances above the threshold are modelled by the Generalised Pareto Distribution:

$$G(x) = 1 - \left(1 + \xi \frac{x}{\psi}\right)^{-1/\xi}$$

Estimated parameters:
* Shape $\xi = 0.2077$ (heavy‑tailed, Fréchet domain)
* Scale $\psi = 0.6662$

A positive $\xi$ confirms the loss distribution has a fat tail.

![QQ-plot of exceedances vs GPD](QQ%E2%80%91plot%20of%20exceedances%20vs%20GPD.png)  
*Figure 6: QQ‑plot of exceedances vs GPD – good fit except one extreme outlier.*

### 5.3 Extremal Index & Adjustment
The extremal index $\theta$ is estimated using the runs estimator (run length = 3). $\theta = 0.8403$ indicates mild clustering of extremes. The effective sample size becomes $T_{\text{adj}} = T \times \theta$, increasing the VaR/ES estimates.

### 5.4 Unconditional POT VaR & ES

| Confidence | VaR (unadj) | VaR ($\theta$‑adj) | ES ($\theta$‑adj) |
| :--- | :--- | :--- | :--- |
| **99%** | 2.6883% | 2.8532% | 4.0711% |
| **99.9%** | 5.4363% | 5.7023% | 7.6673% |

The $\theta$‑adjusted 99.9% ES of 7.67% means that in the worst 1‑in‑1,000 day, the average loss could exceed 7.7% of the portfolio. This is the most conservative tail‑risk measure and is essential for stress testing.

---

## 6. Backtesting (Out‑of‑Sample, 250 Days)

### 6.1 Setup
The last 250 trading days are held out. A rolling expanding‑window GARCH(1,1)‑t model re‑estimates and forecasts 1‑day 95% VaR for each day in the test set.

### 6.2 VaR Backtesting

* Expected breaches (5%): 12.5
* Actual breaches: 5
* Kupiec POF test: $\text{LR} = 12.14$, $p$-value = 0.0137 (reject correct coverage $\rightarrow$ model is conservative)
* Christoffersen CC test: $\text{LR} = 0.205$, $p$-value = 0.6508 (fail to reject independence of breaches)

The VaR model over‑estimates risk (too few breaches), but breaches are independent. This is acceptable from a prudential standpoint.

### 6.3 ES Backtesting (McNeil‑Frey)

The standardized excess $(L_t - \text{ES}_t) / \sigma_t$ on breach days has mean –0.5478, std 0.5161. A one‑sample t‑test yields $p$-value = 0.1010. We fail to reject the null that the mean excess is zero $\rightarrow$ **the ES forecasts are unbiased** (though slightly conservative).

**Final Verdict: MODEL VALID** (conservative, but robust).

---

## 7. Regulatory Capital Calculation (FRTB IMA)

### 7.1 Expected Shortfall at 97.5%
* Latest conditional ES (GARCH‑t): 2.9091%
* 12‑month average ES (from rolling forecasts): 3.3095%
* POT‑based unconditional ES: 3.0549%

The binding constraint is the **12‑month average ES** (3.31%), to avoid procyclicality.

### 7.2 Backtesting Multiplier
Five exceptions place the model in the **Yellow zone** (5‑9 exceptions). The multiplier is:

$$m_c = 1.5 + 0.1 \times (5 - 4) = 1.60$$

*(Note: The model's conservatism may allow a lower multiplier upon regulatory review, but we use the formula for safety.)*

### 7.3 Capital Charge

$$\mathrm{Capital}_{1\text{-day}} = \max(\mathrm{ES}_{\text{latest}}, \mathrm{ES}_{\text{avg}}) \times m_c = 3.3095\% \times 1.60 = 5.2953\%$$

**For a ₹10 crore portfolio, daily regulatory capital = ₹52.95 lakhs.**  
A 10‑day charge (scaled by $\sqrt{10}$) would be ~16.74% of portfolio value.

---

## 8. Factor Analysis (PCA)

### 8.1 Correlation Matrix

| Asset | Crypto | Gold | USDINR | Treasury | NIFTY |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Crypto** | 1.00 | 0.10 | 0.03 | –0.02 | 0.09 |
| **Gold** | 0.10 | 1.00 | –0.01 | 0.20 | 0.06 |
| **USDINR** | 0.03 | –0.01 | 1.00 | –0.01 | –0.15 |
| **Treasury** | –0.02 | 0.20 | –0.01 | 1.00 | –0.08 |
| **NIFTY** | 0.09 | 0.06 | –0.15 | –0.08 | 1.00 |

![Correlation Matrix](Correlation%20Matrix.png)  
Overall low correlations confirm effective diversification.

### 8.2 Principal Components

| PC | Eigenvalue Ratio | Cumulative | Interpretation (loadings) |
| :--- | :--- | :--- | :--- |
| **1** | 24.4% | 24.4% | **Risk‑Off**: Gold (+0.70), Treasury (+0.50) |
| **2** | 23.6% | 48.0% | **Risk‑On**: NIFTY (+0.67), Crypto (+0.21) |
| **3** | 20.8% | 68.8% | **Crypto/FX**: Crypto (+0.73), USDINR (+0.61) |
| **4** | 16.3% | 85.1% | Residual |
| **5** | 14.9% | 100% | Noise |

The first three components explain 68.7% of total variance, indicating three distinct risk factors.

![biplot](biplot.png)  
*Figure 8: The biplot visually separates "Risk‑On" assets (NIFTY, Crypto) from "Safe‑Haven" assets (Gold, Treasury), with USDINR acting as an orthogonal diversifier.*

---

## 9. Dynamic Correlation Analysis

Using a 238‑day rolling window (approximately one trading year), pairwise correlations are computed for all asset pairs.

![Rolling correlations – all 10 pairs](Rolling%20correlations%20%E2%80%93%20all%2010%20pairs.png)  
*Figure 9: Rolling correlations – all 10 pairs.*

**Key Observations:**
* **NIFTY–Crypto** correlation fluctuates between –0.1 and +0.7, spiking during the COVID crash (2020) and the 2022 crypto winter. This confirms that diversification evaporates during systemic crises.
* **Gold–Treasury** correlation remains moderately positive, while **Gold–NIFTY** has turned positive recently (2024‑2026), suggesting Gold's traditional safe‑haven role may be weakening.
* **USDINR** shows low correlation with all assets, reinforcing its diversification benefit.

This dynamic behaviour justifies the use of conditional risk models and highlights the danger of assuming static correlations in portfolio construction.

---

## 10. Conclusion & Recommendations

### 10.1 Summary
The equally weighted multi‑asset portfolio delivers solid historical returns (~15% annualised) but carries significant tail risk, primarily driven by Crypto and equity components. The GARCH(1,1)‑t model provides a well‑specified, validated framework for measuring both conditional VaR/ES and, when supplemented with EVT, unconditional catastrophic risk. Backtesting confirms the model's adequacy: it is conservative but independent, and the ES backtest shows no significant bias.

### 10.2 Regulatory Capital
Under FRTB IMA, the one‑day capital charge is **5.30%** of portfolio value. This figure can be directly used for internal capital allocation and regulatory reporting.

### 10.3 Recommendations
* **Monitor the conservatism** – if the model continues to over‑estimate risk, recalibrate the t‑distribution degrees of freedom or explore skewed‑t alternatives.
* **Integrate DCC‑GARCH** – To fully capture time‑varying correlations for active risk budgeting.
* **Stress testing** – Use EVT‑based ES (99.9%) of 7.67% as a starting point for scenario design.
* **Model governance** – The backtesting results support a model‑approval submission, but the Yellow zone multiplier (1.6) should be discussed with regulators to potentially revert to 1.5 given the conservatism.

### 10.4 Future Work
* Extend the model to incorporate intra‑horizon risk (e.g., value‑at‑risk over multiple days).
* Implement a Dynamic Conditional Correlation (DCC) or Copula‑GARCH model to better capture correlation breakdowns.
* Apply machine learning for regime‑switching volatility to reduce conservatism.
