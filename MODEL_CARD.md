# Model card: Marktradar

## Purpose

Forecast how the market share of each Swiss health insurer (mandatory basic insurance)
changes from the latest year with insured data (*t*) to *t+1* and *t+2*, per canton and
nationally, including scenarios for the premium change relative to the market.
Intended for sales and market analysis. **Not** intended for pricing, decisions about
individuals, or as the sole basis for a decision.

## Data

- BAG premium open data (opendata.swiss, "health-insurance-premiums"):
  approved premiums 2011–2027 and average insured persons per insurer and canton 2009–2025.
- Public and aggregated per insurer and canton. No personal data.
- Reference premium: adults, standard model, CHF 300 deductible, with accident cover,
  averaged over premium regions within a canton.
- Excluded from training: insurer-canton combinations below 500 insured persons, and
  share changes above ±50 % per year (mergers and portfolio transfers, not switching).

## Target and features

- Target: change of the log market share within the canton from *t* to *t+h*.
- Features: premium level relative to the canton market; last premium change relative to
  the market; approved premium change(s) relative to the market for the forecast year(s)
  (the scenario lever); average premium increase of the canton market; previous share
  change; share within the canton.
- All features are known at forecast time: approved premiums are published at the end of
  September, before the switching deadline on 30 November.
- Insurer financials (reserves, administrative costs, risk equalisation) were tested and
  dropped, because they did not improve the backtest.

## Model

One model per horizon (1 and 2 years): scikit-learn `HistGradientBoostingRegressor` with
absolute-error loss, weighted by the square root of insured persons, with monotonic
constraints (a relatively higher premium never increases predicted growth).
Prediction interval: 10 % and 90 % quantile models, recalibrated with conformalized
quantile regression on out-of-time errors. National figures are insured-weighted sums of
the canton forecasts.

## Evaluation

Rolling-origin backtest: for base year *T*, the model is trained only on base years whose
outcome was known at *T*. Errors are for the national market share per insurer.

| Horizon | Test years | Error model | Error "share unchanged" | Improvement | Rank correlation | Interval coverage (target 80 %) |
|---|---|---|---|---|---|---|
| 1 year | 2017–2025 | 0.047 | 0.058 | 20 % | 0.61 | 76 % |
| 2 years | 2018–2025 | 0.097 | 0.115 | 16 % | 0.54 | 72 % |

Per year, the improvement is smallest for 2023–2025, the years after the large premium
increases, when switching patterns changed.

## Alternatives considered

The same backtest was run for linear regression, random forest, LightGBM, TabPFN v2 and
Chronos-2 (zero-shot and fine-tuned); see `docs/model_comparison.md`. TabPFN v2 was
slightly more accurate (+18 % vs. +16 % two years ahead, +20 % vs. +19 % one year ahead),
a difference within year-to-year noise, at about 15 times the compute and without
monotonic guarantees. Chronos-2 did not beat the "share unchanged" baseline. Gradient
boosting therefore remains the production model.

## Known limitations

1. **Lots of switching is not explained by premiums.** Service, brand, brokers and
   supplementary insurance are not in public data. Treat forecasts as one input among others.
2. **Regime changes.** After the premium shocks of 2023/2024 accuracy dropped and the
   intervals covered less than 80 %.
3. **Reference premium only.** Prices of alternative models (HMO, telemedicine) and
   other deductibles are not used; insurers with a different model mix may react differently.
4. **Constant market size.** `bestand_prognose` assumes the size of each canton market
   stays the same.
5. **National intervals are conservative.** They assume all cantons of an insurer move
   together, which overstates the uncertainty somewhat.
6. **No causal claims.** The model learns associations from observational data. A
   scenario shows what similar situations looked like in the past, not a guaranteed effect.
7. **Earlier version.** A previous version used realised average premiums from the
   supervisory data and reported higher accuracy (rank correlation 0.76). That figure was
   too optimistic, because realised premiums partly reflect who switched. The current
   numbers use only information available in advance.

## Accountability

Every model version stores its metrics, data fingerprint, configuration and git commit
in `models/<version>/metadata.json`. New versions only go into production after passing
the quality gate; the live check on new data is recorded in `models/monitoring.json`.
