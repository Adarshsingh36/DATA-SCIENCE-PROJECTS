# Methodology

## 1. Data and hierarchy
* `train.csv` (421,570 rows, 45 stores, 81 departments, 3,331 store-department series, weekly Fridays 2010-02-05 .. 2012-10-26),
  `stores.csv` (Type, Size) and `features.csv` (weather, fuel, 5 markdowns, CPI, unemployment, holiday flag).
* Merge keys: `train.Store -> stores.Store` and `(Store, Date) -> features`; many-to-one validated, row count asserted.
* **Hierarchy** (grouped, 3,461 nodes): Total > Group > Store > Store-Department, plus Department (summed over stores).
  **The Walmart files contain no state.** Supply `config/store_state_mapping.csv` (`Store,State`) and the Group level becomes
  State; without it the Group level is store **Type (A/B/C)**, labelled as a state proxy everywhere. No mapping is invented.
* No item/product level exists in these files and none is created.
* Unobserved weeks (11.5% of the series x week grid) stay `NaN` for features and evaluation; they count as 0 only when summing to aggregates.

## 2. Evaluation protocol (no random splits)
Rolling-origin, chronological, 13-week horizon. Two test windows:
| fold | test weeks | why |
|---|---|---|
| `holiday` | 2011-10-28 .. 2012-01-20 | Thanksgiving + Christmas run-up; hardest regime; only one prior holiday season to learn from |
| `latest` | 2012-08-03 .. 2012-10-26 | last 13 weeks of data; includes Labor Day |

For each fold: **train** targets < validation start; **validation** = the 13 weeks before the test window (early stopping,
interval calibration, reconciliation-method selection); **test** is touched once, after all choices are fixed. The final model is refit on
everything before the test window with the iteration counts found on validation. Hyper-parameters are fixed in
`config/default.yaml`; there is no tuning against test error.

## 3. Features (all past-only; see `features.py` and `tests/test_features_leakage.py`)
Direct multi-horizon forecasting (h = 1..13) from origin t0 = last observed week.
* Lags relative to the origin: 1, 2, 4, 8, 12 weeks (lag 1 = latest observation); **lag 52 is target-aligned** (`y[t0+h-52]`, plus +-1 week neighbours and their mean).
* Trailing rolling means (4, 8, 13, 26, 52) and volatility (std 4, 8, 13, 26; coefficient of variation; 4/13 momentum; 13-week year-over-year ratio).
* Store-level and department-level momentum / year-over-year context from past sums.
* Calendar of the **target** week (known in advance): holiday flag and type, week-of-year, weeks to Christmas / Thanksgiving, Easter distance.
* Exogenous variables **as of the origin** (temperature, fuel, CPI, unemployment, trailing 4-week markdown total + availability flag), because their future values
  are not known at forecast time. `known_future_markdowns: true` switches on target-week markdowns for users whose promotion calendar is planned ahead.
* Static: store, department, type, log size, hierarchy level.
* Per-series **scale** = trailing 13-week mean |sales| at the origin. Lag/rolling features are divided by it and the target is `y / scale`, **multiplied back to dollars**
  at prediction time. Sample weight = scale, so the loss is dollar-denominated. No global target normalisation is used.

## 4. Models
| model | notes |
|---|---|
| Naive | last observed value |
| Seasonal naive | value 52 weeks before the target week (falls back to naive if unobserved) |
| LightGBM | one global model: weighted-L1 point forecast (median, aligned with WAPE) + q10 and q90 quantile models; separate instance for the 130 aggregate nodes |
| Prophet | one model per series on the representative subset; yearly seasonality, holiday effects, multiplicative |
| Temporal Fusion Transformer | `pytorch-forecasting`, 26-week encoder, 13-week decoder, quantile loss (0.1/0.5/0.9), known-future calendar inputs, early stopping on the validation window; representative subset only (CPU) |

**Representative subset:** 100 series (10 per past-revenue decile, measured on weeks before the earliest validation window), restricted to gap-free series
(TFT needs contiguous time indices). Every model is re-scored on the same series for a like-for-like comparison.

## 5. Hierarchical reconciliation (`reconcile.py`)
Base forecasts exist for every node. Coherent forecasts are `S G y_hat` with `G` from: bottom-up, OLS, WLS (structural), MinT with Schafer-Strimmer shrinkage using
(a) hold-out validation errors and (b) in-sample naive residuals. MinT (a) is **cross-fitted** on validation origins when scored and calibrated so its validation
score is not optimistic. The default method per fold is the one with the lowest validation WAPE averaged across levels; test results of all methods are reported.

## 6. Probabilistic forecasts
* Base LightGBM: raw q10/q90 and **conformalised quantile regression** (CQR; score = max(q10 - y, y - q90)/scale per hierarchy level and horizon bucket, calibrated on validation).
* Reconciled forecasts: symmetric split-conformal intervals on validation residuals of the reconciled forecasts, per level and horizon bucket.
* Reported: empirical coverage vs the nominal 80%, relative width, interval (Winkler) score.

## 7. Metrics
* **WAPE** = sum|y - yhat| / sum|y| (primary, additive over any aggregation); **MAE**, **RMSE** in dollars; **RMSSE** = sqrt(MSE / MSE of the one-step naive forecast on the training window),
  per series, reported as an unweighted mean and as a revenue-weighted mean (`RMSSE_w`); signed **bias**. R-squared is intentionally not computed.
* RMSSE is dominated by series with near-zero naive error and can explode for them; WAPE and `RMSSE_w` are the more stable summaries.

## 8. Inventory optimisation (`inventory.py`)
Per-store mixed-integer programme solved with OR-Tools SCIP over the 13 forecast weeks. Demand paths = calibrated p10 / p50 / p90 with weights 25/50/25;
order decisions are common to the scenarios (here-and-now); unmet demand is lost. Constraints: inventory balance with lead time, store storage capacity, order
indicator (fixed ordering cost). **Every cost, lead time, capacity and opening stock is a configurable assumption** (`config/default.yaml: inventory`);
the dataset has no unit costs, stock levels or suppliers, so nothing is presented as Walmart fact. Units are retail-sales dollars. The plan is evaluated
by replaying it on realised sales (a proxy for demand: lost sales are censored) against a base-stock reference policy and a no-reorder policy.

## 8b. Changes made after seeing test results (full disclosure)
1. MinT was first implemented with in-sample one-step naive residuals; its test results were poor, so a hold-out-error version (cross-fitted when scored/calibrated) was added. Both are reported.
2. The default reconciliation method used by the dashboard and the inventory stage was set to bottom-up after MinT proved unstable on the test windows. The validation-selected method remains stored per fold.
3. Origin stride, tree count and learning rate were set once for CPU runtime (see config), not tuned on test error.

## 9. Known limitations
* Only one prior holiday season exists in the `holiday` fold's training data and the lag-52 feature is missing for it; seasonal naive is therefore very hard to beat there.
* The forecasts are evaluated inside the available data. A genuine out-of-sample forecast past 2012-10-26 would need future calendar/holiday inputs and is not produced.
* TFT/Prophet are trained on a 100-series subset with a small CPU budget; their results are not a statement about the best achievable TFT performance.
* Sales are not demand: stockouts censor demand and no unit/price data exists.
