# Review of the reference notebook (`walmart-sales-forecasting.ipynb`)

The notebook was inspected cell by cell and used **only as a reference for the dataset and the original workflow**:
load `train/stores/features` -> fill CPI/Unemployment with the median and MarkDown with 0 -> merge -> drop |z|>2.5 rows ->
add per-(Store,Dept) max/min/mean/median/std -> one-hot encode Store/Dept/Type -> MinMax-scale every numeric column
including the target -> pick 26 features with a random forest -> **random** 80/20 split -> Linear / Random Forest / KNN / XGBoost,
compared by `score()*100` ("accuracy") plus MAE/MSE/RMSE.

| # | What the notebook does | Why it is unsafe for forecasting | What this project does instead |
|---|---|---|---|
| 1 | `train_test_split(random_state=50)` | Neighbouring weeks of the same series land on both sides; the model sees the future. Scores are interpolation scores. | Rolling-origin chronological folds: train < validation < test, always. |
| 2 | Adds per-series `max/min/mean/median/std` of **Weekly_Sales over the whole dataset** as features | Each row carries statistics that include its own target and all future weeks (direct target leakage). | Only trailing windows ending at the forecast origin; unit-tested by destroying all post-origin data. |
| 3 | `MinMaxScaler.fit_transform` on the full data, **target included**, one scaler object re-fit per column | Full-data statistics before splitting; predictions are in 0-1 units, unusable for operations. | No global target scaling; the model divides by a past-only per-series level and multiplies back, so outputs are dollars. |
| 4 | Fills CPI/Unemployment with the global median; MarkDown NaN with 0 (negative -> 0) | Median uses future values; MarkDown NaN before Nov-2011 means "not recorded", not "no promotion". | Past-only forward fill; `markdown_available` indicator; features rows after the training window dropped. |
| 5 | Removes rows with \|z\| > 2.5 on sales, size, weather, markdown | Deletes precisely the holiday peaks the business must stock for; z-scores use the whole dataset. | No removal. Spikes investigated (about 49% fall in the 11% of weeks around holidays); negative sales kept as returns. |
| 6 | One-hot encodes Store (45) + Dept (81) + Type | Large sparse design, no notion of hierarchy or similarity. | Native categorical handling in LightGBM plus size/type/level features. |
| 7 | Feature selection with a random forest fit on the full data | Selection leaks the target distribution of the test period. | No data-driven selection before splitting; importance reported afterwards. |
| 8 | Reports `lr.score(X_test, y_test)*100` as "accuracy" | R-squared is not forecast accuracy and says nothing about scale-free error or horizon. | WAPE, RMSSE, MAE, RMSE (+ bias, interval coverage); R-squared is not computed anywhere. |
| 9 | KNN with `n_neighbors=1` | Memorises neighbouring weeks of the same series (leaky under random split). | Not used. |
| 10 | Only one-step "tabular" regression; no lags, no horizon, no baselines | Cannot tell whether a model beats "same week last year". | Naive and seasonal-naive baselines; direct 1-13 week horizons; lags 1/2/4/8/12/52. |
| 11 | No hierarchy, no uncertainty, no inventory decision | Forecast alone does not support planning. | Reconciliation, conformal intervals, OR-Tools inventory plan, dashboard. |

What was kept: the table join logic (`train + stores` on Store, `+ features` on Store/Date), the observation that `IsHoliday`
is identical in train and features, and the EDA themes (monthly seasonality, store/department averages, holiday share).
