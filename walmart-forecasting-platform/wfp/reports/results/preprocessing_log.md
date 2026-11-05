| # | Step | Decision | Rows/cells affected | Rationale |
|---|---|---|---|---|
| 1 | features window | dropped features rows after last training date 2012-10-26 | 1755 | Those rows belong to Kaggle's hidden test period; they contain NaN CPI/Unemployment and no sales. |
| 2 | MarkDown1-5 NaN | filled with 0 and added indicator `markdown_available` (1 from 2011-11-11) | 21952 | Structural missingness: markdowns were not recorded before the programme. Median-filling (as in the reference notebook) would invent promotions. The indicator lets models distinguish 'no markdown' from 'not recorded'. |
| 3 | MarkDown negatives | clipped to 0 | 23 | Negative promotional spend is not meaningful. |
| 4 | CPI/Unemployment NaN | forward-filled within store (past-only); none remain inside the training window | 0 | No global median/mean imputation, which would leak future/global statistics. |
| 5 | merge | train LEFT JOIN stores ON Store; LEFT JOIN features ON (Store, Date); IsHoliday taken from train (verified identical to features.IsHoliday) | 421570 | validate='many_to_one' asserts no row multiplication. |
| 6 | negative Weekly_Sales | retained, not clipped or dropped | 1285 | Returns/adjustments are genuine records; WAPE/MAE/RMSE are computed on signed values. |
| 7 | outliers / spikes | NOT removed; investigated in EDA (see docs/eda.md) | 0 | The reference notebook dropped |z|>2.5 rows, which deletes exactly the holiday peaks inventory planning needs. |
| 8 | target scaling | no global normalisation of Weekly_Sales; models forecast dollars | 0 | Inside LightGBM the target is divided by a past-only per-series scale and multiplied back (invertible, dollars out). |
| 9 | grid completeness | series x week grid 3331 x 143; unobserved cells kept as NaN (not zero) for features/evaluation; treated as 0 only when summing to aggregates | 54763 | 11.5% of cells are unobserved (series start late, end early or have gaps). |
| 10 | geographic level | hierarchy group level = Type (state proxy; no state in Walmart files) | 3 | stores.csv has no state/region. Supply config/store_state_mapping.csv (Store,State) to use real states; no mapping is invented. |