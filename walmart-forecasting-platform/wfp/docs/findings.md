# Findings (read before using any number)

All figures are out-of-sample 13-week-ahead results from `docs/results.md`; WAPE unless stated.

1. **Latest window (2012-08-03 to 2012-10-26): the global LightGBM model beats the baselines.** Store x Department WAPE 8.40% vs seasonal naive 10.70% and naive 14.29%.
   On the 100-series subset: LightGBM 7.96%, Prophet 9.28%, TFT 10.86%, seasonal naive 10.92%.
2. **Holiday window (2011-10-28 to 2012-01-20): seasonal naive is as good as or better than every learned model.** Store x Department: seasonal naive 12.08%, LightGBM 12.52%.
   Total level: seasonal naive 3.92% vs LightGBM bottom-up 5.00%. The training data holds only one earlier holiday season and lag-52 is missing for it, so the model cannot learn series-specific holiday lifts.
   Prophet (15.46%) and the TFT (15.62%) were worse than seasonal naive on the subset (11.53%).
3. **Reconciliation helps aggregates, not the bottom level.** Bottom-up lowers Department-level WAPE from 5.70% to 4.48% and total from 1.69% to 1.38% (latest window) at no cost to the bottom level.
   Covariance-based MinT did not generalise: on the holiday window its Store x Department WAPE was 26.95% (hold-out-error covariance estimated from only 28 validation samples across 3,461 nodes) and its
   intervals under-covered (46.5%). It was the validation winner in both windows, i.e. the validation window was not a reliable guide to test behaviour.
   **Disclosure:** the operational default was changed to bottom-up (`hierarchy.default_method`) *after* seeing these test results; the validation-selected method is stored in each fold's `info.json`.
4. **Intervals:** LightGBM raw q10/q90 under-cover (72.3% and 71.0% at store-department level, nominal 80%); conformalised intervals reach 79.2% and 79.4%. Reconciled intervals reach 79.5% (bottom-up) at store-department level
   in the latest window but are wider (relative width 46% vs 27%). Total-level coverage is based on 13 points per fold, so treat it as indicative only.
5. **TFT and Prophet are under-resourced here**: 100 series, 8 epochs maximum, 1 CPU core, no hyper-parameter search. They are included as implemented baselines, not as the best attainable result.
6. **RMSSE** is very large for OLS/WLS at the bottom level because a few series with tiny naive-error denominators receive large errors; compare `RMSSE_w` or WAPE.
7. **Inventory (scenario assumptions, replay on realised sales, latest window, 45 stores / 3,000+ store-department items):** the optimised plan used 17,352 orders against 35,840 for the weekly base-stock reference, with total modelled cost
   $16.76M vs $18.11M and fill rate 92.8% vs 93.5%; no reordering gives 7.7% fill. The cost saving comes from fewer orders under an assumed $150 fixed cost, and fill rate is slightly lower. These numbers depend entirely on the assumed costs, lead time (1 week), opening stock (1 week of cover)
   and capacity in `config/default.yaml`, and sales are used as a proxy for demand. Solver status was FEASIBLE (time-limited, median MIP gap 1.3%, worst store 17.4% because that solve ran while the CPU was shared; re-run for a tighter gap.
