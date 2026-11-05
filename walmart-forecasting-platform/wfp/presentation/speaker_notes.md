# Speaker notes (10 minutes, about 1 minute per slide)

## Slide 1: Forecasting weekly demand across Walmart's store-department hierarchy
Open with the one-line promise: a forecasting system you can trust because every number is a chronological, out-of-sample result, including the unflattering ones. About 10 minutes: problem, method, results, inventory, what to take away.

## Slide 2: Planning needs weekly forecasts at every level, with uncertainty
Explain scale and the two honesty points: no state in the data, so the Group level is store type; no item IDs, so no item-level forecasts. Also mention the holiday window has only one earlier holiday season to learn from.

## Slide 3: One pipeline, five stages, every stage past-only
Walk the diagram left to right. Stress the leakage control: a unit test destroys all sales after the forecast origin and asserts the features do not change.

## Slide 4: Evaluation: chronological folds, test window touched once
Two windows on purpose: the latest 13 weeks, and the Thanksgiving to Christmas season where forecasting is hardest.

## Slide 5: Latest window: LightGBM beats every baseline
Headline result. Be explicit that Prophet and TFT were deliberately small and are scored on the same 100 series as LightGBM, so the comparison is fair but not a verdict on the best achievable TFT.

## Slide 6: Holiday window: seasonal naive is hard to beat
This is the slide that builds trust. We report that the learned model does not clearly beat seasonal naive in the holiday window, and explain why. With another year of data the picture should change; we have not tested that.

## Slide 7: Reconciliation sharpens aggregates; MinT was unstable
Reconciliation makes the forecasts add up and helps the aggregates. The covariance-based MinT overfit its tiny validation sample, so the operational default is bottom-up. We disclose that this default was chosen after observing test results.

## Slide 8: Prediction intervals need calibration
Coverage is the honest test of an interval. The raw quantile models are too narrow; conformal calibration fixes that for the base model. Reconciled intervals are wider and in the holiday window still under-cover.

## Slide 9: Inventory optimiser: a decision tool driven by assumptions
Be clear this is a demonstration under stated assumptions: the Walmart files contain no unit costs, stock, suppliers or capacities. The saving comes mostly from fewer orders under an assumed fixed ordering cost.

## Slide 10: Takeaways
Close on what is solid and what is not. Invite questions on the holiday window and the inventory assumptions.
