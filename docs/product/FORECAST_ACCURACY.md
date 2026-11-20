# Expense forecast

The forecast estimates recorded net expenses for the next calendar month. The latest month available to the model is the preceding closed calendar month, so the prediction horizon is two monthly steps. The open month and future entries are excluded. A missing month breaks the sequence. A month with an explicit zero-value expense is valid, and refunds reduce net expenses.

Candidates include the last closed month, recent means and medians, weighted and exponentially smoothed levels, recent trends, and same-month-last-year when enough comparable observations exist. The candidate with the lowest mean absolute error across comparable past targets is selected. The backtest replays the selection procedure at each cutoff and excludes both the target month and its immediately preceding month. The most recent 12 usable targets are compared with the last-closed-month baseline. At least 14 contiguous closed months are required, giving at least three outer tests.

The page shows absolute error in currency units and as a percentage of total absolute spending across the test window. That percentage is undefined when total absolute spending is zero. With at least eight observations the page also shows the empirical 80th percentile of historical absolute error. This is a historical reference rather than a calibrated prediction interval.

## Development benchmark

The synthetic preview account had 29 consecutive completed months. Across the same 12 historical targets in November 2025 through October 2026, mean absolute error was $2,934.89 for repeating the last closed month, $2,392.23 for the previous LSTM architecture with two recursive steps and a fixed seed of 42, and $2,356.70 for the final selection procedure. The preview history informed candidate development. The results are exploratory and performance on real accounts remains unverified.

The engine forecasts the recorded ledger series. Unrecorded transactions, changes in expense patterns, and exceptional future spending affect accuracy. Scheduled reminders remain separate from recorded expenses.

[Time-series cross-validation and simple forecasting methods](https://otexts.com/fpp3/toolbox.html)

## Stress checks

Run the reproducible benchmark with `python forecast_benchmark.py`. It uses 32 synthetic series per scenario and 12 historical targets per series. No account data is read. Seeds 100–131 were used for development, and seeds 200–231 were used for this comparison. The two seed sets use the same scenario definitions.

| Scenario | Previous MAE | Updated MAE | Last-closed-month baseline MAE |
| --- | ---: | ---: | ---: |
| Stable with noise | 49.80 | 50.61 | 60.57 |
| Upward trend | 78.86 | 54.68 | 70.16 |
| Seasonal | 62.75 | 62.75 | 211.85 |
| One-off spikes | 215.70 | 215.70 | 357.75 |
| Sudden sustained change | 177.68 | 173.41 | 135.86 |

The updated selector adds a 12-month linear regression and can favor the recent level when three consecutive closed months differ substantially from the preceding nine. The larger error after sudden changes warrants caution. This benchmark tests synthetic patterns and cannot establish accuracy for a particular live ledger. The forecast page now flags historical underperformance against the baseline and reports signed bias in the backtest details.

### Sustained change follow-up

The selector now checks for two to six consecutive closed months that all exceed the same upper or lower threshold relative to the preceding nine months. One exceptional month cannot trigger this rule. The earlier seasonal-method guard is retained.

A repeat comparison with seeds 400–431 used the same five scenario definitions and 1,920 forecast targets. Mean absolute error under a sustained increase fell from 180.75 to 161.50. The last-closed-month baseline scored 128.56. The other four scenarios had identical mean absolute errors to the reported hundredth. This remains synthetic validation, including the level-shift scenario, which simulates an increase. Unit tests also verify detection of repeated decreases.
