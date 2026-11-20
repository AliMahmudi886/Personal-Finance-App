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

### Comparing the baseline estimate

The benchmark also covers a persistent decrease and a temporary eight-month increase. In exploratory runs, emphasizing recent errors improved the sustained increase case but worsened stable, trending, and intermittent-spike cases. A longer shift-detection window produced mixed results across sustained increases, decreases, and temporary changes on seeds 700–731. The selection policy therefore retains its previous windows.

The forecast page now shows the baseline estimate for the same target month beside its historical error. It repeats the last closed month's total, so users can compare both current estimates and their past errors at the same horizon.

### What-if estimates

An optional signed adjustment lets users explore a known expense or refund for the target month. The server validates the amount to cents and rejects a submission for a period that has rolled over. The adjusted amount is shown separately from the model estimate. It does not enter training, backtests, or the ledger, and disappears on a fresh GET.

### Combining comparable estimates

The selector can now average up to three nonseasonal methods within 4% of the leading method’s historical error, provided the combination has lower error over the same earlier validation targets. A seasonal winner, perfect historical score, or detected sustained shift keeps its individual estimate.

Reproduce the synthetic runs with `python forecast_benchmark.py --seed 1100` in this revision and the previous revision `fc3ba81` (call `benchmark(1100)` directly there). The 2,688 targets use 32 series per scenario and 12 targets per series. The seed range 1100–1131 was separate from the experiments used to choose the rule, while the scenario definitions are shared.

| Scenario | Previous MAE | Updated MAE |
| --- | ---: | ---: |
| Stable | 48.08 | 48.27 |
| Upward trend | 53.63 | 52.05 |
| Seasonal | 56.95 | 56.95 |
| One-off spikes | 209.71 | 207.85 |
| Sustained increase | 155.28 | 155.89 |
| Sustained decrease | 80.30 | 81.51 |
| Temporary shift | 157.79 | 157.37 |
| Aggregate | 108.82 | 108.56 |

The synthetic preview account's MAE changed from 2,170.98 to 2,167.87 over 12 targets. These are small descriptive improvements. Accuracy on real accounts remains unverified.

### Earlier response to a possible spending shift

After one moderate break from the previous level, the forecast moves 40% of the way from the selected method toward that observation. The break must exceed the existing threshold and stay within 60% of the preceding median. Larger single observations require corroboration. A confirmed run now remains eligible for up to 12 months. The seasonal-method guard remains active.

Compare this revision with `d32ff34` using the updated benchmark sampler and `python forecast_benchmark.py --seed 1500 --vary-shifts`. The earlier model uses the same new sampler. These 2,688 synthetic targets use fresh seeds 1500–1531, with varied shift onsets, shift sizes and spike sizes. The thresholds were chosen on earlier fixed-shape synthetic scenarios.

| Scenario | Previous MAE | Updated MAE | Last-closed-month baseline MAE |
| --- | ---: | ---: | ---: |
| Stable | 48.93 | 48.93 | 57.52 |
| Upward trend | 54.22 | 54.22 | 71.62 |
| Seasonal | 52.85 | 52.85 | 215.38 |
| One-off spikes | 148.18 | 158.05 | 210.01 |
| Sustained increase | 170.28 | 160.04 | 129.08 |
| Sustained decrease | 166.74 | 157.99 | 127.11 |
| Temporary shift | 232.34 | 222.44 | 189.99 |
| Aggregate | 124.79 | 122.08 | |

The new strategy trades higher error on one-off spikes for a quicker response to changes. The last-closed-month baseline still beats both model versions on the three shift scenarios. Real-account performance remains unverified.
