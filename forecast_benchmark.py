"""Run deterministic forecast stress checks without a database or account data."""

from datetime import date
from decimal import Decimal
from random import Random
from statistics import mean

from forecast import build_forecast, month_index, month_label

SCENARIOS = (
    "stable",
    "trend",
    "seasonal",
    "one-off spikes",
    "level shift",
    "downward shift",
    "temporary shift",
)


def simulated_spending(scenario, seed):
    random = Random(seed)
    seasonal = (320, 100, -170, 0, -200, 140, -90, -30, 50, 60, -120, 400)
    months = []
    for index in range(48):
        level = 1200
        if scenario == "trend":
            level += 24 * index
        elif scenario == "seasonal":
            level += seasonal[index % 12]
        elif scenario == "one-off spikes" and index in (11, 26, 39, 44):
            level += 950
        elif scenario == "level shift" and index >= 39:
            level += 500
        elif scenario == "downward shift" and index >= 34:
            level -= 500
        elif scenario == "temporary shift" and 34 <= index < 42:
            level += 500
        months.append(Decimal(level + random.randint(-85, 85)))
    return months


def benchmark(first_seed=200):
    today = date(2026, 1, 1)
    start = month_index(today) - 48
    for scenario in SCENARIOS:
        reports = [
            build_forecast(
                [
                    (month_label(start + index), value)
                    for index, value in enumerate(simulated_spending(scenario, seed))
                ],
                today,
            )
            for seed in range(first_seed, first_seed + 32)
        ]
        yield (
            scenario,
            len(reports) * len(reports[0].backtest),
            mean(report.mean_error for report in reports),
            mean(report.baseline_error for report in reports),
        )


if __name__ == "__main__":
    print(f"{'Scenario':<18} {'Targets':>7} {'Forecast MAE':>13} {'Baseline MAE':>13}")
    for scenario, targets, error, baseline in benchmark():
        print(f"{scenario:<18} {targets:>7} {error:>13.2f} {baseline:>13.2f}")
