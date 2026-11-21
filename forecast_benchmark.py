"""Run deterministic forecast stress checks without a database or account data."""

from argparse import ArgumentParser
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


def simulated_spending(scenario, seed, *, vary_shifts=False, vary_precursor=False):
    random = Random(seed)
    seasonal = (320, 100, -170, 0, -200, 140, -90, -30, 50, 60, -120, 400)
    settings = Random(seed + 100_000)
    onset = settings.randint(34, 41) if vary_shifts else None
    shift = settings.randint(290, 790) if vary_shifts else 500
    spikes = (
        settings.sample(range(33, 48), 2) + [11, 26]
        if vary_shifts
        else (11, 26, 39, 44)
    )
    mixed_onset = 39 if onset is None else onset
    precursor_random = Random(seed + 200_000)
    precursor_month = (
        precursor_random.randint(25, mixed_onset - 3) if vary_precursor else 30
    )
    precursor_amount = precursor_random.randint(290, 1050) if vary_precursor else shift
    months = []
    for index in range(48):
        level = 1200
        if scenario == "trend":
            level += 24 * index
        elif scenario == "seasonal":
            level += seasonal[index % 12]
        elif scenario == "one-off spikes" and index in spikes:
            level += settings.randint(290, 1050) if vary_shifts else 950
        elif scenario == "shift after spike":
            if index == precursor_month:
                level += precursor_amount
            if index >= mixed_onset:
                level += shift
        elif scenario == "level shift" and index >= (
            onset if onset is not None else 39
        ):
            level += shift
        elif scenario == "downward shift" and index >= (
            onset if onset is not None else 34
        ):
            level -= shift
        elif (
            scenario == "temporary shift"
            and (onset if onset is not None else 34)
            <= index
            < (onset if onset is not None else 34) + 8
        ):
            level += shift
        months.append(Decimal(level + random.randint(-85, 85)))
    return months


def benchmark(
    first_seed=200, *, vary_shifts=False, include_mixed=False, vary_precursor=False
):
    today = date(2026, 1, 1)
    start = month_index(today) - 48
    scenarios = (*SCENARIOS, "shift after spike") if include_mixed else SCENARIOS
    for scenario in scenarios:
        reports = [
            build_forecast(
                [
                    (month_label(start + index), value)
                    for index, value in enumerate(
                        simulated_spending(
                            scenario,
                            seed,
                            vary_shifts=vary_shifts,
                            vary_precursor=vary_precursor,
                        )
                    )
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
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=200,
        help="First seed of 32 synthetic series per scenario.",
    )
    parser.add_argument(
        "--vary-shifts",
        action="store_true",
        help="Vary timing and size of abrupt changes and spikes.",
    )
    parser.add_argument(
        "--include-mixed",
        action="store_true",
        help="Include a permanent increase after a prior isolated spike.",
    )
    parser.add_argument(
        "--vary-precursor",
        action="store_true",
        help="Vary the time and amount of the earlier spike independently.",
    )
    args = parser.parse_args()
    print(f"{'Scenario':<18} {'Targets':>7} {'Forecast MAE':>13} {'Baseline MAE':>13}")
    for scenario, targets, error, baseline in benchmark(
        args.seed,
        vary_shifts=args.vary_shifts,
        include_mixed=args.include_mixed,
        vary_precursor=args.vary_precursor,
    ):
        print(f"{scenario:<18} {targets:>7} {error:>13.2f} {baseline:>13.2f}")
