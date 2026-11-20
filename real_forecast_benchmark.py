"""Replay forecast revisions on public PKDD'99 debit totals without emitting account data."""

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from random import Random
from statistics import median
from types import ModuleType

from forecast import HORIZON, MIN_HISTORY, select_method

REFERENCE_REVISION = "25ede3568ed62f7adf6ecac5dc7876d725a77a5b"


def read_monthly_debits(path):
    months = defaultdict(lambda: defaultdict(Decimal))
    initial_months = {}
    debit_rows = 0
    with path.open(encoding="latin-1", newline="") as file:
        for row in csv.DictReader(file, delimiter=";"):
            account = row["account_id"]
            year, month = int(row["date"][:2]), int(row["date"][2:4])
            index = year * 12 + month - 1
            initial_months[account] = min(index, initial_months.get(account, index))
            if row["type"] in {"VYDAJ", "VYBER"}:
                months[account][index] += Decimal(row["amount"])
                debit_rows += 1
    for account, initial in initial_months.items():
        if account in months:
            months[account].pop(initial, None)
    return months, debit_rows


def contiguous_runs(months):
    run = []
    last = None
    for month, value in sorted(months.items()):
        if last is not None and month != last + 1:
            yield run
            run = []
        run.append(value)
        last = month
    if run:
        yield run


def event_at(values, target):
    # The first high month is the latest month known at the forecast cutoff.
    # The observed outcomes label cohorts retrospectively and never train a forecast.
    onset = target - HORIZON
    previous = values[:onset]
    window = previous[-9:]
    center = median(window)
    spread = median(abs(value - center) for value in window)
    threshold = max(4 * spread, abs(center) / 4)
    if (
        threshold <= 0
        or abs(previous[-1] - center) > threshold / 2
        or values[onset] <= center + threshold
    ):
        return None

    if all(
        abs(value - center) <= threshold / 2 for value in values[onset + 1 : target + 1]
    ):
        return "isolated spike (returned to normal)"
    if not all(value > center + threshold for value in values[onset + 1 : target + 1]):
        return None

    preceding = previous[-36:]
    reversed_spikes = sum(
        abs(before - center) <= threshold / 2
        and spike > center + threshold
        and abs(after - center) <= threshold / 2
        for before, spike, after in zip(
            preceding, preceding[1:], preceding[2:], strict=False
        )
    )
    if reversed_spikes == 1:
        return "sustained increase after one prior spike"
    return "other sustained increases"


def evaluate(monthly, previous_selector):
    groups = defaultdict(dict)
    for account, months in monthly.items():
        eligible = []
        for run in contiguous_runs(months):
            for index in range(MIN_HISTORY + HORIZON - 1, len(run)):
                eligible.append((run, index, event_at(run, index)))

        recent_start = max(0, len(eligible) - 12)
        for position, (run, target, event) in enumerate(eligible):
            if event is None and position < recent_start:
                continue
            known = run[: target - HORIZON + 1]
            actual = run[target]
            previous = previous_selector(known)[1]
            current = select_method(known)[1]
            errors = (
                abs(previous - actual),
                abs(current - actual),
                abs(known[-1] - actual),
            )
            labels = []
            if position >= recent_start:
                labels.append("latest 12 eligible targets per account")
            if event is not None:
                labels.append(event)
            for label in labels:
                summary = groups[label].setdefault(
                    account, [0, Decimal(0), Decimal(0), Decimal(0), 0]
                )
                summary[0] += 1
                for column, error in enumerate(errors, 1):
                    summary[column] += error
                summary[4] += previous != current
    return groups


def paired_interval(accounts, comparator, seed=2026):
    if len(accounts) < 20:
        return None
    rng = Random(seed)
    pairs = [(int(row[0]), float(row[2] - row[comparator])) for row in accounts]
    estimates = []
    for _ in range(1000):
        draw = rng.choices(pairs, k=len(pairs))
        estimates.append(
            sum(diff for _, diff in draw) / sum(count for count, _ in draw)
        )
    estimates.sort()
    return estimates[24], estimates[974]


def load_reference():
    # Fixed, repository-owned reference, independent of the checked-out implementation.
    root = Path(__file__).resolve().parent
    code = subprocess.check_output(
        ["git", "show", f"{REFERENCE_REVISION}:forecast.py"], cwd=root
    )
    module = ModuleType("forecast_benchmark_reference")
    sys.modules[module.__name__] = module
    # The source is the fixed Git object above, never the input data or a CLI value.
    exec(compile(code, "reference_forecast.py", "exec"), module.__dict__)  # noqa: S102
    return module.select_method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "transactions", type=Path, help="Path to the PKDD'99 trans.asc file"
    )
    args = parser.parse_args()
    months, rows = read_monthly_debits(args.transactions)
    print(f"{rows:,} debits; {len(months):,} accounts with debit records", flush=True)
    for label, entries in sorted(evaluate(months, load_reference()).items()):
        accounts = list(entries.values())
        targets = sum(row[0] for row in accounts)
        errors = [sum(row[i] for row in accounts) / targets for i in (1, 2, 3)]
        print(
            f"{label}: accounts={len(accounts):,}, targets={targets:,}, changed={sum(row[4] for row in accounts):,}"
        )
        print(
            f"  previous={errors[0]:.2f}, current={errors[1]:.2f}, baseline={errors[2]:.2f} CZK"
        )
        for comparator, name in ((1, "previous"), (3, "baseline")):
            interval = paired_interval(accounts, comparator)
            if interval is not None:
                print(
                    f"  current minus {name}, paired 95% interval: [{interval[0]:.2f}, {interval[1]:.2f}] CZK"
                )


if __name__ == "__main__":
    main()
