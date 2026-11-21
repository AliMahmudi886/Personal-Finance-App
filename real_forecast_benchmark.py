"""Evaluate forecasts on public transaction histories without emitting account data."""

import argparse
import csv
import hashlib
import subprocess
import sys
from collections import defaultdict
from contextlib import closing
from datetime import datetime
from decimal import Decimal
from io import BytesIO, TextIOWrapper
from pathlib import Path
from random import Random
from statistics import median
from types import ModuleType
from zipfile import ZipFile

from forecast import HORIZON, MIN_HISTORY, estimate_level_shift, select_method

REFERENCE_REVISION = "25ede3568ed62f7adf6ecac5dc7876d725a77a5b"
EXTERNAL_REFERENCE_REVISION = "18a499e1bb08afe2022bbe0bada6d34acb271091"


def account_partition(account):
    fingerprint = hashlib.sha256(f"berka-forecast-v1:{account}".encode()).digest()
    return "development" if fingerprint[0] < 154 else "test"


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


def read_online_retail(path):
    from openpyxl import load_workbook

    source = path
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            names = [
                name for name in archive.namelist() if name.lower().endswith(".xlsx")
            ]
            if len(names) != 1:
                raise ValueError("Expected exactly one workbook in the UCI archive")
            source = BytesIO(archive.read(names[0]))

    monthly = defaultdict(lambda: defaultdict(Decimal))
    rows = 0
    with closing(load_workbook(source, read_only=True, data_only=True)) as workbook:
        for sheet in workbook.worksheets:
            data = sheet.values
            header = next(data)
            indexes = [
                header.index(name)
                for name in ("Customer ID", "InvoiceDate", "Quantity", "Price")
            ]
            for row in data:
                account, entered_date, quantity, price = (row[i] for i in indexes)
                if account is None or not isinstance(entered_date, datetime):
                    continue
                if (
                    not isinstance(quantity, (float, int))
                    or not isinstance(price, (float, int))
                    or price < 0
                ):
                    continue
                index = entered_date.year * 12 + entered_date.month - 1
                monthly[str(account)][index] += Decimal(str(quantity)) * Decimal(
                    str(price)
                )
                rows += 1
    # The release ends on December 9, 2011. A customer's first observed month may also be partial.
    incomplete = 2011 * 12 + 11
    for months in monthly.values():
        months.pop(min(months), None)
        months.pop(incomplete, None)
    return monthly, rows


def read_complete_journey(path):
    monthly = defaultdict(lambda: defaultdict(Decimal))
    rows = 0
    final_day = 0
    with ZipFile(path) as archive:
        names = [
            name
            for name in archive.namelist()
            if Path(name).name.lower() == "transaction_data.csv"
        ]
        if len(names) != 1:
            raise ValueError(
                "Expected one transaction_data.csv in the Complete Journey archive"
            )
        with TextIOWrapper(
            archive.open(names[0]), encoding="utf-8-sig", newline=""
        ) as file:
            reader = csv.reader(file)
            header = [name.lower() for name in next(reader)]
            indexes = [
                header.index(name) for name in ("household_key", "day", "sales_value")
            ]
            for row in reader:
                household, ordinal_day, sales = (row[i] for i in indexes)
                day = int(ordinal_day)
                if day < 1:
                    raise ValueError("Expected positive source day index")
                monthly[household][(day - 1) // 30] += Decimal(sales)
                final_day = max(final_day, day)
                rows += 1
    for months in monthly.values():
        # Ordinal dates have no calendar anchor. Drop partial 30-day periods.
        months.pop(min(months), None)
        months.pop(final_day // 30, None)
    return monthly, rows


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


def upward_signal(known):
    previous = known[:-1]
    window = previous[-9:]
    center = median(window)
    spread = median(abs(value - center) for value in window)
    threshold = max(4 * spread, abs(center) / 4)
    if (
        threshold <= 0
        or abs(previous[-1] - center) > threshold / 2
        or known[-1] <= center + threshold
    ):
        return None
    reversals = [
        index
        for index in range(max(1, len(previous) - 35), len(previous) - 1)
        if abs(previous[index - 1] - center) <= threshold / 2
        and previous[index] > center + threshold
        and abs(previous[index + 1] - center) <= threshold / 2
    ]
    return center, threshold, reversals


def event_at(values, target):
    # The first high month is the latest month known at the forecast cutoff.
    # The observed outcomes label cohorts retrospectively and never train a forecast.
    onset = target - HORIZON
    signal = upward_signal(values[: onset + 1])
    if signal is None:
        return None
    center, threshold, reversals = signal
    outcomes = values[onset + 1 : target + 1]
    if all(abs(value - center) <= threshold / 2 for value in outcomes):
        return "isolated spike (returned to normal)"
    if not all(value > center + threshold for value in outcomes):
        return None
    if len(reversals) == 1:
        return "sustained increase after one prior spike"
    return "other sustained increases"


def tempered_increase(values):
    previous_level = median(values[-10:-1])
    rise = values[-1] - previous_level
    if (
        rise > 0
        and estimate_level_shift(values, require_single_reversal=True) is not None
    ):
        return "Tempered increase after an earlier spike", previous_level + Decimal(
            "0.60"
        ) * rise
    return select_method(values)


def analog_followthrough(values):
    selected = select_method(values)
    signal = estimate_level_shift(values, require_single_reversal=True)
    if signal is None or signal[0] != "Possible level shift":
        return selected

    center = median(values[-10:-1])
    change = values[-1] - center
    if center <= 0 or change <= 0:
        return selected
    relative_change = change / center
    analogs = []
    for index in range(max(4, len(values) - 37), len(values) - HORIZON - 1):
        earlier_center = median(values[max(0, index - 9) : index])
        earlier_change = values[index] - earlier_center
        if earlier_center <= 0 or earlier_change <= 0:
            continue
        relative = earlier_change / earlier_center
        if not relative_change / 2 <= relative <= relative_change * 2:
            continue
        realized = (values[index + HORIZON] - earlier_center) / earlier_change
        analogs.append((abs(relative - relative_change), max(0, min(1, realized))))
    if not analogs:
        return selected
    analogs.sort(key=lambda pair: pair[0])
    persistence = median(value for _, value in analogs[:3])
    analog = center + change * persistence
    # Keep two thirds of the incumbent forecast when the reference history is sparse.
    return "Matched historical follow-through", (2 * selected[1] + analog) / 3


def reversal_followthrough(values):
    selected = select_method(values)
    signal = upward_signal(values)
    if signal is None:
        return selected
    center, _, reversals = signal
    if len(reversals) != 1:
        return selected

    index = reversals[0]
    earlier_center = median(values[max(0, index - 9) : index])
    earlier_rise = values[index] - earlier_center
    if earlier_rise <= 0 or index + HORIZON >= len(values) - 1:
        return selected
    persistence = max(
        0, min(1, (values[index + HORIZON] - earlier_center) / earlier_rise)
    )
    analog = center + (values[-1] - center) * persistence
    return "Earlier spike follow-through", (2 * selected[1] + analog) / 3


def evaluate(monthly, previous_selector, current_selector=select_method):
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
            current = current_selector(known)[1]
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


def load_reference(revision=REFERENCE_REVISION):
    # Fixed, repository-owned reference, independent of the checked-out implementation.
    root = Path(__file__).resolve().parent
    code = subprocess.check_output(["git", "show", f"{revision}:forecast.py"], cwd=root)
    module = ModuleType("forecast_benchmark_reference")
    sys.modules[module.__name__] = module
    # The CLI uses only the two fixed Git revisions above.
    exec(compile(code, "reference_forecast.py", "exec"), module.__dict__)  # noqa: S102
    return module.select_method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "transactions",
        type=Path,
        help="Local Berka, Online Retail II, or Complete Journey data file",
    )
    parser.add_argument(
        "--dataset",
        choices=("berka", "online-retail-ii", "complete-journey"),
        default="berka",
    )
    parser.add_argument("--account-split", choices=("development", "test"))
    parser.add_argument(
        "--experiment",
        choices=("tempered-increase", "analog-followthrough", "reversal-followthrough"),
        help="Evaluate an experimental rule outside the application.",
    )
    args = parser.parse_args()
    if args.account_split and args.dataset != "berka":
        parser.error("--account-split applies only to Berka")
    if args.dataset == "online-retail-ii":
        months, rows = read_online_retail(args.transactions)
        unit = "GBP"
        print(
            f"{rows:,} usable purchase lines; {len(months):,} customer histories",
            flush=True,
        )
    elif args.dataset == "complete-journey":
        months, rows = read_complete_journey(args.transactions)
        unit = "sales units"
        print(
            f"{rows:,} purchase lines; {len(months):,} household histories", flush=True
        )
    else:
        months, rows = read_monthly_debits(args.transactions)
        unit = "CZK"
        print(
            f"{rows:,} debits; {len(months):,} accounts with debit records", flush=True
        )
    revision = REFERENCE_REVISION
    if args.account_split:
        months = {
            account: values
            for account, values in months.items()
            if account_partition(account) == args.account_split
        }
        revision = EXTERNAL_REFERENCE_REVISION
        print(
            f"{args.account_split}: {len(months):,} accounts before eligibility filtering",
            flush=True,
        )
    selector = {
        None: select_method,
        "tempered-increase": tempered_increase,
        "analog-followthrough": analog_followthrough,
        "reversal-followthrough": reversal_followthrough,
    }[args.experiment]
    if args.experiment:
        print(f"current column evaluates experiment: {args.experiment}", flush=True)
    reference = select_method if args.dataset != "berka" else load_reference(revision)
    results = evaluate(months, reference, selector)
    for label, entries in sorted(results.items()):
        accounts = list(entries.values())
        targets = sum(row[0] for row in accounts)
        errors = [sum(row[i] for row in accounts) / targets for i in (1, 2, 3)]
        print(
            f"{label}: accounts={len(accounts):,}, targets={targets:,}, changed={sum(row[4] for row in accounts):,}"
        )
        print(
            f"  previous={errors[0]:.2f}, current={errors[1]:.2f}, baseline={errors[2]:.2f} {unit}"
        )
        for comparator, name in ((1, "previous"), (3, "baseline")):
            interval = paired_interval(accounts, comparator)
            if interval is not None:
                print(
                    f"  current minus {name}, paired 95% interval: [{interval[0]:.2f}, {interval[1]:.2f}] {unit}"
                )


if __name__ == "__main__":
    main()
