from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from statistics import mean, median

from models import create_connection

MIN_HISTORY = 14
HORIZON = 2
MAX_TEST_MONTHS = 12
MIN_SELECTION_MONTHS = 3


@dataclass(frozen=True)
class Forecast:
    target: str
    history: list[tuple[str, Decimal]]
    method: str
    amount: Decimal
    backtest: list[tuple[str, Decimal, Decimal, Decimal]]
    mean_error: Decimal
    baseline_error: Decimal
    bias: Decimal
    wape: Decimal | None
    typical_error: Decimal | None


def month_index(day):
    return day.year * 12 + day.month - 1


def month_label(index):
    year, month = divmod(index, 12)
    return f"{year:04d}-{month + 1:02d}"


def closed_month_history(rows, today):
    """Use the trailing contiguous run of closed months with recorded activity."""
    current = month_index(today)
    totals = defaultdict(Decimal)
    for entered_date, raw_amount in rows:
        month = month_index(date.fromisoformat(entered_date))
        if month >= current:
            continue
        amount = Decimal(str(raw_amount))
        if not amount.is_finite():
            raise ValueError("Invalid recorded amount")
        totals[month] += amount

    month = current - 1
    history = []
    while month in totals:
        history.append((month_label(month), totals[month]))
        month -= 1
    return history[::-1]


def forecast_candidates(values):
    recent = values[-6:]
    month_changes = [right - left for left, right in pairwise(recent)]
    drift = median(month_changes)
    candidates = {
        # Prefer the more stable median when historical errors are identical.
        "Six-month median": median(recent),
        "Last closed month": values[-1],
        "Three-month average": mean(values[-3:]),
        "Long recent average": mean(values[-12:]),
        "Long recent median": median(values[-12:]),
        "Weighted six-month average": sum(
            weight * value for weight, value in enumerate(recent, start=1)
        )
        / 21,
        "Linear recent trend": median(values[-3:]) + 3 * drift,
        "Damped recent trend": median(values[-3:]) + 2 * drift,
    }
    if len(values) >= 12:
        window = values[-12:]
        average = mean(window)
        slope = sum((2 * i - 11) * value for i, value in enumerate(window)) / 286
        candidates["Twelve-month regression"] = average + Decimal("7.5") * slope
    for alpha in (Decimal("0.2"), Decimal("0.4")):
        level = mean(values[:3])
        for value in values[3:]:
            level += alpha * (value - level)
        candidates[f"Exponential smoothing ({alpha})"] = level
    if len(values) >= 12:
        candidates["Same month last year"] = values[HORIZON - 13]
    return candidates


def has_repeated_reversed_spikes(previous, typical, threshold, direction):
    # Only count completed excursions whose neighbors returned to the old level.
    normal_band = threshold / 2
    reversals = 0
    for before, unusual, after in zip(
        previous, previous[1:], previous[2:], strict=False
    ):
        if (
            abs(before - typical) <= normal_band
            and direction * (unusual - typical) > threshold
            and abs(after - typical) <= normal_band
        ):
            reversals += 1
            if reversals == 2:
                return True
    return False


def estimate_level_shift(values):
    for length in range(12, 0, -1):
        if len(values) < length + 9:
            continue
        previous = values[-length - 9 : -length]
        typical = median(previous)
        spread = median(abs(value - typical) for value in previous)
        conservative_threshold = max(4 * spread, abs(typical) / 5)
        recent = values[-length:]
        direction = 1 if recent[-1] > typical else -1
        repeated = has_repeated_reversed_spikes(
            values[-length - 36 : -length], typical, conservative_threshold, direction
        )
        if repeated and length < 3:
            continue
        threshold = (
            conservative_threshold
            if repeated
            else max(3 * spread, abs(typical) * Decimal("0.15"))
        )
        if any(direction * (value - typical) <= threshold for value in recent):
            continue
        if length == 1:
            return "Possible level shift", recent[-1]
        return "Recent level shift", median(recent)
    return None


def select_method(values):
    # Each candidate is scored over the same dates, using only closed months
    # that would have been known at that prediction's cutoff.
    candidates = forecast_candidates(values)
    first = 12 if len(values) >= 12 + HORIZON + MIN_SELECTION_MONTHS - 1 else 6
    if first == 6:
        candidates.pop("Same month last year", None)
        candidates.pop("Twelve-month regression", None)
    last = len(values) - HORIZON
    origins = range(max(first, last - MAX_TEST_MONTHS + 1), last + 1)
    errors = {name: [] for name in candidates}
    for cutoff in origins:
        estimates = forecast_candidates(values[:cutoff])
        actual = values[cutoff + HORIZON - 1]
        for name, observations in errors.items():
            observations.append(estimates[name] - actual)

    scores = {
        name: mean(abs(error) for error in observations)
        for name, observations in errors.items()
    }
    method = min(scores, key=scores.__getitem__)
    # Protect a strong seasonal pattern, while letting a weak seasonal win yield to a shift.
    seasonal = "Same month last year"
    if method != seasonal or scores[method] >= Decimal("0.85") * min(
        score for name, score in scores.items() if name != seasonal
    ):
        shift = estimate_level_shift(values)
        if shift is not None:
            return shift
    if method == "Same month last year" or scores[method] == 0:
        return method, candidates[method]

    # Average only near-ties. A distinctive seasonal winner keeps its own estimate.
    ceiling = scores[method] * Decimal("1.04")
    close_methods = sorted(
        (
            name
            for name in scores
            if name != "Same month last year" and scores[name] <= ceiling
        ),
        key=scores.__getitem__,
    )[:3]
    if len(close_methods) > 1:
        blended_error = mean(
            abs(mean(residuals))
            for residuals in zip(*(errors[name] for name in close_methods), strict=True)
        )
        if blended_error >= scores[method]:
            return method, candidates[method]
        return "Average of comparable methods", mean(
            candidates[name] for name in close_methods
        )
    return method, candidates[method]


def build_forecast(history, today):
    if len(history) < MIN_HISTORY:
        return None
    values = [value for _, value in history]
    last = len(values) - HORIZON
    backtest = []
    # The selection policy is replayed at each cutoff, so the held-out target
    # and its immediately preceding month cannot influence its estimate.
    for cutoff in range(max(10, last - MAX_TEST_MONTHS + 1), last + 1):
        _, estimate = select_method(values[:cutoff])
        month, actual = history[cutoff + HORIZON - 1]
        backtest.append((month, actual, estimate, values[cutoff - 1]))
    errors = [abs(estimate - actual) for _, actual, estimate, _ in backtest]
    baseline_errors = [abs(baseline - actual) for _, actual, _, baseline in backtest]
    actual_volume = sum(abs(actual) for _, actual, _, _ in backtest)
    method, amount = select_method(values)
    # A small sample is not enough to display an empirical spread.
    typical_error = (
        sorted(errors)[(4 * len(errors) + 4) // 5 - 1] if len(errors) >= 8 else None
    )
    return Forecast(
        target=month_label(month_index(today) + 1),
        history=history,
        method=method,
        amount=amount,
        backtest=backtest,
        mean_error=mean(errors),
        baseline_error=mean(baseline_errors),
        bias=mean(estimate - actual for _, actual, estimate, _ in backtest),
        wape=sum(errors) / actual_volume * 100 if actual_volume else None,
        typical_error=typical_error,
    )


def forecast_for_user(user_id, today):
    with closing(create_connection()) as db:
        rows = db.execute(
            "SELECT date, amount FROM expenses WHERE user_id = ?", (user_id,)
        )
        try:
            history = closed_month_history(rows, today)
        except (InvalidOperation, ValueError) as error:
            raise ValueError("Invalid expense history") from error
    return build_forecast(history, today), len(history)
