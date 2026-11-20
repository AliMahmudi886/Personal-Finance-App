import unittest
from datetime import date
from decimal import Decimal

from forecast import (
    MIN_HISTORY,
    build_forecast,
    closed_month_history,
    month_index,
    month_label,
    select_method,
)


def history_for(values):
    first = month_index(date(2023, 1, 1))
    return [
        (month_label(first + index), Decimal(value))
        for index, value in enumerate(values)
    ]


class ForecastTest(unittest.TestCase):
    def test_excludes_open_and_future_entries_and_stops_at_a_gap(self):
        history = closed_month_history(
            [
                ("2026-07-01", "110"),
                ("2026-09-02", "31.15"),
                ("2026-10-01", "123.40"),
                ("2026-10-15", "-20.35"),
                ("2026-11-01", "999999"),
                ("2026-12-01", "999999"),
            ],
            date(2026, 11, 30),
        )
        self.assertEqual(
            history, [("2026-09", Decimal("31.15")), ("2026-10", Decimal("103.05"))]
        )
        self.assertIsNone(build_forecast(history, date(2026, 11, 30)))

    def test_stale_history_does_not_produce_a_forecast(self):
        self.assertEqual(
            closed_month_history([("2026-08-01", "40")], date(2026, 11, 1)), []
        )

    def test_calendar_target_horizon_and_seasonal_accuracy(self):
        pattern = [180, 320, 110, 400, 250, 725, 200, 295, 990, 195, 380, 1100]
        today = date(2026, 1, 1)
        report = build_forecast(history_for(pattern * 3), today)
        self.assertEqual(report.target, "2026-02")
        self.assertEqual(report.method, "Same month last year")
        self.assertEqual(report.amount, Decimal(320))
        self.assertLess(report.mean_error, report.baseline_error)
        self.assertEqual(len(report.backtest), 12)

    def test_held_out_month_and_gap_month_cannot_change_its_own_estimate(self):
        today = date(2026, 1, 1)
        original = list(range(100, 460, 10))
        before = build_forecast(history_for(original), today)
        changed = original[:-2] + [86000, 95000]
        after = build_forecast(history_for(changed), today)
        self.assertEqual(before.backtest[-1][2], after.backtest[-1][2])
        self.assertNotEqual(before.backtest[-1][1], after.backtest[-1][1])

    def test_regime_change_requires_two_consistent_closed_months(self):
        stable = [Decimal(100)] * 20
        _, spike_estimate = select_method(stable + [Decimal(175)])
        self.assertEqual(spike_estimate, 100)
        for amount in (Decimal(175), Decimal(25)):
            for months in (2, 6, 12):
                with self.subTest(amount=amount, months=months):
                    method, shift_estimate = select_method(stable + [amount] * months)
                    self.assertEqual(method, "Recent level shift")
                    self.assertEqual(shift_estimate, amount)

    def test_single_moderate_change_only_partially_updates_forecast(self):
        history = [Decimal(100)] * 20
        for observation, expected in ((140, 116), (60, 84)):
            with self.subTest(observation=observation):
                method, estimate = select_method([*history, Decimal(observation)])
                self.assertEqual(method, "Possible level shift")
                self.assertEqual(estimate, expected)

    def test_close_methods_are_averaged_only_when_the_blend_reduces_past_error(self):
        from forecast_benchmark import simulated_spending

        values = simulated_spending("stable", seed=5)
        method, estimate = select_method(values)
        self.assertEqual(method, "Average of comparable methods")
        self.assertEqual(estimate.quantize(Decimal("0.01")), Decimal("1176.33"))

        method, estimate = select_method(simulated_spending("stable", seed=1))
        self.assertEqual(method, "Long recent average")
        self.assertEqual(estimate.quantize(Decimal("0.01")), Decimal("1205.42"))

    def test_signed_bias_reports_mean_over_or_underestimation(self):
        today = date(2025, 1, 1)
        report = build_forecast(history_for([100] * 23 + [200]), today)
        self.assertLess(report.bias, 0)

    def test_zero_history_has_no_percentage_error_or_small_sample_range(self):
        today = date(2024, 3, 1)
        report = build_forecast(history_for([0] * MIN_HISTORY), today)
        self.assertEqual(report.amount, 0)
        self.assertEqual(report.mean_error, 0)
        self.assertEqual(len(report.backtest), 3)
        self.assertIsNone(report.wape)
        self.assertIsNone(report.typical_error)


if __name__ == "__main__":
    unittest.main()
