import unittest
from decimal import Decimal

from real_forecast_benchmark import (
    account_partition,
    contiguous_runs,
    evaluate,
    event_at,
    tempered_increase,
)


class RealForecastBenchmarkTest(unittest.TestCase):
    def test_account_split_is_stable(self):
        self.assertEqual(account_partition("1"), "test")
        self.assertEqual(account_partition("2"), "development")

    def test_experiment_combines_local_level_and_latest_increase(self):
        from forecast_benchmark import simulated_spending

        values = simulated_spending("shift after spike", seed=3301, vary_shifts=True)[
            :43
        ]
        method, estimate = tempered_increase(values)
        self.assertEqual(method, "Tempered increase after an earlier spike")
        self.assertEqual(estimate, Decimal("1409.80"))

    def test_event_labels_use_one_completed_earlier_spike_and_future_outcomes(self):
        known = [Decimal(100)] * 14
        known[3] = Decimal(160)
        known.append(Decimal(160))
        self.assertEqual(
            event_at(known + [Decimal(160)] * 2, 16),
            "sustained increase after one prior spike",
        )
        self.assertEqual(
            event_at(known + [Decimal(100)] * 2, 16),
            "isolated spike (returned to normal)",
        )

    def test_future_outcomes_do_not_reach_reference_selector(self):
        amounts = [Decimal(100)] * 16
        amounts[3] = Decimal(160)
        amounts += [Decimal(160)] * 3
        seen = []

        def reference(values):
            seen.append(values)
            return "baseline", values[-1]

        result = evaluate({"fixture": dict(enumerate(amounts))}, reference)
        self.assertEqual(seen[-1], amounts[:-2])
        self.assertIn("sustained increase after one prior spike", result)

    def test_missing_debit_month_breaks_history(self):
        self.assertEqual(
            list(contiguous_runs({1: Decimal(0), 2: Decimal(12), 4: Decimal(16)})),
            [[Decimal(0), Decimal(12)], [Decimal(16)]],
        )


if __name__ == "__main__":
    unittest.main()
