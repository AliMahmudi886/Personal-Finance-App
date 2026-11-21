import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from real_forecast_benchmark import (
    account_partition,
    analog_followthrough,
    contiguous_runs,
    evaluate,
    event_at,
    leave_one_account_out,
    read_complete_journey,
    reversal_followthrough,
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

    def test_analog_experiment_uses_completed_similar_spike(self):
        values = [Decimal(value) for value in [100] * 10 + [150] + [100] * 10 + [150]]
        method, estimate = analog_followthrough(values)
        self.assertEqual(method, "Matched historical follow-through")
        self.assertEqual(estimate.quantize(Decimal("0.01")), Decimal("133.33"))

    def test_analog_experiment_falls_back_with_no_earlier_reversal(self):
        values = [Decimal(100)] * 20 + [Decimal(150)]
        self.assertEqual(
            analog_followthrough(values), ("Possible level shift", Decimal(150))
        )

    def test_reversal_experiment_works_on_cutoff_prefix(self):
        known = [Decimal(100)] * 14
        known[3] = Decimal(160)
        known.append(Decimal(160))
        method, estimate = reversal_followthrough(known)
        self.assertEqual(method, "Earlier spike follow-through")
        self.assertLess(estimate, Decimal(160))

    def test_household_day_buckets_exclude_first_and_partial_final_periods(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "transactions.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr(
                    "transaction_data.csv",
                    "household_key,DAY,SALES_VALUE\nfixture,1,5\nfixture,31,10.20\n"
                    "fixture,60,1.25\nfixture,90,3.00\nfixture,91,100\n",
                )
            monthly, rows = read_complete_journey(path)
            shifted, _ = read_complete_journey(path, offset=15)
        self.assertEqual(rows, 5)
        self.assertEqual(monthly["fixture"], {1: Decimal("11.45"), 2: Decimal(3)})
        self.assertEqual(shifted["fixture"], {1: Decimal("1.25")})

    def test_leave_one_out_keeps_remaining_target_weights(self):
        accounts = [
            (1, Decimal(3), Decimal(5)),
            (2, Decimal(8), Decimal(4)),
            (1, Decimal(2), Decimal(2)),
        ]
        self.assertEqual(
            leave_one_account_out(accounts, 1),
            (Decimal(-4) / 3, Decimal(1)),
        )
        self.assertIsNone(leave_one_account_out(accounts[:1], 1))

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
