from collections import defaultdict
from contextlib import closing
from datetime import date
from decimal import Decimal

from models import create_connection


def financial_overview(user_id, start, end):
    """Summarize the selected ledger window without changing stored amounts."""
    totals = [Decimal(0), Decimal(0)]
    months = defaultdict(lambda: [Decimal(0), Decimal(0)])
    recent = []
    with closing(create_connection()) as db:
        rows = db.execute(
            """
            SELECT date, amount, source AS label, 0 AS kind, id FROM income
            WHERE user_id = ? AND date BETWEEN ? AND ?
            UNION ALL
            SELECT date, amount, category, 1, id FROM expenses
            WHERE user_id = ? AND date BETWEEN ? AND ?
            ORDER BY date DESC, id DESC, kind
            """,
            (user_id, start, end, user_id, start, end),
        )
        for entered_date, raw_amount, label, kind, _ in rows:
            month = date.fromisoformat(entered_date).isoformat()[:7]
            amount = Decimal(str(raw_amount))
            if not amount.is_finite():
                raise ValueError("Unusable ledger amount")
            totals[kind] += amount
            months[month][kind] += amount
            if len(recent) < 7:
                recent.append((entered_date, label, kind, amount))

    scale = max((abs(amount) for pair in months.values() for amount in pair), default=0)
    trend = [
        (
            month,
            [
                (amount, float(abs(amount) / scale * 100) if scale else 0)
                for amount in amounts
            ],
        )
        for month, amounts in sorted(months.items())
    ]
    return {
        "income": totals[0],
        "expenses": totals[1],
        "cash_flow": totals[0] - totals[1],
        "trend": trend,
        "recent": recent,
    }
