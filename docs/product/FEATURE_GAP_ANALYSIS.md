# Feature gap analysis

| Capability | Status | Priority and evidence |
| --- | --- | --- |
| Income, expenses and goal creation | Partially implemented | High. Local form workflows work. Editing, deletion, category governance and transaction history are missing. |
| Expense search and pagination | Fully implemented for text search | High. Indexed owner and date order, case-insensitive substring matching for categories and descriptions, 40 records per page. Structured filters, sorting choices and exports remain open. |
| Financial overview | Partially implemented | High. Date filters, income, expenses, net cash flow, monthly trend and recent activity read the signed-in account's ledger. The app has no assets or liabilities model. |
| Currency and precision | Technically inadequate | Critical. The dashboard aggregates decimal representations of values read from the existing SQLite REAL columns. Exact persisted amounts and multi-currency accounting need a reviewed ledger model. |
| Accounts, transfers and net worth | Missing | High. Define reconciliation and transfer semantics first. |
| Budgeting and rollover | Missing | High. Depends on category rules, reporting periods and settlement semantics. |
| Goal management | Partially implemented | Medium. Target and current amounts are recorded. Due dates, contribution history and progress displays remain open. |
| Recurring transaction management | Missing from the interface | High. A table exists. Account-scoped CRUD, posting and duplicate protection are needed. |
| Forecast and anomaly detection | Partially implemented | Medium. Forecast, anomaly and cluster routes work in tests. Retraining cost and validation of forecast quality remain open. |
| Import, export and bulk actions | Missing | Medium. Requires validation, access controls and duplicate handling. |
| Authentication and profile | Partially implemented | High. Local flow works. Google remains unavailable pending account linking and production controls. |
| Accessibility and personalization | Partially implemented | Medium. Shared focus styles, skip link and responsive layouts are present. Per-user preferences and an assistive-technology review remain open. |
