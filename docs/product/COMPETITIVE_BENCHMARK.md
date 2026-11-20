# Product benchmark

The following patterns were reviewed in official product documentation. The opportunity column describes how a pattern could apply to this repository.

| Product | Observed pattern | Application opportunity |
| --- | --- | --- |
| Monarch Money | Period-based income, spending and net cash-flow reporting with category breakdowns | Connect summaries to transaction drill-down and reuse shared date filters. [Cash Flow](https://help.monarch.com/hc/en-us/articles/20504904768020-Cash-Flow) |
| YNAB | Spending categories with explicit funding and targets | Prioritize a documented allocation model before offering budgeting controls. [YNAB](https://www.ynab.com/) |
| Copilot Money | Dashboard sections for incoming transactions, budget trends and upcoming payments | Keep recent activity and high-frequency actions close to the summary. [Dashboard overview](https://help.copilot.money/en/articles/6045480-dashboard-tab-overview) |
| Revolut | Separate income and spending views with comparative periods | Make time controls obvious and add prior-period comparisons once partial-month rules are specified. [Spending and income analytics](https://help.revolut.com/help/accounts/budget-and-analytics/how-can-i-see-my-spending-and-income-analytics/) |
| Empower Personal Dashboard | Current and previous cash flow alongside account-derived net worth | Keep cash flow and net worth distinct. This app currently derives cash flow from recorded income and expenses. [Dashboard overview](https://support-personalwealth.empower.com/hc/en-us/articles/201169740-Dashboard-Overview) |
| Wallet by BudgetBakers | Scheduled payments and optional confirmation before creating records | Plan recurrence separately from posting, with an idempotent confirmation workflow. [Planned payments](https://support.budgetbakers.com/hc/en-us/articles/7149523920786-Setup-Planned-Payments) |
| PocketGuard | Bill matching to reduce budgeting double-counting | Define a single posting source and match scheduled items to existing transactions. [Bills](https://pocketguard.com/help/bills/) |
| Toshl Finance | Filtering and exports by account, category and tag | Start with searchable, bounded lists and reuse filters for exports when introduced. [Toshl FAQ](https://toshl.com/blog/frequently-asked-questions-faq/) |
