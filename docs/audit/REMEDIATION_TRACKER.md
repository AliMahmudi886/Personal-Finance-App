# Remediation tracker

| Priority | Work | Exit criteria | Status |
| --- | --- | --- | --- |
| P0 | PF-001 public repository data containment | Owner establishes affected data and copies, rotates affected credentials, documents incident disposition, and approves any visibility or history action | Fresh branch tree sanitized. Owner response pending. |
| P0 | PF-002/PF-003 financial precision and history | Agree supported currency, rounding, sign and bounds, reconcile a read-only backup, and pass reversible migration and exact-sum tests | Pending approval for any change affecting existing values or totals. |
| P1 | PF-004/PF-005/PF-006/PF-007 branch security rollout | Verify deployment configuration and old uploads, approve session rotation and Google account-linking policy, and exercise staging login and upload flows | Branch changes implemented. Production approval pending. |
| P1 | PF-009/PF-010 analytics reliability | Backtest against held-out months, define minimum useful history, and bound CPU work outside interactive requests | Crash and startup paths improved. The savings heuristic now displays a labeled spending reference. Forecast validation pending. |
| P1 | PF-011/PF-013 operational readiness | Restore a backup in staging, run deployment smoke, and demonstrate concurrent write isolation before rollout | Local WSGI and schema fixtures available. Staging pending. |
| P2 | PF-012 UI quality | Exercise mobile, tablet, desktop, keyboard, zoom and charts in a real browser | Template feedback improved. Full visual review pending. |
| P2 | PF-014/PF-015 maintenance | Confirm legacy callers and settle delivery scope for missing finance workflows | Open. |
