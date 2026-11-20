# Personal Finance App

A Flask and SQLite application for recording income, expenses and savings goals. It also provides experimental expense forecasts, anomaly views, expense clustering and a historical spending reference.

## Run locally on Windows

The verified Python minor version is 3.12. Create a virtual environment and install the version-pinned runtime dependencies.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force instance | Out-Null
$env:FINANCE_DB_PATH = Join-Path (Get-Location) 'instance/personal_finance.sqlite3'
$env:FINANCE_SECRET_KEY = & .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
$env:FINANCE_HTTPS = '0'
.\.venv\Scripts\python.exe -m flask --app app:app run
```

Open http://127.0.0.1:5000 and register a local account. Keep the generated secret private and stable across restarts when session continuity matters. Google sign-in is unavailable pending review of account linking.

To reuse a trusted existing ledger, back it up and set `FINANCE_DB_PATH` to its absolute path. Pay particular attention to this when switching from a checkout that tracked SQLite files. Analytics are experimental and use the existing SQLite storage format. See the [audit](docs/audit/MASTER_AUDIT_REPORT.md) for financial accuracy limitations.

- [Architecture](docs/ARCHITECTURE.md)
- [Testing](docs/TESTING.md)
- [Deployment and recovery](docs/DEPLOYMENT.md)
- [Security](docs/SECURITY.md)
- [Remediation tracker](docs/audit/REMEDIATION_TRACKER.md)
