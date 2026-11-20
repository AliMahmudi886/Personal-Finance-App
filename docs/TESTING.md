# Testing

From an environment with `requirements.txt` installed, run the active application checks on Python 3.12.

```powershell
python -m pip install -r requirements-dev.txt
ruff check app.py ml_model.py models.py overview.py forecast.py forecast_benchmark.py real_forecast_benchmark.py app/init.py tests
ruff format --check app.py ml_model.py models.py overview.py forecast.py forecast_benchmark.py real_forecast_benchmark.py app/init.py tests
python -m compileall -q app.py ml_model.py models.py overview.py forecast.py forecast_benchmark.py real_forecast_benchmark.py app tests
python -m pip check
$env:FINANCE_TEST_ML = '1'
$env:TF_NUM_INTRAOP_THREADS = '1'
$env:TF_NUM_INTEROP_THREADS = '1'
python -m unittest discover -s tests -v
python -m pip_audit -r requirements.txt
```

The suite creates temporary SQLite databases and synthetic records. It checks form submission, user isolation, malformed input, schema compatibility, private uploads, logout, and analytics routes. With `FINANCE_TEST_ML=1` it trains both TensorFlow models on a small fixture. The application uses Flask's test client for route integration tests. An interactive browser is needed for viewport, keyboard, chart-rendering and assistive-technology checks.
