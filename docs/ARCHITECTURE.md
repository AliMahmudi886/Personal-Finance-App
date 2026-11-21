# Architecture

The application entry point is `app:app`. Flask renders Jinja templates from `templates/` and serves CSS and illustrations from `static/`. Shared navigation, notifications and account controls live in `base.html`. The templates use Bootstrap and some analytics pages use Chart.js from external CDNs.

Flask-Bcrypt verifies local passwords. Flask's signed session cookie carries a local account ID. The request hook resolves that ID to an existing account. Forms use Flask-WTF CSRF tokens. New profile images are decoded, re-encoded and stored under the Flask instance directory. The image route resolves the current user's stored image.

`models.py` owns the active SQLite connection and schema. Users own expense, income, goal and recurring-expense rows through `user_id`. CRUD handlers currently support creation and listing. SQLite table creation and the two additive compatibility checks run on startup. `ml_model.py` uses the same connection configuration and parameterized account filtering. It provides numeric monthly aggregation, LSTM forecasting, isolation forest and autoencoder anomaly detection, and text-assisted clustering. The homepage renders independently of TensorFlow. The spending reference reads the account's most recent expense with an indexed lookup.

`database.py`, `analysis.py`, `gui.py`, `app/init.py`, `app/routes.py`, the empty `main.py` and `recommendations.py` are legacy or incomplete paths. The current WSGI entry point does not import them. Confirm external usage before retiring them.
