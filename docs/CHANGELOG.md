# Changelog

## Initial maintenance cycle

- Restored expense submission with descriptions and validated numeric and date input before new financial writes.
- Added fresh-schema initialization and configurable SQLite location.
- Required an environment-provided session key, added CSRF protection and made logout a protected POST.
- Made new profile images private and removed database files and generated caches from the maintenance branch tree.
- Fixed empty and short analytics paths, bounded clustering parameters and moved TensorFlow off the homepage request path.
- Added route and SQLite regression coverage, runtime pins and a repeatable quality workflow.

## Follow-up maintenance

- Show the latest recorded expense as a spending reference on the existing route.
- Index per-account expense, income and goal reads. Existing records and amount fields remain intact.
