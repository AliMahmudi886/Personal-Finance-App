# Security

Use a stable, randomly generated `FINANCE_SECRET_KEY`. Set `FINANCE_HTTPS=0` for loopback HTTP development only. Keep database files, backups, exports, images and runtime configuration outside Git and outside publicly served directories.

Earlier public commits contain a SQLite database with user and financial records. The maintenance branch removes database files from its current tree. Historical commits and copies remain accessible. Coordinate the containment response with the owner, including repository visibility, ownership of the records, credential rotation, and any notification or history cleanup required. Treat the old session key as compromised if it was used on a deployed instance. Avoid putting record values or credentials into issue reports.

Local account sessions and form protection are tested using synthetic fixtures. Google sign-in requires a reviewed mapping from provider identity to a local account before it can be enabled. Remaining work includes abuse throttling, production transport and proxy verification, and inventory of legacy uploads. Keep dependency scans active and review alerts before deploying.
