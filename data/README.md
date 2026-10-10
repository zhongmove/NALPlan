# Public Data

This release contains CSV/WKT exports of the public portions of the two datasets used by NALPlan.

- `berlintest`: BerlinMOD-derived public transport, geometry, and train trajectory data. Moving-object tables include `Trains.csv` and `Trains__Trip_units.csv`.
- `nanjingtest`: public Nanjing geospatial layers only. Taxi and other moving-object relations are intentionally excluded.

Spatial values use OGC WKT in CSV columns. Moving-point temporal units are represented as ordinary tabular records with timestamps and coordinates. The expanded GeoJSON, segment, vertex, and sampled-point encodings are omitted to keep this code release compact.

The public data are intended for inspection, preprocessing, and reproducible examples. Executing the SECONDO plans requires a separately installed SECONDO instance and the corresponding database setup described in the root README.
