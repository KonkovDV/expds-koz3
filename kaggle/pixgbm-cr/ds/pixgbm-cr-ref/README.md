# koz3-pixgbm-cr-ref

**REF = v2 (`eb268d6`) ready OUTPUT only.**

Drop here (do **not** add EN `geo_pixgbm.py` or `models/`):

```
result.geojson          # preferred top-level name
# or
geo/result.geojson
out_ref/result.geojson
```

Must cover regions **200–217** (MagGr/EtGr/TerFl gate).

## Status on box (bot-6)

**No v2 result.geojson covering 200–217 was found** on this machine
(`bot2_v2` docker mock was `101_*` only). «1» / PC must produce the REF
output and place it in this dataset before the gate is runnable end-to-end.

EN code and tip weights were removed from this staging folder on purpose.
