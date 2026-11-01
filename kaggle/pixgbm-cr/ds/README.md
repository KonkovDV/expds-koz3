# Upload-ready dataset folders (live under `/workspace/kaggle/push/pixgbm-cr/ds/`)

Payloads are **not** mirrored into git:

- `geo_pixgbm.py` — runtime black-box only (gitignored / never commit)
- `models/pixgbm/all/*.txt`, `models_pix_clean/*` — weights
- `*.whl`, `*.zip` — wheels / submission archives

On the box, create/upload from:

```
/workspace/kaggle/push/pixgbm-cr/ds/pixgbm-cr-ref/
/workspace/kaggle/push/pixgbm-cr/ds/pixgbm-cr-cand/
/workspace/kaggle/push/pixgbm-cr/ds/pixgbm-cr-wheels/
```

See parent `README.md` § Push steps for «1».
