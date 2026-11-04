# Kaggle data matrix 2026-11-01 (MSK)

Goal: kernels run without Dmitrij's PC. No S3 creds on CPU accounts.

| Dataset | dmitrijkv (GPU) | elenakonkova (GPU) | usertest1234 | aaaadg4gg4g4g | aaaa0000004 | Notes |
|---|---|---|---|---|---|---|
| koz3-s3-cred | YES (private) | YES (private) | — | — | — | Do NOT copy to CPU |
| koz3-gt-train-v2 | YES | YES (created 01.11) | YES (created 01.11) | YES (created 01.11) | YES (created 01.11) | ~38MB |
| koz3-fr1h-chk-20261101 (v2 eb268d6 / fr1h.zip.bin) | YES | YES (created 01.11) | READER on dmitrijkv | READER | READER | ~501MB; CPU via share |
| koz3-fr1h-wheels | YES | — | — | — | — | small; optional |
| koz3-w3-geo5-feat-base | YES | NO (create 499 after file upload) | NO (same 499) | NO (attempt aborted) | YES | ~548MB many files; retry as single zip later |
| koz3-w3-geo5-feat-n40r8 | — | YES | YES | YES | YES | ~2.3GB; >2GB GPU cross-copy skipped |

## Other notes
- dmitrijkv **cannot** read `elenakonkova/koz3-w2-gate` (no access).
- Real bot7 geo-gate + sat-recall running on elenakonkova since **20:15 MSK**.
- li-post-full COMPLETE on elenakonkova.
- large-gate push rejected HTTP 400 (script ~3.6MB).
- Skipped bulk GPU↔GPU clone of other <2GB research datasets (xfer shards, maps, etc.) to save quota/disk; critical set above.

## Created this session
- elenakonkova/koz3-gt-train-v2
- usertest1234/koz3-gt-train-v2
- aaaadg4gg4g4g/koz3-gt-train-v2
- aaaa0000004/koz3-gt-train-v2
- elenakonkova/koz3-fr1h-chk-20261101
