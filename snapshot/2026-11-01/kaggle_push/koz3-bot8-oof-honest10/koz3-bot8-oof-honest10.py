#!/usr/bin/env python3
"""KOZ3 · bot-8 / GPU-5: OOF-инференс моделей сдачи на трейне для калибровки порогов (honest-10).

Slug: dmitrijkv/koz3-bot8-oof-honest10
Ресурс: NvidiaTeslaT4, internet, dataset dmitrijkv/koz3-s3-cred.
Возобновляемость: /kaggle/working/oof_honest10/jobs/<id>/done.json + oof/*.geojson;
STATUS.txt RUNNING|DONE|FAILED; SHA256SUMS; пропуск завершённых при рестарте.

Режим OOF_MODE=weights (default): scripts/nn_oof.py --weights <pt> на train root.
OOF_MODE=fold: --predict-only --resume при наличии fold-весов в work/cache (иначе FAIL job).
OOF_SMOKE=1: CPU synthetic без S3/GPU (тот же контракт STATUS/sha256).

Код репо + models/*.pt: attach snapshot (см. README) или S3 (OOF_CODE_S3_KEY / models prefix).
Ключи S3 не печатаются и не пишутся в /kaggle/working.
"""
from __future__ import annotations


# vendored sibling (Kaggle script ships only code_file)
import base64
from pathlib import Path
_p = Path(__file__).resolve().parent / "oof_kaggle_jobs.py"
if not _p.is_file():
    _p.write_bytes(base64.b64decode(b'IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiLQn9C70LDQvSDQt9Cw0LTQsNC90LjQuSBPT0Yt0LjQvdGE0LXRgNC10L3RgdCwINC00LvRjyBLYWdnbGUt0Y/QtNGA0LAgYm90LTggKEdQVS01KS4KCtCf0L7RgtGA0LXQsdC40YLQtdC70Lg6IHNjcmlwdHMvc2NvcmVib2FyZC5weSAoLS1ob25lc3QgMTApLCBkZW5zaXR5X3RocmVzaG9sZHMucHksCm9vZl90cmFuc2Zvcm0gY2FsaWIsIGJvdDQvYm90NyDQutCw0LvQuNCx0YDQvtCy0LrQsCDQv9C+0YDQvtCz0L7QsiDQv9C+INGB0LvQvtGOLgoK0KDQtdC20LjQvCDQv9C+INGD0LzQvtC70YfQsNC90LjRjiDigJQgYGB3ZWlnaHRzYGA6INGB0LTQsNC90L3Ri9C1INCy0LXRgdCwINC40LcgY29uZmlncy9kZWZhdWx0LnRvbWwgW21vZGVsc10K0L3QsCDQstGB0LXRhSDRg9GH0LDRgdGC0LrQsNGFINGC0YDQtdC50L3QsCAo0L3QtSBmb2xkLU9PRikuINCn0LXRgdGC0L3QvtGB0YLRjCBob25lc3QtMTAg0YLQvtCz0LTQsCDQsiDQv9C+0LTQsdC+0YDQtQrQv9C+0YDQvtCz0LAg0L3QsCDRh9GD0LbQuNGFINGB0LXQvNGM0Y/RhSwg0L3QtSDQsiDQstC10YHQsNGFLiBGb2xkLU9PRiAoYGBubl9vb2YgLS1wcmVkaWN0LW9ubHkvLS1yZXN1bWVgYCkK0LLQutC70Y7Rh9Cw0LXRgtGB0Y8gZW52IE9PRl9NT0RFPWZvbGQsINC60L7Qs9C00LAg0L3QsCBpbnB1dC9TMyDQtdGB0YLRjCB3b3JrL2ZvbGQqINGBINCy0LXRgdCw0LzQuC4KIiIiCmZyb20gX19mdXR1cmVfXyBpbXBvcnQgYW5ub3RhdGlvbnMKCmltcG9ydCBoYXNobGliCmltcG9ydCBqc29uCmltcG9ydCB0aW1lCmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aApmcm9tIHR5cGluZyBpbXBvcnQgSXRlcmFibGUKCiMg0JjQvNC10L3QsCDQstGL0YXQvtC00L7QsiDigJQg0LrQsNC6INC20LTRkdGCIHNjb3JlYm9hcmQgLyBub3RlcyAo0YHQvC4gc2NyaXB0cy9zY29yZWJvYXJkLnB5KS4KSk9CUzogbGlzdFtkaWN0XSA9IFsKICAgICMgb3V0X25hbWUsIHdlaWdodHMgKHJlbCB0byBtb2RlbHMvKSwgbWFya3VwIGZvciBubl9vb2YsIG9wdGlvbmFsIHByZWRpY3RfbWFya3VwcwogICAgeyJpZCI6ICJzcG9yX2dvciIsICJvdXQiOiAic3Bvcl9nb3Jfb29mLmdlb2pzb24iLCAid2VpZ2h0cyI6IFsic3Bvcl9nb3JfcjE4LnB0Il0sICJtYXJrdXAiOiAic3Bfb3IifSwKICAgIHsiaWQiOiAic3Bvcl9mb2NhbCIsICJvdXQiOiAic3Bvcl9mb2NhbF9vb2YuZ2VvanNvbiIsICJ3ZWlnaHRzIjogWyJzcG9yX2ZvY2FsX3IxOC5wdCJdLCAibWFya3VwIjogInNwX29yIn0sCiAgICB7ImlkIjogInNwb3JfYXJjaCIsICJvdXQiOiAic3Bvcl9hcmNoX29vZi5nZW9qc29uIiwgIndlaWdodHMiOiBbInN1cmZfYXJjaF9yMTgucHQiXSwgIm1hcmt1cCI6ICJzcF9vciJ9LAogICAgeyJpZCI6ICJzcG9yX2ZvY2FsX3dpZGUiLCAib3V0IjogInNwb3JfZm9jYWxfd2lkZV9vb2YuZ2VvanNvbiIsICJ3ZWlnaHRzIjogWyJzcG9yX2ZvY2FsX3dpZGVfcjE4LnB0Il0sICJtYXJrdXAiOiAic3Bfb3IifSwKICAgIHsiaWQiOiAibGlfZ29yX21hc2siLCAib3V0IjogImxpX2dvcl9tYXNrX29vZi5nZW9qc29uIiwgIndlaWdodHMiOiBbImxpX2dvcl9tYXNrX3IxOC5wdCJdLCAibWFya3VwIjogImxpIn0sCiAgICB7ImlkIjogImxpX2ZvcnRfbWFzayIsICJvdXQiOiAibGlfZm9ydF9tYXNrX29vZi5nZW9qc29uIiwgIndlaWdodHMiOiBbImxpX2ZvcnRfbWFza19yMTgucHQiXSwgIm1hcmt1cCI6ICJsaSJ9LAogICAgeyJpZCI6ICJsaV9kZXYiLCAib3V0IjogImxpX2Rldl9vb2YuZ2VvanNvbiIsICJ3ZWlnaHRzIjogWyJsaV9kZXZfcjE4LnB0Il0sICJtYXJrdXAiOiAibGkifSwKICAgIHsiaWQiOiAibGlfaGFyZDIiLCAib3V0IjogImxpX2hhcmQyX29vZi5nZW9qc29uIiwgIndlaWdodHMiOiBbImxpX3IxOF9oYXJkMi5wdCJdLCAibWFya3VwIjogImxpIn0sCiAgICB7ImlkIjogImFlX2ZvcnRfbWFzayIsICJvdXQiOiAiYWVfZm9ydF9tYXNrX29vZi5nZW9qc29uIiwgIndlaWdodHMiOiBbImFlX2ZvcnRfbWFza19yMTgucHQiXSwgIm1hcmt1cCI6ICJhZSJ9LAogICAgeyJpZCI6ICJvcl9hcmNoIiwgIm91dCI6ICJvcl9hcmNoX29vZi5nZW9qc29uIiwgIndlaWdodHMiOiBbInN1cmZfYXJjaF9yMTgucHQiXSwgIm1hcmt1cCI6ICJvciJ9LApdCgoKZGVmIGpvYl9kb25lX21hcmtlcih3b3JrOiBQYXRoLCBqb2JfaWQ6IHN0cikgLT4gUGF0aDoKICAgIHJldHVybiB3b3JrIC8gImpvYnMiIC8gam9iX2lkIC8gImRvbmUuanNvbiIKCgpkZWYgam9iX291dF9wYXRoKHdvcms6IFBhdGgsIG91dF9uYW1lOiBzdHIpIC0+IFBhdGg6CiAgICByZXR1cm4gd29yayAvICJvb2YiIC8gb3V0X25hbWUKCgpkZWYgaXNfam9iX2NvbXBsZXRlKHdvcms6IFBhdGgsIGpvYjogZGljdCkgLT4gYm9vbDoKICAgICIiItCY0LTQtdC80L/QvtGC0LXQvdGC0L3QvtGB0YLRjDogZG9uZS5qc29uICsg0LLRi9GF0L7QtNC90L7QuSBnZW9qc29uINGBINGC0LXQvCDQttC1IHNoYTI1Ni4iIiIKICAgIG1hcmtlciA9IGpvYl9kb25lX21hcmtlcih3b3JrLCBqb2JbImlkIl0pCiAgICBvdXQgPSBqb2Jfb3V0X3BhdGgod29yaywgam9iWyJvdXQiXSkKICAgIGlmIG5vdCBtYXJrZXIuaXNfZmlsZSgpIG9yIG5vdCBvdXQuaXNfZmlsZSgpOgogICAgICAgIHJldHVybiBGYWxzZQogICAgdHJ5OgogICAgICAgIG1ldGEgPSBqc29uLmxvYWRzKG1hcmtlci5yZWFkX3RleHQoZW5jb2Rpbmc9InV0Zi04IikpCiAgICBleGNlcHQganNvbi5KU09ORGVjb2RlRXJyb3I6CiAgICAgICAgcmV0dXJuIEZhbHNlCiAgICByZXR1cm4gYm9vbChtZXRhLmdldCgic2hhMjU2IikpIGFuZCBtZXRhWyJzaGEyNTYiXSA9PSBzaGEyNTZfZmlsZShvdXQpCgoKZGVmIHNoYTI1Nl9maWxlKHBhdGg6IFBhdGgpIC0+IHN0cjoKICAgIGggPSBoYXNobGliLnNoYTI1NigpCiAgICB3aXRoIHBhdGgub3BlbigicmIiKSBhcyBmaDoKICAgICAgICBmb3IgY2h1bmsgaW4gaXRlcihsYW1iZGE6IGZoLnJlYWQoMSA8PCAyMCksIGIiIik6CiAgICAgICAgICAgIGgudXBkYXRlKGNodW5rKQogICAgcmV0dXJuIGguaGV4ZGlnZXN0KCkKCgpkZWYgd3JpdGVfZG9uZSh3b3JrOiBQYXRoLCBqb2I6IGRpY3QsIHNoYTogc3RyLCBuX2ZlYXR1cmVzOiBpbnQsIHNlY29uZHM6IGZsb2F0KSAtPiBOb25lOgogICAgZCA9IGpvYl9kb25lX21hcmtlcih3b3JrLCBqb2JbImlkIl0pCiAgICBkLnBhcmVudC5ta2RpcihwYXJlbnRzPVRydWUsIGV4aXN0X29rPVRydWUpCiAgICBwYXlsb2FkID0gewogICAgICAgICJpZCI6IGpvYlsiaWQiXSwKICAgICAgICAib3V0Ijogam9iWyJvdXQiXSwKICAgICAgICAic2hhMjU2Ijogc2hhLAogICAgICAgICJuX2ZlYXR1cmVzIjogbl9mZWF0dXJlcywKICAgICAgICAic2Vjb25kcyI6IHJvdW5kKHNlY29uZHMsIDMpLAogICAgICAgICJ0cyI6IHRpbWUuc3RyZnRpbWUoIiVZLSVtLSVkVCVIOiVNOiVTJXoiKSwKICAgICAgICAid2VpZ2h0cyI6IGpvYlsid2VpZ2h0cyJdLAogICAgICAgICJtYXJrdXAiOiBqb2JbIm1hcmt1cCJdLAogICAgfQogICAgZC53cml0ZV90ZXh0KGpzb24uZHVtcHMocGF5bG9hZCwgZW5zdXJlX2FzY2lpPUZhbHNlLCBpbmRlbnQ9MikgKyAiXG4iLCBlbmNvZGluZz0idXRmLTgiKQoKCmRlZiB3cml0ZV9zdGF0dXMob3V0X3Jvb3Q6IFBhdGgsIHN0YXRlOiBzdHIsICoqZXh0cmEpIC0+IE5vbmU6CiAgICAiIiJTVEFUVVMudHh0OiDQv9C10YDQstCw0Y8g0YHRgtGA0L7QutCwIFJVTk5JTkd8RE9ORXxGQUlMRUQgKCsg0LLRgNC10LzRjyksINC00LDQu9C10LUga2V5PXZhbHVlLiIiIgogICAgb3V0X3Jvb3QubWtkaXIocGFyZW50cz1UcnVlLCBleGlzdF9vaz1UcnVlKQogICAgdHMgPSB0aW1lLnN0cmZ0aW1lKCIlWS0lbS0lZFQlSDolTTolUyV6IikKICAgIGxpbmVzID0gW3N0YXRlLCBmInRzPXt0c30iXQogICAgZm9yIGssIHYgaW4gZXh0cmEuaXRlbXMoKToKICAgICAgICBsaW5lcy5hcHBlbmQoZiJ7a309e3Z9IikKICAgIChvdXRfcm9vdCAvICJTVEFUVVMudHh0Iikud3JpdGVfdGV4dCgiXG4iLmpvaW4obGluZXMpICsgIlxuIiwgZW5jb2Rpbmc9InV0Zi04IikKICAgIHBheWxvYWQgPSB7InN0YXR1cyI6IHN0YXRlLCAidHMiOiB0cywgKipleHRyYX0KICAgIChvdXRfcm9vdCAvICJTVEFUVVMuanNvbiIpLndyaXRlX3RleHQoCiAgICAgICAganNvbi5kdW1wcyhwYXlsb2FkLCBlbnN1cmVfYXNjaWk9RmFsc2UsIGluZGVudD0yKSArICJcbiIsIGVuY29kaW5nPSJ1dGYtOCIKICAgICkKCgpkZWYgc2VsZWN0X2pvYnMoaWRzOiBJdGVyYWJsZVtzdHJdIHwgTm9uZSA9IE5vbmUpIC0+IGxpc3RbZGljdF06CiAgICBpZiBub3QgaWRzOgogICAgICAgIHJldHVybiBsaXN0KEpPQlMpCiAgICB3YW50ID0gc2V0KGlkcykKICAgIGZvdW5kID0gW2ogZm9yIGogaW4gSk9CUyBpZiBqWyJpZCJdIGluIHdhbnRdCiAgICBtaXNzaW5nID0gd2FudCAtIHtqWyJpZCJdIGZvciBqIGluIGZvdW5kfQogICAgaWYgbWlzc2luZzoKICAgICAgICByYWlzZSBLZXlFcnJvcihmItC90LXQuNC30LLQtdGB0YLQvdGL0LUgam9iIGlkOiB7c29ydGVkKG1pc3NpbmcpfSIpCiAgICByZXR1cm4gZm91bmQKCgpkZWYgY291bnRfZmVhdHVyZXMoZ2VvanNvbl9wYXRoOiBQYXRoKSAtPiBpbnQ6CiAgICBkYXRhID0ganNvbi5sb2FkcyhnZW9qc29uX3BhdGgucmVhZF90ZXh0KGVuY29kaW5nPSJ1dGYtOCIpKQogICAgcmV0dXJuIGxlbihkYXRhLmdldCgiZmVhdHVyZXMiKSBvciBbXSkKCgpkZWYgd3JpdGVfc2hhMjU2c3Vtcyh3b3JrOiBQYXRoLCBqb2JzOiBsaXN0W2RpY3RdKSAtPiBQYXRoOgogICAgbGluZXMgPSBbXQogICAgZm9yIGpvYiBpbiBqb2JzOgogICAgICAgIG91dCA9IGpvYl9vdXRfcGF0aCh3b3JrLCBqb2JbIm91dCJdKQogICAgICAgIGlmIG91dC5pc19maWxlKCk6CiAgICAgICAgICAgIGxpbmVzLmFwcGVuZChmIntzaGEyNTZfZmlsZShvdXQpfSAgb29mL3tqb2JbJ291dCddfSIpCiAgICBwYXRoID0gd29yayAvICJTSEEyNTZTVU1TIgogICAgcGF0aC53cml0ZV90ZXh0KCJcbiIuam9pbihsaW5lcykgKyAoIlxuIiBpZiBsaW5lcyBlbHNlICIiKSwgZW5jb2Rpbmc9InV0Zi04IikKICAgIHJldHVybiBwYXRoCgoKZGVmIHN5bnRoZXRpY19mYyhuOiBpbnQgPSAyKSAtPiBkaWN0OgogICAgIiIi0JrRgNC+0YjQtdGH0L3Ri9C5IEdlb0pTT04g0LTQu9GPIENQVSBzbW9rZSAo0LHQtdC3IHRvcmNoL1MzKS4iIiIKICAgIGZlYXRzID0gW10KICAgIGZvciBpIGluIHJhbmdlKG4pOgogICAgICAgIHgwLCB5MCA9IDRfMjAwXzAwMC4wICsgaSAqIDEwMCwgN181MDBfMDAwLjAKICAgICAgICByaW5nID0gW1t4MCwgeTBdLCBbeDAgKyA0MCwgeTBdLCBbeDAgKyA0MCwgeTAgKyA0MF0sIFt4MCwgeTAgKyA0MF0sIFt4MCwgeTBdXQogICAgICAgIGZlYXRzLmFwcGVuZCh7CiAgICAgICAgICAgICJ0eXBlIjogIkZlYXR1cmUiLAogICAgICAgICAgICAiZ2VvbWV0cnkiOiB7InR5cGUiOiAiUG9seWdvbiIsICJjb29yZGluYXRlcyI6IFtyaW5nXX0sCiAgICAgICAgICAgICJwcm9wZXJ0aWVzIjogewogICAgICAgICAgICAgICAgImZpZCI6IGksCiAgICAgICAgICAgICAgICAiY2xhc3NfbmFtZSI6ICJrdXJnYW55X3RzZWx5ZSIsCiAgICAgICAgICAgICAgICAiY29uZmlkZW5jZSI6IDAuOSwKICAgICAgICAgICAgICAgICJyZWdpb25fbmFtZSI6ICJzbW9rZSIsCiAgICAgICAgICAgICAgICAibWFya3VwX3R5cGUiOiAic3Bfb3IiLAogICAgICAgICAgICAgICAgImNycyI6ICJFUFNHOjM4NTciLAogICAgICAgICAgICB9LAogICAgICAgIH0pCiAgICByZXR1cm4geyJ0eXBlIjogIkZlYXR1cmVDb2xsZWN0aW9uIiwgImZlYXR1cmVzIjogZmVhdHN9CgoKZGVmIHJ1bl9zbW9rZSh3b3JrOiBQYXRoKSAtPiBkaWN0OgogICAgIiIiQ1BVIHNtb2tlOiDQvtC00LjQvSBzeW50aGV0aWMgam9iLCBTVEFUVVMsIHNoYTI1Niwgc2tpcC1vbi1yZXJ1bi4iIiIKICAgIHdvcmsgPSBQYXRoKHdvcmspCiAgICB3b3JrLm1rZGlyKHBhcmVudHM9VHJ1ZSwgZXhpc3Rfb2s9VHJ1ZSkKICAgIHdyaXRlX3N0YXR1cyh3b3JrLCAiUlVOTklORyIsIHBoYXNlPSJzbW9rZSIpCiAgICBqb2IgPSBzZWxlY3Rfam9icyhbInNwb3JfZm9jYWwiXSlbMF0KICAgIG91dCA9IGpvYl9vdXRfcGF0aCh3b3JrLCBqb2JbIm91dCJdKQogICAgb3V0LnBhcmVudC5ta2RpcihwYXJlbnRzPVRydWUsIGV4aXN0X29rPVRydWUpCiAgICBpZiBpc19qb2JfY29tcGxldGUod29yaywgam9iKToKICAgICAgICB3cml0ZV9zdGF0dXMod29yaywgIkRPTkUiLCBwaGFzZT0ic21va2UiLCBub3RlPSJhbHJlYWR5X2NvbXBsZXRlIikKICAgICAgICByZXR1cm4geyJza2lwcGVkIjogVHJ1ZSwgInNoYTI1NiI6IHNoYTI1Nl9maWxlKG91dCl9CiAgICB0MCA9IHRpbWUucGVyZl9jb3VudGVyKCkKICAgIG91dC53cml0ZV90ZXh0KGpzb24uZHVtcHMoc3ludGhldGljX2ZjKDIpLCBlbnN1cmVfYXNjaWk9RmFsc2UpLCBlbmNvZGluZz0idXRmLTgiKQogICAgc2hhID0gc2hhMjU2X2ZpbGUob3V0KQogICAgbiA9IGNvdW50X2ZlYXR1cmVzKG91dCkKICAgIHdyaXRlX2RvbmUod29yaywgam9iLCBzaGEsIG4sIHRpbWUucGVyZl9jb3VudGVyKCkgLSB0MCkKICAgIHdyaXRlX3NoYTI1NnN1bXMod29yaywgW2pvYl0pCiAgICAjIHNlY29uZCBjYWxsIG11c3Qgc2tpcAogICAgYXNzZXJ0IGlzX2pvYl9jb21wbGV0ZSh3b3JrLCBqb2IpCiAgICB3cml0ZV9zdGF0dXMod29yaywgIkRPTkUiLCBwaGFzZT0ic21va2UiLCBuX2ZlYXR1cmVzPW4sIHNoYTI1Nj1zaGEpCiAgICByZXR1cm4geyJza2lwcGVkIjogRmFsc2UsICJzaGEyNTYiOiBzaGEsICJuX2ZlYXR1cmVzIjogbn0KCgppZiBfX25hbWVfXyA9PSAiX19tYWluX18iOgogICAgaW1wb3J0IHRlbXBmaWxlCiAgICB3aXRoIHRlbXBmaWxlLlRlbXBvcmFyeURpcmVjdG9yeSgpIGFzIHRkOgogICAgICAgIHByaW50KGpzb24uZHVtcHMocnVuX3Ntb2tlKFBhdGgodGQpKSwgZW5zdXJlX2FzY2lpPUZhbHNlKSkK'))

import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

OUT = Path(os.environ.get("OOF_OUT", "/kaggle/working/oof_honest10"))
LOG = OUT / "logs"
T0 = time.time()
MODE = os.environ.get("OOF_MODE", "weights")  # weights | fold
SMOKE = os.environ.get("OOF_SMOKE", "").strip() in ("1", "true", "yes")
DEADLINE_H = float(os.environ.get("OOF_DEADLINE_H", "11.0"))
JOB_FILTER = [x for x in os.environ.get("OOF_JOBS", "").split(",") if x.strip()]
BUCKET = os.environ.get("OOF_S3_BUCKET", "train-expds-3")
ENDPOINT = os.environ.get("OOF_S3_ENDPOINT", "https://s3.ru-7.storage.selcloud.ru")
SECRET_ID, SECRET_KEY = "KOZ3_S3_ACCESS_KEY_ID", "KOZ3_S3_SECRET_ACCESS_KEY"
DOCKER_HINT = "gcr.io/kaggle-private-byod/python"


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def _load_jobs_lib(repo: Path | None = None):
    candidates = []
    if repo:
        candidates.append(repo / "scripts" / "oof_kaggle_jobs.py")
    here = Path(__file__).resolve().parent
    candidates += [
        here / "oof_kaggle_jobs.py",
        Path("/kaggle/working/expds-koz3/scripts/oof_kaggle_jobs.py"),
        Path("scripts/oof_kaggle_jobs.py"),
    ]
    for p in candidates:
        if p.is_file():
            spec = importlib.util.spec_from_file_location("oof_kaggle_jobs", p)
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            sys.modules["oof_kaggle_jobs"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("scripts/oof_kaggle_jobs.py не найден (attach repo snapshot)")


def find_repo() -> Path:
    for p in (
        Path(os.environ["OOF_REPO"]) if os.environ.get("OOF_REPO") else None,
        Path("/kaggle/working/expds-koz3"),
        Path("/kaggle/input/expds-koz3"),
        Path("."),
    ):
        if p is None:
            continue
        if (p / "scripts" / "nn_oof.py").is_file() and (p / "scripts" / "oof_kaggle_jobs.py").is_file():
            return p.resolve()
    # dataset that contains the repo as a subfolder
    for base in Path("/kaggle/input").glob("*"):
        for cand in base.rglob("scripts/nn_oof.py"):
            root = cand.parent.parent
            if (root / "scripts" / "oof_kaggle_jobs.py").is_file():
                return root.resolve()
    raise FileNotFoundError(
        "репозиторий с scripts/nn_oof.py + oof_kaggle_jobs.py не найден; "
        "прикрепите private dataset snapshot или задайте OOF_REPO"
    )


def find_models(repo: Path) -> Path:
    env = os.environ.get("OOF_MODELS_ROOT")
    if env and Path(env).is_dir():
        return Path(env)
    if (repo / "models").is_dir() and any((repo / "models").glob("*.pt")):
        return repo / "models"
    for base in Path("/kaggle/input").glob("*"):
        for name in ("models", "koz3-models"):
            cand = base / name
            if cand.is_dir() and any(cand.glob("*.pt")):
                return cand
        if any(base.glob("*.pt")):
            return base
    raise FileNotFoundError("models/*.pt не найдены (OOF_MODELS_ROOT или dataset)")


def find_train_root() -> Path:
    env = os.environ.get("OOF_DATA_ROOT")
    if env and Path(env).is_dir():
        return Path(env)
    for base in Path("/kaggle/input").glob("*"):
        for name in ("train-expds-3", "train_expds_3", "TRAIN"):
            cand = base / name
            if cand.is_dir():
                return cand
    # after S3 sync
    syn = Path("/kaggle/working/train-expds-3")
    if syn.is_dir():
        return syn
    raise FileNotFoundError(
        "train-expds-3 не найден; sync через S3 (koz3-s3-cred) или attach dataset; OOF_DATA_ROOT"
    )


def _cred():
    kid = sec = None
    try:
        from kaggle_secrets import UserSecretsClient
        u = UserSecretsClient()
        kid, sec = u.get_secret(SECRET_ID), u.get_secret(SECRET_KEY)
    except Exception as e:
        log("Kaggle Secrets:", type(e).__name__)
    if not kid or not sec:
        kid, sec = os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not kid or not sec:
        for p in sorted(glob.glob("/kaggle/input/**/cred.json", recursive=True)):
            try:
                c = json.loads(Path(p).read_text(encoding="utf-8"))
            except Exception:
                continue
            kid = c.get("id") or c.get("aws_access_key_id")
            sec = c.get("key") or c.get("aws_secret_access_key")
            if kid and sec:
                log("S3 keys from private dataset")
                break
    if not kid or not sec:
        raise SystemExit(f"нет ключей S3: Secrets {SECRET_ID}/{SECRET_KEY} или dmitrijkv/koz3-s3-cred")
    return kid, sec


def s3_client():
    import boto3
    from botocore.config import Config
    kid, sec = _cred()
    return boto3.client(
        "s3", endpoint_url=ENDPOINT, aws_access_key_id=kid, aws_secret_access_key=sec,
        region_name="ru-7",
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}, max_pool_connections=32),
    )


def maybe_sync_train(dst: Path, prefix: str = "") -> Path:
    """Опционально: OOF_S3_SYNC=1 скачивает префикс бакета в dst (резюмируемо)."""
    if os.environ.get("OOF_S3_SYNC", "").strip() not in ("1", "true", "yes"):
        return find_train_root()
    dst.mkdir(parents=True, exist_ok=True)
    cli = s3_client()
    from boto3.s3.transfer import TransferConfig
    cfg = TransferConfig(max_concurrency=4)
    token = None
    n = 0
    while True:
        kw = dict(Bucket=BUCKET, Prefix=prefix, MaxKeys=1000)
        if token:
            kw["ContinuationToken"] = token
        resp = cli.list_objects_v2(**kw)
        for obj in resp.get("Contents") or []:
            key = obj["Key"]
            if key.endswith("/"):
                continue
            target = dst / key
            if target.is_file() and target.stat().st_size == obj["Size"]:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".part")
            cli.download_file(BUCKET, key, str(tmp), Config=cfg)
            tmp.rename(target)
            n += 1
            if n % 50 == 0:
                log("s3 synced", n, "files")
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    log("s3 sync done, files touched/new≈", n)
    return dst


def pip_deps():
    need = []
    for mod, pin in {
        "rasterio": "rasterio==1.4.3",
        "shapely": "shapely==2.1.1",
        "boto3": "boto3==1.40.61",
        "segmentation_models_pytorch": "segmentation-models-pytorch==0.3.4",
    }.items():
        try:
            importlib.import_module(mod)
        except Exception:
            need.append(pin)
    if need:
        log("pip install", need)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need])
        if r.returncode:
            raise SystemExit(f"pip failed: {need}")


def run_weights_job(repo: Path, root: Path, models: Path, work: Path, job: dict, lib) -> None:
    out = lib.job_out_path(work, job["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    wpaths = []
    for w in job["weights"]:
        p = models / w
        if not p.is_file():
            raise FileNotFoundError(f"нет весов {p}")
        wpaths.append(str(p))
    cmd = [
        sys.executable, "-X", "utf8", str(repo / "scripts" / "nn_oof.py"),
        "--root", str(root),
        "--out", str(out),
        "--markup", job["markup"],
        "--config", str(repo / "configs" / "default.toml"),
        "--weights", *wpaths,
    ]
    if os.environ.get("OOF_TTA"):
        cmd += ["--tta", os.environ["OOF_TTA"]]
    log("run", job["id"], cmd[-6:])
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(repo)}
    logf = LOG / f"{job['id']}.log"
    with logf.open("w", encoding="utf-8") as fh:
        r = subprocess.run(cmd, cwd=str(repo), env=env, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"nn_oof rc={r.returncode} see {logf}")
    if not out.is_file():
        raise RuntimeError(f"нет выхода {out}")


def run_fold_job(repo: Path, root: Path, work: Path, job: dict, lib) -> None:
    """Ожидает кэш и fold-веса в work/fold_work/<id>/ (OOF_FOLD_WORK)."""
    cache = Path(os.environ.get("OOF_CACHE", work / "cache" / job["id"]))
    fold_work = Path(os.environ.get("OOF_FOLD_WORK", work / "fold_work" / job["id"]))
    out = lib.job_out_path(work, job["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if not (cache / "index.csv").is_file():
        raise FileNotFoundError(f"fold mode: нет кэша {cache}/index.csv")
    cmd = [
        sys.executable, "-X", "utf8", str(repo / "scripts" / "nn_oof.py"),
        "--root", str(root),
        "--cache", str(cache),
        "--work", str(fold_work),
        "--out", str(out),
        "--markup", job["markup"],
        "--config", str(repo / "configs" / "default.toml"),
        "--folds", os.environ.get("OOF_FOLDS", "5"),
        "--predict-only", "--resume",
    ]
    log("run fold", job["id"])
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(repo)}
    logf = LOG / f"{job['id']}_fold.log"
    with logf.open("w", encoding="utf-8") as fh:
        r = subprocess.run(cmd, cwd=str(repo), env=env, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"nn_oof fold rc={r.returncode} see {logf}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    # smoke early — no repo/S3 required if lib is beside the kernel or in PYTHONPATH
    if SMOKE:
        # prefer lib next to this file (shipped) or from repo
        lib_path = Path(__file__).resolve().parent / "oof_kaggle_jobs.py"
        if not lib_path.is_file():
            # copy from cwd scripts if present
            src = Path("scripts/oof_kaggle_jobs.py")
            if src.is_file():
                shutil.copy2(src, lib_path)
        try:
            lib = _load_jobs_lib()
        except FileNotFoundError:
            # inline minimal: import from scripts if cwd is repo
            sys.path.insert(0, str(Path(".").resolve()))
            lib = _load_jobs_lib(Path("."))
        lib.write_status(OUT, "RUNNING", phase="smoke", mode="smoke")
        result = lib.run_smoke(OUT)
        log("smoke", result)
        return 0

    lib = None
    try:
        pip_deps()
        repo = find_repo()
        lib = _load_jobs_lib(repo)
        lib.write_status(OUT, "RUNNING", phase="boot", mode=MODE)
        log("repo", repo)
        models = find_models(repo)
        log("models", models)
        if os.environ.get("OOF_S3_SYNC", "").strip() in ("1", "true", "yes"):
            root = maybe_sync_train(Path("/kaggle/working/train-expds-3"))
        else:
            try:
                root = find_train_root()
            except FileNotFoundError:
                log("train root missing — trying S3 sync")
                root = maybe_sync_train(Path("/kaggle/working/train-expds-3"))
        log("train", root)

        jobs = lib.select_jobs(JOB_FILTER or None)
        deadline = T0 + DEADLINE_H * 3600
        done_n = fail_n = skip_n = 0
        for job in jobs:
            if time.time() > deadline:
                log("deadline — stop, resume later")
                lib.write_status(OUT, "RUNNING", phase="deadline", done=done_n, failed=fail_n, skipped=skip_n)
                return 0
            if lib.is_job_complete(OUT, job):
                log("skip done", job["id"])
                skip_n += 1
                continue
            t1 = time.perf_counter()
            try:
                if MODE == "fold":
                    run_fold_job(repo, root, OUT, job, lib)
                else:
                    run_weights_job(repo, root, models, OUT, job, lib)
                out = lib.job_out_path(OUT, job["out"])
                sha = lib.sha256_file(out)
                n = lib.count_features(out)
                lib.write_done(OUT, job, sha, n, time.perf_counter() - t1)
                done_n += 1
                log("done", job["id"], "feats", n, "sha", sha[:12])
            except Exception as e:
                fail_n += 1
                log("FAIL", job["id"], type(e).__name__, e)
                (LOG / f"{job['id']}_error.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
                # continue other jobs
        lib.write_sha256sums(OUT, jobs)
        state = "DONE" if fail_n == 0 else ("FAILED" if done_n == 0 else "DONE")
        # partial success still DONE with failed=; hard FAILED only if nothing new and failures
        if fail_n and done_n == 0 and skip_n == 0:
            state = "FAILED"
        elif fail_n:
            state = "DONE"  # partial; details in STATUS.json
        lib.write_status(
            OUT, state, phase="finish", mode=MODE,
            done=done_n, failed=fail_n, skipped=skip_n,
            elapsed_h=round((time.time() - T0) / 3600, 3),
        )
        log("finish", state, "done", done_n, "fail", fail_n, "skip", skip_n)
        return 0 if state != "FAILED" else 1
    except Exception:
        log("FATAL", traceback.format_exc())
        if lib:
            lib.write_status(OUT, "FAILED", phase="fatal", error="see logs")
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "STATUS.txt").write_text(
                f"FAILED\nts={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\nphase=fatal\n",
                encoding="utf-8",
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
