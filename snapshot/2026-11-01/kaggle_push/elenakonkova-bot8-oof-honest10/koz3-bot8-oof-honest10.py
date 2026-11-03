#!/usr/bin/env python3
"""KOZ3 · bot-8 / GPU-5: OOF-инференс моделей сдачи на трейне для калибровки порогов (honest-10).

Slug: elenakonkova/koz3-bot8-oof-honest10
Ресурс: NvidiaTeslaT4, internet; datasets: koz3-bot8-snap + koz3-s3-cred +
dmitrijkv/koz3-xfer-s000..s009 + dmitrijkv/koz3-gt-train-v2 (collab reader).
Train tree: rebuild_train_tree.py → /kaggle/working/train-expds-3 (PIXGBM_REBUILD_ALL=1).
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
_p = Path("/kaggle/working/oof_kaggle_jobs.py")
if not _p.is_file():
    _p.write_bytes(base64.b64decode(b'IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiLQn9C70LDQvSDQt9Cw0LTQsNC90LjQuSBPT0Yt0LjQvdGE0LXRgNC10L3RgdCwINC00LvRjyBLYWdnbGUt0Y/QtNGA0LAgYm90LTggKEdQVS01KS4KCtCf0L7RgtGA0LXQsdC40YLQtdC70Lg6IHNjcmlwdHMvc2NvcmVib2FyZC5weSAoLS1ob25lc3QgMTApLCBkZW5zaXR5X3RocmVzaG9sZHMucHksCm9vZl90cmFuc2Zvcm0gY2FsaWIsIGJvdDQvYm90NyDQutCw0LvQuNCx0YDQvtCy0LrQsCDQv9C+0YDQvtCz0L7QsiDQv9C+INGB0LvQvtGOLgoK0KDQtdC20LjQvCDQv9C+INGD0LzQvtC70YfQsNC90LjRjiDigJQgYGB3ZWlnaHRzYGA6INGB0LTQsNC90L3Ri9C1INCy0LXRgdCwINC40LcgY29uZmlncy9kZWZhdWx0LnRvbWwgW21vZGVsc10K0L3QsCDQstGB0LXRhSDRg9GH0LDRgdGC0LrQsNGFINGC0YDQtdC50L3QsCAo0L3QtSBmb2xkLU9PRikuINCn0LXRgdGC0L3QvtGB0YLRjCBob25lc3QtMTAg0YLQvtCz0LTQsCDQsiDQv9C+0LTQsdC+0YDQtQrQv9C+0YDQvtCz0LAg0L3QsCDRh9GD0LbQuNGFINGB0LXQvNGM0Y/RhSwg0L3QtSDQsiDQstC10YHQsNGFLiBGb2xkLU9PRiAoYGBubl9vb2YgLS1wcmVkaWN0LW9ubHkvLS1yZXN1bWVgYCkK0LLQutC70Y7Rh9Cw0LXRgtGB0Y8gZW52IE9PRl9NT0RFPWZvbGQsINC60L7Qs9C00LAg0L3QsCBpbnB1dC9TMyDQtdGB0YLRjCB3b3JrL2ZvbGQqINGBINCy0LXRgdCw0LzQuC4KIiIiCmZyb20gX19mdXR1cmVfXyBpbXBvcnQgYW5ub3RhdGlvbnMKCmltcG9ydCBoYXNobGliCmltcG9ydCBqc29uCmltcG9ydCB0aW1lCmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aApmcm9tIHR5cGluZyBpbXBvcnQgSXRlcmFibGUKCiMg0JjQvNC10L3QsCDQstGL0YXQvtC00L7QsiDigJQg0LrQsNC6INC20LTRkdGCIHNjb3JlYm9hcmQgLyBub3RlcyAo0YHQvC4gc2NyaXB0cy9zY29yZWJvYXJkLnB5KS4KSk9CUzogbGlzdFtkaWN0XSA9IFsKICAgICMgb3V0X25hbWUsIHdlaWdodHMgKHJlbCB0byBtb2RlbHMvKSwgbWFya3VwIGZvciBubl9vb2YsIG9wdGlvbmFsIHByZWRpY3RfbWFya3VwcwogICAgeyJpZCI6ICJzcG9yX2dvciIsICJvdXQiOiAic3Bvcl9nb3Jfb29mLmdlb2pzb24iLCAid2VpZ2h0cyI6IFsic3Bvcl9nb3JfcjE4LnB0Il0sICJtYXJrdXAiOiAic3Bfb3IifSwKICAgIHsiaWQiOiAic3Bvcl9mb2NhbCIsICJvdXQiOiAic3Bvcl9mb2NhbF9vb2YuZ2VvanNvbiIsICJ3ZWlnaHRzIjogWyJzcG9yX2ZvY2FsX3IxOC5wdCJdLCAibWFya3VwIjogInNwX29yIn0sCiAgICB7ImlkIjogInNwb3JfYXJjaCIsICJvdXQiOiAic3Bvcl9hcmNoX29vZi5nZW9qc29uIiwgIndlaWdodHMiOiBbInN1cmZfYXJjaF9yMTgucHQiXSwgIm1hcmt1cCI6ICJzcF9vciJ9LAogICAgeyJpZCI6ICJzcG9yX2ZvY2FsX3dpZGUiLCAib3V0IjogInNwb3JfZm9jYWxfd2lkZV9vb2YuZ2VvanNvbiIsICJ3ZWlnaHRzIjogWyJzcG9yX2ZvY2FsX3dpZGVfcjE4LnB0Il0sICJtYXJrdXAiOiAic3Bfb3IifSwKICAgIHsiaWQiOiAibGlfZ29yX21hc2siLCAib3V0IjogImxpX2dvcl9tYXNrX29vZi5nZW9qc29uIiwgIndlaWdodHMiOiBbImxpX2dvcl9tYXNrX3IxOC5wdCJdLCAibWFya3VwIjogImxpIn0sCiAgICB7ImlkIjogImxpX2ZvcnRfbWFzayIsICJvdXQiOiAibGlfZm9ydF9tYXNrX29vZi5nZW9qc29uIiwgIndlaWdodHMiOiBbImxpX2ZvcnRfbWFza19yMTgucHQiXSwgIm1hcmt1cCI6ICJsaSJ9LAogICAgeyJpZCI6ICJsaV9kZXYiLCAib3V0IjogImxpX2Rldl9vb2YuZ2VvanNvbiIsICJ3ZWlnaHRzIjogWyJsaV9kZXZfcjE4LnB0Il0sICJtYXJrdXAiOiAibGkifSwKICAgIHsiaWQiOiAibGlfaGFyZDIiLCAib3V0IjogImxpX2hhcmQyX29vZi5nZW9qc29uIiwgIndlaWdodHMiOiBbImxpX3IxOF9oYXJkMi5wdCJdLCAibWFya3VwIjogImxpIn0sCiAgICB7ImlkIjogImFlX2ZvcnRfbWFzayIsICJvdXQiOiAiYWVfZm9ydF9tYXNrX29vZi5nZW9qc29uIiwgIndlaWdodHMiOiBbImFlX2ZvcnRfbWFza19yMTgucHQiXSwgIm1hcmt1cCI6ICJhZSJ9LAogICAgeyJpZCI6ICJvcl9hcmNoIiwgIm91dCI6ICJvcl9hcmNoX29vZi5nZW9qc29uIiwgIndlaWdodHMiOiBbInN1cmZfYXJjaF9yMTgucHQiXSwgIm1hcmt1cCI6ICJvciJ9LApdCgoKZGVmIGpvYl9kb25lX21hcmtlcih3b3JrOiBQYXRoLCBqb2JfaWQ6IHN0cikgLT4gUGF0aDoKICAgIHJldHVybiB3b3JrIC8gImpvYnMiIC8gam9iX2lkIC8gImRvbmUuanNvbiIKCgpkZWYgam9iX291dF9wYXRoKHdvcms6IFBhdGgsIG91dF9uYW1lOiBzdHIpIC0+IFBhdGg6CiAgICByZXR1cm4gd29yayAvICJvb2YiIC8gb3V0X25hbWUKCgpkZWYgaXNfam9iX2NvbXBsZXRlKHdvcms6IFBhdGgsIGpvYjogZGljdCkgLT4gYm9vbDoKICAgICIiItCY0LTQtdC80L/QvtGC0LXQvdGC0L3QvtGB0YLRjDogZG9uZS5qc29uICsg0LLRi9GF0L7QtNC90L7QuSBnZW9qc29uINGBINGC0LXQvCDQttC1IHNoYTI1Ni4iIiIKICAgIG1hcmtlciA9IGpvYl9kb25lX21hcmtlcih3b3JrLCBqb2JbImlkIl0pCiAgICBvdXQgPSBqb2Jfb3V0X3BhdGgod29yaywgam9iWyJvdXQiXSkKICAgIGlmIG5vdCBtYXJrZXIuaXNfZmlsZSgpIG9yIG5vdCBvdXQuaXNfZmlsZSgpOgogICAgICAgIHJldHVybiBGYWxzZQogICAgdHJ5OgogICAgICAgIG1ldGEgPSBqc29uLmxvYWRzKG1hcmtlci5yZWFkX3RleHQoZW5jb2Rpbmc9InV0Zi04IikpCiAgICBleGNlcHQganNvbi5KU09ORGVjb2RlRXJyb3I6CiAgICAgICAgcmV0dXJuIEZhbHNlCiAgICByZXR1cm4gYm9vbChtZXRhLmdldCgic2hhMjU2IikpIGFuZCBtZXRhWyJzaGEyNTYiXSA9PSBzaGEyNTZfZmlsZShvdXQpCgoKZGVmIHNoYTI1Nl9maWxlKHBhdGg6IFBhdGgpIC0+IHN0cjoKICAgIGggPSBoYXNobGliLnNoYTI1NigpCiAgICB3aXRoIHBhdGgub3BlbigicmIiKSBhcyBmaDoKICAgICAgICBmb3IgY2h1bmsgaW4gaXRlcihsYW1iZGE6IGZoLnJlYWQoMSA8PCAyMCksIGIiIik6CiAgICAgICAgICAgIGgudXBkYXRlKGNodW5rKQogICAgcmV0dXJuIGguaGV4ZGlnZXN0KCkKCgpkZWYgd3JpdGVfZG9uZSh3b3JrOiBQYXRoLCBqb2I6IGRpY3QsIHNoYTogc3RyLCBuX2ZlYXR1cmVzOiBpbnQsIHNlY29uZHM6IGZsb2F0KSAtPiBOb25lOgogICAgZCA9IGpvYl9kb25lX21hcmtlcih3b3JrLCBqb2JbImlkIl0pCiAgICBkLnBhcmVudC5ta2RpcihwYXJlbnRzPVRydWUsIGV4aXN0X29rPVRydWUpCiAgICBwYXlsb2FkID0gewogICAgICAgICJpZCI6IGpvYlsiaWQiXSwKICAgICAgICAib3V0Ijogam9iWyJvdXQiXSwKICAgICAgICAic2hhMjU2Ijogc2hhLAogICAgICAgICJuX2ZlYXR1cmVzIjogbl9mZWF0dXJlcywKICAgICAgICAic2Vjb25kcyI6IHJvdW5kKHNlY29uZHMsIDMpLAogICAgICAgICJ0cyI6IHRpbWUuc3RyZnRpbWUoIiVZLSVtLSVkVCVIOiVNOiVTJXoiKSwKICAgICAgICAid2VpZ2h0cyI6IGpvYlsid2VpZ2h0cyJdLAogICAgICAgICJtYXJrdXAiOiBqb2JbIm1hcmt1cCJdLAogICAgfQogICAgZC53cml0ZV90ZXh0KGpzb24uZHVtcHMocGF5bG9hZCwgZW5zdXJlX2FzY2lpPUZhbHNlLCBpbmRlbnQ9MikgKyAiXG4iLCBlbmNvZGluZz0idXRmLTgiKQoKCmRlZiB3cml0ZV9zdGF0dXMob3V0X3Jvb3Q6IFBhdGgsIHN0YXRlOiBzdHIsICoqZXh0cmEpIC0+IE5vbmU6CiAgICAiIiJTVEFUVVMudHh0OiDQv9C10YDQstCw0Y8g0YHRgtGA0L7QutCwIFJVTk5JTkd8RE9ORXxGQUlMRUQgKCsg0LLRgNC10LzRjyksINC00LDQu9C10LUga2V5PXZhbHVlLiIiIgogICAgb3V0X3Jvb3QubWtkaXIocGFyZW50cz1UcnVlLCBleGlzdF9vaz1UcnVlKQogICAgdHMgPSB0aW1lLnN0cmZ0aW1lKCIlWS0lbS0lZFQlSDolTTolUyV6IikKICAgIGxpbmVzID0gW3N0YXRlLCBmInRzPXt0c30iXQogICAgZm9yIGssIHYgaW4gZXh0cmEuaXRlbXMoKToKICAgICAgICBsaW5lcy5hcHBlbmQoZiJ7a309e3Z9IikKICAgIChvdXRfcm9vdCAvICJTVEFUVVMudHh0Iikud3JpdGVfdGV4dCgiXG4iLmpvaW4obGluZXMpICsgIlxuIiwgZW5jb2Rpbmc9InV0Zi04IikKICAgIHBheWxvYWQgPSB7InN0YXR1cyI6IHN0YXRlLCAidHMiOiB0cywgKipleHRyYX0KICAgIChvdXRfcm9vdCAvICJTVEFUVVMuanNvbiIpLndyaXRlX3RleHQoCiAgICAgICAganNvbi5kdW1wcyhwYXlsb2FkLCBlbnN1cmVfYXNjaWk9RmFsc2UsIGluZGVudD0yKSArICJcbiIsIGVuY29kaW5nPSJ1dGYtOCIKICAgICkKCgpkZWYgc2VsZWN0X2pvYnMoaWRzOiBJdGVyYWJsZVtzdHJdIHwgTm9uZSA9IE5vbmUpIC0+IGxpc3RbZGljdF06CiAgICBpZiBub3QgaWRzOgogICAgICAgIHJldHVybiBsaXN0KEpPQlMpCiAgICB3YW50ID0gc2V0KGlkcykKICAgIGZvdW5kID0gW2ogZm9yIGogaW4gSk9CUyBpZiBqWyJpZCJdIGluIHdhbnRdCiAgICBtaXNzaW5nID0gd2FudCAtIHtqWyJpZCJdIGZvciBqIGluIGZvdW5kfQogICAgaWYgbWlzc2luZzoKICAgICAgICByYWlzZSBLZXlFcnJvcihmItC90LXQuNC30LLQtdGB0YLQvdGL0LUgam9iIGlkOiB7c29ydGVkKG1pc3NpbmcpfSIpCiAgICByZXR1cm4gZm91bmQKCgpkZWYgY291bnRfZmVhdHVyZXMoZ2VvanNvbl9wYXRoOiBQYXRoKSAtPiBpbnQ6CiAgICBkYXRhID0ganNvbi5sb2FkcyhnZW9qc29uX3BhdGgucmVhZF90ZXh0KGVuY29kaW5nPSJ1dGYtOCIpKQogICAgcmV0dXJuIGxlbihkYXRhLmdldCgiZmVhdHVyZXMiKSBvciBbXSkKCgpkZWYgd3JpdGVfc2hhMjU2c3Vtcyh3b3JrOiBQYXRoLCBqb2JzOiBsaXN0W2RpY3RdKSAtPiBQYXRoOgogICAgbGluZXMgPSBbXQogICAgZm9yIGpvYiBpbiBqb2JzOgogICAgICAgIG91dCA9IGpvYl9vdXRfcGF0aCh3b3JrLCBqb2JbIm91dCJdKQogICAgICAgIGlmIG91dC5pc19maWxlKCk6CiAgICAgICAgICAgIGxpbmVzLmFwcGVuZChmIntzaGEyNTZfZmlsZShvdXQpfSAgb29mL3tqb2JbJ291dCddfSIpCiAgICBwYXRoID0gd29yayAvICJTSEEyNTZTVU1TIgogICAgcGF0aC53cml0ZV90ZXh0KCJcbiIuam9pbihsaW5lcykgKyAoIlxuIiBpZiBsaW5lcyBlbHNlICIiKSwgZW5jb2Rpbmc9InV0Zi04IikKICAgIHJldHVybiBwYXRoCgoKZGVmIHN5bnRoZXRpY19mYyhuOiBpbnQgPSAyKSAtPiBkaWN0OgogICAgIiIi0JrRgNC+0YjQtdGH0L3Ri9C5IEdlb0pTT04g0LTQu9GPIENQVSBzbW9rZSAo0LHQtdC3IHRvcmNoL1MzKS4iIiIKICAgIGZlYXRzID0gW10KICAgIGZvciBpIGluIHJhbmdlKG4pOgogICAgICAgIHgwLCB5MCA9IDRfMjAwXzAwMC4wICsgaSAqIDEwMCwgN181MDBfMDAwLjAKICAgICAgICByaW5nID0gW1t4MCwgeTBdLCBbeDAgKyA0MCwgeTBdLCBbeDAgKyA0MCwgeTAgKyA0MF0sIFt4MCwgeTAgKyA0MF0sIFt4MCwgeTBdXQogICAgICAgIGZlYXRzLmFwcGVuZCh7CiAgICAgICAgICAgICJ0eXBlIjogIkZlYXR1cmUiLAogICAgICAgICAgICAiZ2VvbWV0cnkiOiB7InR5cGUiOiAiUG9seWdvbiIsICJjb29yZGluYXRlcyI6IFtyaW5nXX0sCiAgICAgICAgICAgICJwcm9wZXJ0aWVzIjogewogICAgICAgICAgICAgICAgImZpZCI6IGksCiAgICAgICAgICAgICAgICAiY2xhc3NfbmFtZSI6ICJrdXJnYW55X3RzZWx5ZSIsCiAgICAgICAgICAgICAgICAiY29uZmlkZW5jZSI6IDAuOSwKICAgICAgICAgICAgICAgICJyZWdpb25fbmFtZSI6ICJzbW9rZSIsCiAgICAgICAgICAgICAgICAibWFya3VwX3R5cGUiOiAic3Bfb3IiLAogICAgICAgICAgICAgICAgImNycyI6ICJFUFNHOjM4NTciLAogICAgICAgICAgICB9LAogICAgICAgIH0pCiAgICByZXR1cm4geyJ0eXBlIjogIkZlYXR1cmVDb2xsZWN0aW9uIiwgImZlYXR1cmVzIjogZmVhdHN9CgoKZGVmIHJ1bl9zbW9rZSh3b3JrOiBQYXRoKSAtPiBkaWN0OgogICAgIiIiQ1BVIHNtb2tlOiDQvtC00LjQvSBzeW50aGV0aWMgam9iLCBTVEFUVVMsIHNoYTI1Niwgc2tpcC1vbi1yZXJ1bi4iIiIKICAgIHdvcmsgPSBQYXRoKHdvcmspCiAgICB3b3JrLm1rZGlyKHBhcmVudHM9VHJ1ZSwgZXhpc3Rfb2s9VHJ1ZSkKICAgIHdyaXRlX3N0YXR1cyh3b3JrLCAiUlVOTklORyIsIHBoYXNlPSJzbW9rZSIpCiAgICBqb2IgPSBzZWxlY3Rfam9icyhbInNwb3JfZm9jYWwiXSlbMF0KICAgIG91dCA9IGpvYl9vdXRfcGF0aCh3b3JrLCBqb2JbIm91dCJdKQogICAgb3V0LnBhcmVudC5ta2RpcihwYXJlbnRzPVRydWUsIGV4aXN0X29rPVRydWUpCiAgICBpZiBpc19qb2JfY29tcGxldGUod29yaywgam9iKToKICAgICAgICB3cml0ZV9zdGF0dXMod29yaywgIkRPTkUiLCBwaGFzZT0ic21va2UiLCBub3RlPSJhbHJlYWR5X2NvbXBsZXRlIikKICAgICAgICByZXR1cm4geyJza2lwcGVkIjogVHJ1ZSwgInNoYTI1NiI6IHNoYTI1Nl9maWxlKG91dCl9CiAgICB0MCA9IHRpbWUucGVyZl9jb3VudGVyKCkKICAgIG91dC53cml0ZV90ZXh0KGpzb24uZHVtcHMoc3ludGhldGljX2ZjKDIpLCBlbnN1cmVfYXNjaWk9RmFsc2UpLCBlbmNvZGluZz0idXRmLTgiKQogICAgc2hhID0gc2hhMjU2X2ZpbGUob3V0KQogICAgbiA9IGNvdW50X2ZlYXR1cmVzKG91dCkKICAgIHdyaXRlX2RvbmUod29yaywgam9iLCBzaGEsIG4sIHRpbWUucGVyZl9jb3VudGVyKCkgLSB0MCkKICAgIHdyaXRlX3NoYTI1NnN1bXMod29yaywgW2pvYl0pCiAgICAjIHNlY29uZCBjYWxsIG11c3Qgc2tpcAogICAgYXNzZXJ0IGlzX2pvYl9jb21wbGV0ZSh3b3JrLCBqb2IpCiAgICB3cml0ZV9zdGF0dXMod29yaywgIkRPTkUiLCBwaGFzZT0ic21va2UiLCBuX2ZlYXR1cmVzPW4sIHNoYTI1Nj1zaGEpCiAgICByZXR1cm4geyJza2lwcGVkIjogRmFsc2UsICJzaGEyNTYiOiBzaGEsICJuX2ZlYXR1cmVzIjogbn0KCgppZiBfX25hbWVfXyA9PSAiX19tYWluX18iOgogICAgaW1wb3J0IHRlbXBmaWxlCiAgICB3aXRoIHRlbXBmaWxlLlRlbXBvcmFyeURpcmVjdG9yeSgpIGFzIHRkOgogICAgICAgIHByaW50KGpzb24uZHVtcHMocnVuX3Ntb2tlKFBhdGgodGQpKSwgZW5zdXJlX2FzY2lpPUZhbHNlKSkK'))

# vendored rebuild_train_tree.py (bot-6) — Kaggle script may ship only code_file
_rebuild = Path("/kaggle/working/rebuild_train_tree.py")
if not _rebuild.is_file():
    _rebuild.write_bytes(base64.b64decode(b"""IiIiUmVidWlsZCDQmtCe0Jcg4oSWMyB0cmFpbiB0cmVlIGZyb20gS2FnZ2xlIHhmZXIgc2hhcmRzICsgR1QgZGF0YXNldCAoYm90LTYpLgoKU2hhcmRzIChib3QtMSk6CiAgZG1pdHJpamt2L2tvejMteGZlci1zMDAwICDigJQgc21hbGxfZmlsZXMudGFyLnpzdCAoKyBtYW5pZmVzdDsgbGFiZWxzL1VUTS9leHRyYSkKICBkbWl0cmlqa3Yva296My14ZmVyLXMwMDEg4oCmIHMwMDkg4oCUIGxhcmdlIGJsb2JzIGYjIyMjLnRpZiAvIGYjIyMjLmxhejsKICAgICAgZWFjaCBzaGFyZCBoYXMgYSBtYW5pZmVzdCBtYXBwaW5nIGJsb2Ig4oaSIG9yaWdpbmFsIHJlbGF0aXZlIHBhdGgKICBkbWl0cmlqa3Yva296My1ndC10cmFpbi12MiDigJQgY29uc29saWRhdGVkIEdUIChvcHRpb25hbCBvdmVybGF5IGZvciBQSVhHQk07CiAgICAgIHBlci1yZWdpb24gT2JqZWN0UG9seS9GaW5kc1BvaW50cyBhbHJlYWR5IGxpdmUgaW4gczAwMCBzbWFsbCB0YXIpCgpVbnRpbCBzaGFyZCBtYW5pZmVzdHMgZm9yIHMwMDHigJNzMDA5IGFyZSBrbm93biBvZmZsaW5lLCBhdHRhY2ggQUxMIHNoYXJkczsKYXQgcnVudGltZSBvbmx5IG1hdGVyaWFsaXplIHBhdGhzIHdob3NlIHRvcCBmb2xkZXIgaXMgZ2VvIDIwMOKAkzIxNwooUElYR0JNIEdFT19SRUdJT05TKSwgdW5sZXNzIFBJWEdCTV9SRUJVSUxEX0FMTD0xLgoiIiIKZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwoKaW1wb3J0IGpzb24KaW1wb3J0IG9zCmltcG9ydCByZQppbXBvcnQgc2h1dGlsCmltcG9ydCBzdWJwcm9jZXNzCmltcG9ydCB0YXJmaWxlCmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aApmcm9tIHR5cGluZyBpbXBvcnQgQW55LCBJdGVyYXRvcgoKR0VPX1BSRUZJWEVTID0gdHVwbGUoZiJ7bjowM2R9XyIgZm9yIG4gaW4gcmFuZ2UoMjAwLCAyMTgpKQpCTE9CX1JFID0gcmUuY29tcGlsZShyIl5mXGR7NH0oXC5bQS1aYS16MC05XSspPyQiKQoKCmRlZiBfaXNfZ2VvX3JlbChyZWw6IHN0cikgLT4gYm9vbDoKICAgIHJlbCA9IHJlbC5yZXBsYWNlKCJcXCIsICIvIikubHN0cmlwKCIuLyIpCiAgICB0b3AgPSByZWwuc3BsaXQoIi8iLCAxKVswXQogICAgcmV0dXJuIGFueSh0b3Auc3RhcnRzd2l0aChwKSBmb3IgcCBpbiBHRU9fUFJFRklYRVMpCgoKZGVmIF9sb2FkX2pzb24ocGF0aDogUGF0aCkgLT4gQW55OgogICAgcmV0dXJuIGpzb24ubG9hZHMocGF0aC5yZWFkX3RleHQoZW5jb2Rpbmc9InV0Zi04IikpCgoKZGVmIGZpbmRfaW5wdXRfZGF0YXNldHMoaW5wdXRfcm9vdDogUGF0aCB8IE5vbmUgPSBOb25lKSAtPiBkaWN0W3N0ciwgUGF0aF06CiAgICAiIiJNYXAgc2hvcnQgbmFtZSDihpIgL2thZ2dsZS9pbnB1dC88c2x1Zz4gZGlyZWN0b3J5LiIiIgogICAgcm9vdCA9IGlucHV0X3Jvb3Qgb3IgUGF0aCgiL2thZ2dsZS9pbnB1dCIpCiAgICBvdXQ6IGRpY3Rbc3RyLCBQYXRoXSA9IHt9CiAgICBpZiBub3Qgcm9vdC5pc19kaXIoKToKICAgICAgICByZXR1cm4gb3V0CiAgICBmb3IgcCBpbiBzb3J0ZWQocm9vdC5pdGVyZGlyKCkpOgogICAgICAgIGlmIG5vdCBwLmlzX2RpcigpOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIG5hbWUgPSBwLm5hbWUubG93ZXIoKQogICAgICAgIG91dFtuYW1lXSA9IHAKICAgIHJldHVybiBvdXQKCgpkZWYgZmluZF94ZmVyX3NoYXJkcyhkYXRhc2V0czogZGljdFtzdHIsIFBhdGhdKSAtPiBsaXN0W3R1cGxlW2ludCwgUGF0aF1dOgogICAgZm91bmQ6IGxpc3RbdHVwbGVbaW50LCBQYXRoXV0gPSBbXQogICAgZm9yIG5hbWUsIHBhdGggaW4gZGF0YXNldHMuaXRlbXMoKToKICAgICAgICBtID0gcmUuc2VhcmNoKHIia296My14ZmVyLXMoXGR7M30pJCIsIG5hbWUpCiAgICAgICAgaWYgbToKICAgICAgICAgICAgZm91bmQuYXBwZW5kKChpbnQobS5ncm91cCgxKSksIHBhdGgpKQogICAgcmV0dXJuIHNvcnRlZChmb3VuZCkKCgpkZWYgZmluZF9ndF9kYXRhc2V0KGRhdGFzZXRzOiBkaWN0W3N0ciwgUGF0aF0pIC0+IFBhdGggfCBOb25lOgogICAgZm9yIGtleSBpbiAoImtvejMtZ3QtdHJhaW4tdjIiLCAia296My1ndC10cmFpbiIpOgogICAgICAgIGlmIGtleSBpbiBkYXRhc2V0czoKICAgICAgICAgICAgcmV0dXJuIGRhdGFzZXRzW2tleV0KICAgIGZvciBuYW1lLCBwYXRoIGluIGRhdGFzZXRzLml0ZW1zKCk6CiAgICAgICAgaWYgImd0LXRyYWluIiBpbiBuYW1lIG9yICJndF90cmFpbiIgaW4gbmFtZToKICAgICAgICAgICAgcmV0dXJuIHBhdGgKICAgIHJldHVybiBOb25lCgoKZGVmIGZpbmRfbWFuaWZlc3Qoc2hhcmQ6IFBhdGgpIC0+IFBhdGggfCBOb25lOgogICAgZm9yIGNhbmQgaW4gKAogICAgICAgIHNoYXJkIC8gIm1hbmlmZXN0Lmpzb24iLAogICAgICAgIHNoYXJkIC8gZiJ7c2hhcmQubmFtZX1fbWFuaWZlc3QuanNvbiIsCiAgICAgICAgc2hhcmQgLyAia296My14ZmVyLW1hbmlmZXN0Lmpzb24iLAogICAgKToKICAgICAgICBpZiBjYW5kLmlzX2ZpbGUoKToKICAgICAgICAgICAgcmV0dXJuIGNhbmQKICAgICMgYW55ICptYW5pZmVzdCouanNvbiBhdCB0b3AgbGV2ZWwKICAgIGhpdHMgPSBzb3J0ZWQoc2hhcmQuZ2xvYigiKm1hbmlmZXN0Ki5qc29uIikpCiAgICByZXR1cm4gaGl0c1swXSBpZiBoaXRzIGVsc2UgTm9uZQoKCmRlZiBmaW5kX3NtYWxsX3RhcihzaGFyZDogUGF0aCkgLT4gUGF0aCB8IE5vbmU6CiAgICBmb3IgY2FuZCBpbiAoCiAgICAgICAgc2hhcmQgLyAic21hbGxfZmlsZXMudGFyLnpzdCIsCiAgICAgICAgc2hhcmQgLyAic21hbGxfZmlsZXMudGFyLnpzdGQiLAogICAgICAgIHNoYXJkIC8gInNtYWxsX2ZpbGVzLnRhciIsCiAgICApOgogICAgICAgIGlmIGNhbmQuaXNfZmlsZSgpOgogICAgICAgICAgICByZXR1cm4gY2FuZAogICAgaGl0cyA9IGxpc3Qoc2hhcmQuZ2xvYigic21hbGxfZmlsZXMudGFyKiIpKQogICAgcmV0dXJuIGhpdHNbMF0gaWYgaGl0cyBlbHNlIE5vbmUKCgpkZWYgX2lzX2Jsb2JfaWQoczogc3RyKSAtPiBib29sOgogICAgbmFtZSA9IFBhdGgoc3RyKHMpKS5uYW1lCiAgICByZXR1cm4gYm9vbChCTE9CX1JFLm1hdGNoKG5hbWUpIG9yIHJlLm1hdGNoKHIiXmZcZHs0fSQiLCBuYW1lKSkKCgpkZWYgaXRlcl9ibG9iX21hcHBpbmdzKG1hbmlmZXN0OiBBbnkpIC0+IEl0ZXJhdG9yW3R1cGxlW3N0ciwgc3RyXV06CiAgICAiIiJZaWVsZCAoYmxvYl9iYXNlbmFtZSwgcmVsYXRpdmVfcGF0aCkgZnJvbSBmbGV4aWJsZSBtYW5pZmVzdCBzaGFwZXMuIiIiCiAgICBpZiBub3QgaXNpbnN0YW5jZShtYW5pZmVzdCwgZGljdCk6CiAgICAgICAgcmV0dXJuCgogICAgZGVmIGVtaXQoYmxvYjogc3RyLCByZWw6IHN0cik6CiAgICAgICAgcmVsID0gcmVsLnJlcGxhY2UoIlxcIiwgIi8iKS5sc3RyaXAoIi4vIikKICAgICAgICBibG9iID0gUGF0aChzdHIoYmxvYikpLm5hbWUKICAgICAgICBpZiByZWwgYW5kIGJsb2I6CiAgICAgICAgICAgIHlpZWxkIGJsb2IsIHJlbAoKICAgIGRlZiBmcm9tX21lbWJlcnMobWVtYmVyczogZGljdCk6CiAgICAgICAgZm9yIGssIHYgaW4gbWVtYmVycy5pdGVtcygpOgogICAgICAgICAgICBpZiBpc2luc3RhbmNlKHYsIHN0cik6CiAgICAgICAgICAgICAgICBpZiBfaXNfYmxvYl9pZChrKToKICAgICAgICAgICAgICAgICAgICB5aWVsZCBmcm9tIGVtaXQoaywgdikKICAgICAgICAgICAgICAgIGVsaWYgX2lzX2Jsb2JfaWQodik6CiAgICAgICAgICAgICAgICAgICAgeWllbGQgZnJvbSBlbWl0KHYsIGspCiAgICAgICAgICAgIGVsaWYgaXNpbnN0YW5jZSh2LCBkaWN0KToKICAgICAgICAgICAgICAgIHJlbCA9IHYuZ2V0KCJyZWwiKSBvciB2LmdldCgicGF0aCIpIG9yIHYuZ2V0KCJvcmlnIikgb3Igdi5nZXQoIm9yaWdpbmFsIikKICAgICAgICAgICAgICAgIGJsb2IgPSB2LmdldCgiYmxvYiIpIG9yIHYuZ2V0KCJuYW1lIikgb3Igdi5nZXQoImZpbGUiKSBvciB2LmdldCgiaWQiKSBvciB2LmdldCgia2V5IikKICAgICAgICAgICAgICAgIGlmIF9pc19ibG9iX2lkKGspIGFuZCByZWw6CiAgICAgICAgICAgICAgICAgICAgeWllbGQgZnJvbSBlbWl0KGssIHN0cihyZWwpKQogICAgICAgICAgICAgICAgZWxpZiBibG9iIGFuZCByZWw6CiAgICAgICAgICAgICAgICAgICAgeWllbGQgZnJvbSBlbWl0KHN0cihibG9iKSwgc3RyKHJlbCkpCiAgICAgICAgICAgICAgICBlbGlmIGJsb2IgYW5kICIvIiBpbiBzdHIoayk6CiAgICAgICAgICAgICAgICAgICAgeWllbGQgZnJvbSBlbWl0KHN0cihibG9iKSwgc3RyKGspKQoKICAgIGZvciBrZXkgaW4gKCJmaWxlcyIsICJsYXJnZSIsICJibG9icyIsICJtZW1iZXJzIiwgIm1hcCIsICJtYXBwaW5nIik6CiAgICAgICAgYmxvY2sgPSBtYW5pZmVzdC5nZXQoa2V5KQogICAgICAgIGlmIGlzaW5zdGFuY2UoYmxvY2ssIGRpY3QpOgogICAgICAgICAgICBpZiBpc2luc3RhbmNlKGJsb2NrLmdldCgibWVtYmVycyIpLCBkaWN0KToKICAgICAgICAgICAgICAgIHlpZWxkIGZyb20gZnJvbV9tZW1iZXJzKGJsb2NrWyJtZW1iZXJzIl0pCiAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICB5aWVsZCBmcm9tIGZyb21fbWVtYmVycyhibG9jaykKCiAgICB0b3AgPSB7azogdiBmb3IgaywgdiBpbiBtYW5pZmVzdC5pdGVtcygpIGlmIF9pc19ibG9iX2lkKHN0cihrKSl9CiAgICBpZiB0b3A6CiAgICAgICAgeWllbGQgZnJvbSBmcm9tX21lbWJlcnModG9wKQoKCmRlZiBleHRyYWN0X3NtYWxsX3Rhcih0YXJfcGF0aDogUGF0aCwgZGVzdDogUGF0aCkgLT4gaW50OgogICAgIiIiRXh0cmFjdCBzbWFsbF9maWxlcyB0YXIoLnpzdCkgaW50byBkZXN0LiBSZXR1cm5zIG1lbWJlciBjb3VudC4iIiIKICAgIGRlc3QubWtkaXIocGFyZW50cz1UcnVlLCBleGlzdF9vaz1UcnVlKQogICAgbmFtZSA9IHRhcl9wYXRoLm5hbWUKICAgIGlmIG5hbWUuZW5kc3dpdGgoIi56c3QiKSBvciBuYW1lLmVuZHN3aXRoKCIuenN0ZCIpOgogICAgICAgICMgUHJlZmVyIHpzdGQgQ0xJIChLYWdnbGUgaW1hZ2VzIHVzdWFsbHkgaGF2ZSBpdCk7IGZhbGwgYmFjayB0byBwaXAgenN0YW5kYXJkLgogICAgICAgIHRtcF90YXIgPSBkZXN0LnBhcmVudCAvICJfc21hbGxfZmlsZXMudGFyIgogICAgICAgIGlmIHRtcF90YXIuZXhpc3RzKCk6CiAgICAgICAgICAgIHRtcF90YXIudW5saW5rKCkKICAgICAgICB0cnk6CiAgICAgICAgICAgIHN1YnByb2Nlc3MuY2hlY2tfY2FsbChbInpzdGQiLCAiLWQiLCAiLWYiLCAiLW8iLCBzdHIodG1wX3RhciksIHN0cih0YXJfcGF0aCldKQogICAgICAgIGV4Y2VwdCAoRmlsZU5vdEZvdW5kRXJyb3IsIHN1YnByb2Nlc3MuQ2FsbGVkUHJvY2Vzc0Vycm9yKToKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgaW1wb3J0IHpzdGFuZGFyZCBhcyB6c3RkICAjIHR5cGU6IGlnbm9yZQogICAgICAgICAgICBleGNlcHQgSW1wb3J0RXJyb3I6CiAgICAgICAgICAgICAgICBzdWJwcm9jZXNzLmNoZWNrX2NhbGwoW3N5c19leGVjdXRhYmxlKCksICItbSIsICJwaXAiLCAiaW5zdGFsbCIsICItcSIsICJ6c3RhbmRhcmQiXSkKICAgICAgICAgICAgICAgIGltcG9ydCB6c3RhbmRhcmQgYXMgenN0ZCAgIyB0eXBlOiBpZ25vcmUKICAgICAgICAgICAgZGN0eCA9IHpzdGQuWnN0ZERlY29tcHJlc3NvcigpCiAgICAgICAgICAgIHdpdGggdGFyX3BhdGgub3BlbigicmIiKSBhcyBzcmMsIHRtcF90YXIub3Blbigid2IiKSBhcyBvdXQ6CiAgICAgICAgICAgICAgICBkY3R4LmNvcHlfc3RyZWFtKHNyYywgb3V0KQogICAgICAgIHBhdGhfZm9yX3RhciA9IHRtcF90YXIKICAgIGVsc2U6CiAgICAgICAgcGF0aF9mb3JfdGFyID0gdGFyX3BhdGgKCiAgICBuID0gMAogICAgd2l0aCB0YXJmaWxlLm9wZW4ocGF0aF9mb3JfdGFyLCBtb2RlPSJyOioiKSBhcyB0ZjoKICAgICAgICBmb3IgbSBpbiB0Zi5nZXRtZW1iZXJzKCk6CiAgICAgICAgICAgIGlmIG5vdCBtLmlzZmlsZSgpOgogICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgcmVsID0gbS5uYW1lLmxzdHJpcCgiLi8iKQogICAgICAgICAgICBpZiBub3QgX3dhbnRfcmVsKHJlbCk6CiAgICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgICAgICB0Zi5leHRyYWN0KG0sIHBhdGg9ZGVzdCwgZmlsdGVyPSJkYXRhIikKICAgICAgICAgICAgbiArPSAxCiAgICBpZiBwYXRoX2Zvcl90YXIgIT0gdGFyX3BhdGggYW5kIHBhdGhfZm9yX3Rhci5leGlzdHMoKToKICAgICAgICBwYXRoX2Zvcl90YXIudW5saW5rKG1pc3Npbmdfb2s9VHJ1ZSkKICAgIHJldHVybiBuCgoKZGVmIHN5c19leGVjdXRhYmxlKCkgLT4gc3RyOgogICAgaW1wb3J0IHN5cwoKICAgIHJldHVybiBzeXMuZXhlY3V0YWJsZQoKCmRlZiBfd2FudF9yZWwocmVsOiBzdHIpIC0+IGJvb2w6CiAgICBpZiBvcy5lbnZpcm9uLmdldCgiUElYR0JNX1JFQlVJTERfQUxMIiwgIiIpLnN0cmlwKCkgaW4gKCIxIiwgInRydWUiLCAieWVzIik6CiAgICAgICAgcmV0dXJuIFRydWUKICAgIHJldHVybiBfaXNfZ2VvX3JlbChyZWwpCgoKZGVmIHJlc29sdmVfYmxvYihzaGFyZDogUGF0aCwgYmxvYjogc3RyKSAtPiBQYXRoIHwgTm9uZToKICAgICIiIkZpbmQgZiMjIyMgb3IgZiMjIyMuZXh0IGluc2lkZSBzaGFyZCBkaXJlY3RvcnkuIiIiCiAgICBiID0gUGF0aChibG9iKS5uYW1lCiAgICBkaXJlY3QgPSBzaGFyZCAvIGIKICAgIGlmIGRpcmVjdC5pc19maWxlKCk6CiAgICAgICAgcmV0dXJuIGRpcmVjdAogICAgc3RlbSA9IGIuc3BsaXQoIi4iKVswXQogICAgIyB0cnkgZXhhY3Qgc3RlbSB3aXRoIGFueSBzdWZmaXgKICAgIGhpdHMgPSBsaXN0KHNoYXJkLmdsb2Ioc3RlbSArICIuKiIpKSArIGxpc3Qoc2hhcmQuZ2xvYihzdGVtKSkKICAgIGZvciBoIGluIGhpdHM6CiAgICAgICAgaWYgaC5pc19maWxlKCk6CiAgICAgICAgICAgIHJldHVybiBoCiAgICAjIG5lc3RlZCBvbmUgbGV2ZWwKICAgIGZvciBoIGluIHNoYXJkLmdsb2IoIiovIiArIHN0ZW0gKyAiLioiKToKICAgICAgICBpZiBoLmlzX2ZpbGUoKToKICAgICAgICAgICAgcmV0dXJuIGgKICAgIHJldHVybiBOb25lCgoKZGVmIGxpbmtfb3JfY29weShzcmM6IFBhdGgsIGRzdDogUGF0aCkgLT4gc3RyOgogICAgZHN0LnBhcmVudC5ta2RpcihwYXJlbnRzPVRydWUsIGV4aXN0X29rPVRydWUpCiAgICBpZiBkc3QuZXhpc3RzKCkgb3IgZHN0LmlzX3N5bWxpbmsoKToKICAgICAgICByZXR1cm4gImV4aXN0cyIKICAgIHRyeToKICAgICAgICBvcy5zeW1saW5rKHNyYywgZHN0KQogICAgICAgIHJldHVybiAic3ltbGluayIKICAgIGV4Y2VwdCBPU0Vycm9yOgogICAgICAgIHNodXRpbC5jb3B5MihzcmMsIGRzdCkKICAgICAgICByZXR1cm4gImNvcHkiCgoKZGVmIG1hdGVyaWFsaXplX3NoYXJkX2Jsb2JzKHNoYXJkOiBQYXRoLCB0cmVlOiBQYXRoLCBtYW5pZmVzdF9wYXRoOiBQYXRoIHwgTm9uZSkgLT4gZGljdDoKICAgIHN0YXRzID0geyJzaGFyZCI6IHNoYXJkLm5hbWUsICJsaW5rZWQiOiAwLCAiY29waWVkIjogMCwgImV4aXN0cyI6IDAsICJtaXNzaW5nX2Jsb2IiOiAwLCAic2tpcHBlZF9ub25fZ2VvIjogMCwgIm1hcHBlZCI6IDB9CiAgICBpZiBtYW5pZmVzdF9wYXRoIGlzIE5vbmU6CiAgICAgICAgc3RhdHNbImVycm9yIl0gPSAibm9fbWFuaWZlc3QiCiAgICAgICAgcmV0dXJuIHN0YXRzCiAgICBtYW4gPSBfbG9hZF9qc29uKG1hbmlmZXN0X3BhdGgpCiAgICBtYXBwaW5ncyA9IGxpc3QoaXRlcl9ibG9iX21hcHBpbmdzKG1hbikpCiAgICBzdGF0c1sibWFwcGVkIl0gPSBsZW4obWFwcGluZ3MpCiAgICBpZiBub3QgbWFwcGluZ3M6CiAgICAgICAgc3RhdHNbImVycm9yIl0gPSAibm9fYmxvYl9tYXBwaW5nc19pbl9tYW5pZmVzdCIKICAgICAgICByZXR1cm4gc3RhdHMKICAgIGZvciBibG9iLCByZWwgaW4gbWFwcGluZ3M6CiAgICAgICAgaWYgbm90IF93YW50X3JlbChyZWwpOgogICAgICAgICAgICBzdGF0c1sic2tpcHBlZF9ub25fZ2VvIl0gKz0gMQogICAgICAgICAgICBjb250aW51ZQogICAgICAgIHNyYyA9IHJlc29sdmVfYmxvYihzaGFyZCwgYmxvYikKICAgICAgICBpZiBzcmMgaXMgTm9uZToKICAgICAgICAgICAgIyBibG9iIGlkIHdpdGhvdXQgZXh0ZW5zaW9uIOKAlCB0cnkgYXMtaXMKICAgICAgICAgICAgc3JjID0gcmVzb2x2ZV9ibG9iKHNoYXJkLCBibG9iLnNwbGl0KCIuIilbMF0pCiAgICAgICAgaWYgc3JjIGlzIE5vbmU6CiAgICAgICAgICAgIHN0YXRzWyJtaXNzaW5nX2Jsb2IiXSArPSAxCiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgIyBwcmVzZXJ2ZSBleHRlbnNpb24gZnJvbSBhY3R1YWwgZmlsZSBpZiByZWwgaGFzIG5vbmUKICAgICAgICByZWxfcGF0aCA9IHJlbC5yZXBsYWNlKCJcXCIsICIvIikKICAgICAgICBpZiBub3QgUGF0aChyZWxfcGF0aCkuc3VmZml4IGFuZCBzcmMuc3VmZml4OgogICAgICAgICAgICByZWxfcGF0aCA9IHJlbF9wYXRoICsgc3JjLnN1ZmZpeAogICAgICAgIGFjdGlvbiA9IGxpbmtfb3JfY29weShzcmMsIHRyZWUgLyByZWxfcGF0aCkKICAgICAgICBzdGF0c1thY3Rpb25dID0gc3RhdHMuZ2V0KGFjdGlvbiwgMCkgKyAxCiAgICByZXR1cm4gc3RhdHMKCgpkZWYgb3ZlcmxheV9ndChndF9yb290OiBQYXRoLCB0cmVlOiBQYXRoKSAtPiBkaWN0OgogICAgIiIiQmVzdC1lZmZvcnQ6IGNvcHkgZ2VvanNvbi9hbm5vdGF0aW9ucyBmcm9tIGd0IGRhdGFzZXQgaW50byB0cmVlIGlmIG1pc3NpbmcuIiIiCiAgICBzdGF0cyA9IHsiZ3QiOiBndF9yb290Lm5hbWUsICJjb3BpZWQiOiAwLCAic2tpcHBlZCI6IDB9CiAgICAjIElmIGRhdGFzZXQgaXMgYSBzaW5nbGUgZ2VvanNvbiDigJQgbGVhdmUgZm9yIG90aGVyIHBpcGVsaW5lczsgUElYR0JNIHVzZXMgcGVyLXJlZ2lvbiBmaWxlcy4KICAgIGZvciBwIGluIGd0X3Jvb3Qucmdsb2IoIioiKToKICAgICAgICBpZiBub3QgcC5pc19maWxlKCk6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgdHJ5OgogICAgICAgICAgICByZWwgPSBwLnJlbGF0aXZlX3RvKGd0X3Jvb3QpLmFzX3Bvc2l4KCkKICAgICAgICBleGNlcHQgVmFsdWVFcnJvcjoKICAgICAgICAgICAgY29udGludWUKICAgICAgICBpZiBub3QgX3dhbnRfcmVsKHJlbCkgYW5kIG5vdCByZWwuZW5kc3dpdGgoIi5nZW9qc29uIik6CiAgICAgICAgICAgICMgYWxsb3cgZ2VvanNvbiBvdmVybGF5cyBvbmx5IHVuZGVyIGdlbyByZWdpb25zCiAgICAgICAgICAgIGlmIG5vdCBhbnkocGFydC5zdGFydHN3aXRoKEdFT19QUkVGSVhFUykgZm9yIHBhcnQgaW4gUGF0aChyZWwpLnBhcnRzKToKICAgICAgICAgICAgICAgIHN0YXRzWyJza2lwcGVkIl0gKz0gMQogICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICBpZiBub3QgX3dhbnRfcmVsKHJlbCk6CiAgICAgICAgICAgIHN0YXRzWyJza2lwcGVkIl0gKz0gMQogICAgICAgICAgICBjb250aW51ZQogICAgICAgIGRzdCA9IHRyZWUgLyByZWwKICAgICAgICBpZiBkc3QuZXhpc3RzKCk6CiAgICAgICAgICAgIHN0YXRzWyJza2lwcGVkIl0gKz0gMQogICAgICAgICAgICBjb250aW51ZQogICAgICAgIGRzdC5wYXJlbnQubWtkaXIocGFyZW50cz1UcnVlLCBleGlzdF9vaz1UcnVlKQogICAgICAgIHNodXRpbC5jb3B5MihwLCBkc3QpCiAgICAgICAgc3RhdHNbImNvcGllZCJdICs9IDEKICAgIHJldHVybiBzdGF0cwoKCmRlZiByZWJ1aWxkX3RyYWluX3RyZWUoCiAgICBkZXN0OiBQYXRoIHwgTm9uZSA9IE5vbmUsCiAgICBpbnB1dF9yb290OiBQYXRoIHwgTm9uZSA9IE5vbmUsCikgLT4gZGljdDoKICAgICIiIkJ1aWxkIHRyYWluIHJvb3QgdW5kZXIgZGVzdDsgcmV0dXJuIHN0YXRzIGRpY3QgaW5jbHVkaW5nICdyb290JyBwYXRoLiIiIgogICAgZGVzdCA9IFBhdGgoZGVzdCBvciBvcy5lbnZpcm9uLmdldCgiUElYR0JNX1RSRUUiLCAiL2thZ2dsZS93b3JraW5nL3RyYWluLWV4cGRzLTMiKSkKICAgIGRhdGFzZXRzID0gZmluZF9pbnB1dF9kYXRhc2V0cyhpbnB1dF9yb290KQogICAgc2hhcmRzID0gZmluZF94ZmVyX3NoYXJkcyhkYXRhc2V0cykKICAgIHJlcG9ydDogZGljdFtzdHIsIEFueV0gPSB7CiAgICAgICAgImRlc3QiOiBzdHIoZGVzdCksCiAgICAgICAgInNoYXJkc19mb3VuZCI6IFtmInN7aWR4OjAzZH06e3BhdGgubmFtZX0iIGZvciBpZHgsIHBhdGggaW4gc2hhcmRzXSwKICAgICAgICAic3RlcHMiOiBbXSwKICAgIH0KICAgIGlmIG5vdCBzaGFyZHM6CiAgICAgICAgcmFpc2UgRmlsZU5vdEZvdW5kRXJyb3IoCiAgICAgICAgICAgICJObyBrb3ozLXhmZXItc05OTiBkYXRhc2V0cyB1bmRlciAva2FnZ2xlL2lucHV0LiAiCiAgICAgICAgICAgICJBdHRhY2ggZG1pdHJpamt2L2tvejMteGZlci1zMDAwIOKApiBzMDA5LiIKICAgICAgICApCgogICAgZGVzdC5ta2RpcihwYXJlbnRzPVRydWUsIGV4aXN0X29rPVRydWUpCiAgICAjIHMwMDAgc21hbGwgdGFyIGZpcnN0CiAgICBmb3IgaWR4LCBwYXRoIGluIHNoYXJkczoKICAgICAgICBpZiBpZHggIT0gMDoKICAgICAgICAgICAgY29udGludWUKICAgICAgICB0YXIgPSBmaW5kX3NtYWxsX3RhcihwYXRoKQogICAgICAgIGlmIHRhciBpcyBOb25lOgogICAgICAgICAgICByZXBvcnRbInN0ZXBzIl0uYXBwZW5kKHsiczAwMCI6IHBhdGgubmFtZSwgImVycm9yIjogInNtYWxsX2ZpbGVzLnRhciogbWlzc2luZyJ9KQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIG4gPSBleHRyYWN0X3NtYWxsX3Rhcih0YXIsIGRlc3QpCiAgICAgICAgICAgIHJlcG9ydFsic3RlcHMiXS5hcHBlbmQoeyJzMDAwX2V4dHJhY3QiOiB0YXIubmFtZSwgIm1lbWJlcnNfZ2VvIjogbn0pCgogICAgIyBsYXJnZSBzaGFyZHMgKCsgczAwMCBpZiBpdCBhbHNvIGhhcyBibG9iIG1hcHMpCiAgICBmb3IgaWR4LCBwYXRoIGluIHNoYXJkczoKICAgICAgICBtYW4gPSBmaW5kX21hbmlmZXN0KHBhdGgpCiAgICAgICAgaWYgaWR4ID09IDAgYW5kIG1hbiBpcyBOb25lOgogICAgICAgICAgICAjIG9wdGlvbmFsOiBzaGFyZWQgY29weSBwYXRoIG5hbWUKICAgICAgICAgICAgY29udGludWUKICAgICAgICBzdCA9IG1hdGVyaWFsaXplX3NoYXJkX2Jsb2JzKHBhdGgsIGRlc3QsIG1hbikKICAgICAgICByZXBvcnRbInN0ZXBzIl0uYXBwZW5kKHN0KQoKICAgIGd0ID0gZmluZF9ndF9kYXRhc2V0KGRhdGFzZXRzKQogICAgaWYgZ3QgaXMgbm90IE5vbmU6CiAgICAgICAgcmVwb3J0WyJzdGVwcyJdLmFwcGVuZChvdmVybGF5X2d0KGd0LCBkZXN0KSkKICAgIGVsc2U6CiAgICAgICAgcmVwb3J0WyJzdGVwcyJdLmFwcGVuZCh7Imd0IjogTm9uZSwgIm5vdGUiOiAia296My1ndC10cmFpbi12MiBub3QgYXR0YWNoZWQifSkKCiAgICAjIHNhbml0eTogY291bnQgZ2VvIHJlZ2lvbiBkaXJzICgyMDDigJMyMTcpIGFuZCBhbnkgcmVnaW9uIGRpcnMKICAgIGdlb19kaXJzID0gc29ydGVkKAogICAgICAgIHAubmFtZSBmb3IgcCBpbiBkZXN0Lml0ZXJkaXIoKSBpZiBwLmlzX2RpcigpIGFuZCBhbnkocC5uYW1lLnN0YXJ0c3dpdGgocHJlZikgZm9yIHByZWYgaW4gR0VPX1BSRUZJWEVTKQogICAgKQogICAgYW55X3JlZ2lvbl9kaXJzID0gc29ydGVkKAogICAgICAgIHAubmFtZSBmb3IgcCBpbiBkZXN0Lml0ZXJkaXIoKSBpZiBwLmlzX2RpcigpIGFuZCBsZW4ocC5uYW1lKSA+PSA0IGFuZCBwLm5hbWVbOjNdLmlzZGlnaXQoKQogICAgKQogICAgcmVwb3J0WyJnZW9fcmVnaW9uX2RpcnMiXSA9IGdlb19kaXJzCiAgICByZXBvcnRbImFueV9yZWdpb25fZGlyc19uIl0gPSBsZW4oYW55X3JlZ2lvbl9kaXJzKQogICAgcmVwb3J0WyJhbnlfcmVnaW9uX2RpcnNfc2FtcGxlIl0gPSBhbnlfcmVnaW9uX2RpcnNbOjQwXQogICAgcmVwb3J0WyJyb290Il0gPSBzdHIoZGVzdCkKICAgIHJlYnVpbGRfYWxsID0gb3MuZW52aXJvbi5nZXQoIlBJWEdCTV9SRUJVSUxEX0FMTCIsICIiKS5zdHJpcCgpIGluICgiMSIsICJ0cnVlIiwgInllcyIpCiAgICBpZiBub3QgZ2VvX2RpcnM6CiAgICAgICAgaWYgcmVidWlsZF9hbGwgYW5kIGFueV9yZWdpb25fZGlyczoKICAgICAgICAgICAgIyBob25lc3QxMCAvIGZ1bGwgdHJlZTogZ2VvcyAx4oCTMzMgcmFzdGVycyBsaXZlIG91dHNpZGUgMjAw4oCTMjE3OyAyMDDigJMyMTcgb2Z0ZW4gaG9sbG93CiAgICAgICAgICAgIHJlcG9ydFsibm90ZSJdID0gKAogICAgICAgICAgICAgICAgIlBJWEdCTV9SRUJVSUxEX0FMTD0xOiBubyBnZW8gMjAw4oCTMjE3IGRpcnMgd2l0aCBjb250ZW50IGdhdGUgc2tpcHBlZDsgIgogICAgICAgICAgICAgICAgZiJ1c2luZyB7bGVuKGFueV9yZWdpb25fZGlycyl9IHJlZ2lvbiBkaXJzIgogICAgICAgICAgICApCiAgICAgICAgZWxzZToKICAgICAgICAgICAgcmFpc2UgRmlsZU5vdEZvdW5kRXJyb3IoCiAgICAgICAgICAgICAgICBmIlJlYnVpbGQgcHJvZHVjZWQgbm8gZ2VvIDIwMOKAkzIxNyBkaXJzIHVuZGVyIHtkZXN0fS4gIgogICAgICAgICAgICAgICAgIkNoZWNrIHNoYXJkIG1hbmlmZXN0cyAvIHNtYWxsX2ZpbGVzIGV4dHJhY3QuIgogICAgICAgICAgICApCiAgICByZXR1cm4gcmVwb3J0CgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIGltcG9ydCBwcHJpbnQKCiAgICBwcHJpbnQucHAocmVidWlsZF90cmFpbl90cmVlKCkpCg=="""))

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
        Path("/kaggle/working/oof_kaggle_jobs.py"),
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



def ensure_train_tree() -> Path:
    """Rebuild /kaggle/working/train-expds-3 from attached xfer shards + gt-train-v2.

    Uses sibling rebuild_train_tree.py (bot-6). Sets PIXGBM_REBUILD_ALL=1 so geos
    outside 200–217 (li/ae/or/sp_or on 001–033) are materialized — required for
    honest-10 OOF. Known gap: geo 200–217 rasters are hollow in xfer shards.
    """
    syn = Path("/kaggle/working/train-expds-3")
    # reuse existing rebuild if it already has region dirs
    if syn.is_dir() and any(p.is_dir() and p.name[:3].isdigit() for p in syn.iterdir()):
        log("reuse existing train tree", syn)
        return syn
    os.environ.setdefault("PIXGBM_REBUILD_ALL", "1")
    # load sibling module shipped with this kernel push
    candidates = [
        Path(__file__).resolve().parent / "rebuild_train_tree.py",
        Path("/kaggle/working/rebuild_train_tree.py"),
        Path("rebuild_train_tree.py"),
    ]
    mod_path = next((c for c in candidates if c.is_file()), None)
    if mod_path is None:
        raise FileNotFoundError(
            "rebuild_train_tree.py не найден рядом с ядром; "
            "прикрепите xfer shards и перезалейте kernel package"
        )
    spec = importlib.util.spec_from_file_location("rebuild_train_tree", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    log("rebuild train tree via", mod_path, "PIXGBM_REBUILD_ALL=", os.environ.get("PIXGBM_REBUILD_ALL"))
    report = mod.rebuild_train_tree(dest=syn)
    (Path("/kaggle/working") / "rebuild_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(
        "rebuild done regions≈",
        report.get("any_region_dirs_n"),
        "geo200-217=",
        len(report.get("geo_region_dirs") or []),
        "shards=",
        report.get("shards_found"),
    )
    return Path(report["root"])



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
        # Prefer xfer-shard rebuild when koz3-xfer-s* attached (acc2 = collab reader).
        # OOF_S3_SYNC=1 still allowed as override for selective bucket sync.
        xfer_attached = any(
            p.is_dir() and "koz3-xfer-s" in p.name.lower()
            for p in Path("/kaggle/input").glob("*")
        ) if Path("/kaggle/input").is_dir() else False
        if os.environ.get("OOF_S3_SYNC", "").strip() in ("1", "true", "yes"):
            root = maybe_sync_train(Path("/kaggle/working/train-expds-3"))
        elif xfer_attached:
            log("xfer shards attached — rebuild train-expds-3")
            root = ensure_train_tree()
        else:
            try:
                root = find_train_root()
            except FileNotFoundError:
                log("train root missing — trying rebuild then S3")
                try:
                    root = ensure_train_tree()
                except Exception as e:
                    raise FileNotFoundError(
                        "train-expds-3 отсутствует и rebuild из xfer не удался: "
                        f"{type(e).__name__}: {e}"
                    ) from e
        log("train", root)

        # Staging / CPU push: keep source+datasets updated while acc2 GPU 2/2 is full.
        # Full OOF needs CUDA; set OOF_FORCE_CPU=1 only for intentional CPU runs.
        try:
            import torch
            has_cuda = bool(torch.cuda.is_available())
        except Exception:
            has_cuda = False
        force_cpu = os.environ.get("OOF_FORCE_CPU", "").strip() in ("1", "true", "yes")
        if not has_cuda and not force_cpu:
            n_reg = sum(
                1 for x in Path(root).iterdir()
                if x.is_dir() and len(x.name) >= 4 and x.name[:3].isdigit()
            ) if Path(root).is_dir() else 0
            lib.write_status(
                OUT, "DONE", phase="staged_no_gpu", mode=MODE,
                train=str(root), n_regions=n_reg,
                note="source+xfer rebuild OK; re-push with GPU when slot free",
            )
            log("staged_no_gpu — rebuild OK, regions≈", n_reg, "; wait for GPU slot")
            return 0

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
