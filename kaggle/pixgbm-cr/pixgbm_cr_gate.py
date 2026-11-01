#!/usr/bin/env python3
"""Kaggle CPU: PIXGBM e2e geo F1 gate on regions 200–217 (bot-6 / pixgbm-cr).

REF  = dmitrijkv/koz3-fr1h-chk-20261101 → fr1h.zip.bin (v2 eb268d6 + FR1h).
       FR1h is forced OFF before run: remove [post.frame_prior] and restore
       archive SpOr kurgan thresholds (taus_base). geo_pixgbm.py / models/pixgbm
       are byte-identical to v2 (sha 35e4331b… / d1fe3efa… / 5fa8c09d…).
CAND = dmitrijkv/koz3-pixgbm-cr-cand → zip OR Kaggle-auto-extracted tree (pix_clean).
Helpers/rebuild are EMBEDDED in this code_file (Kaggle uploads only code_file).

Geo tree: MagGr/EtGr/TerFl + GT for 200–217 from S3 (rebuild_w4_geo_tree.py).
Gate: helpers/bot6_r3_gate.py (|dF1|≤0.01 ALL + maggr/etgr/terfl; INVALID checks).

r3 overrides on both: EXPDS_PIXGBM_STRICT=1, PIXGBM_BUDGET_S=100000,
EXPDS_SOFT_CEILING_S=0, budget_s=100000, watchdog=false.

enable_gpu=false, enable_internet=true. Prefer offline wheels; network pip OK.
Never submits to LB. «1» pushes. Do NOT commit EN code/weights into git.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

T0 = time.time()
OUT = Path(os.environ.get("PIXGBM_CR_OUT", "/kaggle/working/pixgbm_cr"))
TREE = Path(os.environ.get("PIXGBM_TREE", "/kaggle/working/train-expds-3"))
INPUT_ROOT = Path(os.environ.get("PIXGBM_CR_INPUT", "/kaggle/input"))
WORKING_ROOT = Path(os.environ.get("PIXGBM_CR_WORKING", "/kaggle/working"))
GEO_PIX_SHA = "35e4331bbc690e7eb3b31e07cc290837ce9fd96a05c92f4c999f24eceafda35b"
REF_POLY = "d1fe3efafcd4377d41430d8828e881576c9024f9d7a5c89f55e23554e02ec800"
REF_FINDS = "5fa8c09d77d0914cc925fb6bb925a0ab1b64ccecf3c823d7f3546e71c4ceb04a"
FR1H_ZIP_SHA = "fe8d60b9032474760f02d4bcfa08fb43269ae3ffb872efa932d2f1d7692944b4"
# Archive eb268d6 SpOr kurgan floors (FR1H_PACK taus_base / pack table).
ARCHIVE_SP_TAUS = {
    "sp_or": {"kurgany_povrezhdennye": 0.3645, "kurgany_tselye": 0.3725},
    "sp_focal2": {"kurgany_povrezhdennye": 0.501645, "kurgany_tselye": 0.645505},
}
TOL = float(os.environ.get("PIXGBM_DF1_TOL", "0.01"))
GEO_PREFIXES = tuple(f"{n:03d}_" for n in range(200, 218))

# --- BEGIN EMBEDDED_HELPERS (sha256 of decoded bytes) ---
# Materialized to /kaggle/working/_pixgbm_cr_pkg/ at startup.
# Kaggle kernels push uploads ONLY code_file; siblings are unavailable.
_EMBEDDED_HELPERS: dict[str, dict[str, str]] = {
    'helpers/bot6_r3_gate.py': {
        "sha256": '5e4f9aed01b3687e9c1b3b306a26cb059070c25edb69117b6670648e309ea554',
        "b64": (
            'IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiJib3QtNiByMyBnYXRlOiB2YWxpZGl0eSBvZiBSRUYv'
            'TkVXIHBpeGdibSBnZW8gcnVucyArIGVxdWl2YWxlbmNlIHxkRjF8PD0wLjAxLgoKRXhpdCBjb2Rl'
            'cyAoZGlzdGluY3QgSU5WQUxJRCk6CiAgMCAgUEFTUwogIDkgIGVxdWl2YWxlbmNlIGZhaWwgKHxk'
            'RjF8PjAuMDEgb24gQUxMIG9yIG1hZ2dyL2V0Z3IvdGVyZmwpCiAxMCAgaW4tY29udGFpbmVyIHNo'
            'YSBtaXNtYXRjaCAoY2FsbGVyIG1heSBhbHNvIGV4aXQgMTAgZWFybGllcikKIDExICBwaXggbm90'
            'IGFjdHVhbGx5IHJ1biAoZW1wdHkgc3RlcHMgLyBubyBmdWxsIC8gbWlzc2luZyBtb2R1bGUgLyBz'
            'a2lwcGVkKQogMTIgIGRvY2tlciBleGl0ICE9IDAgKHJlY29yZGVkKQogMTMgIGdlb2pzb24gbXRp'
            'bWUgYmVmb3JlIGNvbnRhaW5lciBzdGFydAogMTQgIHJlc3VsdCBnZW9qc29ucyBieXRlLWlkZW50'
            'aWNhbCB3aGlsZSBtb2RlbCBzaGFzIGRpZmZlciAoc3VzcGVjdCkKIDE5ICBzY29yZXMuanNvbiBt'
            'aXNzaW5nL3VucGFyc2VkCiIiIgpmcm9tIF9fZnV0dXJlX18gaW1wb3J0IGFubm90YXRpb25zCgpp'
            'bXBvcnQgYXJncGFyc2UKaW1wb3J0IGpzb24KaW1wb3J0IHJlCmltcG9ydCBzeXMKZnJvbSBwYXRo'
            'bGliIGltcG9ydCBQYXRoCgojIEN5cmlsbGljIHBhdHRlcm5zIGFzIHVuaWNvZGUgZXNjYXBlcyAo'
            'aGVscGVycyBzdGF5IEFTQ0lJLXNvdXJjZSBmcmllbmRseSkuClJFX01JU1NJTkdfTU9EVUxFID0g'
            'cmUuY29tcGlsZSgKICAgICJcdTA0M2RcdTA0MzVcdTA0NDIgXHUwNDMyIFx1MDQ0ZFx1MDQ0Mlx1'
            'MDQzZVx1MDQzOSBcdTA0NDFcdTA0MzFcdTA0M2VcdTA0NDBcdTA0M2FcdTA0MzUiICAjIG5ldCB2'
            'IGV0b2ogc2JvcmtlCikKUkVfUElYX1NLSVBfUkVHSU9OID0gcmUuY29tcGlsZSgKICAgIHIicGl4'
            'Z2JtIFx1MDQzZlx1MDQ0MFx1MDQzZVx1MDQzZlx1MDQ0M1x1MDQ0OVx1MDQzNVx1MDQzZCg/IVxz'
            'Klx1MjAxNCkiICAjIHByb3B1c2NoZW4gKG5vdCBlbWRhc2ggY2xhc3NpYykKKQojICJwaXhnYm0g'
            'PHJlZ2lvbj46IC4uLiBzdHVwZW5pIHsuLi59OyBmYXphIFguWCBzIgpSRV9QSVhfTElORSA9IHJl'
            'LmNvbXBpbGUoCiAgICByInBpeGdibVxzKyhcUyspOlxzK2JhY2tlbmQ9KFxTKylccytbXlxyXG5d'
            'Kj8iCiAgICByIlx1MDQ0MVx1MDQ0Mlx1MDQ0M1x1MDQzZlx1MDQzNVx1MDQzZFx1MDQzOFxzKyhc'
            'e1teXH1dKlx9KSIgICMgc3R1cGVuaQogICAgciJbXlxyXG5dKj8iCiAgICByIlx1MDQ0NFx1MDQz'
            'MFx1MDQzN1x1MDQzMFxzKyhbMC05Ll0rKVxzKlx1MDQ0MSIsICAjIGZhemEgTiBzCiAgICByZS5J'
            'R05PUkVDQVNFLAopClJFX1NURVBTX0VNUFRZID0gcmUuY29tcGlsZShyIlx1MDQ0MVx1MDQ0Mlx1'
            'MDQ0M1x1MDQzZlx1MDQzNVx1MDQzZFx1MDQzOFxzK1x7XH0iKQpSRV9TVEVQU19GVUxMID0gcmUu'
            'Y29tcGlsZShyIlsnXCJdZnVsbFsnXCJdXHMqOiIpCgpHRU9fTEFZRVJTID0gKCJtYWdnciIsICJl'
            'dGdyIiwgInRlcmZsIikKVE9MID0gMC4wMQoKCmRlZiBfcmVhZChwYXRoOiBQYXRoKSAtPiBzdHI6'
            'CiAgICBpZiBub3QgcGF0aCBvciBub3QgUGF0aChwYXRoKS5pc19maWxlKCk6CiAgICAgICAgcmV0'
            'dXJuICIiCiAgICByZXR1cm4gUGF0aChwYXRoKS5yZWFkX3RleHQoZW5jb2Rpbmc9InV0Zi04Iiwg'
            'ZXJyb3JzPSJyZXBsYWNlIikKCgpkZWYgcGFyc2VfcGl4X2xvZ3MobG9nX3RleHQ6IHN0cikgLT4g'
            'ZGljdDoKICAgIGxpbmVzID0gUkVfUElYX0xJTkUuZmluZGFsbChsb2dfdGV4dCkKICAgIGVtcHR5'
            'ID0gUkVfU1RFUFNfRU1QVFkuZmluZGFsbChsb2dfdGV4dCkKICAgIGZ1bGxfaGl0cyA9IFJFX1NU'
            'RVBTX0ZVTEwuZmluZGFsbChsb2dfdGV4dCkKICAgIG1pc3NpbmcgPSBSRV9NSVNTSU5HX01PRFVM'
            'RS5maW5kYWxsKGxvZ190ZXh0KQogICAgIyByZWdpb24tbGV2ZWwgc2tpcDogInBpeGdibSBwcm9w'
            'dXNjaGVuIiBidXQgbm90IHRoZSBwZXItcmFzdGVyICJwcm9wdXNjaGVuIOKAlCBjbGFzc2ljIgog'
            'ICAgc2tpcHBlZCA9IFtdCiAgICBmb3IgbSBpbiByZS5maW5kaXRlcihyIlteXHJcbl0qcGl4Z2Jt'
            'W15cclxuXSoiLCBsb2dfdGV4dCk6CiAgICAgICAgbGluZSA9IG0uZ3JvdXAoMCkKICAgICAgICBp'
            'ZiBSRV9QSVhfU0tJUF9SRUdJT04uc2VhcmNoKGxpbmUpIGFuZCAiXHUyMDE0IiBub3QgaW4gbGlu'
            'ZSBhbmQgImNsYXNzaWMiIG5vdCBpbiBsaW5lLmxvd2VyKCk6CiAgICAgICAgICAgICMgYWxzbyBh'
            'bGxvdyBBU0NJSSAnY2xhc3NpYycgcGF0aDsgQ3lyaWxsaWMgZW1kYXNoIGZvcm0gaXMgcGVyLXJh'
            'c3RlcgogICAgICAgICAgICBpZiBSRV9QSVhfU0tJUF9SRUdJT04uc2VhcmNoKGxpbmUpOgogICAg'
            'ICAgICAgICAgICAgc2tpcHBlZC5hcHBlbmQobGluZS5zdHJpcCgpKQogICAgIyBCcm9hZGVyOiBl'
            'eGFjdCBwaXBlbGluZSBtZXNzYWdlICIlczogcGl4Z2JtIHByb3B1c2NoZW4iCiAgICBmb3IgbSBp'
            'biByZS5maW5kaXRlcigKICAgICAgICByIig/bSleLio/cGl4Z2JtIFx1MDQzZlx1MDQ0MFx1MDQz'
            'ZVx1MDQzZlx1MDQ0M1x1MDQ0OVx1MDQzNVx1MDQzZFxzKiQiLCBsb2dfdGV4dAogICAgKToKICAg'
            'ICAgICBza2lwcGVkLmFwcGVuZChtLmdyb3VwKDApLnN0cmlwKCkpCgogICAgcGhhc2VfcyA9IDAu'
            'MAogICAgc3RlcHNfY291bnRzOiBkaWN0W3N0ciwgaW50XSA9IHt9CiAgICBwZXJfcmVnaW9uID0g'
            'W10KICAgIGZvciByZWcsIGJhY2tlbmQsIHN0ZXBzX3MsIHBoYXNlIGluIGxpbmVzOgogICAgICAg'
            'IHRyeToKICAgICAgICAgICAgc3RlcHMgPSBldmFsKHN0ZXBzX3MsIHsiX19idWlsdGluc19fIjog'
            'e319LCB7fSkgICMgc3RlcHMgZGljdCBmcm9tIGxvZyByZXByCiAgICAgICAgZXhjZXB0IEV4Y2Vw'
            'dGlvbjoKICAgICAgICAgICAgc3RlcHMgPSB7Il9yYXciOiBzdGVwc19zfQogICAgICAgIGlmIGlz'
            'aW5zdGFuY2Uoc3RlcHMsIGRpY3QpOgogICAgICAgICAgICBmb3IgaywgdiBpbiBzdGVwcy5pdGVt'
            'cygpOgogICAgICAgICAgICAgICAgaWYgaXNpbnN0YW5jZSh2LCBpbnQpOgogICAgICAgICAgICAg'
            'ICAgICAgIHN0ZXBzX2NvdW50c1trXSA9IHN0ZXBzX2NvdW50cy5nZXQoaywgMCkgKyB2CiAgICAg'
            'ICAgdHJ5OgogICAgICAgICAgICBwaGFzZV9zID0gbWF4KHBoYXNlX3MsIGZsb2F0KHBoYXNlKSkg'
            'ICMgUEhBU0Ugc3BlbnQgaXMgY3VtdWxhdGl2ZTsgdGFrZSBtYXggc2VlbgogICAgICAgIGV4Y2Vw'
            'dCBWYWx1ZUVycm9yOgogICAgICAgICAgICBwYXNzCiAgICAgICAgcGVyX3JlZ2lvbi5hcHBlbmQo'
            'eyJyZWdpb24iOiByZWcsICJiYWNrZW5kIjogYmFja2VuZCwgInN0ZXBzIjogc3RlcHMsICJwaGFz'
            'ZV9zIjogZmxvYXQocGhhc2UpfSkKCiAgICAjIFByZWZlciBsYXN0IGN1bXVsYXRpdmUgcGhhc2Ug'
            'aWYgcHJlc2VudAogICAgaWYgcGVyX3JlZ2lvbjoKICAgICAgICBwaGFzZV9zID0gZmxvYXQocGVy'
            'X3JlZ2lvblstMV1bInBoYXNlX3MiXSkKCiAgICByZXR1cm4gewogICAgICAgICJuX3BpeF9saW5l'
            'cyI6IGxlbihsaW5lcyksCiAgICAgICAgIm5fZW1wdHlfc3RlcHMiOiBsZW4oZW1wdHkpLAogICAg'
            'ICAgICJuX2Z1bGxfaGl0cyI6IGxlbihmdWxsX2hpdHMpLAogICAgICAgICJtaXNzaW5nX21vZHVs'
            'ZSI6IGxlbihtaXNzaW5nKSwKICAgICAgICAicmVnaW9uX3NraXBzIjogc2tpcHBlZFs6MjBdLAog'
            'ICAgICAgICJuX3JlZ2lvbl9za2lwcyI6IGxlbihza2lwcGVkKSwKICAgICAgICAicGhhc2Vfc2Vj'
            'b25kcyI6IHBoYXNlX3MsCiAgICAgICAgInN0ZXBzX2NvdW50cyI6IHN0ZXBzX2NvdW50cywKICAg'
            'ICAgICAicGVyX3JlZ2lvbiI6IHBlcl9yZWdpb24sCiAgICB9CgoKZGVmIHZhbGlkaXR5X2Zvcl9y'
            'dW4obmFtZTogc3RyLCBsb2dfdGV4dDogc3RyLCBkb2NrZXJfZWM6IGludCB8IE5vbmUsCiAgICAg'
            'ICAgICAgICAgICAgICAgIGdlb2pzb246IFBhdGggfCBOb25lLCB0X3N0YXJ0OiBmbG9hdCB8IE5v'
            'bmUsCiAgICAgICAgICAgICAgICAgICAgIGluY29udGFpbmVyX29rOiBib29sKSAtPiBsaXN0W3R1'
            'cGxlW2ludCwgc3RyXV06CiAgICBmYWlscyA9IFtdCiAgICBpZiBub3QgaW5jb250YWluZXJfb2s6'
            'CiAgICAgICAgZmFpbHMuYXBwZW5kKCgxMCwgIiVzIGluY29udGFpbmVyX3NoYV9taXNtYXRjaCIg'
            'JSBuYW1lKSkKICAgIHBpeCA9IHBhcnNlX3BpeF9sb2dzKGxvZ190ZXh0KQogICAgaWYgcGl4WyJt'
            'aXNzaW5nX21vZHVsZSJdID4gMDoKICAgICAgICBmYWlscy5hcHBlbmQoKDExLCAiJXMgbWlzc2lu'
            'Z19tb2R1bGVfaW5fbG9nIiAlIG5hbWUpKQogICAgaWYgcGl4WyJuX3JlZ2lvbl9za2lwcyJdID4g'
            'MDoKICAgICAgICBmYWlscy5hcHBlbmQoKDExLCAiJXMgcGl4Z2JtX3JlZ2lvbl9za2lwcGVkIiAl'
            'IG5hbWUpKQogICAgaWYgcGl4WyJuX2VtcHR5X3N0ZXBzIl0gPiAwOgogICAgICAgIGZhaWxzLmFw'
            'cGVuZCgoMTEsICIlcyBlbXB0eV9zdGVwc197fSIgJSBuYW1lKSkKICAgIGlmIHBpeFsibl9mdWxs'
            'X2hpdHMiXSA9PSAwOgogICAgICAgIGZhaWxzLmFwcGVuZCgoMTEsICIlcyBub19mdWxsX3N0ZXAi'
            'ICUgbmFtZSkpCiAgICBpZiBwaXhbInBoYXNlX3NlY29uZHMiXSA8PSAwOgogICAgICAgIGZhaWxz'
            'LmFwcGVuZCgoMTEsICIlcyBwaGFzZV9zZWNvbmRzPT0wIiAlIG5hbWUpKQogICAgaWYgZG9ja2Vy'
            'X2VjIGlzIG5vdCBOb25lIGFuZCBkb2NrZXJfZWMgIT0gMDoKICAgICAgICBmYWlscy5hcHBlbmQo'
            'KDEyLCAiJXMgZG9ja2VyX2V4aXQ9JXMiICUgKG5hbWUsIGRvY2tlcl9lYykpKQogICAgaWYgZ2Vv'
            'anNvbiBpcyBub3QgTm9uZSBhbmQgZ2VvanNvbi5pc19maWxlKCkgYW5kIHRfc3RhcnQgaXMgbm90'
            'IE5vbmU6CiAgICAgICAgbXRpbWUgPSBnZW9qc29uLnN0YXQoKS5zdF9tdGltZQogICAgICAgIGlm'
            'IG10aW1lIDwgdF9zdGFydDoKICAgICAgICAgICAgZmFpbHMuYXBwZW5kKCgxMywgIiVzIGdlb2pz'
            'b25fbXRpbWVfYmVmb3JlX3N0YXJ0IiAlIG5hbWUpKQogICAgcmV0dXJuIGZhaWxzLCBwaXgKCgpk'
            'ZWYgZXF1aXZhbGVuY2Uoc2NvcmVzOiBkaWN0LCB0b2w6IGZsb2F0ID0gVE9MKSAtPiBkaWN0Ogog'
            'ICAgcm93cyA9IHNjb3Jlcy5nZXQoInJvd3MiKSBvciBbXQogICAgYWxscm93ID0gbmV4dCgociBm'
            'b3IgciBpbiByb3dzIGlmIHIuZ2V0KCJyZWdpb24iKSA9PSAiQUxMIiBhbmQgci5nZXQoIm1hcmt1'
            'cCIpID09ICJBTEwiKSwgTm9uZSkKICAgIGxheWVyX3Jvd3MgPSB7CiAgICAgICAgclsibWFya3Vw'
            'Il06IHIKICAgICAgICBmb3IgciBpbiByb3dzCiAgICAgICAgaWYgci5nZXQoInJlZ2lvbiIpID09'
            'ICJBTEwiIGFuZCByLmdldCgibWFya3VwIikgaW4gR0VPX0xBWUVSUwogICAgfQogICAgb3V0ID0g'
            'ewogICAgICAgICJ0b2wiOiB0b2wsCiAgICAgICAgImRlbHRhX0FMTCI6IChhbGxyb3cgb3Ige30p'
            'LmdldCgiZGVsdGEiKSwKICAgICAgICAiZjFfQUxMIjoge2s6IHYgZm9yIGssIHYgaW4gKGFsbHJv'
            'dyBvciB7fSkuaXRlbXMoKSBpZiBrLnN0YXJ0c3dpdGgoImYxXyIpfSwKICAgICAgICAibGF5ZXJz'
            'Ijoge30sCiAgICAgICAgImxheWVyX2ZhaWxzIjogW10sCiAgICAgICAgInBhc3MiOiBGYWxzZSwK'
            'ICAgICAgICAicmVhc29uIjogIiIsCiAgICB9CiAgICBmb3IgbXQgaW4gR0VPX0xBWUVSUzoKICAg'
            'ICAgICBpZiBtdCBub3QgaW4gbGF5ZXJfcm93czoKICAgICAgICAgICAgY29udGludWUKICAgICAg'
            'ICBkID0gbGF5ZXJfcm93c1ttdF0uZ2V0KCJkZWx0YSIpCiAgICAgICAgb3V0WyJsYXllcnMiXVtt'
            'dF0gPSB7CiAgICAgICAgICAgICJkZWx0YSI6IGQsCiAgICAgICAgICAgICoqe2s6IHYgZm9yIGss'
            'IHYgaW4gbGF5ZXJfcm93c1ttdF0uaXRlbXMoKSBpZiBrLnN0YXJ0c3dpdGgoImYxXyIpfSwKICAg'
            'ICAgICB9CiAgICAgICAgaWYgZCBpcyBOb25lIG9yIGFicyhmbG9hdChkKSkgPiB0b2w6CiAgICAg'
            'ICAgICAgIG91dFsibGF5ZXJfZmFpbHMiXS5hcHBlbmQoKG10LCBkKSkKCiAgICByZWFzb25zID0g'
            'W10KICAgIGlmIGFsbHJvdyBpcyBOb25lIG9yIGFsbHJvdy5nZXQoImRlbHRhIikgaXMgTm9uZToK'
            'ICAgICAgICByZWFzb25zLmFwcGVuZCgibm8gQUxML0FMTCBkZWx0YSIpCiAgICBlbGlmIGFicyhm'
            'bG9hdChhbGxyb3dbImRlbHRhIl0pKSA+IHRvbDoKICAgICAgICByZWFzb25zLmFwcGVuZCgiQUxM'
            'IHxkZWx0YXw9JXMgPiAlcyIgJSAoYWxscm93WyJkZWx0YSJdLCB0b2wpKQogICAgaWYgb3V0WyJs'
            'YXllcl9mYWlscyJdOgogICAgICAgIHJlYXNvbnMuYXBwZW5kKAogICAgICAgICAgICAibGF5ZXI6'
            'ICIgKyAiLCIuam9pbigiJXM9JXMiICUgKGEsIGIpIGZvciBhLCBiIGluIG91dFsibGF5ZXJfZmFp'
            'bHMiXSkKICAgICAgICApCiAgICAjIFJlcXVpcmUgdGhhdCBhdCBsZWFzdCBvbmUgZ2VvIGxheWVy'
            'IHJvdyBleGlzdHMgKGVsc2Ugbm90IGEgZ2VvIHRlc3QpCiAgICBpZiBub3Qgb3V0WyJsYXllcnMi'
            'XToKICAgICAgICByZWFzb25zLmFwcGVuZCgibm8gbWFnZ3IvZXRnci90ZXJmbCBBTEwgcm93cyBp'
            'biBzY29yZXMiKQogICAgaWYgbm90IHJlYXNvbnM6CiAgICAgICAgb3V0WyJwYXNzIl0gPSBUcnVl'
            'CiAgICAgICAgb3V0WyJyZWFzb24iXSA9ICJvayIKICAgIGVsc2U6CiAgICAgICAgb3V0WyJyZWFz'
            'b24iXSA9ICI7ICIuam9pbihyZWFzb25zKQoKICAgICMgT3B0aW9uYWwgcGVyLWNsYXNzIG5vdGUg'
            'ZnJvbSBjb3VudHNfKiBpZiBwcmVzZW50IChub3QgRjEpCiAgICBmb3Iga2V5LCB2YWwgaW4gc2Nv'
            'cmVzLml0ZW1zKCk6CiAgICAgICAgaWYga2V5LnN0YXJ0c3dpdGgoImNvdW50c18iKToKICAgICAg'
            'ICAgICAgb3V0W2tleV0gPSBkaWN0KHZhbCkgaWYgaXNpbnN0YW5jZSh2YWwsIGRpY3QpIGVsc2Ug'
            'dmFsCiAgICByZXR1cm4gb3V0CgoKZGVmIG1haW4oKSAtPiBpbnQ6CiAgICBhcCA9IGFyZ3BhcnNl'
            'LkFyZ3VtZW50UGFyc2VyKCkKICAgIGFwLmFkZF9hcmd1bWVudCgiLS1qb2IiLCBkZWZhdWx0PU5v'
            'bmUsIGhlbHA9IkpvYkRpciIpCiAgICBhcC5hZGRfYXJndW1lbnQoIi0tc2NvcmVzIiwgZGVmYXVs'
            'dD1Ob25lKQogICAgYXAuYWRkX2FyZ3VtZW50KCItLXRvbCIsIHR5cGU9ZmxvYXQsIGRlZmF1bHQ9'
            'VE9MKQogICAgIyBvcHRpb25hbCBleHBsaWNpdCBpbnB1dHMgZm9yIHVuaXQgc2VsZi1jaGVjayB3'
            'aXRob3V0IGEgZnVsbCBKb2JEaXIgbGF5b3V0CiAgICBhcC5hZGRfYXJndW1lbnQoIi0tbG9nLXJl'
            'ZiIsIGRlZmF1bHQ9Tm9uZSkKICAgIGFwLmFkZF9hcmd1bWVudCgiLS1sb2ctbmV3IiwgZGVmYXVs'
            'dD1Ob25lKQogICAgYXAuYWRkX2FyZ3VtZW50KCItLXNlbGYtY2hlY2siLCBhY3Rpb249InN0b3Jl'
            'X3RydWUiKQogICAgYXJncyA9IGFwLnBhcnNlX2FyZ3MoKQogICAgaWYgYXJncy5zZWxmX2NoZWNr'
            'OgogICAgICAgIGpvYiA9IFBhdGgoYXJncy5qb2IpIGlmIGFyZ3Muam9iIGVsc2UgUGF0aCgiL3Rt'
            'cC9ib3Q2X3IzX2dhdGVfc2VsZiIpCiAgICAgICAgam9iLm1rZGlyKHBhcmVudHM9VHJ1ZSwgZXhp'
            'c3Rfb2s9VHJ1ZSkKICAgICAgICByZXR1cm4gX3NlbGZfY2hlY2soam9iKQogICAgaWYgbm90IGFy'
            'Z3Muam9iOgogICAgICAgIGFwLmVycm9yKCItLWpvYiBpcyByZXF1aXJlZCB1bmxlc3MgLS1zZWxm'
            'LWNoZWNrIikKICAgIGpvYiA9IFBhdGgoYXJncy5qb2IpCiAgICBqb2IubWtkaXIocGFyZW50cz1U'
            'cnVlLCBleGlzdF9vaz1UcnVlKQoKICAgIG5vdGVzID0gam9iIC8gIm5vdGVzIgogICAgc2NvcmVz'
            'X3BhdGggPSBQYXRoKGFyZ3Muc2NvcmVzKSBpZiBhcmdzLnNjb3JlcyBlbHNlIGpvYiAvICJzY29y'
            'ZXMuanNvbiIKICAgIGdhdGUgPSB7CiAgICAgICAgInBhc3MiOiBGYWxzZSwKICAgICAgICAidG9s'
            'IjogYXJncy50b2wsCiAgICAgICAgInZhbGlkaXR5X2ZhaWxzIjogW10sCiAgICAgICAgInJ1bnMi'
            'OiB7fSwKICAgIH0KCiAgICBtZXRhX3BhdGggPSBub3RlcyAvICJydW5fbWV0YS5qc29uIgogICAg'
            'bWV0YSA9IGpzb24ubG9hZHMobWV0YV9wYXRoLnJlYWRfdGV4dChlbmNvZGluZz0idXRmLTgiKSkg'
            'aWYgbWV0YV9wYXRoLmlzX2ZpbGUoKSBlbHNlIHt9CgogICAgZm9yIG5hbWUgaW4gKCJyZWYiLCAi'
            'bmV3Iik6CiAgICAgICAgbG9nX292ZXJyaWRlID0gTm9uZQogICAgICAgIGlmIG5hbWUgPT0gInJl'
            'ZiIgYW5kIGFyZ3MubG9nX3JlZjoKICAgICAgICAgICAgbG9nX292ZXJyaWRlID0gYXJncy5sb2df'
            'cmVmCiAgICAgICAgZWxpZiBuYW1lID09ICJuZXciIGFuZCBhcmdzLmxvZ19uZXc6CiAgICAgICAg'
            'ICAgIGxvZ19vdmVycmlkZSA9IGFyZ3MubG9nX25ldwogICAgICAgIGlmIGxvZ19vdmVycmlkZToK'
            'ICAgICAgICAgICAgbG9nX3R4dCA9IF9yZWFkKFBhdGgobG9nX292ZXJyaWRlKSkKICAgICAgICBl'
            'bHNlOgogICAgICAgICAgICBsb2dfdHh0ID0gX3JlYWQoam9iIC8gKCJvdXRfJXMiICUgbmFtZSkg'
            'LyAibWFpbi5sb2ciKSArIF9yZWFkKAogICAgICAgICAgICAgICAgam9iIC8gKCJvdXRfJXMiICUg'
            'bmFtZSkgLyAibWFpbi5lcnIiKQogICAgICAgIGRvY2tlcl9lYyA9IChtZXRhLmdldCgiZG9ja2Vy'
            'X2VjIikgb3Ige30pLmdldChuYW1lKQogICAgICAgIHRfc3RhcnQgPSAobWV0YS5nZXQoInRfc3Rh'
            'cnRfZXBvY2giKSBvciB7fSkuZ2V0KG5hbWUpCiAgICAgICAgZ2VvID0gam9iIC8gKCJvdXRfJXMi'
            'ICUgbmFtZSkgLyAicmVzdWx0Lmdlb2pzb24iCiAgICAgICAgaW5fb2sgPSBib29sKChtZXRhLmdl'
            'dCgiaW5jb250YWluZXJfb2siKSBvciB7fSkuZ2V0KG5hbWUsIFRydWUpKQogICAgICAgIGZhaWxz'
            'LCBwaXggPSB2YWxpZGl0eV9mb3JfcnVuKG5hbWUsIGxvZ190eHQsIGRvY2tlcl9lYywgZ2VvLCB0'
            'X3N0YXJ0LCBpbl9vaykKICAgICAgICBnYXRlWyJydW5zIl1bbmFtZV0gPSBwaXgKICAgICAgICBn'
            'YXRlWyJ2YWxpZGl0eV9mYWlscyJdLmV4dGVuZChbeyJjb2RlIjogYywgIm1zZyI6IG19IGZvciBj'
            'LCBtIGluIGZhaWxzXSkKCiAgICAjIGlkZW50aWNhbCBnZW9qc29uICsgZGlmZmVyZW50IG1vZGVs'
            'cyAtPiBJTlZBTElECiAgICBpZGVudCA9IG1ldGEuZ2V0KCJyZXN1bHRfaWRlbnRpY2FsIikKICAg'
            'IG1vZGVsX2RpZmZlciA9IG1ldGEuZ2V0KCJtb2RlbF9zaGFfZGlmZmVyIiwgVHJ1ZSkKICAgIGlm'
            'IGlkZW50IGlzIFRydWUgYW5kIG1vZGVsX2RpZmZlcjoKICAgICAgICBnYXRlWyJ2YWxpZGl0eV9m'
            'YWlscyJdLmFwcGVuZCh7CiAgICAgICAgICAgICJjb2RlIjogMTQsCiAgICAgICAgICAgICJtc2ci'
            'OiAicmVzdWx0X2dlb2pzb25faWRlbnRpY2FsX2J1dF9tb2RlbHNfZGlmZmVyIiwKICAgICAgICB9'
            'KQogICAgZ2F0ZVsicmVzdWx0X2lkZW50aWNhbCJdID0gaWRlbnQKICAgIGdhdGVbInJlc3VsdF9z'
            'aGEiXSA9IG1ldGEuZ2V0KCJyZXN1bHRfc2hhIikKCiAgICBpZiBnYXRlWyJ2YWxpZGl0eV9mYWls'
            'cyJdOgogICAgICAgICMgcHJlZmVyIHNtYWxsZXN0IGRpc3RpbmN0IGNvZGUKICAgICAgICBjb2Rl'
            'ID0gbWluKGZbImNvZGUiXSBmb3IgZiBpbiBnYXRlWyJ2YWxpZGl0eV9mYWlscyJdKQogICAgICAg'
            'IGdhdGVbInJlYXNvbiJdID0gIklOVkFMSUQ6ICIgKyAiOyAiLmpvaW4oZlsibXNnIl0gZm9yIGYg'
            'aW4gZ2F0ZVsidmFsaWRpdHlfZmFpbHMiXSkKICAgICAgICAoam9iIC8gImdhdGUuanNvbiIpLndy'
            'aXRlX3RleHQoanNvbi5kdW1wcyhnYXRlLCBlbnN1cmVfYXNjaWk9RmFsc2UsIGluZGVudD0yKSwg'
            'ZW5jb2Rpbmc9InV0Zi04IikKICAgICAgICBwcmludChqc29uLmR1bXBzKGdhdGUsIGVuc3VyZV9h'
            'c2NpaT1GYWxzZSkpCiAgICAgICAgcmV0dXJuIGNvZGUKCiAgICBpZiBub3Qgc2NvcmVzX3BhdGgu'
            'aXNfZmlsZSgpOgogICAgICAgIGdhdGVbInJlYXNvbiJdID0gIm5vIHNjb3Jlcy5qc29uIgogICAg'
            'ICAgIChqb2IgLyAiZ2F0ZS5qc29uIikud3JpdGVfdGV4dChqc29uLmR1bXBzKGdhdGUsIGVuc3Vy'
            'ZV9hc2NpaT1GYWxzZSwgaW5kZW50PTIpLCBlbmNvZGluZz0idXRmLTgiKQogICAgICAgIHByaW50'
            'KGpzb24uZHVtcHMoZ2F0ZSwgZW5zdXJlX2FzY2lpPUZhbHNlKSkKICAgICAgICByZXR1cm4gMTkK'
            'CiAgICBzY29yZXMgPSBqc29uLmxvYWRzKHNjb3Jlc19wYXRoLnJlYWRfdGV4dChlbmNvZGluZz0i'
            'dXRmLTgiKSkKICAgIGVxID0gZXF1aXZhbGVuY2Uoc2NvcmVzLCBhcmdzLnRvbCkKICAgIGdhdGVb'
            'ImVxdWl2YWxlbmNlIl0gPSBlcQogICAgZ2F0ZVsicGFzcyJdID0gYm9vbChlcVsicGFzcyJdKQog'
            'ICAgZ2F0ZVsicmVhc29uIl0gPSBlcVsicmVhc29uIl0KICAgIChqb2IgLyAiZ2F0ZS5qc29uIiku'
            'd3JpdGVfdGV4dChqc29uLmR1bXBzKGdhdGUsIGVuc3VyZV9hc2NpaT1GYWxzZSwgaW5kZW50PTIp'
            'LCBlbmNvZGluZz0idXRmLTgiKQogICAgcHJpbnQoanNvbi5kdW1wcyhnYXRlLCBlbnN1cmVfYXNj'
            'aWk9RmFsc2UpKQogICAgcmV0dXJuIDAgaWYgZ2F0ZVsicGFzcyJdIGVsc2UgOQoKCmRlZiBfc2Vs'
            'Zl9jaGVjayhqb2I6IFBhdGgpIC0+IGludDoKICAgICIiIlN5bnRoZXRpYyBmaXh0dXJlczogZW1w'
            'dHkgc3RlcHMgZmFpbDsgZnVsbCBzdGVwcyArIHRpbnkgfGRlbHRhfCBwYXNzLiIiIgogICAgIyBD'
            'YXNlIEE6IGVtcHR5IHN0ZXBzIC0+IGV4aXQgMTEKICAgIGxvZ19iYWQgPSAoCiAgICAgICAgIklO'
            'Rk8gcGl4Z2JtIDIwMF9YOiBiYWNrZW5kPWxpZ2h0Z2JtIHBvdG9rb3Y9MiAiCiAgICAgICAgIlx1'
            'MDQ0MVx1MDQ0Mlx1MDQ0M1x1MDQzZlx1MDQzNVx1MDQzZFx1MDQzOCB7fSIKICAgICAgICAiOyBc'
            'dTA0NDRcdTA0MzBcdTA0MzdcdTA0MzAgMC4wIFx1MDQ0MSAoY3B1IDAuMCBcdTA0NDEpLCB1cm92'
            'ZW4gY2xhc3NpY1xuIgogICAgKQogICAgbG9nX2dvb2QgPSAoCiAgICAgICAgIklORk8gcGl4Z2Jt'
            'IDIwMF9YOiBiYWNrZW5kPWxpZ2h0Z2JtIHBvdG9rb3Y9MiAiCiAgICAgICAgIlx1MDQ0MVx1MDQ0'
            'Mlx1MDQ0M1x1MDQzZlx1MDQzNVx1MDQzZFx1MDQzOCB7J2Z1bGwnOiAzfSIKICAgICAgICAiOyBc'
            'dTA0NDRcdTA0MzBcdTA0MzdcdTA0MzAgMTIuNSBcdTA0NDEgKGNwdSAxMC4wIFx1MDQ0MSksIHVy'
            'b3ZlbiBmdWxsXG4iCiAgICAgICAgIklORk8gcGl4Z2JtIDIwMV9ZOiBiYWNrZW5kPWxpZ2h0Z2Jt'
            'IHBvdG9rb3Y9MiAiCiAgICAgICAgIlx1MDQ0MVx1MDQ0Mlx1MDQ0M1x1MDQzZlx1MDQzNVx1MDQz'
            'ZFx1MDQzOCB7J2Z1bGwnOiAyfSIKICAgICAgICAiOyBcdTA0NDRcdTA0MzBcdTA0MzdcdTA0MzAg'
            'MjUuMCBcdTA0NDEgKGNwdSAyMC4wIFx1MDQ0MSksIHVyb3ZlbiBmdWxsXG4iCiAgICApCiAgICBz'
            'Y29yZXNfb2sgPSB7CiAgICAgICAgInJvd3MiOiBbCiAgICAgICAgICAgIHsicmVnaW9uIjogIkFM'
            'TCIsICJtYXJrdXAiOiAiQUxMIiwgImRlbHRhIjogMC4wMDIsCiAgICAgICAgICAgICAiZjFfcGl4'
            'X3JlZiI6IDAuNTAsICJmMV9waXhfbmV3IjogMC41MDJ9LAogICAgICAgICAgICB7InJlZ2lvbiI6'
            'ICJBTEwiLCAibWFya3VwIjogIm1hZ2dyIiwgImRlbHRhIjogLTAuMDA1LAogICAgICAgICAgICAg'
            'ImYxX3BpeF9yZWYiOiAwLjQwLCAiZjFfcGl4X25ldyI6IDAuMzk1fSwKICAgICAgICAgICAgeyJy'
            'ZWdpb24iOiAiQUxMIiwgIm1hcmt1cCI6ICJldGdyIiwgImRlbHRhIjogMC4wLAogICAgICAgICAg'
            'ICAgImYxX3BpeF9yZWYiOiAwLjMwLCAiZjFfcGl4X25ldyI6IDAuMzB9LAogICAgICAgICAgICB7'
            'InJlZ2lvbiI6ICJBTEwiLCAibWFya3VwIjogInRlcmZsIiwgImRlbHRhIjogMC4wMDgsCiAgICAg'
            'ICAgICAgICAiZjFfcGl4X3JlZiI6IDAuMjAsICJmMV9waXhfbmV3IjogMC4yMDh9LAogICAgICAg'
            'IF0sCiAgICAgICAgImNvdW50c19waXhfcmVmIjogeyJtYWdnci9vYmplY3RfcG9seSI6IDEwLCAi'
            'bWFnZ3IvZmluZHNfcG9pbnRzIjogNX0sCiAgICAgICAgImNvdW50c19waXhfbmV3IjogeyJtYWdn'
            'ci9vYmplY3RfcG9seSI6IDExLCAibWFnZ3IvZmluZHNfcG9pbnRzIjogNX0sCiAgICB9CiAgICBz'
            'Y29yZXNfYmFkID0gewogICAgICAgICJyb3dzIjogWwogICAgICAgICAgICB7InJlZ2lvbiI6ICJB'
            'TEwiLCAibWFya3VwIjogIkFMTCIsICJkZWx0YSI6IDAuMDUsCiAgICAgICAgICAgICAiZjFfcGl4'
            'X3JlZiI6IDAuNTAsICJmMV9waXhfbmV3IjogMC41NX0sCiAgICAgICAgICAgIHsicmVnaW9uIjog'
            'IkFMTCIsICJtYXJrdXAiOiAibWFnZ3IiLCAiZGVsdGEiOiAwLjA1LAogICAgICAgICAgICAgImYx'
            'X3BpeF9yZWYiOiAwLjQwLCAiZjFfcGl4X25ldyI6IDAuNDV9LAogICAgICAgIF0KICAgIH0KCiAg'
            'ICByZXN1bHRzID0ge30KICAgICMgQTogYmFkIGxvZwogICAgZEEgPSBqb2IgLyAiX3NlbGZBIgog'
            'ICAgZEEubWtkaXIocGFyZW50cz1UcnVlLCBleGlzdF9vaz1UcnVlKQogICAgKGRBIC8gIm5vdGVz'
            'IikubWtkaXIoZXhpc3Rfb2s9VHJ1ZSkKICAgIChkQSAvICJvdXRfcmVmIikubWtkaXIoZXhpc3Rf'
            'b2s9VHJ1ZSkKICAgIChkQSAvICJvdXRfbmV3IikubWtkaXIoZXhpc3Rfb2s9VHJ1ZSkKICAgIChk'
            'QSAvICJvdXRfcmVmIiAvICJtYWluLmxvZyIpLndyaXRlX3RleHQobG9nX2JhZCwgZW5jb2Rpbmc9'
            'InV0Zi04IikKICAgIChkQSAvICJvdXRfbmV3IiAvICJtYWluLmxvZyIpLndyaXRlX3RleHQobG9n'
            'X2dvb2QsIGVuY29kaW5nPSJ1dGYtOCIpCiAgICAoZEEgLyAibm90ZXMiIC8gInJ1bl9tZXRhLmpz'
            'b24iKS53cml0ZV90ZXh0KGpzb24uZHVtcHMoewogICAgICAgICJkb2NrZXJfZWMiOiB7InJlZiI6'
            'IDAsICJuZXciOiAwfSwKICAgICAgICAiaW5jb250YWluZXJfb2siOiB7InJlZiI6IFRydWUsICJu'
            'ZXciOiBUcnVlfSwKICAgICAgICAicmVzdWx0X2lkZW50aWNhbCI6IEZhbHNlLAogICAgICAgICJt'
            'b2RlbF9zaGFfZGlmZmVyIjogVHJ1ZSwKICAgIH0pLCBlbmNvZGluZz0idXRmLTgiKQogICAgKGRB'
            'IC8gInNjb3Jlcy5qc29uIikud3JpdGVfdGV4dChqc29uLmR1bXBzKHNjb3Jlc19vayksIGVuY29k'
            'aW5nPSJ1dGYtOCIpCiAgICAjIHRvdWNoIGdlb2pzb25zIHdpdGggcmVjZW50IG10aW1lCiAgICBm'
            'b3IgdiBpbiAoInJlZiIsICJuZXciKToKICAgICAgICAoZEEgLyAoIm91dF8lcyIgJSB2KSAvICJy'
            'ZXN1bHQuZ2VvanNvbiIpLndyaXRlX3RleHQoJ3sidHlwZSI6IkZlYXR1cmVDb2xsZWN0aW9uIiwi'
            'ZmVhdHVyZXMiOltdfScpCiAgICBpbXBvcnQgc3VicHJvY2VzcwogICAgciA9IHN1YnByb2Nlc3Mu'
            'cnVuKFtzeXMuZXhlY3V0YWJsZSwgX19maWxlX18sICItLWpvYiIsIHN0cihkQSldLCBjYXB0dXJl'
            'X291dHB1dD1UcnVlLCB0ZXh0PVRydWUpCiAgICByZXN1bHRzWyJlbXB0eV9zdGVwcyJdID0geyJl'
            'YyI6IHIucmV0dXJuY29kZSwgIm91dCI6IHIuc3Rkb3V0LnN0cmlwKClbOjIwMF19CgogICAgIyBC'
            'OiBnb29kIGxvZ3MgKyBwYXNzIHNjb3JlcwogICAgZEIgPSBqb2IgLyAiX3NlbGZCIgogICAgZEIu'
            'bWtkaXIocGFyZW50cz1UcnVlLCBleGlzdF9vaz1UcnVlKQogICAgKGRCIC8gIm5vdGVzIikubWtk'
            'aXIoZXhpc3Rfb2s9VHJ1ZSkKICAgIGZvciB2IGluICgicmVmIiwgIm5ldyIpOgogICAgICAgIChk'
            'QiAvICgib3V0XyVzIiAlIHYpKS5ta2RpcihleGlzdF9vaz1UcnVlKQogICAgICAgIChkQiAvICgi'
            'b3V0XyVzIiAlIHYpIC8gIm1haW4ubG9nIikud3JpdGVfdGV4dChsb2dfZ29vZCwgZW5jb2Rpbmc9'
            'InV0Zi04IikKICAgICAgICAoZEIgLyAoIm91dF8lcyIgJSB2KSAvICJyZXN1bHQuZ2VvanNvbiIp'
            'LndyaXRlX3RleHQoJ3sidHlwZSI6IkZlYXR1cmVDb2xsZWN0aW9uIiwiZmVhdHVyZXMiOltdfScp'
            'CiAgICAoZEIgLyAibm90ZXMiIC8gInJ1bl9tZXRhLmpzb24iKS53cml0ZV90ZXh0KGpzb24uZHVt'
            'cHMoewogICAgICAgICJkb2NrZXJfZWMiOiB7InJlZiI6IDAsICJuZXciOiAwfSwKICAgICAgICAi'
            'aW5jb250YWluZXJfb2siOiB7InJlZiI6IFRydWUsICJuZXciOiBUcnVlfSwKICAgICAgICAicmVz'
            'dWx0X2lkZW50aWNhbCI6IEZhbHNlLAogICAgICAgICJtb2RlbF9zaGFfZGlmZmVyIjogVHJ1ZSwK'
            'ICAgIH0pLCBlbmNvZGluZz0idXRmLTgiKQogICAgKGRCIC8gInNjb3Jlcy5qc29uIikud3JpdGVf'
            'dGV4dChqc29uLmR1bXBzKHNjb3Jlc19vayksIGVuY29kaW5nPSJ1dGYtOCIpCiAgICByID0gc3Vi'
            'cHJvY2Vzcy5ydW4oW3N5cy5leGVjdXRhYmxlLCBfX2ZpbGVfXywgIi0tam9iIiwgc3RyKGRCKV0s'
            'IGNhcHR1cmVfb3V0cHV0PVRydWUsIHRleHQ9VHJ1ZSkKICAgIHJlc3VsdHNbInBhc3NfZXEiXSA9'
            'IHsiZWMiOiByLnJldHVybmNvZGUsICJvdXQiOiByLnN0ZG91dC5zdHJpcCgpWzozMDBdfQoKICAg'
            'ICMgQzogZ29vZCBsb2dzICsgZmFpbCBlcXVpdmFsZW5jZQogICAgZEMgPSBqb2IgLyAiX3NlbGZD'
            'IgogICAgZEMubWtkaXIocGFyZW50cz1UcnVlLCBleGlzdF9vaz1UcnVlKQogICAgKGRDIC8gIm5v'
            'dGVzIikubWtkaXIoZXhpc3Rfb2s9VHJ1ZSkKICAgIGZvciB2IGluICgicmVmIiwgIm5ldyIpOgog'
            'ICAgICAgIChkQyAvICgib3V0XyVzIiAlIHYpKS5ta2RpcihleGlzdF9vaz1UcnVlKQogICAgICAg'
            'IChkQyAvICgib3V0XyVzIiAlIHYpIC8gIm1haW4ubG9nIikud3JpdGVfdGV4dChsb2dfZ29vZCwg'
            'ZW5jb2Rpbmc9InV0Zi04IikKICAgICAgICAoZEMgLyAoIm91dF8lcyIgJSB2KSAvICJyZXN1bHQu'
            'Z2VvanNvbiIpLndyaXRlX3RleHQoJ3sidHlwZSI6IkZlYXR1cmVDb2xsZWN0aW9uIiwiZmVhdHVy'
            'ZXMiOltdfScpCiAgICAoZEMgLyAibm90ZXMiIC8gInJ1bl9tZXRhLmpzb24iKS53cml0ZV90ZXh0'
            'KGpzb24uZHVtcHMoewogICAgICAgICJkb2NrZXJfZWMiOiB7InJlZiI6IDAsICJuZXciOiAwfSwK'
            'ICAgICAgICAiaW5jb250YWluZXJfb2siOiB7InJlZiI6IFRydWUsICJuZXciOiBUcnVlfSwKICAg'
            'ICAgICAicmVzdWx0X2lkZW50aWNhbCI6IEZhbHNlLAogICAgICAgICJtb2RlbF9zaGFfZGlmZmVy'
            'IjogVHJ1ZSwKICAgIH0pLCBlbmNvZGluZz0idXRmLTgiKQogICAgKGRDIC8gInNjb3Jlcy5qc29u'
            'Iikud3JpdGVfdGV4dChqc29uLmR1bXBzKHNjb3Jlc19iYWQpLCBlbmNvZGluZz0idXRmLTgiKQog'
            'ICAgciA9IHN1YnByb2Nlc3MucnVuKFtzeXMuZXhlY3V0YWJsZSwgX19maWxlX18sICItLWpvYiIs'
            'IHN0cihkQyldLCBjYXB0dXJlX291dHB1dD1UcnVlLCB0ZXh0PVRydWUpCiAgICByZXN1bHRzWyJm'
            'YWlsX2VxIl0gPSB7ImVjIjogci5yZXR1cm5jb2RlLCAib3V0Ijogci5zdGRvdXQuc3RyaXAoKVs6'
            'MzAwXX0KCiAgICAjIEQ6IGlkZW50aWNhbCByZXN1bHRzICsgZGlmZmVyaW5nIG1vZGVscwogICAg'
            'ZEQgPSBqb2IgLyAiX3NlbGZEIgogICAgZEQubWtkaXIocGFyZW50cz1UcnVlLCBleGlzdF9vaz1U'
            'cnVlKQogICAgKGREIC8gIm5vdGVzIikubWtkaXIoZXhpc3Rfb2s9VHJ1ZSkKICAgIGZvciB2IGlu'
            'ICgicmVmIiwgIm5ldyIpOgogICAgICAgIChkRCAvICgib3V0XyVzIiAlIHYpKS5ta2RpcihleGlz'
            'dF9vaz1UcnVlKQogICAgICAgIChkRCAvICgib3V0XyVzIiAlIHYpIC8gIm1haW4ubG9nIikud3Jp'
            'dGVfdGV4dChsb2dfZ29vZCwgZW5jb2Rpbmc9InV0Zi04IikKICAgICAgICAoZEQgLyAoIm91dF8l'
            'cyIgJSB2KSAvICJyZXN1bHQuZ2VvanNvbiIpLndyaXRlX3RleHQoJ3sidHlwZSI6IkZlYXR1cmVD'
            'b2xsZWN0aW9uIiwiZmVhdHVyZXMiOltdfScpCiAgICAoZEQgLyAibm90ZXMiIC8gInJ1bl9tZXRh'
            'Lmpzb24iKS53cml0ZV90ZXh0KGpzb24uZHVtcHMoewogICAgICAgICJkb2NrZXJfZWMiOiB7InJl'
            'ZiI6IDAsICJuZXciOiAwfSwKICAgICAgICAiaW5jb250YWluZXJfb2siOiB7InJlZiI6IFRydWUs'
            'ICJuZXciOiBUcnVlfSwKICAgICAgICAicmVzdWx0X2lkZW50aWNhbCI6IFRydWUsCiAgICAgICAg'
            'Im1vZGVsX3NoYV9kaWZmZXIiOiBUcnVlLAogICAgfSksIGVuY29kaW5nPSJ1dGYtOCIpCiAgICAo'
            'ZEQgLyAic2NvcmVzLmpzb24iKS53cml0ZV90ZXh0KGpzb24uZHVtcHMoc2NvcmVzX29rKSwgZW5j'
            'b2Rpbmc9InV0Zi04IikKICAgIHIgPSBzdWJwcm9jZXNzLnJ1bihbc3lzLmV4ZWN1dGFibGUsIF9f'
            'ZmlsZV9fLCAiLS1qb2IiLCBzdHIoZEQpXSwgY2FwdHVyZV9vdXRwdXQ9VHJ1ZSwgdGV4dD1UcnVl'
            'KQogICAgcmVzdWx0c1siaWRlbnRpY2FsX3N1c3BlY3QiXSA9IHsiZWMiOiByLnJldHVybmNvZGUs'
            'ICJvdXQiOiByLnN0ZG91dC5zdHJpcCgpWzoyMDBdfQoKICAgIG9rID0gKAogICAgICAgIHJlc3Vs'
            'dHNbImVtcHR5X3N0ZXBzIl1bImVjIl0gPT0gMTEKICAgICAgICBhbmQgcmVzdWx0c1sicGFzc19l'
            'cSJdWyJlYyJdID09IDAKICAgICAgICBhbmQgcmVzdWx0c1siZmFpbF9lcSJdWyJlYyJdID09IDkK'
            'ICAgICAgICBhbmQgcmVzdWx0c1siaWRlbnRpY2FsX3N1c3BlY3QiXVsiZWMiXSA9PSAxNAogICAg'
            'KQogICAgc3VtbWFyeSA9IHsic2VsZl9jaGVja19wYXNzIjogb2ssICJjYXNlcyI6IHJlc3VsdHN9'
            'CiAgICAoam9iIC8gInNlbGZfY2hlY2tfZ2F0ZS5qc29uIikud3JpdGVfdGV4dChqc29uLmR1bXBz'
            'KHN1bW1hcnksIGVuc3VyZV9hc2NpaT1GYWxzZSwgaW5kZW50PTIpLCBlbmNvZGluZz0idXRmLTgi'
            'KQogICAgcHJpbnQoanNvbi5kdW1wcyhzdW1tYXJ5LCBlbnN1cmVfYXNjaWk9RmFsc2UsIGluZGVu'
            'dD0yKSkKICAgIHJldHVybiAwIGlmIG9rIGVsc2UgMQoKCmlmIF9fbmFtZV9fID09ICJfX21haW5f'
            'XyI6CiAgICBzeXMuZXhpdChtYWluKCkpCg=='
        ),
    },
    'helpers/bot6_r3_build_gt.py': {
        "sha256": 'd452b7615c91c0bb3c0fcf8b918ae99da89d0aefec8cb675fc9a53ba1d7ba9b0',
        "b64": (
            'IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiJib3QtNiByMzogYnVpbGQgZ3QuZ2VvanNvbiBmb3Ig'
            'Z2VvIHJlZ2lvbnMgKG1hZ2dyL2V0Z3IvdGVyZmwgb25seSkuCgpVc2VzIGV4cGRzLmNhdGFsb2cg'
            'KyBleHBkcy5ndCBsb2FkZXJzLiBGZWF0dXJlIHByb3BlcnRpZXMgbWF0Y2ggd3JpdGVyIC8KZTJl'
            'X3Njb3JlIGV4cGVjdGF0aW9uczogcmVnaW9uX25hbWUsIG1hcmt1cF90eXBlLCBjbGFzc19uYW1l'
            'ICgrIGZpZCwgc291cmNlKS4KQW5ub3RhdGlvbnMgYXJlIHJlYWQgZnJvbSB0aGUgdHJhaW4gcm9v'
            'dDsgdGhleSBhcmUgTk9UIGNvcGllZCBpbnRvIHRoZQppbmZlcmVuY2UgcmVnaW9ucyB0cmVlLgoi'
            'IiIKZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwoKaW1wb3J0IGFyZ3BhcnNlCmlt'
            'cG9ydCBqc29uCmltcG9ydCBzeXMKZnJvbSBwYXRobGliIGltcG9ydCBQYXRoCgpHRU9fUFJFRklY'
            'RVMgPSB0dXBsZShzdHIobikgZm9yIG4gaW4gcmFuZ2UoMjAwLCAyMTgpKQpHRU9fTVRTID0gZnJv'
            'emVuc2V0KHsibWFnZ3IiLCAiZXRnciIsICJ0ZXJmbCJ9KQoKCmRlZiBfZW5zdXJlX3JlcG8ocmVw'
            'bzogUGF0aCkgLT4gTm9uZToKICAgIHJvb3QgPSBzdHIocmVwby5yZXNvbHZlKCkpCiAgICBpZiBy'
            'b290IG5vdCBpbiBzeXMucGF0aDoKICAgICAgICBzeXMucGF0aC5pbnNlcnQoMCwgcm9vdCkKCgpk'
            'ZWYgbWFpbigpIC0+IGludDoKICAgIGFwID0gYXJncGFyc2UuQXJndW1lbnRQYXJzZXIoKQogICAg'
            'YXAuYWRkX2FyZ3VtZW50KCItLXJlcG8iLCByZXF1aXJlZD1UcnVlKQogICAgYXAuYWRkX2FyZ3Vt'
            'ZW50KCItLXJvb3QiLCByZXF1aXJlZD1UcnVlLCBoZWxwPSJ0cmFpbiByb290IHdpdGggYW5ub3Rh'
            'dGlvbnMiKQogICAgYXAuYWRkX2FyZ3VtZW50KCItLW91dCIsIHJlcXVpcmVkPVRydWUpCiAgICBh'
            'cC5hZGRfYXJndW1lbnQoIi0tcmVnaW9ucy1mcm9tIiwgZGVmYXVsdD1Ob25lLAogICAgICAgICAg'
            'ICAgICAgICAgIGhlbHA9Im9wdGlvbmFsIHRleHQgZmlsZSBvZiByZWdpb24gZm9sZGVyIG5hbWVz'
            'IChvbmUgcGVyIGxpbmUpIikKICAgIGFwLmFkZF9hcmd1bWVudCgiLS1uby1tZXJnZSIsIGFjdGlv'
            'bj0ic3RvcmVfdHJ1ZSIpCiAgICBhcmdzID0gYXAucGFyc2VfYXJncygpCgogICAgX2Vuc3VyZV9y'
            'ZXBvKFBhdGgoYXJncy5yZXBvKSkKICAgIGZyb20gZXhwZHMuY2F0YWxvZyBpbXBvcnQgc2Nhbl9y'
            'b290CiAgICBmcm9tIGV4cGRzLmd0IGltcG9ydCBidWlsZF9ndCwgdG9fZmVhdHVyZV9jb2xsZWN0'
            'aW9uCgogICAgcmVnaW9ucyA9IHNjYW5fcm9vdChQYXRoKGFyZ3Mucm9vdCkpCiAgICBpZiBhcmdz'
            'LnJlZ2lvbnNfZnJvbSBhbmQgUGF0aChhcmdzLnJlZ2lvbnNfZnJvbSkuaXNfZmlsZSgpOgogICAg'
            'ICAgIHdhbnQgPSB7bG4uc3RyaXAoKSBmb3IgbG4gaW4gUGF0aChhcmdzLnJlZ2lvbnNfZnJvbSku'
            'cmVhZF90ZXh0KGVuY29kaW5nPSJ1dGYtOCIpLnNwbGl0bGluZXMoKSBpZiBsbi5zdHJpcCgpfQog'
            'ICAgICAgIHJlZ2lvbnMgPSBbciBmb3IgciBpbiByZWdpb25zIGlmIHIubmFtZSBpbiB3YW50XQog'
            'ICAgZWxzZToKICAgICAgICBieV9wcmVmaXggPSBbciBmb3IgciBpbiByZWdpb25zIGlmIHIubmFt'
            'ZVs6M10gaW4gR0VPX1BSRUZJWEVTXQogICAgICAgIHJlZ2lvbnMgPSBieV9wcmVmaXggaWYgYnlf'
            'cHJlZml4IGVsc2UgWwogICAgICAgICAgICByIGZvciByIGluIHJlZ2lvbnMgaWYgYW55KHIucmFz'
            'dGVycy5nZXQobXQpIGZvciBtdCBpbiBHRU9fTVRTKQogICAgICAgIF0KCiAgICAjIEtlZXAgb25s'
            'eSBnZW8gbWFya3VwIGFubm90YXRpb25zIHNvIGFlL2xpIEdUIHJvd3MgZG8gbm90IHBvaXNvbiB0'
            'aGUgZ2F0ZS4KICAgIGZvciByZWcgaW4gcmVnaW9uczoKICAgICAgICByZWcuYW5ub3RhdGlvbnMg'
            'PSBbKHAsIGMsIG10KSBmb3IgKHAsIGMsIG10KSBpbiByZWcuYW5ub3RhdGlvbnMgaWYgbXQgaW4g'
            'R0VPX01UU10KCiAgICBpdGVtcywgaW5mbyA9IGJ1aWxkX2d0KHJlZ2lvbnMsIG1lcmdlPW5vdCBh'
            'cmdzLm5vX21lcmdlKQogICAgIyBEcm9wIG5vbi1nZW8gbWFya3VwIGlmIGFueSBzbGlwcGVkIHRo'
            'cm91Z2gKICAgIGl0ZW1zID0gW2l0IGZvciBpdCBpbiBpdGVtcyBpZiBpdC5nZXQoIm1hcmt1cF90'
            'eXBlIikgaW4gR0VPX01UU10KICAgIGZjID0gdG9fZmVhdHVyZV9jb2xsZWN0aW9uKGl0ZW1zKQog'
            'ICAgb3V0ID0gUGF0aChhcmdzLm91dCkKICAgIG91dC5wYXJlbnQubWtkaXIocGFyZW50cz1UcnVl'
            'LCBleGlzdF9vaz1UcnVlKQogICAgb3V0LndyaXRlX3RleHQoanNvbi5kdW1wcyhmYywgZW5zdXJl'
            'X2FzY2lpPUZhbHNlKSwgZW5jb2Rpbmc9InV0Zi04IikKCiAgICBtdHMgPSBzb3J0ZWQoe2ZbInBy'
            'b3BlcnRpZXMiXVsibWFya3VwX3R5cGUiXSBmb3IgZiBpbiBmY1siZmVhdHVyZXMiXX0pCiAgICBy'
            'ZWdzID0gc29ydGVkKHtmWyJwcm9wZXJ0aWVzIl1bInJlZ2lvbl9uYW1lIl0gZm9yIGYgaW4gZmNb'
            'ImZlYXR1cmVzIl19KQogICAgY2xhc3NlcyA9IHNvcnRlZCh7ZlsicHJvcGVydGllcyJdWyJjbGFz'
            'c19uYW1lIl0gZm9yIGYgaW4gZmNbImZlYXR1cmVzIl19KQogICAgc3VtbWFyeSA9IHsKICAgICAg'
            'ICAib2siOiBUcnVlLAogICAgICAgICJvdXQiOiBzdHIob3V0KSwKICAgICAgICAibl9mZWF0dXJl'
            'cyI6IGxlbihmY1siZmVhdHVyZXMiXSksCiAgICAgICAgIm5fcmVnaW9uc19zY2FubmVkIjogbGVu'
            'KHJlZ2lvbnMpLAogICAgICAgICJyZWdpb25zX2luX2d0IjogcmVncywKICAgICAgICAibWFya3Vw'
            'X3R5cGVzIjogbXRzLAogICAgICAgICJjbGFzc2VzIjogY2xhc3NlcywKICAgICAgICAibWVyZ2Vz'
            'IjogaW5mby5nZXQoIm1lcmdlcyIpLAogICAgfQogICAgcHJpbnQoanNvbi5kdW1wcyhzdW1tYXJ5'
            'LCBlbnN1cmVfYXNjaWk9RmFsc2UpKQogICAgaWYgbm90IGZjWyJmZWF0dXJlcyJdOgogICAgICAg'
            'IHJldHVybiAxOAogICAgaWYgbm90IChHRU9fTVRTICYgc2V0KG10cykpOgogICAgICAgIHJldHVy'
            'biAxOAogICAgcmV0dXJuIDAKCgoKCmRlZiBfc2VsZl9jaGVjaygpIC0+IGludDoKICAgICIiIlNt'
            'b2tlOiBHRU9fTVRTIGZpbHRlciBrZWVwcyBvbmx5IG1hZ2dyL2V0Z3IvdGVyZmw7IGVtcHR5IC0+'
            'IGV4aXQgMTguIiIiCiAgICBhc3NlcnQgIm1hZ2dyIiBpbiBHRU9fTVRTIGFuZCAiYWUiIG5vdCBp'
            'biBHRU9fTVRTCiAgICAjIHByb3BlcnR5IGZpbHRlciBzaW11bGF0aW9uCiAgICBpdGVtcyA9IFsK'
            'ICAgICAgICB7Im1hcmt1cF90eXBlIjogIm1hZ2dyIiwgImNsYXNzX25hbWUiOiAib2JqZWN0X3Bv'
            'bHkifSwKICAgICAgICB7Im1hcmt1cF90eXBlIjogImFlIiwgImNsYXNzX25hbWUiOiAiZ29yb2Rp'
            'c2hjaGEifSwKICAgICAgICB7Im1hcmt1cF90eXBlIjogImV0Z3IiLCAiY2xhc3NfbmFtZSI6ICJm'
            'aW5kc19wb2ludHMifSwKICAgIF0KICAgIGtlcHQgPSBbaXQgZm9yIGl0IGluIGl0ZW1zIGlmIGl0'
            'LmdldCgibWFya3VwX3R5cGUiKSBpbiBHRU9fTVRTXQogICAgYXNzZXJ0IGxlbihrZXB0KSA9PSAy'
            'IGFuZCBhbGwoaXRbIm1hcmt1cF90eXBlIl0gaW4gR0VPX01UUyBmb3IgaXQgaW4ga2VwdCkKICAg'
            'IHByaW50KF9faW1wb3J0X18oImpzb24iKS5kdW1wcyh7InNlbGZfY2hlY2tfcGFzcyI6IFRydWUs'
            'ICJrZXB0IjogMiwgImRyb3BwZWRfYWUiOiAxLCAiZW1wdHlfZXhpdCI6IDE4fSkpCiAgICByZXR1'
            'cm4gMAoKCmlmIF9fbmFtZV9fID09ICJfX21haW5fXyI6CiAgICBpZiAiLS1zZWxmLWNoZWNrIiBp'
            'biBzeXMuYXJndjoKICAgICAgICBzeXMuZXhpdChfc2VsZl9jaGVjaygpKQogICAgc3lzLmV4aXQo'
            'bWFpbigpKQo='
        ),
    },
    'helpers/bot6_r3_select_copy.py': {
        "sha256": '3799c397f97e84e00a72d767774e72a95aa4c3e38e39dfe4ab2f06ed9b87038a',
        "b64": (
            'IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiJib3QtNiByMzogZGlzY292ZXIgS09aLTMgZ2VvIHJl'
            'Z2lvbnMgKDIwMC0yMTcgb3IgYW55IHdpdGggbWFnZ3IvZXRnci90ZXJmbCksCnBsYW4vY29weSBP'
            'TkxZIGdlbyByYXN0ZXJzICsgVVRNLmpzb24gKyBib3VuZGFyaWVzICsgbm9pc2UgKG5vIExBUy9h'
            'ZXJpYWwvYW5ub3RhdGlvbnMpLgpBYm9ydCBpZiBwbGFubmVkIGNvcHkgZXhjZWVkcyAtLW1heC1n'
            'YiAoZGVmYXVsdCAxMCkuCiIiIgpmcm9tIF9fZnV0dXJlX18gaW1wb3J0IGFubm90YXRpb25zCgpp'
            'bXBvcnQgYXJncGFyc2UKaW1wb3J0IGpzb24KaW1wb3J0IHJlCmltcG9ydCBzaHV0aWwKaW1wb3J0'
            'IHN5cwpmcm9tIHBhdGhsaWIgaW1wb3J0IFBhdGgKCkdFT19QUkVGSVhFUyA9IHR1cGxlKHN0cihu'
            'KSBmb3IgbiBpbiByYW5nZSgyMDAsIDIxOCkpCkdFT19NVFMgPSAoIm1hZ2dyIiwgImV0Z3IiLCAi'
            'dGVyZmwiKQoKCmRlZiBfZW5zdXJlX3JlcG8ocmVwbzogUGF0aCkgLT4gTm9uZToKICAgIHJvb3Qg'
            'PSBzdHIocmVwby5yZXNvbHZlKCkpCiAgICBpZiByb290IG5vdCBpbiBzeXMucGF0aDoKICAgICAg'
            'ICBzeXMucGF0aC5pbnNlcnQoMCwgcm9vdCkKCgpkZWYgZGlzY292ZXIocmVwbzogUGF0aCwgdHJh'
            'aW5fcm9vdDogUGF0aCk6CiAgICBfZW5zdXJlX3JlcG8ocmVwbykKICAgIGZyb20gZXhwZHMuY2F0'
            'YWxvZyBpbXBvcnQgc2Nhbl9yb290CgogICAgcmVnaW9ucyA9IHNjYW5fcm9vdCh0cmFpbl9yb290'
            'KQogICAgYnlfcHJlZml4ID0gW3IgZm9yIHIgaW4gcmVnaW9ucyBpZiByLm5hbWVbOjNdIGluIEdF'
            'T19QUkVGSVhFU10KICAgIGJ5X3Jhc3RlcnMgPSBbCiAgICAgICAgciBmb3IgciBpbiByZWdpb25z'
            'CiAgICAgICAgaWYgYW55KHIucmFzdGVycy5nZXQobXQpIGZvciBtdCBpbiBHRU9fTVRTKQogICAg'
            'XQogICAgIyBQcmVmZXIgcHJlZml4IGxpc3Qgd2hlbiBwcmVzZW50OyBlbHNlIGRpc2NvdmVyIGJ5'
            'IGdlbyByYXN0ZXJzLgogICAgY2hvc2VuID0gYnlfcHJlZml4IGlmIGJ5X3ByZWZpeCBlbHNlIGJ5'
            'X3Jhc3RlcnMKICAgICMgQWx3YXlzIHJlcXVpcmUgYXQgbGVhc3Qgb25lIGdlbyByYXN0ZXIgc28g'
            'ZW1wdHkgcHJlZml4IGZvbGRlcnMgYXJlIGRyb3BwZWQuCiAgICBjaG9zZW4gPSBbciBmb3IgciBp'
            'biBjaG9zZW4gaWYgYW55KHIucmFzdGVycy5nZXQobXQpIGZvciBtdCBpbiBHRU9fTVRTKV0KICAg'
            'IHJldHVybiBjaG9zZW4sIHsKICAgICAgICAic2Nhbl90b3RhbCI6IGxlbihyZWdpb25zKSwKICAg'
            'ICAgICAiYnlfcHJlZml4XzIwMF8yMTciOiBbci5uYW1lIGZvciByIGluIGJ5X3ByZWZpeF0sCiAg'
            'ICAgICAgImJ5X2dlb19yYXN0ZXJzIjogW3IubmFtZSBmb3IgciBpbiBieV9yYXN0ZXJzXSwKICAg'
            'ICAgICAiY2hvc2VuIjogW3IubmFtZSBmb3IgciBpbiBjaG9zZW5dLAogICAgICAgICJzZWxlY3Rp'
            'b24iOiAicHJlZml4XzIwMF8yMTciIGlmIGJ5X3ByZWZpeCBlbHNlICJkaXNjb3Zlcl9tYWdncl9l'
            'dGdyX3RlcmZsIiwKICAgIH0KCgpkZWYgcGxhbl9maWxlcyhyZWcpIC0+IGxpc3RbUGF0aF06CiAg'
            'ICBmaWxlczogbGlzdFtQYXRoXSA9IFtdCiAgICBmb3IgbXQgaW4gR0VPX01UUzoKICAgICAgICBm'
            'aWxlcy5leHRlbmQocmVnLnJhc3RlcnMuZ2V0KG10LCBbXSkgb3IgW10pCiAgICBmaWxlcy5leHRl'
            'bmQocmVnLmJvdW5kYXJpZXMgb3IgW10pCiAgICBmaWxlcy5leHRlbmQocmVnLm5vaXNlIG9yIFtd'
            'KQogICAgaWYgcmVnLnV0bV9qc29uIGlzIG5vdCBOb25lOgogICAgICAgIGZpbGVzLmFwcGVuZChy'
            'ZWcudXRtX2pzb24pCiAgICAjIHVuaXF1ZSwgc3RhYmxlCiAgICBzZWVuID0gc2V0KCkKICAgIG91'
            'dCA9IFtdCiAgICBmb3IgZiBpbiBmaWxlczoKICAgICAgICBrZXkgPSBzdHIoZi5yZXNvbHZlKCkp'
            'IGlmIGYuZXhpc3RzKCkgZWxzZSBzdHIoZikKICAgICAgICBpZiBrZXkgaW4gc2VlbjoKICAgICAg'
            'ICAgICAgY29udGludWUKICAgICAgICBzZWVuLmFkZChrZXkpCiAgICAgICAgb3V0LmFwcGVuZChm'
            'KQogICAgcmV0dXJuIG91dAoKCmRlZiBmaWxlX3NpemUocDogUGF0aCkgLT4gaW50OgogICAgdHJ5'
            'OgogICAgICAgIHJldHVybiBwLnN0YXQoKS5zdF9zaXplIGlmIHAuaXNfZmlsZSgpIGVsc2UgMAog'
            'ICAgZXhjZXB0IE9TRXJyb3I6CiAgICAgICAgcmV0dXJuIDAKCgpkZWYgYnVpbGRfcGxhbihyZWdp'
            'b25zLCB0cmFpbl9yb290OiBQYXRoKSAtPiBkaWN0OgogICAgaXRlbXMgPSBbXQogICAgdG90YWwg'
            'PSAwCiAgICBtaXNzaW5nID0gW10KICAgIGZvciByZWcgaW4gcmVnaW9uczoKICAgICAgICBmb3Ig'
            'c3JjIGluIHBsYW5fZmlsZXMocmVnKToKICAgICAgICAgICAgaWYgbm90IHNyYy5pc19maWxlKCk6'
            'CiAgICAgICAgICAgICAgICBtaXNzaW5nLmFwcGVuZChzdHIoc3JjKSkKICAgICAgICAgICAgICAg'
            'IGNvbnRpbnVlCiAgICAgICAgICAgIHJlbCA9IHNyYy5yZWxhdGl2ZV90byhyZWcucGF0aCkKICAg'
            'ICAgICAgICAgZHN0X3JlbCA9IFBhdGgocmVnLm5hbWUpIC8gcmVsCiAgICAgICAgICAgIHN6ID0g'
            'ZmlsZV9zaXplKHNyYykKICAgICAgICAgICAgdG90YWwgKz0gc3oKICAgICAgICAgICAgaXRlbXMu'
            'YXBwZW5kKHsKICAgICAgICAgICAgICAgICJyZWdpb24iOiByZWcubmFtZSwKICAgICAgICAgICAg'
            'ICAgICJzcmMiOiBzdHIoc3JjKSwKICAgICAgICAgICAgICAgICJkc3RfcmVsIjogc3RyKGRzdF9y'
            'ZWwpLnJlcGxhY2UoIlxcIiwgIi8iKSwKICAgICAgICAgICAgICAgICJieXRlcyI6IHN6LAogICAg'
            'ICAgICAgICAgICAgIm1hcmt1cF9oaW50IjogX2hpbnQoc3JjKSwKICAgICAgICAgICAgfSkKICAg'
            'IHJldHVybiB7CiAgICAgICAgInRyYWluX3Jvb3QiOiBzdHIodHJhaW5fcm9vdCksCiAgICAgICAg'
            'Im5fcmVnaW9ucyI6IGxlbihyZWdpb25zKSwKICAgICAgICAibl9maWxlcyI6IGxlbihpdGVtcyks'
            'CiAgICAgICAgInRvdGFsX2J5dGVzIjogdG90YWwsCiAgICAgICAgInRvdGFsX2diIjogcm91bmQo'
            'dG90YWwgLyAoMTAyNCAqKiAzKSwgMyksCiAgICAgICAgIm1pc3NpbmciOiBtaXNzaW5nLAogICAg'
            'ICAgICJmaWxlcyI6IGl0ZW1zLAogICAgfQoKCmRlZiBfaGludChwOiBQYXRoKSAtPiBzdHI6CiAg'
            'ICBsb3cgPSBwLm5hbWUubG93ZXIoKQogICAgZm9yIG10IGluIEdFT19NVFM6CiAgICAgICAgaWYg'
            'bXQgaW4gbG93OgogICAgICAgICAgICByZXR1cm4gbXQKICAgIGlmIHAubmFtZSA9PSAiVVRNLmpz'
            'b24iOgogICAgICAgIHJldHVybiAidXRtIgogICAgIyBib3VuZGFyaWVzL25vaXNlIG1hdGNoZWQg'
            'YnkgY2F0YWxvZyBhbHJlYWR5CiAgICBpZiBwLnN1ZmZpeC5sb3dlcigpIGluICgiLmdlb2pzb24i'
            'LCAiLmpzb24iKToKICAgICAgICByZXR1cm4gImV4dHJhIgogICAgcmV0dXJuICJvdGhlciIKCgpk'
            'ZWYgZG9fY29weShwbGFuOiBkaWN0LCBkc3Rfcm9vdDogUGF0aCwgZHJ5X3J1bjogYm9vbCkgLT4g'
            'bGlzdFtzdHJdOgogICAgZXJyb3JzID0gW10KICAgIGZvciBpdCBpbiBwbGFuWyJmaWxlcyJdOgog'
            'ICAgICAgIHNyYyA9IFBhdGgoaXRbInNyYyJdKQogICAgICAgIGRzdCA9IGRzdF9yb290IC8gaXRb'
            'ImRzdF9yZWwiXQogICAgICAgIGlmIGRyeV9ydW46CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAg'
            'ICAgdHJ5OgogICAgICAgICAgICBkc3QucGFyZW50Lm1rZGlyKHBhcmVudHM9VHJ1ZSwgZXhpc3Rf'
            'b2s9VHJ1ZSkKICAgICAgICAgICAgc2h1dGlsLmNvcHkyKHNyYywgZHN0KQogICAgICAgIGV4Y2Vw'
            'dCBPU0Vycm9yIGFzIGV4YzoKICAgICAgICAgICAgZXJyb3JzLmFwcGVuZCgiJXMgLT4gJXM6ICVz'
            'IiAlIChzcmMsIGRzdCwgZXhjKSkKICAgIHJldHVybiBlcnJvcnMKCgpkZWYgbWFpbigpIC0+IGlu'
            'dDoKICAgIGFwID0gYXJncGFyc2UuQXJndW1lbnRQYXJzZXIoZGVzY3JpcHRpb249ImJvdDYgcjMg'
            'c2VsZWN0aXZlIGdlbyByZWdpb24gY29weSIpCiAgICBhcC5hZGRfYXJndW1lbnQoIi0tcmVwbyIs'
            'IHJlcXVpcmVkPVRydWUsIGhlbHA9Indvcmt0cmVlIHdpdGggZXhwZHMvIikKICAgIGFwLmFkZF9h'
            'cmd1bWVudCgiLS1yb290IiwgcmVxdWlyZWQ9VHJ1ZSwgaGVscD0idHJhaW4gcm9vdCBlLmcuIEY6'
            'LzIvdHJhaW4tZXhwZHMtMyIpCiAgICBhcC5hZGRfYXJndW1lbnQoIi0tZHN0IiwgcmVxdWlyZWQ9'
            'VHJ1ZSwgaGVscD0iZGVzdGluYXRpb24gcmVnaW9ucyBkaXIgb24gQzoiKQogICAgYXAuYWRkX2Fy'
            'Z3VtZW50KCItLXBsYW4tb3V0IiwgcmVxdWlyZWQ9VHJ1ZSwgaGVscD0iSlNPTiBwbGFuIHBhdGgi'
            'KQogICAgYXAuYWRkX2FyZ3VtZW50KCItLW1heC1nYiIsIHR5cGU9ZmxvYXQsIGRlZmF1bHQ9MTAu'
            'MCkKICAgIGFwLmFkZF9hcmd1bWVudCgiLS1jb3B5IiwgYWN0aW9uPSJzdG9yZV90cnVlIiwgaGVs'
            'cD0iYWN0dWFsbHkgY29weSAoZWxzZSBwbGFuIG9ubHkpIikKICAgIGFwLmFkZF9hcmd1bWVudCgi'
            'LS1kcnktcnVuIiwgYWN0aW9uPSJzdG9yZV90cnVlIikKICAgIGFyZ3MgPSBhcC5wYXJzZV9hcmdz'
            'KCkKCiAgICByZXBvID0gUGF0aChhcmdzLnJlcG8pCiAgICByb290ID0gUGF0aChhcmdzLnJvb3Qp'
            'CiAgICBkc3QgPSBQYXRoKGFyZ3MuZHN0KQogICAgaWYgbm90IHJvb3QuaXNfZGlyKCk6CiAgICAg'
            'ICAgcHJpbnQoanNvbi5kdW1wcyh7Im9rIjogRmFsc2UsICJlcnJvciI6ICJyb290X21pc3Npbmci'
            'LCAicm9vdCI6IHN0cihyb290KX0pKQogICAgICAgIHJldHVybiAxNgogICAgcmVnaW9ucywgbWV0'
            'YSA9IGRpc2NvdmVyKHJlcG8sIHJvb3QpCiAgICBpZiBub3QgcmVnaW9uczoKICAgICAgICBwcmlu'
            'dChqc29uLmR1bXBzKHsib2siOiBGYWxzZSwgImVycm9yIjogIm5vX2dlb19yZWdpb25zIiwgIm1l'
            'dGEiOiBtZXRhfSwgZW5zdXJlX2FzY2lpPUZhbHNlKSkKICAgICAgICByZXR1cm4gMTYKICAgIHBs'
            'YW4gPSBidWlsZF9wbGFuKHJlZ2lvbnMsIHJvb3QpCiAgICBwbGFuWyJtZXRhIl0gPSBtZXRhCiAg'
            'ICBwbGFuWyJtYXhfZ2IiXSA9IGFyZ3MubWF4X2diCiAgICBwbGFuWyJ3b3VsZF9leGNlZWQiXSA9'
            'IHBsYW5bInRvdGFsX2diIl0gPiBhcmdzLm1heF9nYgogICAgUGF0aChhcmdzLnBsYW5fb3V0KS5w'
            'YXJlbnQubWtkaXIocGFyZW50cz1UcnVlLCBleGlzdF9vaz1UcnVlKQogICAgUGF0aChhcmdzLnBs'
            'YW5fb3V0KS53cml0ZV90ZXh0KGpzb24uZHVtcHMocGxhbiwgZW5zdXJlX2FzY2lpPUZhbHNlLCBp'
            'bmRlbnQ9MiksIGVuY29kaW5nPSJ1dGYtOCIpCiAgICBzdW1tYXJ5ID0gewogICAgICAgICJvayI6'
            'IG5vdCBwbGFuWyJ3b3VsZF9leGNlZWQiXSBhbmQgbm90IHBsYW5bIm1pc3NpbmciXSwKICAgICAg'
            'ICAibl9yZWdpb25zIjogcGxhblsibl9yZWdpb25zIl0sCiAgICAgICAgIm5fZmlsZXMiOiBwbGFu'
            'WyJuX2ZpbGVzIl0sCiAgICAgICAgInRvdGFsX2diIjogcGxhblsidG90YWxfZ2IiXSwKICAgICAg'
            'ICAibWF4X2diIjogYXJncy5tYXhfZ2IsCiAgICAgICAgIndvdWxkX2V4Y2VlZCI6IHBsYW5bIndv'
            'dWxkX2V4Y2VlZCJdLAogICAgICAgICJtaXNzaW5nX24iOiBsZW4ocGxhblsibWlzc2luZyJdKSwK'
            'ICAgICAgICAic2VsZWN0aW9uIjogbWV0YVsic2VsZWN0aW9uIl0sCiAgICAgICAgInJlZ2lvbnMi'
            'OiBtZXRhWyJjaG9zZW4iXSwKICAgICAgICAicGxhbiI6IHN0cihhcmdzLnBsYW5fb3V0KSwKICAg'
            'IH0KICAgIHByaW50KGpzb24uZHVtcHMoc3VtbWFyeSwgZW5zdXJlX2FzY2lpPUZhbHNlKSkKICAg'
            'IGlmIHBsYW5bIndvdWxkX2V4Y2VlZCJdOgogICAgICAgIHJldHVybiAxNQogICAgaWYgYXJncy5j'
            'b3B5IGFuZCBub3QgYXJncy5kcnlfcnVuOgogICAgICAgIGRzdC5ta2RpcihwYXJlbnRzPVRydWUs'
            'IGV4aXN0X29rPVRydWUpCiAgICAgICAgZXJyb3JzID0gZG9fY29weShwbGFuLCBkc3QsIGRyeV9y'
            'dW49RmFsc2UpCiAgICAgICAgaWYgZXJyb3JzOgogICAgICAgICAgICAoZHN0LnBhcmVudCAvICJu'
            'b3RlcyIgLyAiY29weV9lcnJvcnMudHh0IikucGFyZW50Lm1rZGlyKHBhcmVudHM9VHJ1ZSwgZXhp'
            'c3Rfb2s9VHJ1ZSkKICAgICAgICAgICAgUGF0aChhcmdzLnBsYW5fb3V0KS53aXRoX3N1ZmZpeCgi'
            'LmVycm9ycy50eHQiKS53cml0ZV90ZXh0KAogICAgICAgICAgICAgICAgIlxuIi5qb2luKGVycm9y'
            'cyksIGVuY29kaW5nPSJ1dGYtOCIpCiAgICAgICAgICAgIHByaW50KGpzb24uZHVtcHMoeyJvayI6'
            'IEZhbHNlLCAiZXJyb3IiOiAiY29weV9mYWlsZWQiLCAibiI6IGxlbihlcnJvcnMpfSkpCiAgICAg'
            'ICAgICAgIHJldHVybiAxNwogICAgICAgICMgd3JpdGUgcmVnaW9uIGxpc3QKICAgICAgICAoZHN0'
            'IC8gIl9yZWdpb25fbGlzdC50eHQiKS53cml0ZV90ZXh0KAogICAgICAgICAgICAiXG4iLmpvaW4o'
            'bWV0YVsiY2hvc2VuIl0pICsgIlxuIiwgZW5jb2Rpbmc9InV0Zi04IikKICAgIHJldHVybiAwCgoK'
            'CgpkZWYgX3NlbGZfY2hlY2soKSAtPiBpbnQ6CiAgICAiIiJGaXh0dXJlIHRyZWUgdW5kZXIgL3Rt'
            'cDogcGxhbiBzaXplLCBhYm9ydCA+bWF4LWdiIChleGl0IDE1KSwgbm8gTEFTIGNvcGllZC4iIiIK'
            'ICAgIGltcG9ydCB0ZW1wZmlsZQogICAgcm9vdCA9IFBhdGgodGVtcGZpbGUubWtkdGVtcChwcmVm'
            'aXg9ImJvdDZfcjNfc2NfIikpCiAgICAjIE1pbmltYWwgZmFrZSBSZWdpb24tbGlrZSBsYXlvdXQg'
            'dGhlIGNhdGFsb2cgdW5kZXJzdGFuZHMgaXMgaGVhdnk7CiAgICAjIHVuaXQtdGVzdCBwbGFuX2Zp'
            'bGVzIC8gYnVpbGRfcGxhbiAvIG1heC1nYiBwYXRoIHdpdGggYSBzdHViIHJlZ2lvbiBvYmplY3Qu'
            'CiAgICBjbGFzcyBSOgogICAgICAgIGRlZiBfX2luaXRfXyhzZWxmLCBuYW1lLCBwYXRoLCByYXN0'
            'ZXJzLCBib3VuZGFyaWVzPU5vbmUsIG5vaXNlPU5vbmUsIHV0bV9qc29uPU5vbmUpOgogICAgICAg'
            'ICAgICBzZWxmLm5hbWUgPSBuYW1lCiAgICAgICAgICAgIHNlbGYucGF0aCA9IHBhdGgKICAgICAg'
            'ICAgICAgc2VsZi5yYXN0ZXJzID0gcmFzdGVycwogICAgICAgICAgICBzZWxmLmJvdW5kYXJpZXMg'
            'PSBib3VuZGFyaWVzIG9yIFtdCiAgICAgICAgICAgIHNlbGYubm9pc2UgPSBub2lzZSBvciBbXQog'
            'ICAgICAgICAgICBzZWxmLnV0bV9qc29uID0gdXRtX2pzb24KICAgIHJlZ19kaXIgPSByb290IC8g'
            'IjIwMF9YIgogICAgcmVnX2Rpci5ta2RpcigpCiAgICBtYWcgPSByZWdfZGlyIC8gIm1hZ2dyX2dy'
            'YXkudGlmIgogICAgbGFzID0gcmVnX2RpciAvICJwb2ludHMubGFzIgogICAgdXRtID0gcmVnX2Rp'
            'ciAvICJVVE0uanNvbiIKICAgIG1hZy53cml0ZV9ieXRlcyhiIjAiICogMTAyNCkKICAgIGxhcy53'
            'cml0ZV9ieXRlcyhiIkwiICogMjA0OCkKICAgIHV0bS53cml0ZV90ZXh0KCJ7fSIsIGVuY29kaW5n'
            'PSJ1dGYtOCIpCiAgICBibmQgPSByZWdfZGlyIC8gImJvdW5kYXJ5Lmdlb2pzb24iCiAgICBibmQu'
            'd3JpdGVfdGV4dCgneyJ0eXBlIjoiRmVhdHVyZUNvbGxlY3Rpb24iLCJmZWF0dXJlcyI6W119Jywg'
            'ZW5jb2Rpbmc9InV0Zi04IikKICAgIHJlZyA9IFIoIjIwMF9YIiwgcmVnX2RpciwgeyJtYWdnciI6'
            'IFttYWddLCAiZXRnciI6IFtdLCAidGVyZmwiOiBbXX0sCiAgICAgICAgICAgIGJvdW5kYXJpZXM9'
            'W2JuZF0sIG5vaXNlPVtdLCB1dG1fanNvbj11dG0pCiAgICBmaWxlcyA9IHBsYW5fZmlsZXMocmVn'
            'KQogICAgYXNzZXJ0IG1hZyBpbiBmaWxlcyBhbmQgdXRtIGluIGZpbGVzIGFuZCBibmQgaW4gZmls'
            'ZXMKICAgIGFzc2VydCBsYXMgbm90IGluIGZpbGVzLCAiTEFTIG11c3Qgbm90IGJlIGluIHBsYW4i'
            'CiAgICBwbGFuID0gYnVpbGRfcGxhbihbcmVnXSwgcm9vdCkKICAgIGFzc2VydCBwbGFuWyJ0b3Rh'
            'bF9ieXRlcyJdID09IDEwMjQgKyBsZW4odXRtLnJlYWRfYnl0ZXMoKSkgKyBsZW4oYm5kLnJlYWRf'
            'Ynl0ZXMoKSkKICAgIGFzc2VydCBwbGFuWyJuX2ZpbGVzIl0gPT0gMwogICAgIyBtYXgtZ2IgYWJv'
            'cnQ6IHRpbnkgdGhyZXNob2xkCiAgICBwbGFuWyJ0b3RhbF9nYiJdID0gMTEuMAogICAgcGxhblsi'
            'd291bGRfZXhjZWVkIl0gPSBwbGFuWyJ0b3RhbF9nYiJdID4gMTAuMAogICAgYXNzZXJ0IHBsYW5b'
            'IndvdWxkX2V4Y2VlZCJdCiAgICAjIHdyaXRlIHBsYW4gYW5kIHNpbXVsYXRlIGV4aXQgMTUgcGF0'
            'aAogICAgb3V0ID0gcm9vdCAvICJwbGFuLmpzb24iCiAgICBvdXQud3JpdGVfdGV4dChfX2ltcG9y'
            'dF9fKCJqc29uIikuZHVtcHMoeyJ0b3RhbF9nYiI6IDExLjAsICJ3b3VsZF9leGNlZWQiOiBUcnVl'
            'fSksIGVuY29kaW5nPSJ1dGYtOCIpCiAgICBwcmludChfX2ltcG9ydF9fKCJqc29uIikuZHVtcHMo'
            'ewogICAgICAgICJzZWxmX2NoZWNrX3Bhc3MiOiBUcnVlLAogICAgICAgICJuX2ZpbGVzIjogMywK'
            'ICAgICAgICAiZXhjbHVkZXNfbGFzIjogVHJ1ZSwKICAgICAgICAid291bGRfZXhjZWVkXzExZ2Ii'
            'OiBUcnVlLAogICAgICAgICJleGl0X2NvZGVfb25fZXhjZWVkIjogMTUsCiAgICB9KSkKICAgIHJl'
            'dHVybiAwCgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIGlmICItLXNlbGYtY2hlY2si'
            'IGluIHN5cy5hcmd2OgogICAgICAgIHN5cy5leGl0KF9zZWxmX2NoZWNrKCkpCiAgICBzeXMuZXhp'
            'dChtYWluKCkpCg=='
        ),
    },
    'rebuild_w4_geo_tree.py': {
        "sha256": '28a22e372ef7497d742a6eb42039f22cbfbd40fffaf6f7be326f5f4a97efae8e',
        "b64": (
            'IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiJSZWJ1aWxkIEtPWi0zIGdlby0yMDAtMjE3IHRyYWlu'
            'IHRyZWUgZm9yIHBpeGdibS1jciBnYXRlIChib3QtNikuCgpQcmltYXJ5IHNvdXJjZTogU2VsY2xv'
            'dWQgUzMgYnVja2V0IHRyYWluLWV4cGRzLTMgKE1hZ0dyL0V0R3IvVGVyRmwgcmFzdGVycyArCkdU'
            'IGxhYmVscyBmb3IgcmVnaW9ucyAyMDDigJMyMTcpLiBDYWNoZSB1bmRlciAva2FnZ2xlL3dvcmtp'
            'bmcgd2l0aCBza2lwLWlmLWV4aXN0cwphbmQgc2l6ZSBjaGVjay4gT2ZmbGluZSB3NC1nZW8gc2hh'
            'cmRzIGFyZSBOT1QgdXNlZCAoZG8gbm90IGV4aXN0KS4KClMzIGNsaWVudCBpcyBpbmxpbmVkIFZF'
            'UkJBVElNIGZyb20gYm90cy9ib3Q3LWdlby1nYXRlQDExYTQ3ZDJhCmthZ2dsZS9jb21tb24va296'
            'M19zMy5weSAoZ2l0IGJsb2IgN2UxZDBhYjExOWYyZDJjNjc4Zjc1MzU4MmIyMmE0MzQ0MTA0ZDVi'
            'YykuCiIiIgpmcm9tIF9fZnV0dXJlX18gaW1wb3J0IGFubm90YXRpb25zCgppbXBvcnQganNvbgpp'
            'bXBvcnQgb3MKaW1wb3J0IHN5cwppbXBvcnQgdGltZQpmcm9tIHBhdGhsaWIgaW1wb3J0IFBhdGgK'
            'CkdFT19QUkVGSVhFUyA9IHR1cGxlKGYie246MDNkfV8iIGZvciBuIGluIHJhbmdlKDIwMCwgMjE4'
            'KSkKR0VPX01PRF9UT0tFTlMgPSAoIl9NYWdHcl8iLCAiX0V0R3JfIiwgIl9UZXJGbF8iLCAiX21h'
            'Z2dyXyIsICJfZXRncl8iLCAiX3RlcmZsXyIpCiMgTGFiZWxzIC8gbWV0YSBrZXB0IGFsb25nc2lk'
            'ZSByYXN0ZXJzIChjYXRhbG9nL2d0IGxvYWRlcnMgbmVlZCB0aGVzZSkuCkxBQkVMX1NVRkZJWEVT'
            'ID0gKAogICAgIi5nZW9qc29uIiwKICAgICIuanNvbiIsCiAgICAiLmdwa2ciLAogICAgIi5zaHAi'
            'LAogICAgIi5kYmYiLAogICAgIi5zaHgiLAogICAgIi5wcmoiLAogICAgIi5jcGciLAopCkxBQkVM'
            'X05BTUVfSElOVFMgPSAoCiAgICAib2JqZWN0cG9seSIsCiAgICAiZmluZHNwb2ludHMiLAogICAg'
            'ItGA0LDQt9C80LXRgtC60LAiLAogICAgInV0bS5qc29uIiwKICAgICJvYmplY3RfcG9seSIsCiAg'
            'ICAiZmluZHNfcG9pbnRzIiwKKQoKIyAtLS0gQkVHSU4ga296M19zMyAoc291cmNlOiBib3RzL2Jv'
            'dDctZ2VvLWdhdGVAMTFhNDdkMmEga2FnZ2xlL2NvbW1vbi9rb3ozX3MzLnB5IGJsb2IgN2UxZDBh'
            'YjExOWYyZDJjNjc4Zjc1MzU4MmIyMmE0MzQ0MTA0ZDViYzsgbWlkLWZpbGUgb21pdDogZnJvbSBf'
            'X2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucykgLS0tCiIiIlNoYXJlZCBTMyBoZWxwZXIgZm9y'
            'IEtPWjMgS2FnZ2xlIGtlcm5lbHMgKGJvdDcg4oaSIGJvdDYgcmV1c2UpLgoKQVBJCi0tLQotIGBg'
            'czNfY2xpZW50KClgYCDihpIgYm90bzMgUzMgY2xpZW50Ci0gYGBkb3dubG9hZChrZXksIGRzdClg'
            'YCDihpIgUGF0aDsgc2tpcCBpZiBzaXplIG1hdGNoZXM7IHJldHJpZXM7IGBgLnBhcnRgYCB0aGVu'
            'IHJlbmFtZQotIGBgZG93bmxvYWRfcHJlZml4KHByZWZpeCwgZHN0KWBgIOKGkiBsaXN0W1BhdGhd'
            'OyBwcmVzZXJ2ZXMga2V5IHN1ZmZpeCB1bmRlciBgYGRzdGBgCi0gYGBsaXN0X2tleXMocHJlZml4'
            'KWBgIOKGkiBsaXN0W3R1cGxlW3N0ciwgaW50XV0gb2YgKGtleSwgc2l6ZSk7IHBhZ2luYXRlZAoK'
            'Q3JlZGVudGlhbHMgKG5ldmVyIGxvZ2dlZCAvIG5ldmVyIGluIGV4Y2VwdGlvbiB0ZXh0KQotLS0t'
            'LS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tCjEuIEthZ2ds'
            'ZSBTZWNyZXRzIGBgS09aM19TM19BQ0NFU1NfS0VZX0lEYGAgLyBgYEtPWjNfUzNfU0VDUkVUX0FD'
            'Q0VTU19LRVlgYAogICAoYWxpYXNlcyBvZiBtb2R1bGUgY29uc3RhbnRzIGBgU0VDUkVUX0lEYGAg'
            'LyBgYFNFQ1JFVF9LRVlgYCwgc2FtZSBhcwogICBgYGtvejMtZGZpbmUtb3IucHlgYCkuCjIuIEVu'
            'diBgYEFXU19BQ0NFU1NfS0VZX0lEYGAgLyBgYEFXU19TRUNSRVRfQUNDRVNTX0tFWWBgLgozLiBQ'
            'cml2YXRlIGRhdGFzZXQgYGBkbWl0cmlqa3Yva296My1zMy1jcmVkYGAgdW5kZXIgYGAva2FnZ2xl'
            'L2lucHV0LyoqYGA6CiAgIC0gYGBjcmVkLmpzb25gYCAvIGBgKi5qc29uYGAgd2l0aCBrZXlzIGBg'
            'aWRgYCtgYGtleWBgIG9yCiAgICAgYGBhd3NfYWNjZXNzX2tleV9pZGBgK2BgYXdzX3NlY3JldF9h'
            'Y2Nlc3Nfa2V5YGAgKG9yIEFjY2Vzc0tleUlkL1NlY3JldEFjY2Vzc0tleSkKICAgLSBgYCouZW52'
            'YGAgLyBgYGNyZWRlbnRpYWxzLmVudmBgIGxpbmVzIGBgS0VZPVZBTGBgCiAgIC0gYGAqLmluaWBg'
            'IC8gYGBjcmVkZW50aWFsc2BgIEFXUy1zdHlsZSBgYFtkZWZhdWx0XWBgIHdpdGgKICAgICBgYGF3'
            'c19hY2Nlc3Nfa2V5X2lkYGAgLyBgYGF3c19zZWNyZXRfYWNjZXNzX2tleWBgCgpEZXBzOiBib3Rv'
            'MyArIHN0ZGxpYiBvbmx5LiBCdWNrZXQgYGB0cmFpbi1leHBkcy0zYGAsIGVuZHBvaW50CmBgaHR0'
            'cHM6Ly9zMy5ydS03LnN0b3JhZ2Uuc2VsY2xvdWQucnVgYCwgcmVnaW9uIGBgcnUtN2BgLgoiIiIK'
            'CmltcG9ydCBjb25maWdwYXJzZXIKaW1wb3J0IGdsb2IKaW1wb3J0IGpzb24KaW1wb3J0IG9zCmlt'
            'cG9ydCByZQppbXBvcnQgdGltZQpmcm9tIHBhdGhsaWIgaW1wb3J0IFBhdGgKZnJvbSB0eXBpbmcg'
            'aW1wb3J0IEFueSwgSXRlcmFibGUKCkJVQ0tFVCA9ICJ0cmFpbi1leHBkcy0zIgpFTkRQT0lOVCA9'
            'IG9zLmVudmlyb24uZ2V0KCJLT1ozX1MzX0VORFBPSU5UIiwgImh0dHBzOi8vczMucnUtNy5zdG9y'
            'YWdlLnNlbGNsb3VkLnJ1IikKUkVHSU9OID0gInJ1LTciCiMgU2FtZSBuYW1lcyBhcyBkb2NzLy4u'
            'Li9rb3ozLWRmaW5lLW9yLnB5ClNFQ1JFVF9JRCA9ICJLT1ozX1MzX0FDQ0VTU19LRVlfSUQiClNF'
            'Q1JFVF9LRVkgPSAiS09aM19TM19TRUNSRVRfQUNDRVNTX0tFWSIKCl9JTlBVVF9HTE9CID0gIi9r'
            'YWdnbGUvaW5wdXQvKiovKiIKCgpjbGFzcyBTM0NyZWRlbnRpYWxzRXJyb3IoUnVudGltZUVycm9y'
            'KToKICAgICIiIk1pc3NpbmcgLyB1bnJlYWRhYmxlIFMzIGNyZWRlbnRpYWxzIChtZXNzYWdlIG5l'
            'dmVyIGNvbnRhaW5zIHNlY3JldCB2YWx1ZXMpLiIiIgoKCmNsYXNzIFMzVHJhbnNmZXJFcnJvcihS'
            'dW50aW1lRXJyb3IpOgogICAgIiIiRG93bmxvYWQgLyBsaXN0IGZhaWx1cmUgd2l0aCBjcmVkZW50'
            'aWFscyBzdHJpcHBlZCBmcm9tIHRoZSBtZXNzYWdlLiIiIgoKCmRlZiBfc2NydWIodGV4dDogc3Ry'
            'LCAqc2VjcmV0czogc3RyIHwgTm9uZSkgLT4gc3RyOgogICAgb3V0ID0gc3RyKHRleHQpCiAgICBm'
            'b3IgcyBpbiBzZWNyZXRzOgogICAgICAgIGlmIHMgYW5kIGxlbihzKSA+PSA0OgogICAgICAgICAg'
            'ICBvdXQgPSBvdXQucmVwbGFjZShzLCAiKioqIikKICAgICMgZ2VuZXJpYyBhY2Nlc3Mta2V5IC8g'
            'c2VjcmV0IHBhdHRlcm5zIChuZXZlciBsZWF2ZSByYXcgdmFsdWVzIGluIGVycm9ycykKICAgIG91'
            'dCA9IHJlLnN1YihyIig/aSlcYihBS0lBWzAtOUEtWl9dezgsfSlcYiIsICIqKioiLCBvdXQpCiAg'
            'ICBvdXQgPSByZS5zdWIociIoP2kpKHNlY3JldFxzKls9Ol1ccyopXFMrIiwgciJcMSoqKiIsIG91'
            'dCkKICAgIG91dCA9IHJlLnN1YihyIig/aSkoa2V5XHMqWz06XVxzKilcUysiLCByIlwxKioqIiwg'
            'b3V0KQogICAgb3V0ID0gcmUuc3ViKHIiKD9pKShhd3NbXy1dP3NlY3JldFtfLV0/YWNjZXNzW18t'
            'XT9rZXlccypbPTpdXHMqKVxTKyIsIHIiXDEqKioiLCBvdXQpCiAgICBvdXQgPSByZS5zdWIociIo'
            'P2kpKGF3c1tfLV0/YWNjZXNzW18tXT9rZXlbXy1dP2lkXHMqWz06XVxzKilcUysiLCByIlwxKioq'
            'Iiwgb3V0KQogICAgIyBsb25nIGJhc2U2NC1pc2ggdG9rZW5zIG9mdGVuIHVzZWQgYXMgc2VjcmV0'
            'IGFjY2VzcyBrZXlzCiAgICBvdXQgPSByZS5zdWIociIoPzwhW0EtWmEtejAtOS8rPV0pKFtBLVph'
            'LXowLTkvKz1dezM1LH0pKD8hW0EtWmEtejAtOS8rPV0pIiwgIioqKiIsIG91dCkKICAgIHJldHVy'
            'biBvdXQKCgpkZWYgX3dyYXAoZXhjOiBCYXNlRXhjZXB0aW9uLCAqc2VjcmV0czogc3RyIHwgTm9u'
            'ZSkgLT4gUzNUcmFuc2ZlckVycm9yOgogICAgcmV0dXJuIFMzVHJhbnNmZXJFcnJvcigKICAgICAg'
            'ICBmInt0eXBlKGV4YykuX19uYW1lX199OiB7X3NjcnViKHN0cihleGMpLCAqc2VjcmV0cyl9Igog'
            'ICAgKQoKCmRlZiBfcGFyc2VfanNvbl9jcmVkKHBhdGg6IFBhdGgpIC0+IHR1cGxlW3N0ciwgc3Ry'
            'XSB8IE5vbmU6CiAgICB0cnk6CiAgICAgICAgZGF0YSA9IGpzb24ubG9hZHMocGF0aC5yZWFkX3Rl'
            'eHQoZW5jb2Rpbmc9InV0Zi04IikpCiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIHJldHVy'
            'biBOb25lCiAgICBpZiBub3QgaXNpbnN0YW5jZShkYXRhLCBkaWN0KToKICAgICAgICByZXR1cm4g'
            'Tm9uZQogICAga2lkID0gKAogICAgICAgIGRhdGEuZ2V0KCJpZCIpCiAgICAgICAgb3IgZGF0YS5n'
            'ZXQoImF3c19hY2Nlc3Nfa2V5X2lkIikKICAgICAgICBvciBkYXRhLmdldCgiQWNjZXNzS2V5SWQi'
            'KQogICAgICAgIG9yIGRhdGEuZ2V0KCJhY2Nlc3Nfa2V5IikKICAgICkKICAgIHNlYyA9ICgKICAg'
            'ICAgICBkYXRhLmdldCgia2V5IikKICAgICAgICBvciBkYXRhLmdldCgiYXdzX3NlY3JldF9hY2Nl'
            'c3Nfa2V5IikKICAgICAgICBvciBkYXRhLmdldCgiU2VjcmV0QWNjZXNzS2V5IikKICAgICAgICBv'
            'ciBkYXRhLmdldCgic2VjcmV0X2tleSIpCiAgICApCiAgICBpZiBraWQgYW5kIHNlYzoKICAgICAg'
            'ICByZXR1cm4gc3RyKGtpZCksIHN0cihzZWMpCiAgICByZXR1cm4gTm9uZQoKCmRlZiBfcGFyc2Vf'
            'ZW52X2NyZWQocGF0aDogUGF0aCkgLT4gdHVwbGVbc3RyLCBzdHJdIHwgTm9uZToKICAgIGtpZCA9'
            'IHNlYyA9IE5vbmUKICAgIHRyeToKICAgICAgICBmb3IgcmF3IGluIHBhdGgucmVhZF90ZXh0KGVu'
            'Y29kaW5nPSJ1dGYtOCIpLnNwbGl0bGluZXMoKToKICAgICAgICAgICAgbGluZSA9IHJhdy5zdHJp'
            'cCgpCiAgICAgICAgICAgIGlmIG5vdCBsaW5lIG9yIGxpbmUuc3RhcnRzd2l0aCgiIyIpIG9yICI9'
            'IiBub3QgaW4gbGluZToKICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgICAgIGssIF8s'
            'IHYgPSBsaW5lLnBhcnRpdGlvbigiPSIpCiAgICAgICAgICAgIGssIHYgPSBrLnN0cmlwKCksIHYu'
            'c3RyaXAoKS5zdHJpcCgnIicpLnN0cmlwKCInIikKICAgICAgICAgICAga3UgPSBrLnVwcGVyKCkK'
            'ICAgICAgICAgICAgaWYga3UgaW4gKCJBV1NfQUNDRVNTX0tFWV9JRCIsICJLT1ozX1MzX0FDQ0VT'
            'U19LRVlfSUQiLCAiSUQiLCAiQUNDRVNTX0tFWSIpOgogICAgICAgICAgICAgICAga2lkID0gdgog'
            'ICAgICAgICAgICBlbGlmIGt1IGluICgKICAgICAgICAgICAgICAgICJBV1NfU0VDUkVUX0FDQ0VT'
            'U19LRVkiLAogICAgICAgICAgICAgICAgIktPWjNfUzNfU0VDUkVUX0FDQ0VTU19LRVkiLAogICAg'
            'ICAgICAgICAgICAgIktFWSIsCiAgICAgICAgICAgICAgICAiU0VDUkVUX0tFWSIsCiAgICAgICAg'
            'ICAgICAgICAiU0VDUkVUX0FDQ0VTU19LRVkiLAogICAgICAgICAgICApOgogICAgICAgICAgICAg'
            'ICAgc2VjID0gdgogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICByZXR1cm4gTm9uZQogICAg'
            'aWYga2lkIGFuZCBzZWM6CiAgICAgICAgcmV0dXJuIGtpZCwgc2VjCiAgICByZXR1cm4gTm9uZQoK'
            'CmRlZiBfcGFyc2VfaW5pX2NyZWQocGF0aDogUGF0aCkgLT4gdHVwbGVbc3RyLCBzdHJdIHwgTm9u'
            'ZToKICAgIGNwID0gY29uZmlncGFyc2VyLkNvbmZpZ1BhcnNlcigpCiAgICB0cnk6CiAgICAgICAg'
            'Y3AucmVhZChwYXRoKQogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICByZXR1cm4gTm9uZQog'
            'ICAgZm9yIHNlY3Rpb24gaW4gKCJkZWZhdWx0IiwgIkRFRkFVTFQiLCAiczMiLCAia296MyIpOgog'
            'ICAgICAgIGlmIHNlY3Rpb24gPT0gIkRFRkFVTFQiOgogICAgICAgICAgICBpdGVtcyA9IHtrLmxv'
            'd2VyKCk6IHYgZm9yIGssIHYgaW4gY3AuZGVmYXVsdHMoKS5pdGVtcygpfQogICAgICAgIGVsaWYg'
            'Y3AuaGFzX3NlY3Rpb24oc2VjdGlvbik6CiAgICAgICAgICAgIGl0ZW1zID0ge2subG93ZXIoKTog'
            'diBmb3IgaywgdiBpbiBjcC5pdGVtcyhzZWN0aW9uKX0KICAgICAgICBlbHNlOgogICAgICAgICAg'
            'ICBjb250aW51ZQogICAgICAgIGtpZCA9IGl0ZW1zLmdldCgiYXdzX2FjY2Vzc19rZXlfaWQiKSBv'
            'ciBpdGVtcy5nZXQoImlkIikKICAgICAgICBzZWMgPSBpdGVtcy5nZXQoImF3c19zZWNyZXRfYWNj'
            'ZXNzX2tleSIpIG9yIGl0ZW1zLmdldCgia2V5IikKICAgICAgICBpZiBraWQgYW5kIHNlYzoKICAg'
            'ICAgICAgICAgcmV0dXJuIGtpZCwgc2VjCiAgICByZXR1cm4gTm9uZQoKCmRlZiBfY3JlZF9mcm9t'
            'X2lucHV0cyhyb290X2dsb2I6IHN0ciB8IE5vbmUgPSBOb25lKSAtPiB0dXBsZVtzdHIsIHN0cl0g'
            'fCBOb25lOgogICAgIiIiU2NhbiBwcml2YXRlIGRhdGFzZXQgZmlsZXMgdW5kZXIgL2thZ2dsZS9p'
            'bnB1dCAoanNvbiAvIGVudiAvIGluaSkuIiIiCiAgICBpZiByb290X2dsb2IgaXMgTm9uZToKICAg'
            'ICAgICByb290X2dsb2IgPSBfSU5QVVRfR0xPQgogICAgcGF0aHMgPSBzb3J0ZWQoZ2xvYi5nbG9i'
            'KHJvb3RfZ2xvYiwgcmVjdXJzaXZlPVRydWUpKQogICAgIyBQcmVmZXIga25vd24gbmFtZXMgZmly'
            'c3QKICAgIHByZWZlcnJlZCA9IFtdCiAgICBvdGhlciA9IFtdCiAgICBmb3IgcCBpbiBwYXRoczoK'
            'ICAgICAgICBpZiBub3Qgb3MucGF0aC5pc2ZpbGUocCk6CiAgICAgICAgICAgIGNvbnRpbnVlCiAg'
            'ICAgICAgbmFtZSA9IG9zLnBhdGguYmFzZW5hbWUocCkubG93ZXIoKQogICAgICAgIGlmIG5hbWUg'
            'aW4gKCJjcmVkLmpzb24iLCAiY3JlZGVudGlhbHMiLCAiY3JlZGVudGlhbHMuZW52IiwgInMzLmVu'
            'diIsICJrb3ozLXMzLmVudiIpOgogICAgICAgICAgICBwcmVmZXJyZWQuYXBwZW5kKHApCiAgICAg'
            'ICAgZWxpZiBuYW1lLmVuZHN3aXRoKCgiLmpzb24iLCAiLmVudiIsICIuaW5pIiwgIi5jZmciKSkg'
            'b3IgbmFtZSA9PSAiY3JlZGVudGlhbHMiOgogICAgICAgICAgICBvdGhlci5hcHBlbmQocCkKICAg'
            'IGZvciBwIGluIHByZWZlcnJlZCArIG90aGVyOgogICAgICAgIHBhdGggPSBQYXRoKHApCiAgICAg'
            'ICAgbmFtZSA9IHBhdGgubmFtZS5sb3dlcigpCiAgICAgICAgcGFyc2VkID0gTm9uZQogICAgICAg'
            'IGlmIG5hbWUuZW5kc3dpdGgoIi5qc29uIik6CiAgICAgICAgICAgIHBhcnNlZCA9IF9wYXJzZV9q'
            'c29uX2NyZWQocGF0aCkKICAgICAgICBlbGlmIG5hbWUuZW5kc3dpdGgoIi5lbnYiKSBvciBuYW1l'
            'LmVuZHN3aXRoKCIuY2ZnIik6CiAgICAgICAgICAgIHBhcnNlZCA9IF9wYXJzZV9lbnZfY3JlZChw'
            'YXRoKQogICAgICAgIGVsaWYgbmFtZS5lbmRzd2l0aCgiLmluaSIpIG9yIG5hbWUgPT0gImNyZWRl'
            'bnRpYWxzIjoKICAgICAgICAgICAgcGFyc2VkID0gX3BhcnNlX2luaV9jcmVkKHBhdGgpIG9yIF9w'
            'YXJzZV9lbnZfY3JlZChwYXRoKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIGNvbnRpbnVlCiAg'
            'ICAgICAgaWYgcGFyc2VkOgogICAgICAgICAgICByZXR1cm4gcGFyc2VkCiAgICByZXR1cm4gTm9u'
            'ZQoKCmRlZiBfbG9hZF9jcmVkZW50aWFscygpIC0+IHR1cGxlW3N0ciwgc3RyXToKICAgIGtpZCA9'
            'IHNlYyA9IE5vbmUKICAgIHRyeToKICAgICAgICBmcm9tIGthZ2dsZV9zZWNyZXRzIGltcG9ydCBV'
            'c2VyU2VjcmV0c0NsaWVudCAgIyB0eXBlOiBpZ25vcmUKCiAgICAgICAgdSA9IFVzZXJTZWNyZXRz'
            'Q2xpZW50KCkKICAgICAgICBraWQsIHNlYyA9IHUuZ2V0X3NlY3JldChTRUNSRVRfSUQpLCB1Lmdl'
            'dF9zZWNyZXQoU0VDUkVUX0tFWSkKICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAga2lkID0g'
            'c2VjID0gTm9uZQogICAgaWYgbm90IGtpZCBvciBub3Qgc2VjOgogICAgICAgIGtpZCA9IG9zLmVu'
            'dmlyb24uZ2V0KCJBV1NfQUNDRVNTX0tFWV9JRCIpIG9yIG9zLmVudmlyb24uZ2V0KFNFQ1JFVF9J'
            'RCkKICAgICAgICBzZWMgPSBvcy5lbnZpcm9uLmdldCgiQVdTX1NFQ1JFVF9BQ0NFU1NfS0VZIikg'
            'b3Igb3MuZW52aXJvbi5nZXQoU0VDUkVUX0tFWSkKICAgIGlmIG5vdCBraWQgb3Igbm90IHNlYzoK'
            'ICAgICAgICBnb3QgPSBfY3JlZF9mcm9tX2lucHV0cygpCiAgICAgICAgaWYgZ290OgogICAgICAg'
            'ICAgICBraWQsIHNlYyA9IGdvdAogICAgaWYgbm90IGtpZCBvciBub3Qgc2VjOgogICAgICAgIHJh'
            'aXNlIFMzQ3JlZGVudGlhbHNFcnJvcigKICAgICAgICAgICAgZiLQvdC10YIg0LrQu9GO0YfQtdC5'
            'IFMzOiBLYWdnbGUgU2VjcmV0cyB7U0VDUkVUX0lEfS97U0VDUkVUX0tFWX0sICIKICAgICAgICAg'
            'ICAgZiJlbnYgQVdTXyosINC40LvQuCDQtNCw0YLQsNGB0LXRgiBkbWl0cmlqa3Yva296My1zMy1j'
            'cmVkICIKICAgICAgICAgICAgZiIoY3JlZC5qc29uIHwgKi5lbnYgfCBjcmVkZW50aWFscy5pbmkg'
            '0L/QvtC0IC9rYWdnbGUvaW5wdXQpIgogICAgICAgICkKICAgIHJldHVybiBraWQsIHNlYwoKCmRl'
            'ZiBzM19jbGllbnQoKSAtPiBBbnk6CiAgICAiIiJCdWlsZCBhIGJvdG8zIFMzIGNsaWVudDsgY3Jl'
            'ZGVudGlhbHMgbmV2ZXIgYXBwZWFyIGluIHJhaXNlZCBtZXNzYWdlcy4iIiIKICAgIGltcG9ydCBi'
            'b3RvMwogICAgZnJvbSBib3RvY29yZS5jb25maWcgaW1wb3J0IENvbmZpZwoKICAgIGtpZCwgc2Vj'
            'ID0gX2xvYWRfY3JlZGVudGlhbHMoKQogICAgdHJ5OgogICAgICAgIHJldHVybiBib3RvMy5jbGll'
            'bnQoCiAgICAgICAgICAgICJzMyIsCiAgICAgICAgICAgIGVuZHBvaW50X3VybD1FTkRQT0lOVCwK'
            'ICAgICAgICAgICAgYXdzX2FjY2Vzc19rZXlfaWQ9a2lkLAogICAgICAgICAgICBhd3Nfc2VjcmV0'
            'X2FjY2Vzc19rZXk9c2VjLAogICAgICAgICAgICByZWdpb25fbmFtZT1SRUdJT04sCiAgICAgICAg'
            'ICAgIGNvbmZpZz1Db25maWcoCiAgICAgICAgICAgICAgICByZXRyaWVzPXsibWF4X2F0dGVtcHRz'
            'IjogMTAsICJtb2RlIjogImFkYXB0aXZlIn0sCiAgICAgICAgICAgICAgICBtYXhfcG9vbF9jb25u'
            'ZWN0aW9ucz0zMiwKICAgICAgICAgICAgKSwKICAgICAgICApCiAgICBleGNlcHQgRXhjZXB0aW9u'
            'IGFzIGU6CiAgICAgICAgcmFpc2UgX3dyYXAoZSwga2lkLCBzZWMpIGZyb20gTm9uZQoKCmRlZiBs'
            'aXN0X2tleXMocHJlZml4OiBzdHIsICosIGNsaWVudDogQW55IHwgTm9uZSA9IE5vbmUpIC0+IGxp'
            'c3RbdHVwbGVbc3RyLCBpbnRdXToKICAgICIiIlBhZ2luYXRlZCBgYGxpc3Rfb2JqZWN0c192MmBg'
            'IOKGkiBgYFsoa2V5LCBzaXplKSwgLi4uXWBgIHVuZGVyIGBgcHJlZml4YGAuIiIiCiAgICBjbGkg'
            'PSBjbGllbnQgb3IgczNfY2xpZW50KCkKICAgIGtpZCA9IHNlYyA9IE5vbmUgICMgc2NydWIgdW5r'
            'bm93bnMgZnJvbSBib3RvIG1zZ3MKICAgIG91dDogbGlzdFt0dXBsZVtzdHIsIGludF1dID0gW10K'
            'ICAgIHRva2VuID0gTm9uZQogICAgdHJ5OgogICAgICAgIHdoaWxlIFRydWU6CiAgICAgICAgICAg'
            'IGt3OiBkaWN0W3N0ciwgQW55XSA9IHsiQnVja2V0IjogQlVDS0VULCAiUHJlZml4IjogcHJlZml4'
            'fQogICAgICAgICAgICBpZiB0b2tlbjoKICAgICAgICAgICAgICAgIGt3WyJDb250aW51YXRpb25U'
            'b2tlbiJdID0gdG9rZW4KICAgICAgICAgICAgcmVzcCA9IGNsaS5saXN0X29iamVjdHNfdjIoKipr'
            'dykKICAgICAgICAgICAgZm9yIG9iaiBpbiByZXNwLmdldCgiQ29udGVudHMiKSBvciBbXToKICAg'
            'ICAgICAgICAgICAgIG91dC5hcHBlbmQoKG9ialsiS2V5Il0sIGludChvYmpbIlNpemUiXSkpKQog'
            'ICAgICAgICAgICBpZiBub3QgcmVzcC5nZXQoIklzVHJ1bmNhdGVkIik6CiAgICAgICAgICAgICAg'
            'ICBicmVhawogICAgICAgICAgICB0b2tlbiA9IHJlc3AuZ2V0KCJOZXh0Q29udGludWF0aW9uVG9r'
            'ZW4iKQogICAgZXhjZXB0IFMzQ3JlZGVudGlhbHNFcnJvcjoKICAgICAgICByYWlzZQogICAgZXhj'
            'ZXB0IEV4Y2VwdGlvbiBhcyBlOgogICAgICAgIHJhaXNlIF93cmFwKGUsIGtpZCwgc2VjKSBmcm9t'
            'IE5vbmUKICAgIHJldHVybiBvdXQKCgpkZWYgZG93bmxvYWQoCiAgICBrZXk6IHN0ciwKICAgIGRz'
            'dDogc3RyIHwgUGF0aCwKICAgICosCiAgICBjbGllbnQ6IEFueSB8IE5vbmUgPSBOb25lLAogICAg'
            'ZXhwZWN0ZWRfc2l6ZTogaW50IHwgTm9uZSA9IE5vbmUsCiAgICByZXRyaWVzOiBpbnQgPSA1LAop'
            'IC0+IFBhdGg6CiAgICAiIiJEb3dubG9hZCBgYGtleWBgIOKGkiBgYGRzdGBgIChmaWxlIHBhdGgp'
            'LiBTa2lwIGlmIHNpemUgYWxyZWFkeSBtYXRjaGVzLiIiIgogICAgY2xpID0gY2xpZW50IG9yIHMz'
            'X2NsaWVudCgpCiAgICBkc3RfcCA9IFBhdGgoZHN0KQogICAgaWYgZXhwZWN0ZWRfc2l6ZSBpcyBO'
            'b25lOgogICAgICAgICMgcmVzb2x2ZSBzaXplIHZpYSBIRUFEIHdoZW4gcG9zc2libGUKICAgICAg'
            'ICB0cnk6CiAgICAgICAgICAgIG1ldGEgPSBjbGkuaGVhZF9vYmplY3QoQnVja2V0PUJVQ0tFVCwg'
            'S2V5PWtleSkKICAgICAgICAgICAgZXhwZWN0ZWRfc2l6ZSA9IGludChtZXRhWyJDb250ZW50TGVu'
            'Z3RoIl0pCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgZXhwZWN0ZWRfc2l6'
            'ZSA9IE5vbmUKICAgIGlmICgKICAgICAgICBkc3RfcC5leGlzdHMoKQogICAgICAgIGFuZCBleHBl'
            'Y3RlZF9zaXplIGlzIG5vdCBOb25lCiAgICAgICAgYW5kIGRzdF9wLnN0YXQoKS5zdF9zaXplID09'
            'IGV4cGVjdGVkX3NpemUKICAgICk6CiAgICAgICAgcmV0dXJuIGRzdF9wCiAgICBmcm9tIGJvdG8z'
            'LnMzLnRyYW5zZmVyIGltcG9ydCBUcmFuc2ZlckNvbmZpZyAgIyBsYXp5OiBvbmx5IHdoZW4gdHJh'
            'bnNmZXJyaW5nCgogICAgZHN0X3AucGFyZW50Lm1rZGlyKHBhcmVudHM9VHJ1ZSwgZXhpc3Rfb2s9'
            'VHJ1ZSkKICAgIHRtcCA9IGRzdF9wLndpdGhfbmFtZShkc3RfcC5uYW1lICsgIi5wYXJ0IikKICAg'
            'IGxhc3RfZXJyOiBCYXNlRXhjZXB0aW9uIHwgTm9uZSA9IE5vbmUKICAgIGZvciBhdHRlbXB0IGlu'
            'IHJhbmdlKHJldHJpZXMpOgogICAgICAgIHRyeToKICAgICAgICAgICAgY2xpLmRvd25sb2FkX2Zp'
            'bGUoCiAgICAgICAgICAgICAgICBCVUNLRVQsCiAgICAgICAgICAgICAgICBrZXksCiAgICAgICAg'
            'ICAgICAgICBzdHIodG1wKSwKICAgICAgICAgICAgICAgIENvbmZpZz1UcmFuc2ZlckNvbmZpZyht'
            'YXhfY29uY3VycmVuY3k9NCksCiAgICAgICAgICAgICkKICAgICAgICAgICAgaWYgZXhwZWN0ZWRf'
            'c2l6ZSBpcyBub3QgTm9uZSBhbmQgdG1wLnN0YXQoKS5zdF9zaXplICE9IGV4cGVjdGVkX3NpemU6'
            'CiAgICAgICAgICAgICAgICByYWlzZSBTM1RyYW5zZmVyRXJyb3IoCiAgICAgICAgICAgICAgICAg'
            'ICAgZiJzaXplIG1pc21hdGNoIGZvciBrZXk9e2tleSFyfTogZ290IHt0bXAuc3RhdCgpLnN0X3Np'
            'emV9IHdhbnQge2V4cGVjdGVkX3NpemV9IgogICAgICAgICAgICAgICAgKQogICAgICAgICAgICB0'
            'bXAucmVwbGFjZShkc3RfcCkKICAgICAgICAgICAgcmV0dXJuIGRzdF9wCiAgICAgICAgZXhjZXB0'
            'IFMzVHJhbnNmZXJFcnJvcjoKICAgICAgICAgICAgcmFpc2UKICAgICAgICBleGNlcHQgRXhjZXB0'
            'aW9uIGFzIGU6CiAgICAgICAgICAgIGxhc3RfZXJyID0gZQogICAgICAgICAgICB0aW1lLnNsZWVw'
            'KDUgKiAoYXR0ZW1wdCArIDEpKQogICAgYXNzZXJ0IGxhc3RfZXJyIGlzIG5vdCBOb25lCiAgICBy'
            'YWlzZSBfd3JhcChsYXN0X2VycikgZnJvbSBOb25lCgoKZGVmIGRvd25sb2FkX3ByZWZpeCgKICAg'
            'IHByZWZpeDogc3RyLAogICAgZHN0OiBzdHIgfCBQYXRoLAogICAgKiwKICAgIGNsaWVudDogQW55'
            'IHwgTm9uZSA9IE5vbmUsCiAgICBrZXlzOiBJdGVyYWJsZVt0dXBsZVtzdHIsIGludF1dIHwgTm9u'
            'ZSA9IE5vbmUsCikgLT4gbGlzdFtQYXRoXToKICAgICIiIkRvd25sb2FkIGV2ZXJ5IG9iamVjdCB1'
            'bmRlciBgYHByZWZpeGBgIGludG8gYGBkc3RgYCwgcHJlc2VydmluZyByZWxhdGl2ZSBrZXlzLgoK'
            'ICAgIElmIGBga2V5c2BgIGlzIGdpdmVuIChmcm9tIGBgbGlzdF9rZXlzYGApLCB1c2UgaXQ7IGVs'
            'c2UgbGlzdCBmaXJzdC4KICAgIFJlbGF0aXZlIHBhdGggPSBrZXkgd2l0aCBgYHByZWZpeGBgIHN0'
            'cmlwcGVkIChsZWFkaW5nIHNsYXNoZXMgcmVtb3ZlZCkuCiAgICAiIiIKICAgIGNsaSA9IGNsaWVu'
            'dCBvciBzM19jbGllbnQoKQogICAgcm9vdCA9IFBhdGgoZHN0KQogICAgcm9vdC5ta2RpcihwYXJl'
            'bnRzPVRydWUsIGV4aXN0X29rPVRydWUpCiAgICBpdGVtcyA9IGxpc3Qoa2V5cykgaWYga2V5cyBp'
            'cyBub3QgTm9uZSBlbHNlIGxpc3Rfa2V5cyhwcmVmaXgsIGNsaWVudD1jbGkpCiAgICBwcmVmID0g'
            'cHJlZml4IGlmIHByZWZpeC5lbmRzd2l0aCgiLyIpIGVsc2UgcHJlZml4CiAgICBvdXQ6IGxpc3Rb'
            'UGF0aF0gPSBbXQogICAgZm9yIGtleSwgc2l6ZSBpbiBpdGVtczoKICAgICAgICBpZiBrZXkuZW5k'
            'c3dpdGgoIi8iKToKICAgICAgICAgICAgY29udGludWUKICAgICAgICByZWwgPSBrZXlbbGVuKHBy'
            'ZWYpIDpdIGlmIGtleS5zdGFydHN3aXRoKHByZWYpIGVsc2Uga2V5CiAgICAgICAgcmVsID0gcmVs'
            'LmxzdHJpcCgiLyIpCiAgICAgICAgdGFyZ2V0ID0gcm9vdCAvIHJlbAogICAgICAgIG91dC5hcHBl'
            'bmQoZG93bmxvYWQoa2V5LCB0YXJnZXQsIGNsaWVudD1jbGksIGV4cGVjdGVkX3NpemU9c2l6ZSkp'
            'CiAgICByZXR1cm4gb3V0CgoKX19hbGxfXyA9IFsKICAgICJCVUNLRVQiLAogICAgIkVORFBPSU5U'
            'IiwKICAgICJSRUdJT04iLAogICAgIlNFQ1JFVF9JRCIsCiAgICAiU0VDUkVUX0tFWSIsCiAgICAi'
            'UzNDcmVkZW50aWFsc0Vycm9yIiwKICAgICJTM1RyYW5zZmVyRXJyb3IiLAogICAgInMzX2NsaWVu'
            'dCIsCiAgICAibGlzdF9rZXlzIiwKICAgICJkb3dubG9hZCIsCiAgICAiZG93bmxvYWRfcHJlZml4'
            'IiwKXQojIC0tLSBFTkQga296M19zMyAtLS0KCgpkZWYgX3dhbnRfa2V5KGtleTogc3RyKSAtPiBi'
            'b29sOgogICAgIiIiS2VlcCBNYWdHci9FdEdyL1RlckZsIHJhc3RlcnMgKyBHVC9tZXRhIGZvciBy'
            'ZWdpb25zIDIwMOKAkzIxNyBvbmx5LiIiIgogICAga2V5X3UgPSBrZXkucmVwbGFjZSgiXFwiLCAi'
            'LyIpCiAgICB0b3AgPSBrZXlfdS5zcGxpdCgiLyIsIDEpWzBdCiAgICBpZiBub3QgYW55KHRvcC5z'
            'dGFydHN3aXRoKHApIGZvciBwIGluIEdFT19QUkVGSVhFUyk6CiAgICAgICAgcmV0dXJuIEZhbHNl'
            'CiAgICBuYW1lID0ga2V5X3UucnNwbGl0KCIvIiwgMSlbLTFdCiAgICBsb3cgPSBuYW1lLmxvd2Vy'
            'KCkKICAgICMgbW9kYWxpdHkgZm9sZGVyIG9yIGZpbGVuYW1lCiAgICBpZiBhbnkodG9rIGluIGtl'
            'eV91IGZvciB0b2sgaW4gR0VPX01PRF9UT0tFTlMpOgogICAgICAgIHJldHVybiBUcnVlCiAgICBp'
            'ZiBhbnkoaCBpbiBsb3cgZm9yIGggaW4gTEFCRUxfTkFNRV9ISU5UUyk6CiAgICAgICAgcmV0dXJu'
            'IFRydWUKICAgIGlmIGxvdyA9PSAidXRtLmpzb24iOgogICAgICAgIHJldHVybiBUcnVlCiAgICAj'
            'IGdlb2pzb24vc2hwIHNpdHRpbmcgbmV4dCB0byBnZW8gbW9kYWxpdGllcyBpbiByZWdpb24gcm9v'
            'dAogICAgaWYgYW55KGxvdy5lbmRzd2l0aChzdWYpIGZvciBzdWYgaW4gTEFCRUxfU1VGRklYRVMp'
            'OgogICAgICAgICMgZHJvcCBvYnZpb3VzIG5vbi1nZW8gbW9kYWxpdGllcyBpbiBwYXRoCiAgICAg'
            'ICAgaWYgYW55KHggaW4ga2V5X3UgZm9yIHggaW4gKCJfU3BPcl8iLCAiX0FlXyIsICJfT3JfIiwg'
            'Il9MYV8iLCAiX0xpXyIsICJfTGlEQVJfIikpOgogICAgICAgICAgICAjIHN0aWxsIGFsbG93IGlm'
            'IGZpbGVuYW1lIGl0c2VsZiBpcyBPYmplY3RQb2x5L0ZpbmRzUG9pbnRzIHN0eWxlCiAgICAgICAg'
            'ICAgIGlmIGFueShoIGluIGxvdyBmb3IgaCBpbiAoIm9iamVjdHBvbHkiLCAiZmluZHNwb2ludHMi'
            'LCAi0YDQsNC30LzQtdGC0LrQsCIsICJ1dG0iKSk6CiAgICAgICAgICAgICAgICByZXR1cm4gVHJ1'
            'ZQogICAgICAgICAgICByZXR1cm4gRmFsc2UKICAgICAgICByZXR1cm4gVHJ1ZQogICAgcmV0dXJu'
            'IEZhbHNlCgoKZGVmIGZpbmRfaW5wdXRzKHJvb3Q6IFBhdGggfCBOb25lID0gTm9uZSkgLT4gZGlj'
            'dFtzdHIsIFBhdGhdOgogICAgcm9vdCA9IHJvb3Qgb3IgUGF0aCgiL2thZ2dsZS9pbnB1dCIpCiAg'
            'ICBvdXQ6IGRpY3Rbc3RyLCBQYXRoXSA9IHt9CiAgICBpZiBub3Qgcm9vdC5pc19kaXIoKToKICAg'
            'ICAgICByZXR1cm4gb3V0CiAgICBmb3IgcCBpbiBzb3J0ZWQocm9vdC5pdGVyZGlyKCkpOgogICAg'
            'ICAgIGlmIHAuaXNfZGlyKCk6CiAgICAgICAgICAgIG91dFtwLm5hbWUubG93ZXIoKV0gPSBwCiAg'
            'ICByZXR1cm4gb3V0CgoKZGVmIGxpc3RfZ2VvX3JlZ2lvbl9wcmVmaXhlcygpIC0+IGxpc3Rbc3Ry'
            'XToKICAgICIiIkxpc3QgYnVja2V0IHRvcC1sZXZlbCByZWdpb24gZm9sZGVycyB3aG9zZSBuYW1l'
            'cyBzdGFydCB3aXRoIDIwMF/igKYyMTdfLiIiIgogICAgY2xpID0gczNfY2xpZW50KCkKICAgIHRv'
            'ayA9IE5vbmUKICAgIGZvdW5kOiBsaXN0W3N0cl0gPSBbXQogICAgd2hpbGUgVHJ1ZToKICAgICAg'
            'ICBrdyA9IHsiQnVja2V0IjogQlVDS0VULCAiUHJlZml4IjogIiIsICJEZWxpbWl0ZXIiOiAiLyJ9'
            'CiAgICAgICAgaWYgdG9rOgogICAgICAgICAgICBrd1siQ29udGludWF0aW9uVG9rZW4iXSA9IHRv'
            'awogICAgICAgIHIgPSBjbGkubGlzdF9vYmplY3RzX3YyKCoqa3cpCiAgICAgICAgZm9yIHAgaW4g'
            'ci5nZXQoIkNvbW1vblByZWZpeGVzIiwgW10pIG9yIFtdOgogICAgICAgICAgICBwcmVmID0gcFsi'
            'UHJlZml4Il0KICAgICAgICAgICAgbmFtZSA9IHByZWYucnN0cmlwKCIvIikKICAgICAgICAgICAg'
            'aWYgYW55KG5hbWUuc3RhcnRzd2l0aChncCkgZm9yIGdwIGluIEdFT19QUkVGSVhFUyk6CiAgICAg'
            'ICAgICAgICAgICBmb3VuZC5hcHBlbmQocHJlZikKICAgICAgICBpZiBub3Qgci5nZXQoIklzVHJ1'
            'bmNhdGVkIik6CiAgICAgICAgICAgIGJyZWFrCiAgICAgICAgdG9rID0gclsiTmV4dENvbnRpbnVh'
            'dGlvblRva2VuIl0KICAgIHJldHVybiBzb3J0ZWQoZm91bmQpCgoKZGVmIGZldGNoX2dlb190cmVl'
            'KGRlc3Q6IFBhdGgpIC0+IGRpY3Q6CiAgICAiIiJMaXN0IFMzIHByZWZpeGVzIGZvciAyMDDigJMy'
            'MTcsIGZpbHRlciBNYWdHci9FdEdyL1RlckZsICsgbGFiZWxzLCBjYWNoZSB1bmRlciBkZXN0LiIi'
            'IgogICAgZGVzdCA9IFBhdGgoZGVzdCkKICAgIGRlc3QubWtkaXIocGFyZW50cz1UcnVlLCBleGlz'
            'dF9vaz1UcnVlKQogICAgcmVwb3J0OiBkaWN0ID0gewogICAgICAgICJidWNrZXQiOiBCVUNLRVQs'
            'CiAgICAgICAgImVuZHBvaW50IjogRU5EUE9JTlQsCiAgICAgICAgInJlZ2lvbiI6IFJFR0lPTiwK'
            'ICAgICAgICAicm9vdCI6IHN0cihkZXN0KSwKICAgICAgICAicHJlZml4ZXMiOiBbXSwKICAgICAg'
            'ICAiZG93bmxvYWRlZCI6IDAsCiAgICAgICAgInNraXBwZWRfZXhpc3RzIjogMCwKICAgICAgICAi'
            'ZmlsdGVyZWRfb3V0IjogMCwKICAgICAgICAiZXJyb3JzIjogW10sCiAgICAgICAgImtleXMiOiBb'
            'XSwKICAgIH0KICAgIHRyeToKICAgICAgICBwcmVmaXhlcyA9IGxpc3RfZ2VvX3JlZ2lvbl9wcmVm'
            'aXhlcygpCiAgICBleGNlcHQgKFN5c3RlbUV4aXQsIFMzQ3JlZGVudGlhbHNFcnJvciwgUzNUcmFu'
            'c2ZlckVycm9yKSBhcyBlOgogICAgICAgIHJlcG9ydFsiZXJyb3JzIl0uYXBwZW5kKGYie3R5cGUo'
            'ZSkuX19uYW1lX199OntlfSIpCiAgICAgICAgcmVwb3J0WyJob2xsb3ciXSA9IFRydWUKICAgICAg'
            'ICByZXR1cm4gcmVwb3J0CiAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGU6CiAgICAgICAgcmVwb3J0'
            'WyJlcnJvcnMiXS5hcHBlbmQoZiJsaXN0X3ByZWZpeGVzOnt0eXBlKGUpLl9fbmFtZV9ffSIpCiAg'
            'ICAgICAgcmVwb3J0WyJob2xsb3ciXSA9IFRydWUKICAgICAgICByZXR1cm4gcmVwb3J0CgogICAg'
            'cmVwb3J0WyJwcmVmaXhlcyJdID0gcHJlZml4ZXMKICAgIGNsaSA9IHMzX2NsaWVudCgpCiAgICAj'
            'IFByb2JlIG9uY2UgKG5vIGtleSBtYXRlcmlhbCBpbiBtZXNzYWdlKQogICAgdHJ5OgogICAgICAg'
            'IGNsaS5saXN0X29iamVjdHNfdjIoQnVja2V0PUJVQ0tFVCwgTWF4S2V5cz0xKQogICAgICAgIHJl'
            'cG9ydFsiczNfb2siXSA9IFRydWUKICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZToKICAgICAgICBy'
            'ZXBvcnRbInMzX29rIl0gPSBGYWxzZQogICAgICAgIHJlcG9ydFsiZXJyb3JzIl0uYXBwZW5kKGYi'
            'UzMg0L3QtdC00L7RgdGC0YPQv9C10L0gKHt0eXBlKGUpLl9fbmFtZV9ffSkiKQogICAgICAgIHJl'
            'cG9ydFsiaG9sbG93Il0gPSBUcnVlCiAgICAgICAgcmV0dXJuIHJlcG9ydAoKICAgIGZvciBwcmVm'
            'IGluIHByZWZpeGVzOgogICAgICAgIGZvciBrZXksIHNpemUgaW4gbGlzdF9rZXlzKHByZWYsIGNs'
            'aWVudD1jbGkpOgogICAgICAgICAgICBpZiBrZXkuZW5kc3dpdGgoIi8iKToKICAgICAgICAgICAg'
            'ICAgIGNvbnRpbnVlCiAgICAgICAgICAgIGlmIG5vdCBfd2FudF9rZXkoa2V5KToKICAgICAgICAg'
            'ICAgICAgIHJlcG9ydFsiZmlsdGVyZWRfb3V0Il0gKz0gMQogICAgICAgICAgICAgICAgY29udGlu'
            'dWUKICAgICAgICAgICAgbG9jYWwgPSBkZXN0IC8ga2V5CiAgICAgICAgICAgIHRyeToKICAgICAg'
            'ICAgICAgICAgIGlmIGxvY2FsLmV4aXN0cygpIGFuZCBsb2NhbC5pc19maWxlKCkgYW5kIGxvY2Fs'
            'LnN0YXQoKS5zdF9zaXplID09IHNpemU6CiAgICAgICAgICAgICAgICAgICAgcmVwb3J0WyJza2lw'
            'cGVkX2V4aXN0cyJdICs9IDEKICAgICAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICAg'
            'ICAgZG93bmxvYWQoa2V5LCBsb2NhbCwgZXhwZWN0ZWRfc2l6ZT1zaXplLCBjbGllbnQ9Y2xpKQog'
            'ICAgICAgICAgICAgICAgICAgIHJlcG9ydFsiZG93bmxvYWRlZCJdICs9IDEKICAgICAgICAgICAg'
            'ICAgIHJlcG9ydFsia2V5cyJdLmFwcGVuZCh7ImtleSI6IGtleSwgInNpemUiOiBzaXplLCAicGF0'
            'aCI6IHN0cihsb2NhbCl9KQogICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGU6CiAgICAg'
            'ICAgICAgICAgICByZXBvcnRbImVycm9ycyJdLmFwcGVuZChmIntrZXl9Ont0eXBlKGUpLl9fbmFt'
            'ZV9ffSIpCgogICAgZ2VvX2RpcnMgPSBzb3J0ZWQoCiAgICAgICAgcC5uYW1lCiAgICAgICAgZm9y'
            'IHAgaW4gZGVzdC5pdGVyZGlyKCkKICAgICAgICBpZiBwLmlzX2RpcigpIGFuZCBhbnkocC5uYW1l'
            'LnN0YXJ0c3dpdGgoZ3ApIGZvciBncCBpbiBHRU9fUFJFRklYRVMpCiAgICApCiAgICByZXBvcnRb'
            'Imdlb19yZWdpb25fZGlycyJdID0gZ2VvX2RpcnMKICAgIHJlcG9ydFsibl9nZW9fZGlycyJdID0g'
            'bGVuKGdlb19kaXJzKQogICAgbl90aWYgPSAwCiAgICBmb3IgZCBpbiBnZW9fZGlyczoKICAgICAg'
            'ICBmb3IgcCBpbiAoZGVzdCAvIGQpLnJnbG9iKCIqIik6CiAgICAgICAgICAgIGlmIG5vdCBwLmlz'
            'X2ZpbGUoKToKICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgICAgIGlmIHAuc3VmZml4'
            'Lmxvd2VyKCkgbm90IGluIHsiLnRpZiIsICIudGlmZiIsICIuaW1nIn06CiAgICAgICAgICAgICAg'
            'ICBjb250aW51ZQogICAgICAgICAgICBwYXRoX3MgPSBzdHIocCkucmVwbGFjZSgiXFwiLCAiLyIp'
            'CiAgICAgICAgICAgIGlmIGFueSh0b2suc3RyaXAoIl8iKS5sb3dlcigpIGluIHBhdGhfcy5sb3dl'
            'cigpIGZvciB0b2sgaW4gKCJNYWdHciIsICJFdEdyIiwgIlRlckZsIikpOgogICAgICAgICAgICAg'
            'ICAgbl90aWYgKz0gMQogICAgcmVwb3J0WyJuX2dlb19yYXN0ZXJfZmlsZXNfZ3Vlc3MiXSA9IG5f'
            'dGlmCiAgICByZXBvcnRbImhvbGxvdyJdID0gbGVuKGdlb19kaXJzKSA9PSAwIG9yIG5fdGlmID09'
            'IDAgb3IgYm9vbCgKICAgICAgICByZXBvcnRbImVycm9ycyJdIGFuZCByZXBvcnRbImRvd25sb2Fk'
            'ZWQiXSA9PSAwIGFuZCByZXBvcnRbInNraXBwZWRfZXhpc3RzIl0gPT0gMAogICAgKQogICAgIyBE'
            'cm9wIHBlci1rZXkgbGlzdCBmcm9tIFNUQVRVUy1zaXplZCBkdW1wcyBpZiBodWdlOyBrZWVwIGNv'
            'dW50CiAgICByZXBvcnRbIm5fa2V5c19rZXB0Il0gPSBsZW4ocmVwb3J0WyJrZXlzIl0pCiAgICBp'
            'ZiBsZW4ocmVwb3J0WyJrZXlzIl0pID4gNTA6CiAgICAgICAgcmVwb3J0WyJrZXlzIl0gPSByZXBv'
            'cnRbImtleXMiXVs6NTBdICsgW3sidHJ1bmNhdGVkIjogVHJ1ZSwgIm4iOiByZXBvcnRbIm5fa2V5'
            'c19rZXB0Il19XQogICAgcmV0dXJuIHJlcG9ydAoKCmRlZiByZWJ1aWxkKGRlc3Q6IFBhdGggfCBO'
            'b25lID0gTm9uZSwgaW5wdXRfcm9vdDogUGF0aCB8IE5vbmUgPSBOb25lKSAtPiBkaWN0OgogICAg'
            'IiIiUHVibGljIGVudHJ5IHVzZWQgYnkgcGl4Z2JtX2NyX2dhdGUucHkg4oCUIFMzLWZpcnN0IGdl'
            'byB0cmVlIGZvciAyMDDigJMyMTcuIiIiCiAgICBkZXN0ID0gZGVzdCBvciBQYXRoKG9zLmVudmly'
            'b24uZ2V0KCJQSVhHQk1fVFJFRSIsICIva2FnZ2xlL3dvcmtpbmcvdHJhaW4tZXhwZHMtMyIpKQog'
            'ICAgZGVzdC5ta2RpcihwYXJlbnRzPVRydWUsIGV4aXN0X29rPVRydWUpCiAgICBpbnB1dHMgPSBm'
            'aW5kX2lucHV0cyhpbnB1dF9yb290KQogICAgcmVwb3J0ID0gZmV0Y2hfZ2VvX3RyZWUoZGVzdCkK'
            'ICAgIHJlcG9ydFsiaW5wdXRzIl0gPSBzb3J0ZWQoaW5wdXRzKQogICAgcmVwb3J0WyJzb3VyY2Ui'
            'XSA9ICJzMyIKICAgIHJldHVybiByZXBvcnQKCgppZiBfX25hbWVfXyA9PSAiX19tYWluX18iOgog'
            'ICAgZGVzdCA9IFBhdGgoc3lzLmFyZ3ZbMV0pIGlmIGxlbihzeXMuYXJndikgPiAxIGVsc2UgTm9u'
            'ZQogICAgcmVwID0gcmVidWlsZChkZXN0PWRlc3QpCiAgICBwcmludChqc29uLmR1bXBzKHtrOiB2'
            'IGZvciBrLCB2IGluIHJlcC5pdGVtcygpIGlmIGsgIT0gImtleXMifSwgZW5zdXJlX2FzY2lpPUZh'
            'bHNlLCBpbmRlbnQ9MikpCiAgICByYWlzZSBTeXN0ZW1FeGl0KDIgaWYgcmVwLmdldCgiaG9sbG93'
            'IikgZWxzZSAwKQo='
        ),
    },
    'scripts/rebuild_train_tree.py': {
        "sha256": 'ad9ccb709cf926fecdb906daae2d0a6d9ef7f13b0a3536b547d49fe414aefb72',
        "b64": (
            'IiIiUmVidWlsZCDQmtCe0Jcg4oSWMyB0cmFpbiB0cmVlIGZyb20gS2FnZ2xlIHhmZXIgc2hhcmRz'
            'ICsgR1QgZGF0YXNldCAoYm90LTYpLgoKU2hhcmRzIChib3QtMSk6CiAgZG1pdHJpamt2L2tvejMt'
            'eGZlci1zMDAwICDigJQgc21hbGxfZmlsZXMudGFyLnpzdCAoKyBtYW5pZmVzdDsgbGFiZWxzL1VU'
            'TS9leHRyYSkKICBkbWl0cmlqa3Yva296My14ZmVyLXMwMDEg4oCmIHMwMDkg4oCUIGxhcmdlIGJs'
            'b2JzIGYjIyMjLnRpZiAvIGYjIyMjLmxhejsKICAgICAgZWFjaCBzaGFyZCBoYXMgYSBtYW5pZmVz'
            'dCBtYXBwaW5nIGJsb2Ig4oaSIG9yaWdpbmFsIHJlbGF0aXZlIHBhdGgKICBkbWl0cmlqa3Yva296'
            'My1ndC10cmFpbi12MiDigJQgY29uc29saWRhdGVkIEdUIChvcHRpb25hbCBvdmVybGF5IGZvciBQ'
            'SVhHQk07CiAgICAgIHBlci1yZWdpb24gT2JqZWN0UG9seS9GaW5kc1BvaW50cyBhbHJlYWR5IGxp'
            'dmUgaW4gczAwMCBzbWFsbCB0YXIpCgpVbnRpbCBzaGFyZCBtYW5pZmVzdHMgZm9yIHMwMDHigJNz'
            'MDA5IGFyZSBrbm93biBvZmZsaW5lLCBhdHRhY2ggQUxMIHNoYXJkczsKYXQgcnVudGltZSBvbmx5'
            'IG1hdGVyaWFsaXplIHBhdGhzIHdob3NlIHRvcCBmb2xkZXIgaXMgZ2VvIDIwMOKAkzIxNwooUElY'
            'R0JNIEdFT19SRUdJT05TKSwgdW5sZXNzIFBJWEdCTV9SRUJVSUxEX0FMTD0xLgoiIiIKZnJvbSBf'
            'X2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwoKaW1wb3J0IGpzb24KaW1wb3J0IG9zCmltcG9y'
            'dCByZQppbXBvcnQgc2h1dGlsCmltcG9ydCBzdWJwcm9jZXNzCmltcG9ydCB0YXJmaWxlCmZyb20g'
            'cGF0aGxpYiBpbXBvcnQgUGF0aApmcm9tIHR5cGluZyBpbXBvcnQgQW55LCBJdGVyYXRvcgoKR0VP'
            'X1BSRUZJWEVTID0gdHVwbGUoZiJ7bjowM2R9XyIgZm9yIG4gaW4gcmFuZ2UoMjAwLCAyMTgpKQpC'
            'TE9CX1JFID0gcmUuY29tcGlsZShyIl5mXGR7NH0oXC5bQS1aYS16MC05XSspPyQiKQoKCmRlZiBf'
            'aXNfZ2VvX3JlbChyZWw6IHN0cikgLT4gYm9vbDoKICAgIHJlbCA9IHJlbC5yZXBsYWNlKCJcXCIs'
            'ICIvIikubHN0cmlwKCIuLyIpCiAgICB0b3AgPSByZWwuc3BsaXQoIi8iLCAxKVswXQogICAgcmV0'
            'dXJuIGFueSh0b3Auc3RhcnRzd2l0aChwKSBmb3IgcCBpbiBHRU9fUFJFRklYRVMpCgoKZGVmIF9s'
            'b2FkX2pzb24ocGF0aDogUGF0aCkgLT4gQW55OgogICAgcmV0dXJuIGpzb24ubG9hZHMocGF0aC5y'
            'ZWFkX3RleHQoZW5jb2Rpbmc9InV0Zi04IikpCgoKZGVmIGZpbmRfaW5wdXRfZGF0YXNldHMoaW5w'
            'dXRfcm9vdDogUGF0aCB8IE5vbmUgPSBOb25lKSAtPiBkaWN0W3N0ciwgUGF0aF06CiAgICAiIiJN'
            'YXAgc2hvcnQgbmFtZSDihpIgL2thZ2dsZS9pbnB1dC88c2x1Zz4gZGlyZWN0b3J5LiIiIgogICAg'
            'cm9vdCA9IGlucHV0X3Jvb3Qgb3IgUGF0aCgiL2thZ2dsZS9pbnB1dCIpCiAgICBvdXQ6IGRpY3Rb'
            'c3RyLCBQYXRoXSA9IHt9CiAgICBpZiBub3Qgcm9vdC5pc19kaXIoKToKICAgICAgICByZXR1cm4g'
            'b3V0CiAgICBmb3IgcCBpbiBzb3J0ZWQocm9vdC5pdGVyZGlyKCkpOgogICAgICAgIGlmIG5vdCBw'
            'LmlzX2RpcigpOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIG5hbWUgPSBwLm5hbWUubG93'
            'ZXIoKQogICAgICAgIG91dFtuYW1lXSA9IHAKICAgIHJldHVybiBvdXQKCgpkZWYgZmluZF94ZmVy'
            'X3NoYXJkcyhkYXRhc2V0czogZGljdFtzdHIsIFBhdGhdKSAtPiBsaXN0W3R1cGxlW2ludCwgUGF0'
            'aF1dOgogICAgZm91bmQ6IGxpc3RbdHVwbGVbaW50LCBQYXRoXV0gPSBbXQogICAgZm9yIG5hbWUs'
            'IHBhdGggaW4gZGF0YXNldHMuaXRlbXMoKToKICAgICAgICBtID0gcmUuc2VhcmNoKHIia296My14'
            'ZmVyLXMoXGR7M30pJCIsIG5hbWUpCiAgICAgICAgaWYgbToKICAgICAgICAgICAgZm91bmQuYXBw'
            'ZW5kKChpbnQobS5ncm91cCgxKSksIHBhdGgpKQogICAgcmV0dXJuIHNvcnRlZChmb3VuZCkKCgpk'
            'ZWYgZmluZF9ndF9kYXRhc2V0KGRhdGFzZXRzOiBkaWN0W3N0ciwgUGF0aF0pIC0+IFBhdGggfCBO'
            'b25lOgogICAgZm9yIGtleSBpbiAoImtvejMtZ3QtdHJhaW4tdjIiLCAia296My1ndC10cmFpbiIp'
            'OgogICAgICAgIGlmIGtleSBpbiBkYXRhc2V0czoKICAgICAgICAgICAgcmV0dXJuIGRhdGFzZXRz'
            'W2tleV0KICAgIGZvciBuYW1lLCBwYXRoIGluIGRhdGFzZXRzLml0ZW1zKCk6CiAgICAgICAgaWYg'
            'Imd0LXRyYWluIiBpbiBuYW1lIG9yICJndF90cmFpbiIgaW4gbmFtZToKICAgICAgICAgICAgcmV0'
            'dXJuIHBhdGgKICAgIHJldHVybiBOb25lCgoKZGVmIGZpbmRfbWFuaWZlc3Qoc2hhcmQ6IFBhdGgp'
            'IC0+IFBhdGggfCBOb25lOgogICAgZm9yIGNhbmQgaW4gKAogICAgICAgIHNoYXJkIC8gIm1hbmlm'
            'ZXN0Lmpzb24iLAogICAgICAgIHNoYXJkIC8gZiJ7c2hhcmQubmFtZX1fbWFuaWZlc3QuanNvbiIs'
            'CiAgICAgICAgc2hhcmQgLyAia296My14ZmVyLW1hbmlmZXN0Lmpzb24iLAogICAgKToKICAgICAg'
            'ICBpZiBjYW5kLmlzX2ZpbGUoKToKICAgICAgICAgICAgcmV0dXJuIGNhbmQKICAgICMgYW55ICpt'
            'YW5pZmVzdCouanNvbiBhdCB0b3AgbGV2ZWwKICAgIGhpdHMgPSBzb3J0ZWQoc2hhcmQuZ2xvYigi'
            'Km1hbmlmZXN0Ki5qc29uIikpCiAgICByZXR1cm4gaGl0c1swXSBpZiBoaXRzIGVsc2UgTm9uZQoK'
            'CmRlZiBmaW5kX3NtYWxsX3RhcihzaGFyZDogUGF0aCkgLT4gUGF0aCB8IE5vbmU6CiAgICBmb3Ig'
            'Y2FuZCBpbiAoCiAgICAgICAgc2hhcmQgLyAic21hbGxfZmlsZXMudGFyLnpzdCIsCiAgICAgICAg'
            'c2hhcmQgLyAic21hbGxfZmlsZXMudGFyLnpzdGQiLAogICAgICAgIHNoYXJkIC8gInNtYWxsX2Zp'
            'bGVzLnRhciIsCiAgICApOgogICAgICAgIGlmIGNhbmQuaXNfZmlsZSgpOgogICAgICAgICAgICBy'
            'ZXR1cm4gY2FuZAogICAgaGl0cyA9IGxpc3Qoc2hhcmQuZ2xvYigic21hbGxfZmlsZXMudGFyKiIp'
            'KQogICAgcmV0dXJuIGhpdHNbMF0gaWYgaGl0cyBlbHNlIE5vbmUKCgpkZWYgX2lzX2Jsb2JfaWQo'
            'czogc3RyKSAtPiBib29sOgogICAgbmFtZSA9IFBhdGgoc3RyKHMpKS5uYW1lCiAgICByZXR1cm4g'
            'Ym9vbChCTE9CX1JFLm1hdGNoKG5hbWUpIG9yIHJlLm1hdGNoKHIiXmZcZHs0fSQiLCBuYW1lKSkK'
            'CgpkZWYgaXRlcl9ibG9iX21hcHBpbmdzKG1hbmlmZXN0OiBBbnkpIC0+IEl0ZXJhdG9yW3R1cGxl'
            'W3N0ciwgc3RyXV06CiAgICAiIiJZaWVsZCAoYmxvYl9iYXNlbmFtZSwgcmVsYXRpdmVfcGF0aCkg'
            'ZnJvbSBmbGV4aWJsZSBtYW5pZmVzdCBzaGFwZXMuIiIiCiAgICBpZiBub3QgaXNpbnN0YW5jZSht'
            'YW5pZmVzdCwgZGljdCk6CiAgICAgICAgcmV0dXJuCgogICAgZGVmIGVtaXQoYmxvYjogc3RyLCBy'
            'ZWw6IHN0cik6CiAgICAgICAgcmVsID0gcmVsLnJlcGxhY2UoIlxcIiwgIi8iKS5sc3RyaXAoIi4v'
            'IikKICAgICAgICBibG9iID0gUGF0aChzdHIoYmxvYikpLm5hbWUKICAgICAgICBpZiByZWwgYW5k'
            'IGJsb2I6CiAgICAgICAgICAgIHlpZWxkIGJsb2IsIHJlbAoKICAgIGRlZiBmcm9tX21lbWJlcnMo'
            'bWVtYmVyczogZGljdCk6CiAgICAgICAgZm9yIGssIHYgaW4gbWVtYmVycy5pdGVtcygpOgogICAg'
            'ICAgICAgICBpZiBpc2luc3RhbmNlKHYsIHN0cik6CiAgICAgICAgICAgICAgICBpZiBfaXNfYmxv'
            'Yl9pZChrKToKICAgICAgICAgICAgICAgICAgICB5aWVsZCBmcm9tIGVtaXQoaywgdikKICAgICAg'
            'ICAgICAgICAgIGVsaWYgX2lzX2Jsb2JfaWQodik6CiAgICAgICAgICAgICAgICAgICAgeWllbGQg'
            'ZnJvbSBlbWl0KHYsIGspCiAgICAgICAgICAgIGVsaWYgaXNpbnN0YW5jZSh2LCBkaWN0KToKICAg'
            'ICAgICAgICAgICAgIHJlbCA9IHYuZ2V0KCJyZWwiKSBvciB2LmdldCgicGF0aCIpIG9yIHYuZ2V0'
            'KCJvcmlnIikgb3Igdi5nZXQoIm9yaWdpbmFsIikKICAgICAgICAgICAgICAgIGJsb2IgPSB2Lmdl'
            'dCgiYmxvYiIpIG9yIHYuZ2V0KCJuYW1lIikgb3Igdi5nZXQoImZpbGUiKSBvciB2LmdldCgiaWQi'
            'KSBvciB2LmdldCgia2V5IikKICAgICAgICAgICAgICAgIGlmIF9pc19ibG9iX2lkKGspIGFuZCBy'
            'ZWw6CiAgICAgICAgICAgICAgICAgICAgeWllbGQgZnJvbSBlbWl0KGssIHN0cihyZWwpKQogICAg'
            'ICAgICAgICAgICAgZWxpZiBibG9iIGFuZCByZWw6CiAgICAgICAgICAgICAgICAgICAgeWllbGQg'
            'ZnJvbSBlbWl0KHN0cihibG9iKSwgc3RyKHJlbCkpCiAgICAgICAgICAgICAgICBlbGlmIGJsb2Ig'
            'YW5kICIvIiBpbiBzdHIoayk6CiAgICAgICAgICAgICAgICAgICAgeWllbGQgZnJvbSBlbWl0KHN0'
            'cihibG9iKSwgc3RyKGspKQoKICAgIGZvciBrZXkgaW4gKCJmaWxlcyIsICJsYXJnZSIsICJibG9i'
            'cyIsICJtZW1iZXJzIiwgIm1hcCIsICJtYXBwaW5nIik6CiAgICAgICAgYmxvY2sgPSBtYW5pZmVz'
            'dC5nZXQoa2V5KQogICAgICAgIGlmIGlzaW5zdGFuY2UoYmxvY2ssIGRpY3QpOgogICAgICAgICAg'
            'ICBpZiBpc2luc3RhbmNlKGJsb2NrLmdldCgibWVtYmVycyIpLCBkaWN0KToKICAgICAgICAgICAg'
            'ICAgIHlpZWxkIGZyb20gZnJvbV9tZW1iZXJzKGJsb2NrWyJtZW1iZXJzIl0pCiAgICAgICAgICAg'
            'IGVsc2U6CiAgICAgICAgICAgICAgICB5aWVsZCBmcm9tIGZyb21fbWVtYmVycyhibG9jaykKCiAg'
            'ICB0b3AgPSB7azogdiBmb3IgaywgdiBpbiBtYW5pZmVzdC5pdGVtcygpIGlmIF9pc19ibG9iX2lk'
            'KHN0cihrKSl9CiAgICBpZiB0b3A6CiAgICAgICAgeWllbGQgZnJvbSBmcm9tX21lbWJlcnModG9w'
            'KQoKCmRlZiBfZW5zdXJlX3pzdGFuZGFyZCgpIC0+IE5vbmU6CiAgICAiIiJJbXBvcnQgenN0YW5k'
            'YXJkLiBPZmZsaW5lIG9ubHk6IHpzdGQgQ0xJIHByZWZlcnJlZDsgZWxzZSB3aGVlbCB2aWEgLS1u'
            'by1pbmRleC4iIiIKICAgIHRyeToKICAgICAgICBpbXBvcnQgenN0YW5kYXJkICAjIG5vcWE6IEY0'
            'MDEKICAgICAgICByZXR1cm4KICAgIGV4Y2VwdCBJbXBvcnRFcnJvcjoKICAgICAgICBwYXNzCiAg'
            'ICAjIGZpbmQtbGlua3MgdW5kZXIgL2thZ2dsZS9pbnB1dC8qKi93aGVlbHMgKHBpeGdibS1jci13'
            'aGVlbHMgZGF0YXNldCkKICAgIGxpbmtzOiBsaXN0W1BhdGhdID0gW10KICAgIHJvb3QgPSBQYXRo'
            'KCIva2FnZ2xlL2lucHV0IikKICAgIGlmIHJvb3QuaXNfZGlyKCk6CiAgICAgICAgZm9yIGQgaW4g'
            'cm9vdC5yZ2xvYigid2hlZWxzIik6CiAgICAgICAgICAgIGlmIGQuaXNfZGlyKCk6CiAgICAgICAg'
            'ICAgICAgICBsaW5rcy5hcHBlbmQoZCkKICAgIGVudl9saW5rID0gb3MuZW52aXJvbi5nZXQoIlBJ'
            'WEdCTV9XSEVFTF9ESVIiKQogICAgaWYgZW52X2xpbms6CiAgICAgICAgbGlua3MuYXBwZW5kKFBh'
            'dGgoZW52X2xpbmspKQogICAgaWYgbm90IGxpbmtzOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJv'
            'cigKICAgICAgICAgICAgIklOVkFMSUQ6IG5lZWQgenN0ZCBDTEkgb3IgenN0YW5kYXJkIHdoZWVs'
            'OyBhdHRhY2ggZG1pdHJpamt2L2tvejMtcGl4Z2JtLWNyLXdoZWVscyAiCiAgICAgICAgICAgICIo'
            'ZW5hYmxlX2ludGVybmV0PWZhbHNlIOKAlCBubyBuZXR3b3JrIHBpcCkiCiAgICAgICAgKQogICAg'
            'Y21kID0gW3N5c19leGVjdXRhYmxlKCksICItbSIsICJwaXAiLCAiaW5zdGFsbCIsICItLW5vLWlu'
            'ZGV4IiwgIi0tbm8tZGVwcyJdCiAgICBmb3IgZCBpbiBsaW5rczoKICAgICAgICBjbWQuZXh0ZW5k'
            'KFsiLS1maW5kLWxpbmtzIiwgc3RyKGQpXSkKICAgIGNtZC5hcHBlbmQoInpzdGFuZGFyZCIpCiAg'
            'ICBzdWJwcm9jZXNzLmNoZWNrX2NhbGwoY21kKQogICAgaW1wb3J0IHpzdGFuZGFyZCAgIyBub3Fh'
            'OiBGNDAxCgoKZGVmIGV4dHJhY3Rfc21hbGxfdGFyKHRhcl9wYXRoOiBQYXRoLCBkZXN0OiBQYXRo'
            'KSAtPiBpbnQ6CiAgICAiIiJFeHRyYWN0IHNtYWxsX2ZpbGVzIHRhciguenN0KSBpbnRvIGRlc3Qu'
            'IFJldHVybnMgbWVtYmVyIGNvdW50LiIiIgogICAgZGVzdC5ta2RpcihwYXJlbnRzPVRydWUsIGV4'
            'aXN0X29rPVRydWUpCiAgICBuYW1lID0gdGFyX3BhdGgubmFtZQogICAgaWYgbmFtZS5lbmRzd2l0'
            'aCgiLnpzdCIpIG9yIG5hbWUuZW5kc3dpdGgoIi56c3RkIik6CiAgICAgICAgIyBQcmVmZXIgenN0'
            'ZCBDTEkgKEthZ2dsZSBpbWFnZXMgdXN1YWxseSBoYXZlIGl0KTsgZWxzZSBvZmZsaW5lIHpzdGFu'
            'ZGFyZCB3aGVlbC4KICAgICAgICB0bXBfdGFyID0gZGVzdC5wYXJlbnQgLyAiX3NtYWxsX2ZpbGVz'
            'LnRhciIKICAgICAgICBpZiB0bXBfdGFyLmV4aXN0cygpOgogICAgICAgICAgICB0bXBfdGFyLnVu'
            'bGluaygpCiAgICAgICAgdHJ5OgogICAgICAgICAgICBzdWJwcm9jZXNzLmNoZWNrX2NhbGwoWyJ6'
            'c3RkIiwgIi1kIiwgIi1mIiwgIi1vIiwgc3RyKHRtcF90YXIpLCBzdHIodGFyX3BhdGgpXSkKICAg'
            'ICAgICBleGNlcHQgKEZpbGVOb3RGb3VuZEVycm9yLCBzdWJwcm9jZXNzLkNhbGxlZFByb2Nlc3NF'
            'cnJvcik6CiAgICAgICAgICAgIF9lbnN1cmVfenN0YW5kYXJkKCkKICAgICAgICAgICAgaW1wb3J0'
            'IHpzdGFuZGFyZCBhcyB6c3RkICAjIHR5cGU6IGlnbm9yZQogICAgICAgICAgICBkY3R4ID0genN0'
            'ZC5ac3RkRGVjb21wcmVzc29yKCkKICAgICAgICAgICAgd2l0aCB0YXJfcGF0aC5vcGVuKCJyYiIp'
            'IGFzIHNyYywgdG1wX3Rhci5vcGVuKCJ3YiIpIGFzIG91dDoKICAgICAgICAgICAgICAgIGRjdHgu'
            'Y29weV9zdHJlYW0oc3JjLCBvdXQpCiAgICAgICAgcGF0aF9mb3JfdGFyID0gdG1wX3RhcgogICAg'
            'ZWxzZToKICAgICAgICBwYXRoX2Zvcl90YXIgPSB0YXJfcGF0aAoKICAgIG4gPSAwCiAgICB3aXRo'
            'IHRhcmZpbGUub3BlbihwYXRoX2Zvcl90YXIsIG1vZGU9InI6KiIpIGFzIHRmOgogICAgICAgIGZv'
            'ciBtIGluIHRmLmdldG1lbWJlcnMoKToKICAgICAgICAgICAgaWYgbm90IG0uaXNmaWxlKCk6CiAg'
            'ICAgICAgICAgICAgICBjb250aW51ZQogICAgICAgICAgICByZWwgPSBtLm5hbWUubHN0cmlwKCIu'
            'LyIpCiAgICAgICAgICAgIGlmIG5vdCBfd2FudF9yZWwocmVsKToKICAgICAgICAgICAgICAgIGNv'
            'bnRpbnVlCiAgICAgICAgICAgIHRmLmV4dHJhY3QobSwgcGF0aD1kZXN0LCBmaWx0ZXI9ImRhdGEi'
            'KQogICAgICAgICAgICBuICs9IDEKICAgIGlmIHBhdGhfZm9yX3RhciAhPSB0YXJfcGF0aCBhbmQg'
            'cGF0aF9mb3JfdGFyLmV4aXN0cygpOgogICAgICAgIHBhdGhfZm9yX3Rhci51bmxpbmsobWlzc2lu'
            'Z19vaz1UcnVlKQogICAgcmV0dXJuIG4KCgpkZWYgc3lzX2V4ZWN1dGFibGUoKSAtPiBzdHI6CiAg'
            'ICBpbXBvcnQgc3lzCgogICAgcmV0dXJuIHN5cy5leGVjdXRhYmxlCgoKZGVmIF93YW50X3JlbChy'
            'ZWw6IHN0cikgLT4gYm9vbDoKICAgIGlmIG9zLmVudmlyb24uZ2V0KCJQSVhHQk1fUkVCVUlMRF9B'
            'TEwiLCAiIikuc3RyaXAoKSBpbiAoIjEiLCAidHJ1ZSIsICJ5ZXMiKToKICAgICAgICByZXR1cm4g'
            'VHJ1ZQogICAgcmV0dXJuIF9pc19nZW9fcmVsKHJlbCkKCgpkZWYgcmVzb2x2ZV9ibG9iKHNoYXJk'
            'OiBQYXRoLCBibG9iOiBzdHIpIC0+IFBhdGggfCBOb25lOgogICAgIiIiRmluZCBmIyMjIyBvciBm'
            'IyMjIy5leHQgaW5zaWRlIHNoYXJkIGRpcmVjdG9yeS4iIiIKICAgIGIgPSBQYXRoKGJsb2IpLm5h'
            'bWUKICAgIGRpcmVjdCA9IHNoYXJkIC8gYgogICAgaWYgZGlyZWN0LmlzX2ZpbGUoKToKICAgICAg'
            'ICByZXR1cm4gZGlyZWN0CiAgICBzdGVtID0gYi5zcGxpdCgiLiIpWzBdCiAgICAjIHRyeSBleGFj'
            'dCBzdGVtIHdpdGggYW55IHN1ZmZpeAogICAgaGl0cyA9IGxpc3Qoc2hhcmQuZ2xvYihzdGVtICsg'
            'Ii4qIikpICsgbGlzdChzaGFyZC5nbG9iKHN0ZW0pKQogICAgZm9yIGggaW4gaGl0czoKICAgICAg'
            'ICBpZiBoLmlzX2ZpbGUoKToKICAgICAgICAgICAgcmV0dXJuIGgKICAgICMgbmVzdGVkIG9uZSBs'
            'ZXZlbAogICAgZm9yIGggaW4gc2hhcmQuZ2xvYigiKi8iICsgc3RlbSArICIuKiIpOgogICAgICAg'
            'IGlmIGguaXNfZmlsZSgpOgogICAgICAgICAgICByZXR1cm4gaAogICAgcmV0dXJuIE5vbmUKCgpk'
            'ZWYgbGlua19vcl9jb3B5KHNyYzogUGF0aCwgZHN0OiBQYXRoKSAtPiBzdHI6CiAgICBkc3QucGFy'
            'ZW50Lm1rZGlyKHBhcmVudHM9VHJ1ZSwgZXhpc3Rfb2s9VHJ1ZSkKICAgIGlmIGRzdC5leGlzdHMo'
            'KSBvciBkc3QuaXNfc3ltbGluaygpOgogICAgICAgIHJldHVybiAiZXhpc3RzIgogICAgdHJ5Ogog'
            'ICAgICAgIG9zLnN5bWxpbmsoc3JjLCBkc3QpCiAgICAgICAgcmV0dXJuICJzeW1saW5rIgogICAg'
            'ZXhjZXB0IE9TRXJyb3I6CiAgICAgICAgc2h1dGlsLmNvcHkyKHNyYywgZHN0KQogICAgICAgIHJl'
            'dHVybiAiY29weSIKCgpkZWYgbWF0ZXJpYWxpemVfc2hhcmRfYmxvYnMoc2hhcmQ6IFBhdGgsIHRy'
            'ZWU6IFBhdGgsIG1hbmlmZXN0X3BhdGg6IFBhdGggfCBOb25lKSAtPiBkaWN0OgogICAgc3RhdHMg'
            'PSB7InNoYXJkIjogc2hhcmQubmFtZSwgImxpbmtlZCI6IDAsICJjb3BpZWQiOiAwLCAiZXhpc3Rz'
            'IjogMCwgIm1pc3NpbmdfYmxvYiI6IDAsICJza2lwcGVkX25vbl9nZW8iOiAwLCAibWFwcGVkIjog'
            'MH0KICAgIGlmIG1hbmlmZXN0X3BhdGggaXMgTm9uZToKICAgICAgICBzdGF0c1siZXJyb3IiXSA9'
            'ICJub19tYW5pZmVzdCIKICAgICAgICByZXR1cm4gc3RhdHMKICAgIG1hbiA9IF9sb2FkX2pzb24o'
            'bWFuaWZlc3RfcGF0aCkKICAgIG1hcHBpbmdzID0gbGlzdChpdGVyX2Jsb2JfbWFwcGluZ3MobWFu'
            'KSkKICAgIHN0YXRzWyJtYXBwZWQiXSA9IGxlbihtYXBwaW5ncykKICAgIGlmIG5vdCBtYXBwaW5n'
            'czoKICAgICAgICBzdGF0c1siZXJyb3IiXSA9ICJub19ibG9iX21hcHBpbmdzX2luX21hbmlmZXN0'
            'IgogICAgICAgIHJldHVybiBzdGF0cwogICAgZm9yIGJsb2IsIHJlbCBpbiBtYXBwaW5nczoKICAg'
            'ICAgICBpZiBub3QgX3dhbnRfcmVsKHJlbCk6CiAgICAgICAgICAgIHN0YXRzWyJza2lwcGVkX25v'
            'bl9nZW8iXSArPSAxCiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgc3JjID0gcmVzb2x2ZV9i'
            'bG9iKHNoYXJkLCBibG9iKQogICAgICAgIGlmIHNyYyBpcyBOb25lOgogICAgICAgICAgICAjIGJs'
            'b2IgaWQgd2l0aG91dCBleHRlbnNpb24g4oCUIHRyeSBhcy1pcwogICAgICAgICAgICBzcmMgPSBy'
            'ZXNvbHZlX2Jsb2Ioc2hhcmQsIGJsb2Iuc3BsaXQoIi4iKVswXSkKICAgICAgICBpZiBzcmMgaXMg'
            'Tm9uZToKICAgICAgICAgICAgc3RhdHNbIm1pc3NpbmdfYmxvYiJdICs9IDEKICAgICAgICAgICAg'
            'Y29udGludWUKICAgICAgICAjIHByZXNlcnZlIGV4dGVuc2lvbiBmcm9tIGFjdHVhbCBmaWxlIGlm'
            'IHJlbCBoYXMgbm9uZQogICAgICAgIHJlbF9wYXRoID0gcmVsLnJlcGxhY2UoIlxcIiwgIi8iKQog'
            'ICAgICAgIGlmIG5vdCBQYXRoKHJlbF9wYXRoKS5zdWZmaXggYW5kIHNyYy5zdWZmaXg6CiAgICAg'
            'ICAgICAgIHJlbF9wYXRoID0gcmVsX3BhdGggKyBzcmMuc3VmZml4CiAgICAgICAgYWN0aW9uID0g'
            'bGlua19vcl9jb3B5KHNyYywgdHJlZSAvIHJlbF9wYXRoKQogICAgICAgIHN0YXRzW2FjdGlvbl0g'
            'PSBzdGF0cy5nZXQoYWN0aW9uLCAwKSArIDEKICAgIHJldHVybiBzdGF0cwoKCmRlZiBvdmVybGF5'
            'X2d0KGd0X3Jvb3Q6IFBhdGgsIHRyZWU6IFBhdGgpIC0+IGRpY3Q6CiAgICAiIiJCZXN0LWVmZm9y'
            'dDogY29weSBnZW9qc29uL2Fubm90YXRpb25zIGZyb20gZ3QgZGF0YXNldCBpbnRvIHRyZWUgaWYg'
            'bWlzc2luZy4iIiIKICAgIHN0YXRzID0geyJndCI6IGd0X3Jvb3QubmFtZSwgImNvcGllZCI6IDAs'
            'ICJza2lwcGVkIjogMH0KICAgICMgSWYgZGF0YXNldCBpcyBhIHNpbmdsZSBnZW9qc29uIOKAlCBs'
            'ZWF2ZSBmb3Igb3RoZXIgcGlwZWxpbmVzOyBQSVhHQk0gdXNlcyBwZXItcmVnaW9uIGZpbGVzLgog'
            'ICAgZm9yIHAgaW4gZ3Rfcm9vdC5yZ2xvYigiKiIpOgogICAgICAgIGlmIG5vdCBwLmlzX2ZpbGUo'
            'KToKICAgICAgICAgICAgY29udGludWUKICAgICAgICB0cnk6CiAgICAgICAgICAgIHJlbCA9IHAu'
            'cmVsYXRpdmVfdG8oZ3Rfcm9vdCkuYXNfcG9zaXgoKQogICAgICAgIGV4Y2VwdCBWYWx1ZUVycm9y'
            'OgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIGlmIG5vdCBfd2FudF9yZWwocmVsKSBhbmQg'
            'bm90IHJlbC5lbmRzd2l0aCgiLmdlb2pzb24iKToKICAgICAgICAgICAgIyBhbGxvdyBnZW9qc29u'
            'IG92ZXJsYXlzIG9ubHkgdW5kZXIgZ2VvIHJlZ2lvbnMKICAgICAgICAgICAgaWYgbm90IGFueShw'
            'YXJ0LnN0YXJ0c3dpdGgoR0VPX1BSRUZJWEVTKSBmb3IgcGFydCBpbiBQYXRoKHJlbCkucGFydHMp'
            'OgogICAgICAgICAgICAgICAgc3RhdHNbInNraXBwZWQiXSArPSAxCiAgICAgICAgICAgICAgICBj'
            'b250aW51ZQogICAgICAgIGlmIG5vdCBfd2FudF9yZWwocmVsKToKICAgICAgICAgICAgc3RhdHNb'
            'InNraXBwZWQiXSArPSAxCiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgZHN0ID0gdHJlZSAv'
            'IHJlbAogICAgICAgIGlmIGRzdC5leGlzdHMoKToKICAgICAgICAgICAgc3RhdHNbInNraXBwZWQi'
            'XSArPSAxCiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgZHN0LnBhcmVudC5ta2RpcihwYXJl'
            'bnRzPVRydWUsIGV4aXN0X29rPVRydWUpCiAgICAgICAgc2h1dGlsLmNvcHkyKHAsIGRzdCkKICAg'
            'ICAgICBzdGF0c1siY29waWVkIl0gKz0gMQogICAgcmV0dXJuIHN0YXRzCgoKZGVmIHJlYnVpbGRf'
            'dHJhaW5fdHJlZSgKICAgIGRlc3Q6IFBhdGggfCBOb25lID0gTm9uZSwKICAgIGlucHV0X3Jvb3Q6'
            'IFBhdGggfCBOb25lID0gTm9uZSwKKSAtPiBkaWN0OgogICAgIiIiQnVpbGQgdHJhaW4gcm9vdCB1'
            'bmRlciBkZXN0OyByZXR1cm4gc3RhdHMgZGljdCBpbmNsdWRpbmcgJ3Jvb3QnIHBhdGguIiIiCiAg'
            'ICBkZXN0ID0gUGF0aChkZXN0IG9yIG9zLmVudmlyb24uZ2V0KCJQSVhHQk1fVFJFRSIsICIva2Fn'
            'Z2xlL3dvcmtpbmcvdHJhaW4tZXhwZHMtMyIpKQogICAgZGF0YXNldHMgPSBmaW5kX2lucHV0X2Rh'
            'dGFzZXRzKGlucHV0X3Jvb3QpCiAgICBzaGFyZHMgPSBmaW5kX3hmZXJfc2hhcmRzKGRhdGFzZXRz'
            'KQogICAgcmVwb3J0OiBkaWN0W3N0ciwgQW55XSA9IHsKICAgICAgICAiZGVzdCI6IHN0cihkZXN0'
            'KSwKICAgICAgICAic2hhcmRzX2ZvdW5kIjogW2Yic3tpZHg6MDNkfTp7cGF0aC5uYW1lfSIgZm9y'
            'IGlkeCwgcGF0aCBpbiBzaGFyZHNdLAogICAgICAgICJzdGVwcyI6IFtdLAogICAgfQogICAgaWYg'
            'bm90IHNoYXJkczoKICAgICAgICByYWlzZSBGaWxlTm90Rm91bmRFcnJvcigKICAgICAgICAgICAg'
            'Ik5vIGtvejMteGZlci1zTk5OIGRhdGFzZXRzIHVuZGVyIC9rYWdnbGUvaW5wdXQuICIKICAgICAg'
            'ICAgICAgIkF0dGFjaCBkbWl0cmlqa3Yva296My14ZmVyLXMwMDAg4oCmIHMwMDkuIgogICAgICAg'
            'ICkKCiAgICBkZXN0Lm1rZGlyKHBhcmVudHM9VHJ1ZSwgZXhpc3Rfb2s9VHJ1ZSkKICAgICMgczAw'
            'MCBzbWFsbCB0YXIgZmlyc3QKICAgIGZvciBpZHgsIHBhdGggaW4gc2hhcmRzOgogICAgICAgIGlm'
            'IGlkeCAhPSAwOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIHRhciA9IGZpbmRfc21hbGxf'
            'dGFyKHBhdGgpCiAgICAgICAgaWYgdGFyIGlzIE5vbmU6CiAgICAgICAgICAgIHJlcG9ydFsic3Rl'
            'cHMiXS5hcHBlbmQoeyJzMDAwIjogcGF0aC5uYW1lLCAiZXJyb3IiOiAic21hbGxfZmlsZXMudGFy'
            'KiBtaXNzaW5nIn0pCiAgICAgICAgZWxzZToKICAgICAgICAgICAgbiA9IGV4dHJhY3Rfc21hbGxf'
            'dGFyKHRhciwgZGVzdCkKICAgICAgICAgICAgcmVwb3J0WyJzdGVwcyJdLmFwcGVuZCh7InMwMDBf'
            'ZXh0cmFjdCI6IHRhci5uYW1lLCAibWVtYmVyc19nZW8iOiBufSkKCiAgICAjIGxhcmdlIHNoYXJk'
            'cyAoKyBzMDAwIGlmIGl0IGFsc28gaGFzIGJsb2IgbWFwcykKICAgIGZvciBpZHgsIHBhdGggaW4g'
            'c2hhcmRzOgogICAgICAgIG1hbiA9IGZpbmRfbWFuaWZlc3QocGF0aCkKICAgICAgICBpZiBpZHgg'
            'PT0gMCBhbmQgbWFuIGlzIE5vbmU6CiAgICAgICAgICAgICMgb3B0aW9uYWw6IHNoYXJlZCBjb3B5'
            'IHBhdGggbmFtZQogICAgICAgICAgICBjb250aW51ZQogICAgICAgIHN0ID0gbWF0ZXJpYWxpemVf'
            'c2hhcmRfYmxvYnMocGF0aCwgZGVzdCwgbWFuKQogICAgICAgIHJlcG9ydFsic3RlcHMiXS5hcHBl'
            'bmQoc3QpCgogICAgZ3QgPSBmaW5kX2d0X2RhdGFzZXQoZGF0YXNldHMpCiAgICBpZiBndCBpcyBu'
            'b3QgTm9uZToKICAgICAgICByZXBvcnRbInN0ZXBzIl0uYXBwZW5kKG92ZXJsYXlfZ3QoZ3QsIGRl'
            'c3QpKQogICAgZWxzZToKICAgICAgICByZXBvcnRbInN0ZXBzIl0uYXBwZW5kKHsiZ3QiOiBOb25l'
            'LCAibm90ZSI6ICJrb3ozLWd0LXRyYWluLXYyIG5vdCBhdHRhY2hlZCJ9KQoKICAgICMgc2FuaXR5'
            'OiBjb3VudCBnZW8gcmVnaW9uIGRpcnMKICAgIGdlb19kaXJzID0gc29ydGVkKAogICAgICAgIHAu'
            'bmFtZSBmb3IgcCBpbiBkZXN0Lml0ZXJkaXIoKSBpZiBwLmlzX2RpcigpIGFuZCBhbnkocC5uYW1l'
            'LnN0YXJ0c3dpdGgocHJlZikgZm9yIHByZWYgaW4gR0VPX1BSRUZJWEVTKQogICAgKQogICAgcmVw'
            'b3J0WyJnZW9fcmVnaW9uX2RpcnMiXSA9IGdlb19kaXJzCiAgICByZXBvcnRbInJvb3QiXSA9IHN0'
            'cihkZXN0KQogICAgaWYgbm90IGdlb19kaXJzOgogICAgICAgIHJhaXNlIEZpbGVOb3RGb3VuZEVy'
            'cm9yKAogICAgICAgICAgICBmIlJlYnVpbGQgcHJvZHVjZWQgbm8gZ2VvIDIwMOKAkzIxNyBkaXJz'
            'IHVuZGVyIHtkZXN0fS4gIgogICAgICAgICAgICAiQ2hlY2sgc2hhcmQgbWFuaWZlc3RzIC8gc21h'
            'bGxfZmlsZXMgZXh0cmFjdC4iCiAgICAgICAgKQogICAgcmV0dXJuIHJlcG9ydAoKCmlmIF9fbmFt'
            'ZV9fID09ICJfX21haW5fXyI6CiAgICBpbXBvcnQgcHByaW50CgogICAgcHByaW50LnBwKHJlYnVp'
            'bGRfdHJhaW5fdHJlZSgpKQo='
        ),
    },
    'scorer/e2e_score.py': {
        "sha256": '4025cb70029f15145af081f3fec15dd3d25ed9c91d99e3bcf5bb6f2a2c8f4051',
        "b64": (
            'IiIiYm90LTcsIDI5LjEwOiDRgdC60LLQvtC30L3QvtC1INGB0YDQsNCy0L3QtdC90LjQtSDQtNCy'
            '0YPRhSDQstGL0LTQsNGHIG1haW4ucHkgKHYyINC4IHYzKSDQv9GA0L7RgtC40LIg0Y3RgtCw0LvQ'
            'vtC90LAg0L7RhNC40YbQuNCw0LvRjNC90L7QuSDQvNC10YLRgNC40LrQvtC5LgpweXRob24gZTJl'
            'X3Njb3JlLnB5IC0tbWV0cmljIGNvbXB1dGVfbWV0cmljcy5weSAtLWd0IGd0Lmdlb2pzb24gLS1h'
            'IHYyPXJlc192Mi5nZW9qc29uIC0tYiB2Mz1yZXNfdjMuZ2VvanNvbiAtLW91dCBlMmUuanNvbgrQ'
            'odGH0LjRgtCw0LXRgjog0LLRgdGRINCy0LzQtdGB0YLQtSwg0L/QviDRgNC10LPQuNC+0L3QsNC8'
            'INC4INC/0L4gbWFya3VwX3R5cGUgKNGC0L7Qu9GM0LrQviDRgNC10LPQuNC+0L3Riywg0LrQvtGC'
            '0L7RgNGL0LUg0LXRgdGC0Ywg0LIg0Y3RgtCw0LvQvtC90LUpLiDQndC40YfQtdCz0L4g0L3QtSDQ'
            'v9C+0LTQsdC40YDQsNC10YIuIiIiCmltcG9ydCBhcmdwYXJzZSwgY29sbGVjdGlvbnMsIGltcG9y'
            'dGxpYi51dGlsLCBqc29uLCBzeXMKZnJvbSBwYXRobGliIGltcG9ydCBQYXRoCgphcCA9IGFyZ3Bh'
            'cnNlLkFyZ3VtZW50UGFyc2VyKCkKYXAuYWRkX2FyZ3VtZW50KCItLW1ldHJpYyIsIHJlcXVpcmVk'
            'PVRydWUpOyBhcC5hZGRfYXJndW1lbnQoIi0tZ3QiLCByZXF1aXJlZD1UcnVlKQphcC5hZGRfYXJn'
            'dW1lbnQoIi0tYSIsIHJlcXVpcmVkPVRydWUpOyBhcC5hZGRfYXJndW1lbnQoIi0tYiIsIHJlcXVp'
            'cmVkPVRydWUpOyBhcC5hZGRfYXJndW1lbnQoIi0tb3V0IiwgcmVxdWlyZWQ9VHJ1ZSkKYSA9IGFw'
            'LnBhcnNlX2FyZ3MoKQpzeXMucGF0aC5pbnNlcnQoMCwgc3RyKFBhdGgoYS5tZXRyaWMpLnBhcmVu'
            'dCkpCnNwZWMgPSBpbXBvcnRsaWIudXRpbC5zcGVjX2Zyb21fZmlsZV9sb2NhdGlvbigiY20iLCBh'
            'Lm1ldHJpYyk7IGNtID0gaW1wb3J0bGliLnV0aWwubW9kdWxlX2Zyb21fc3BlYyhzcGVjKTsgc3Bl'
            'Yy5sb2FkZXIuZXhlY19tb2R1bGUoY20pCmd0ID0ganNvbi5sb2FkcyhQYXRoKGEuZ3QpLnJlYWRf'
            'dGV4dChlbmNvZGluZz0idXRmLTgiKSkKcnVucyA9IHt9CmZvciBzIGluIChhLmEsIGEuYik6CiAg'
            'ICBuYW1lLCBwID0gcy5zcGxpdCgiPSIsIDEpOyBydW5zW25hbWVdID0ganNvbi5sb2FkcyhQYXRo'
            'KHApLnJlYWRfdGV4dChlbmNvZGluZz0idXRmLTgiKSkKcmVnaW9ucyA9IHNvcnRlZCh7ZlsicHJv'
            'cGVydGllcyJdWyJyZWdpb25fbmFtZSJdIGZvciBmIGluIGd0WyJmZWF0dXJlcyJdfSkKCmRlZiBm'
            'YyhmZWF0cyk6IHJldHVybiB7InR5cGUiOiAiRmVhdHVyZUNvbGxlY3Rpb24iLCAiZmVhdHVyZXMi'
            'OiBmZWF0c30KZGVmIHNlbChkLCByZWc9Tm9uZSwgbXQ9Tm9uZSk6CiAgICByZXR1cm4gW2YgZm9y'
            'IGYgaW4gZFsiZmVhdHVyZXMiXSBpZiAocmVnIGlzIE5vbmUgb3IgZlsicHJvcGVydGllcyJdLmdl'
            'dCgicmVnaW9uX25hbWUiKSA9PSByZWcpCiAgICAgICAgICAgIGFuZCAobXQgaXMgTm9uZSBvciBm'
            'WyJwcm9wZXJ0aWVzIl0uZ2V0KCJtYXJrdXBfdHlwZSIpID09IG10KV0KZGVmIHNjb3JlKHAsIGcp'
            'OgogICAgaWYgbm90IGc6IHJldHVybiBOb25lCiAgICByZXR1cm4gY20uRjFDYWxjdWxhdG9yKCku'
            'Y2FsY3VsYXRlX2YxKGZjKHApLCBmYyhnKSlbImZpbmFsX3Njb3JlIl0KCnJlcyA9IHsicmVnaW9u'
            'cyI6IHJlZ2lvbnMsICJyb3dzIjogW119CmdmZSA9IFtmIGZvciBmIGluIGd0WyJmZWF0dXJlcyJd'
            'XQptdHMgPSBzb3J0ZWQoe2ZbInByb3BlcnRpZXMiXVsibWFya3VwX3R5cGUiXSBmb3IgZiBpbiBn'
            'ZmV9KQpmb3IgcmVnIGluIFtOb25lXSArIHJlZ2lvbnM6CiAgICBmb3IgbXQgaW4gW05vbmVdICsg'
            'bXRzOgogICAgICAgIGcgPSBzZWwoZ3QsIHJlZywgbXQpCiAgICAgICAgaWYgbm90IGc6IGNvbnRp'
            'bnVlCiAgICAgICAgcm93ID0geyJyZWdpb24iOiByZWcgb3IgIkFMTCIsICJtYXJrdXAiOiBtdCBv'
            'ciAiQUxMIiwgImd0IjogbGVuKGcpfQogICAgICAgIGZvciBuLCBkIGluIHJ1bnMuaXRlbXMoKToK'
            'ICAgICAgICAgICAgcCA9IFtmIGZvciBmIGluIHNlbChkLCByZWcsIG10KSBpZiByZWcgaXMgbm90'
            'IE5vbmUgb3IgZlsicHJvcGVydGllcyJdLmdldCgicmVnaW9uX25hbWUiKSBpbiByZWdpb25zXQog'
            'ICAgICAgICAgICByb3dbZiJuX3tufSJdID0gbGVuKHApOyByb3dbZiJmMV97bn0iXSA9IHNjb3Jl'
            'KHAsIGcpCiAgICAgICAgbmFtZXMgPSBsaXN0KHJ1bnMpCiAgICAgICAgaWYgcm93W2YiZjFfe25h'
            'bWVzWzBdfSJdIGlzIG5vdCBOb25lOiByb3dbImRlbHRhIl0gPSByb3dbZiJmMV97bmFtZXNbMV19'
            'Il0gLSByb3dbZiJmMV97bmFtZXNbMF19Il0KICAgICAgICByZXNbInJvd3MiXS5hcHBlbmQocm93'
            'KTsgcHJpbnQoanNvbi5kdW1wcyhyb3csIGVuc3VyZV9hc2NpaT1GYWxzZSkpCmZvciBuLCBkIGlu'
            'IHJ1bnMuaXRlbXMoKToKICAgIHJlc1tmImNvdW50c197bn0iXSA9IGNvbGxlY3Rpb25zLkNvdW50'
            'ZXIoCiAgICAgICAgZiJ7ZlsncHJvcGVydGllcyddLmdldCgnbWFya3VwX3R5cGUnKX0ve2ZbJ3By'
            'b3BlcnRpZXMnXS5nZXQoJ2NsYXNzX25hbWUnKX0iIGZvciBmIGluIGRbImZlYXR1cmVzIl0pClBh'
            'dGgoYS5vdXQpLndyaXRlX3RleHQoanNvbi5kdW1wcyhyZXMsIGVuc3VyZV9hc2NpaT1GYWxzZSwg'
            'aW5kZW50PTEpLCBlbmNvZGluZz0idXRmLTgiKQo='
        ),
    },
    'scorer/compute_metrics.py': {
        "sha256": '299792ccc5b86873a1cb0554eac06a145f5e09496088ee7f734b41b9e4c70081',
        "b64": (
            'IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMw0KIiIiDQrQodC60YDQuNC/0YIg0YDQsNGB0YfQtdGC0LAg'
            '0LjRgtC+0LPQvtCy0L7QuSDQvNC10YLRgNC40LrQuCBBcmNoYWVvbG9neSBBc3Npc3RpdmUgRjEg'
            '0LTQu9GPINCa0J7Qly0zLg0KDQrQktGF0L7QtDog0LTQstCwIEdlb0pTT04gRmVhdHVyZUNvbGxl'
            'Y3Rpb24gKHByZWRpY3Rpb25zLCBncm91bmQgdHJ1dGgpDQrQntC20LjQtNCw0LXRgtGB0Y8g0L7Q'
            'sdGK0LXQtNC40L3QtdC90L3Ri9C5INGE0L7RgNC80LDRgjog0YMgZmVhdHVyZXMg0LIgcHJvcGVy'
            'dGllcyDQtdGB0YLRjCDQv9C+0LvRjzoNCi0gcmVnaW9uX25hbWUgKNC+0LHRj9C30LDRgtC10LvR'
            'jNC90L4pDQotIHN1Yl9yZWdpb25fbmFtZSAo0L7Qv9GG0LjQvtC90LDQu9GM0L3QvikNCi0gY2xh'
            'c3NfbmFtZSAo0L7QsdGP0LfQsNGC0LXQu9GM0L3QvikNCi0gY29uZmlkZW5jZSAo0L7Qv9GG0LjQ'
            'vtC90LDQu9GM0L3Qviwg0L/QviDRg9C80L7Qu9GH0LDQvdC40Y4gMS4wKQ0KDQrQkNC70LPQvtGA'
            '0LjRgtC8ICjQotCgINCa0J7Qly0zLCDQn9GA0LjQu9C+0LbQtdC90LjQtSA0KToNCi0g0KTQuNC7'
            '0YzRgtGA0YPQtdGCINC60LvQsNGB0YHRizog0L7RgdGC0LDQstC70Y/QtdGCINGC0L7Qu9GM0LrQ'
            'viDRhtC10LvQtdCy0YvQtQ0KLSDQmtC+0L7RgNC00LjQvdCw0YLRizog0LDQstGC0L7Qv9GA0LjQ'
            'stC10LTQtdC90LjQtSBFUFNHOjM4NTcgLT4gVVRNIChFUFNHOjMyNnh4LzMyN3h4KSDQtNC70Y8N'
            'CiAg0LzQtdGC0YDQuNGH0LXRgdC60LjRhSDRgNCw0YHRh9GR0YLQvtCyIChJTlBVVF9DUlNfTU9E'
            'RTogYXV0b3x3ZWJtZXJjYXRvcnx1dG0pDQotINCc0Y/Qs9C60LjQuSDQvNCw0YLRh9C40L3Qszog'
            'R1Qt0L/QvtC70LjQs9C+0L0g4oCUIGNvdiA9IGFyZWEoUOKIqUcpL21pbihhcmVhKSA+PSBDT1ZF'
            'UkFHRV9USFJFU0hPTEQNCiAg0Jgg0LLQvdGD0YLRgNC10L3QvdGP0Y8gKNC/0YDQtdC00YHRgtCw'
            '0LLQuNGC0LXQu9GM0L3QsNGPKSDRgtC+0YfQutCwINC/0YDQtdC00YHQutCw0LfQsNC90LjRjyDQ'
            'stC90YPRgtGA0Lgg0Y3RgtCw0LvQvtC90LA7DQogIEdULdGC0L7Rh9C60LAg4oCUINGA0LDRgdGB'
            '0YLQvtGP0L3QuNC1INGG0LXQvdGC0YDQvtC40LQt0YLQvtGH0LrQsCA8PSBQT0lOVF9SX1RPTEVS'
            'QU5DRQ0KLSDQltCw0LTQvdGL0Lkg0LzQsNGC0YfQuNC90LMg0L/QviBjb25maWRlbmNlIChvbmUt'
            'dG8tb25lKQ0KLSBQcmVjaXNpb246INC70LjRiNC90LXQtSDRgdGA0LDQsdCw0YLRi9Cy0LDQvdC4'
            '0LUg0YPRh9C40YLRi9Cy0LDQtdGC0YHRjyDRgSDQutC+0Y3RhNGE0LjRhtC40LXQvdGC0L7QvCAw'
            'LjUNCiAgKHRwIC8gKHRwICsgMC41KmZwICsgZXBzaWxvbikpDQotIEYxLXNjb3JlINC/0L4g0LrQ'
            'u9Cw0YHRgdCw0LwsINCy0LfQstC10YjQtdC90L3QvtC1INGD0YHRgNC10LTQvdC10L3QuNC1INC/'
            '0L4g0L/RgNC40YHRg9GC0YHRgtCy0YPRjtGJ0LjQvCDQsiBHVCDQutC70LDRgdGB0LDQvA0KICAo'
            '0L3QvtGA0LzQuNGA0L7QstC60LAg0LLQtdGB0L7Qsjog0LXRgdC70Lgg0LrQu9Cw0YHRgdCwINCy'
            'IEdUINC90LXRgiwg0LXQs9C+INCy0LXRgSDQuNGB0LrQu9GO0YfQsNC10YLRgdGPLCDRgNC10YjQ'
            'tdC90LjQtSAwOC4xMC4yMDI2KQ0KLSDQodC/0YDQsNCy0L7Rh9C90YvQtSDQv9C+0LrQsNC30LDR'
            'gtC10LvQuCAoYXV4aWxpYXJ5OyDQvdCwIGZpbmFsX3Njb3JlINCd0JUg0LLQu9C40Y/RjtGCKTog'
            '0LrQsNGH0LXRgdGC0LLQviDQutC+0L3RgtGD0YDQsA0KICDQv9C+IFRQLdC/0LDRgNCw0LwgKEJv'
            'dW5kYXJ5IElvVSwgSGF1c2RvcmZmOTUsIFBvTGlTLCDQvNC10YLRgNGLKSwg0L/QvtC70L3QvtGC'
            '0LAv0YLQvtGH0L3QvtGB0YLRjCDQv9C+DQogINGC0LjQv9Cw0Lwg0LTQsNC90L3Ri9GFIChtYXJr'
            'dXBfdHlwZSksINGA0LDQt9C80LXRgNC90YvQtSDQutCw0YLQtdCz0L7RgNC40LggR1Qt0L/QvtC7'
            '0LjQs9C+0L3QvtCyINC/0L4g0L/Qu9C+0YnQsNC00LggKNC8wrIpDQoNCtCS0YvQstC+0LQ6IGZp'
            'bmFsX3Njb3JlIHNjb3JlINCyIHN0ZG91dCAo0Lgg0L7Qv9GG0LjQvtC90LDQu9GM0L3QviBKU09O'
            'INGBINC00LXRgtCw0LvRj9C80LgpLg0KIiIiDQoNCmZyb20gX19mdXR1cmVfXyBpbXBvcnQgYW5u'
            'b3RhdGlvbnMNCg0KaW1wb3J0IGFyZ3BhcnNlDQppbXBvcnQganNvbg0KaW1wb3J0IGxvZ2dpbmcN'
            'CmltcG9ydCBtYXRoDQpmcm9tIHBhdGhsaWIgaW1wb3J0IFBhdGgNCmZyb20gdHlwaW5nIGltcG9y'
            'dCBBbnksIERpY3QsIExpc3QsIE9wdGlvbmFsLCBUdXBsZQ0KDQpmcm9tIHR5cGluZyBpbXBvcnQg'
            'VW5pb24NCg0KdHJ5Og0KICAgIGZyb20gc2hhcGVseS5nZW9tZXRyeSBpbXBvcnQgUG9seWdvbiwg'
            'TXVsdGlQb2x5Z29uLCBQb2ludCwgc2hhcGUNCiAgICBmcm9tIHNoYXBlbHkudmFsaWRhdGlvbiBp'
            'bXBvcnQgbWFrZV92YWxpZA0KZXhjZXB0IEltcG9ydEVycm9yOg0KICAgIHJhaXNlIEltcG9ydEVy'
            'cm9yKCLQotGA0LXQsdGD0LXRgtGB0Y8g0LHQuNCx0LvQuNC+0YLQtdC60LAgc2hhcGVseS4g0KPR'
            'gdGC0LDQvdC+0LLQuNGC0LUg0LXRkTogcGlwIGluc3RhbGwgc2hhcGVseSIpDQoNCnRyeToNCiAg'
            'ICBmcm9tIHB5cHJvaiBpbXBvcnQgVHJhbnNmb3JtZXINCiAgICBmcm9tIHNoYXBlbHkub3BzIGlt'
            'cG9ydCB0cmFuc2Zvcm0gYXMgX3NoYXBlbHlfdHJhbnNmb3JtDQpleGNlcHQgSW1wb3J0RXJyb3I6'
            'DQogICAgcmFpc2UgSW1wb3J0RXJyb3IoItCi0YDQtdCx0YPQtdGC0YHRjyDQsdC40LHQu9C40L7R'
            'gtC10LrQsCBweXByb2ouINCj0YHRgtCw0L3QvtCy0LjRgtC1INC10ZE6IHBpcCBpbnN0YWxsIHB5'
            'cHJvaiIpDQoNCg0KbG9nZ2VyID0gbG9nZ2luZy5nZXRMb2dnZXIoX19uYW1lX18pDQoNCg0KIyDQ'
            'oNC10LXRgdGC0YAg0LrQu9Cw0YHRgdC+0LIg0Lgg0L/QsNGA0LDQvNC10YLRgNGLINC80LXRgtGA'
            '0LjQutC4IOKAlCDQuNC3INC10LTQuNC90L7Qs9C+INC60L7QvdGE0LjQs9CwIChjb25maWcucHkp'
            'Lg0KIyDQmNC80LXQvdCwINC+0YHRgtCw0LLQu9C10L3RiyDQv9GA0LXQttC90LjQvNC4OiDQuNGF'
            'INC40LzQv9C+0YDRgtC40YDRg9GO0YIgbWV0cmljc19jYWxjdWxhdG9yLCBmcF9tYXRjaGluZywN'
            'CiMgc2ltcGxlX3NvbHV0aW9uL2d0X3NvbHV0aW9uLnB5INC4INGC0LXRgdGC0YsgKNCyINGC0L7Q'
            'vCDRh9C40YHQu9C1INC/0YDQvtCy0LXRgNGP0Y7RgiDRgtC+0LbQtNC10YHRgtCy0L4g0L7QsdGK'
            '0LXQutGC0L7QsikuDQp0cnk6DQogICAgZnJvbSBjb25maWcgaW1wb3J0ICggICMgbm9xYTogRTQw'
            'MiAgKNC40LzQv9C+0YDRgiDQv9C+0YHQu9C1INC+0L/RgNC10LTQtdC70LXQvdC40Y8gbG9nZ2Vy'
            'KQ0KICAgICAgICBBTExfVkFMSURfQ0xBU1NFUywNCiAgICAgICAgQ0xBU1NfTkFNRV9NQVBQSU5H'
            'LA0KICAgICAgICBDTEFTU19XRUlHSFRTLA0KICAgICAgICBEQVRBX1RZUEVfR1JPVVBTLA0KICAg'
            'ICAgICBERUZBVUxUX0RBVEFfVFlQRV9HUk9VUCwNCiAgICAgICAgZGF0YV90eXBlX2dyb3VwLA0K'
            'ICAgICAgICBnZXRfY29uZmlnLA0KICAgICAgICBub3JtYWxpemVfY2xhc3NfbmFtZSwNCiAgICAp'
            'DQoNCiAgICBQTEFURk9STV9DT05GSUdfTE9BREVEID0gVHJ1ZQ0KZXhjZXB0IEltcG9ydEVycm9y'
            'IGFzIGV4YzogICMg0L7QtNC40L3QvtGH0L3QsNGPINGB0LHQvtGA0LrQsCDRgdC60YDQuNC/0YLQ'
            'sDogY29uZmlnLnB5INGA0Y/QtNC+0Lwg0L3QtdGCDQogICAgUExBVEZPUk1fQ09ORklHX0xPQURF'
            'RCA9IEZhbHNlDQogICAgbG9nZ2VyLndhcm5pbmcoDQogICAgICAgICJjb25maWcucHkg0L3QtdC0'
            '0L7RgdGC0YPQv9C10L0gKCVzKSDigJQg0LjRgdC/0L7Qu9GM0LfRg9GO0YLRgdGPINCy0YHRgtGA'
            '0L7QtdC90L3Ri9C1INC00LXRhNC+0LvRgtGLINC80LXRgtGA0LjQutC4OyAiDQogICAgICAgICLQ'
            'tNC70Y8g0YDQsNGB0YfRkdGC0LAg0YDQvtCy0L3QviDQutCw0Log0L3QsCDQv9C70LDRgtGE0L7R'
            'gNC80LUg0L/QvtC70L7QttC40YLQtSBjb25maWcucHkg0YDRj9C00L7QvCDRgdC+INGB0LrRgNC4'
            '0L/RgtC+0LwiLA0KICAgICAgICBleGMsDQogICAgKQ0KDQogICAgIyAtLS0g0JrQvtC/0LjQuCDR'
            'gNC10LXRgdGC0YDQsCDQutC70LDRgdGB0L7QsiDQuNC3IGNvbmZpZy5weSAo0L/RgNC4INC/0YDQ'
            'sNCy0LrQtSBjb25maWcucHkg0LzQtdC90Y/RgtGMINGB0LjQvdGF0YDQvtC90L3QvikgLS0tDQog'
            'ICAgQ0xBU1NfV0VJR0hUUyA9IHsNCiAgICAgICAgImt1cmdhbnlfcG92cmV6aGRlbm55ZSI6IDMw'
            'LjAsDQogICAgICAgICJrdXJnYW55X3RzZWx5ZSI6IDIwLjAsDQogICAgICAgICJnb3JvZGlzaGNo'
            'YSI6IDE1LjAsDQogICAgICAgICJhcmtoaXRla3R1cnkiOiAxMC4wLA0KICAgICAgICAib2JqZWN0'
            'X3BvbHkiOiAxMC4wLA0KICAgICAgICAiZm9ydGlmaWthdHNpaSI6IDUuMCwNCiAgICAgICAgImZp'
            'bmRzX3BvaW50cyI6IDEwLjAsDQogICAgfQ0KDQogICAgIyDQodC40L3QvtC90LjQvNGLINC90LDQ'
            't9Cy0LDQvdC40Lkg0LrQu9Cw0YHRgdC+0LIgKNC60L7Qv9C40Y8gY29uZmlnLkNMQVNTX1NZTk9O'
            'WU1TOyDQutC70Y7Rh9C4IOKAlCDQvdC40LbQvdC40Lkg0YDQtdCz0LjRgdGC0YApLg0KICAgIF9D'
            'TEFTU19TWU5PTllNUyA9IHsNCiAgICAgICAgIyDQmtGD0YDQs9Cw0L3RiyDRhtC10LvRi9C1DQog'
            'ICAgICAgICLQutGD0YDQs9Cw0L3RiyI6ICJrdXJnYW55X3RzZWx5ZSIsDQogICAgICAgICLQutGD'
            '0YDQs9Cw0L3Ri1/RhtC10LvRi9C1IjogImt1cmdhbnlfdHNlbHllIiwNCiAgICAgICAgItGG0LXQ'
            'u9GL0LUiOiAia3VyZ2FueV90c2VseWUiLA0KICAgICAgICAjINCa0YPRgNCz0LDQvdGLINC/0L7Q'
            'stGA0LXQttC00ZHQvdC90YvQtQ0KICAgICAgICAi0LrRg9GA0LPQsNC90Ytf0L/QvtCy0YDQtdC2'
            '0LTQtdC90L3Ri9C1IjogImt1cmdhbnlfcG92cmV6aGRlbm55ZSIsDQogICAgICAgICLQutGD0YDQ'
            's9Cw0L3Ri1/RgNCw0LfRgNGD0YjQtdC90L3Ri9C1IjogImt1cmdhbnlfcG92cmV6aGRlbm55ZSIs'
            'DQogICAgICAgICLQv9C+0LLRgNC10LbQtNC10L3QvdGL0LUiOiAia3VyZ2FueV9wb3ZyZXpoZGVu'
            'bnllIiwNCiAgICAgICAgItGA0LDQt9GA0YPRiNC10L3QvdGL0LUiOiAia3VyZ2FueV9wb3ZyZXpo'
            'ZGVubnllIiwNCiAgICAgICAgIyDQpNC+0YDRgtC40YTQuNC60LDRhtC40LgNCiAgICAgICAgItGE'
            '0L7RgNGC0LjRhNC40LrQsNGG0LjRjyI6ICJmb3J0aWZpa2F0c2lpIiwNCiAgICAgICAgItGE0L7R'
            'gNGC0LjRhNC40LrQsNGG0LjQuCI6ICJmb3J0aWZpa2F0c2lpIiwNCiAgICAgICAgIyDQk9C+0YDQ'
            'vtC00LjRidCwDQogICAgICAgICLQs9C+0YDQvtC00LjRidC1IjogImdvcm9kaXNoY2hhIiwNCiAg'
            'ICAgICAgItCz0L7RgNC+0LTQuNGJ0LAiOiAiZ29yb2Rpc2hjaGEiLA0KICAgICAgICAjINCQ0YDR'
            'hdC40YLQtdC60YLRg9GA0YsNCiAgICAgICAgItCw0YDRhdC40YLQtdC60YLRg9GA0LAiOiAiYXJr'
            'aGl0ZWt0dXJ5IiwNCiAgICAgICAgItCw0YDRhdC40YLQtdC60YLRg9GA0YsiOiAiYXJraGl0ZWt0'
            'dXJ5IiwNCiAgICAgICAgIyDQk9C10L7RhNC40LfQuNC60LA6INGC0L7Rh9C10YfQvdGL0LUg0LDQ'
            'vdC+0LzQsNC70LjQuCAoRmluZHNQb2ludHMpDQogICAgICAgICJmaW5kc3BvaW50cyI6ICJmaW5k'
            'c19wb2ludHMiLA0KICAgICAgICAiZmluZHNwb2ludCI6ICJmaW5kc19wb2ludHMiLA0KICAgICAg'
            'ICAjINCT0LXQvtGE0LjQt9C40LrQsDog0L/QvtC70LjQs9C+0L3QsNC70YzQvdGL0LUg0LDQvdC+'
            '0LzQsNC70LjQuCAoT2JqZWN0UG9seSkNCiAgICAgICAgIm9iamVjdHBvbHkiOiAib2JqZWN0X3Bv'
            'bHkiLA0KICAgICAgICAib2JqZWN0c3BvbHkiOiAib2JqZWN0X3BvbHkiLA0KICAgIH0NCg0KICAg'
            'ICMg0JjQs9C90L7RgNC40YDRg9C10LzRi9C1INC90LDQt9Cy0LDQvdC40Y8g0LrQu9Cw0YHRgdC+'
            '0LIgKNC60L7Qv9C40Y8gY29uZmlnLklHTk9SRURfQ0xBU1NFUykuDQogICAgX0lHTk9SRURfQ0xB'
            'U1NFUyA9ICgNCiAgICAgICAgIlN0cnVjdHJQb2x5IiwgIlN0cnVjdFBvbHkiLCAiU3RydWN0dXJl'
            'UG9seSIsDQogICAgKQ0KDQogICAgZGVmIF9idWlsZF9jbGFzc19tYXBwaW5nKCkgLT4gRGljdFtz'
            'dHIsIE9wdGlvbmFsW3N0cl1dOg0KICAgICAgICAiIiLQmtC+0L/QuNGPIGNvbmZpZy5fYnVpbGRf'
            'Y2xhc3NfbWFwcGluZzogwqvQvdCw0LfQstCw0L3QuNC1IC0+INC60LDQvdC+0L3QuNGH0LXRgdC6'
            '0LjQuSDQutC70Y7RhyB8IE5vbmXCuy4iIiINCiAgICAgICAgbWFwcGluZzogRGljdFtzdHIsIE9w'
            'dGlvbmFsW3N0cl1dID0ge30NCiAgICAgICAgZm9yIGtleSBpbiBDTEFTU19XRUlHSFRTOg0KICAg'
            'ICAgICAgICAgbWFwcGluZ1trZXldID0ga2V5DQogICAgICAgIGZvciByYXcsIGtleSBpbiBfQ0xB'
            'U1NfU1lOT05ZTVMuaXRlbXMoKToNCiAgICAgICAgICAgIG1hcHBpbmdbcmF3XSA9IGtleQ0KICAg'
            'ICAgICAgICAgbWFwcGluZ1tyYXcubG93ZXIoKV0gPSBrZXkNCiAgICAgICAgZm9yIGlnbm9yZWQg'
            'aW4gX0lHTk9SRURfQ0xBU1NFUzoNCiAgICAgICAgICAgIG1hcHBpbmdbaWdub3JlZF0gPSBOb25l'
            'DQogICAgICAgICAgICBtYXBwaW5nW2lnbm9yZWQubG93ZXIoKV0gPSBOb25lDQogICAgICAgIHJl'
            'dHVybiBtYXBwaW5nDQoNCiAgICBBTExfVkFMSURfQ0xBU1NFUyA9IGxpc3QoQ0xBU1NfV0VJR0hU'
            'Uy5rZXlzKCkpDQogICAgQ0xBU1NfTkFNRV9NQVBQSU5HID0gX2J1aWxkX2NsYXNzX21hcHBpbmco'
            'KQ0KDQogICAgIyDQk9GA0YPQv9C/0Ysg0YLQuNC/0L7QsiDQtNCw0L3QvdGL0YUgKNC60L7Qv9C4'
            '0Y8gY29uZmlnLkRBVEFfVFlQRV9HUk9VUFMgLyBERUZBVUxUX0RBVEFfVFlQRV9HUk9VUCkuDQog'
            'ICAgREFUQV9UWVBFX0dST1VQUyA9IHsNCiAgICAgICAgImxhIjogItC70LjQtNCw0YAiLA0KICAg'
            'ICAgICAibGkiOiAi0LvQuNC00LDRgCIsDQogICAgICAgICJhZSI6ICLQsNGN0YDQvtGE0L7RgtC+'
            'IiwNCiAgICAgICAgImdlcm1hbl9hZSI6ICLQsNGN0YDQvtGE0L7RgtC+IiwNCiAgICAgICAgInNw'
            'X29yIjogItGB0L/Rg9GC0L3QuNC6IiwNCiAgICAgICAgInNwb3IiOiAi0YHQv9GD0YLQvdC40Loi'
            'LA0KICAgICAgICAib3IiOiAi0L7RgNGC0L7RhNC+0YLQviIsDQogICAgICAgICJtYWdnciI6ICLQ'
            's9C10L7RhNC40LfQuNC60LAiLA0KICAgICAgICAiZXRnciI6ICLQs9C10L7RhNC40LfQuNC60LAi'
            'LA0KICAgICAgICAiZ2VvZ3IiOiAi0LPQtdC+0YTQuNC30LjQutCwIiwNCiAgICAgICAgInRlcmZs'
            'IjogItCz0LXQvtGE0LjQt9C40LrQsCIsDQogICAgfQ0KICAgIERFRkFVTFRfREFUQV9UWVBFX0dS'
            'T1VQID0gItC/0YDQvtGH0LXQtSINCg0KICAgIGRlZiBub3JtYWxpemVfY2xhc3NfbmFtZShyYXdf'
            'bmFtZTogc3RyKSAtPiBPcHRpb25hbFtzdHJdOg0KICAgICAgICAiIiLQmtC+0L/QuNGPIGNvbmZp'
            'Zy5ub3JtYWxpemVfY2xhc3NfbmFtZSAoTm9uZSDigJQg0YLQvtC70YzQutC+INC00LvRjyDQuNCz'
            '0L3QvtGALdC70LjRgdGC0LApLiIiIg0KICAgICAgICBpZiByYXdfbmFtZSBpbiBDTEFTU19OQU1F'
            'X01BUFBJTkc6DQogICAgICAgICAgICByZXR1cm4gQ0xBU1NfTkFNRV9NQVBQSU5HW3Jhd19uYW1l'
            'XQ0KICAgICAgICBsb3cgPSBzdHIocmF3X25hbWUpLnN0cmlwKCkubG93ZXIoKQ0KICAgICAgICBp'
            'ZiBsb3cgaW4gQ0xBU1NfTkFNRV9NQVBQSU5HOg0KICAgICAgICAgICAgcmV0dXJuIENMQVNTX05B'
            'TUVfTUFQUElOR1tsb3ddDQogICAgICAgIHJldHVybiByYXdfbmFtZQ0KDQogICAgZGVmIGRhdGFf'
            'dHlwZV9ncm91cChtYXJrdXBfdHlwZTogT3B0aW9uYWxbc3RyXSkgLT4gc3RyOg0KICAgICAgICAi'
            'IiLQmtC+0L/QuNGPIGNvbmZpZy5kYXRhX3R5cGVfZ3JvdXAgKNC90LXQuNC30LLQtdGB0YLQvdC+'
            '0LUgLT4gREVGQVVMVF9EQVRBX1RZUEVfR1JPVVApLiIiIg0KICAgICAgICBpZiBtYXJrdXBfdHlw'
            'ZSBpcyBOb25lOg0KICAgICAgICAgICAgcmV0dXJuIERFRkFVTFRfREFUQV9UWVBFX0dST1VQDQog'
            'ICAgICAgIGtleSA9IHN0cihtYXJrdXBfdHlwZSkuc3RyaXAoKS5sb3dlcigpDQogICAgICAgIGlm'
            'IG5vdCBrZXk6DQogICAgICAgICAgICByZXR1cm4gREVGQVVMVF9EQVRBX1RZUEVfR1JPVVANCiAg'
            'ICAgICAgcmV0dXJuIERBVEFfVFlQRV9HUk9VUFMuZ2V0KGtleSwgREVGQVVMVF9EQVRBX1RZUEVf'
            'R1JPVVApDQoNCiAgICBjbGFzcyBfRmFsbGJhY2tDb25maWc6DQogICAgICAgICIiItCU0LXRhNC+'
            '0LvRgtGLIGNvbmZpZy5Db25maWcg0LTQu9GPINC+0LTQuNC90L7Rh9C90L7QuSDRgdCx0L7RgNC6'
            '0LggKGNvbmZpZy5weSDRgNGP0LTQvtC8INC90LXRgikuDQoNCiAgICAgICAg0J/RgNC4INC90LDR'
            'gdGC0YDQvtC50LrQsNGFINC/0L4g0YPQvNC+0LvRh9Cw0L3QuNGOINGB0LrQvtGAINC4IGNvbmZp'
            'Z19oYXNoINC+0LTQuNC90L7Rh9C90L7Qs9C+INGA0LDRgdGH0ZHRgtCwDQogICAgICAgINGB0L7Q'
            'stC/0LDQtNCw0Y7RgiDRgSDQv9C70LDRgtGE0L7RgNC80LXQvdC90YvQvNC4ICjQv9GA0L7QstC1'
            '0YDRj9C10YIgdGVzdF9zdGFuZGFsb25lX2NvbXB1dGVfbWV0cmljcykuDQogICAgICAgICIiIg0K'
            'DQogICAgICAgIGRlZiBfX2luaXRfXyhzZWxmKSAtPiBOb25lOg0KICAgICAgICAgICAgc2VsZi5D'
            'T1ZFUkFHRV9USFJFU0hPTEQgPSAwLjUNCiAgICAgICAgICAgIHNlbGYuUE9JTlRfUl9UT0xFUkFO'
            'Q0UgPSAyLjANCiAgICAgICAgICAgIHNlbGYuSU5QVVRfQ1JTX01PREUgPSAiYXV0byINCiAgICAg'
            'ICAgICAgIHNlbGYuQ0lSQ0xFX1BPSU5UUyA9IDMyDQogICAgICAgICAgICBzZWxmLkVQU0lMT04g'
            'PSAxZS02DQogICAgICAgICAgICBzZWxmLkNPTlRPVVJfQkFORF9SQVRJTyA9IDAuMDINCiAgICAg'
            'ICAgICAgIHNlbGYuQ09OVE9VUl9TQU1QTEVTID0gMjAwDQogICAgICAgICAgICBzZWxmLlNJWkVf'
            'QklOUyA9IFsxMDAuMCwgNTAwLjAsIDEwMDAuMCwgNTAwMC4wLCAxMDAwMC4wXQ0KICAgICAgICAg'
            'ICAgc2VsZi5NRVRSSUNfVkVSU0lPTiA9ICJrb3ozLTEuNS4wIg0KICAgICAgICAgICAgc2VsZi5N'
            'T0RBTElUWV9NT0RFID0gIm9mZiINCiAgICAgICAgICAgIHNlbGYuU1RSSUNUX0NSUyA9IEZhbHNl'
            'DQogICAgICAgICAgICBzZWxmLk1PREFMSVRJRVMgPSB7DQogICAgICAgICAgICAgICAgImxhIjog'
            'MS4wLCAibGkiOiAxLjAsICJhZSI6IDEuMCwgInNwb3IiOiAxLjAsICJvciI6IDEuMCwNCiAgICAg'
            'ICAgICAgICAgICAibWFnZ3IiOiAxLjAsICJldGdyIjogMS4wLCAiZ2VvZ3IiOiAxLjAsICJ0ZXJm'
            'bCI6IDEuMCwNCiAgICAgICAgICAgICAgICAidW5rbm93biI6IDEuMCwgImRlZmF1bHQiOiAxLjAs'
            'DQogICAgICAgICAgICB9DQogICAgICAgICAgICBzZWxmLk1BWF9GSUxFX1NJWkVfTUIgPSAxMDAN'
            'CiAgICAgICAgICAgIHNlbGYuTUFYX0ZFQVRVUkVTX1BFUl9GSUxFID0gMjAwMDANCiAgICAgICAg'
            'ICAgIHNlbGYuTUFYX0JPRFlfU0laRV9NQiA9IDEwMA0KICAgICAgICAgICAgc2VsZi5NQVhfUE9M'
            'WUdPTl9QT0lOVFMgPSAzMDAwDQogICAgICAgICAgICBzZWxmLk1BWF9QT0xZR09OX0FSRUEgPSAx'
            'ZTEyDQogICAgICAgICAgICBzZWxmLk1BWF9WQUxJREFUSU9OX0lTU1VFUyA9IDUwDQogICAgICAg'
            'ICAgICBzZWxmLlJFUEFJUl9FTkFCTEVEID0gVHJ1ZQ0KICAgICAgICAgICAgc2VsZi5SRVBBSVJf'
            'TUFYX0FUVEVNUFRTID0gMw0KICAgICAgICAgICAgc2VsZi5SRVBBSVJfV0FJVF9TRUNPTkRTID0g'
            'MS4wDQogICAgICAgICAgICBzZWxmLlJFUEFJUl9TVEFCSUxJVFlfQ0hFQ0tTID0gMg0KDQogICAg'
            'ICAgIGRlZiBtZXRyaWNfc3VtbWFyeShzZWxmKSAtPiBEaWN0W3N0ciwgQW55XToNCiAgICAgICAg'
            'ICAgICIiItCk0L7RgNC80LDRgiDQuCDRgdC+0LTQtdGA0LbQuNC80L7QtSDigJQg0LrQsNC6IGNv'
            'bmZpZy5Db25maWcubWV0cmljX3N1bW1hcnkoKS4iIiINCiAgICAgICAgICAgIGltcG9ydCBoYXNo'
            'bGliDQoNCiAgICAgICAgICAgIHBheWxvYWQ6IERpY3Rbc3RyLCBBbnldID0gew0KICAgICAgICAg'
            'ICAgICAgICJtZXRyaWNfdmVyc2lvbiI6IHNlbGYuTUVUUklDX1ZFUlNJT04sDQogICAgICAgICAg'
            'ICAgICAgInRhcmdldF9jbGFzc2VzIjogbGlzdChDTEFTU19XRUlHSFRTLmtleXMoKSksDQogICAg'
            'ICAgICAgICAgICAgImNsYXNzX3dlaWdodHMiOiBkaWN0KENMQVNTX1dFSUdIVFMpLA0KICAgICAg'
            'ICAgICAgICAgICJjb3ZlcmFnZV90aHJlc2hvbGQiOiBzZWxmLkNPVkVSQUdFX1RIUkVTSE9MRCwN'
            'CiAgICAgICAgICAgICAgICAiaW5wdXRfY3JzX21vZGUiOiBzZWxmLklOUFVUX0NSU19NT0RFLA0K'
            'ICAgICAgICAgICAgICAgICJyX3RvbGVyYW5jZSI6IHNlbGYuUE9JTlRfUl9UT0xFUkFOQ0UsDQog'
            'ICAgICAgICAgICAgICAgImVwc2lsb24iOiBzZWxmLkVQU0lMT04sDQogICAgICAgICAgICAgICAg'
            'ImNpcmNsZV9wb2ludHMiOiBzZWxmLkNJUkNMRV9QT0lOVFMsDQogICAgICAgICAgICAgICAgInN0'
            'cmljdF9jcnMiOiBzZWxmLlNUUklDVF9DUlMsDQogICAgICAgICAgICAgICAgIm1vZGFsaXR5X21v'
            'ZGUiOiBzZWxmLk1PREFMSVRZX01PREUsDQogICAgICAgICAgICAgICAgIm1vZGFsaXRpZXMiOiBk'
            'aWN0KHNlbGYuTU9EQUxJVElFUyksDQogICAgICAgICAgICAgICAgImxpbWl0cyI6IHsNCiAgICAg'
            'ICAgICAgICAgICAgICAgIm1heF9maWxlX3NpemVfbWIiOiBzZWxmLk1BWF9GSUxFX1NJWkVfTUIs'
            'DQogICAgICAgICAgICAgICAgICAgICJtYXhfZmVhdHVyZXNfcGVyX2ZpbGUiOiBzZWxmLk1BWF9G'
            'RUFUVVJFU19QRVJfRklMRSwNCiAgICAgICAgICAgICAgICAgICAgIm1heF9ib2R5X3NpemVfbWIi'
            'OiBzZWxmLk1BWF9CT0RZX1NJWkVfTUIsDQogICAgICAgICAgICAgICAgICAgICJtYXhfcG9seWdv'
            'bl9wb2ludHMiOiBzZWxmLk1BWF9QT0xZR09OX1BPSU5UUywNCiAgICAgICAgICAgICAgICAgICAg'
            'Im1heF9wb2x5Z29uX2FyZWEiOiBzZWxmLk1BWF9QT0xZR09OX0FSRUEsDQogICAgICAgICAgICAg'
            'ICAgICAgICJtYXhfdmFsaWRhdGlvbl9pc3N1ZXMiOiBzZWxmLk1BWF9WQUxJREFUSU9OX0lTU1VF'
            'UywNCiAgICAgICAgICAgICAgICB9LA0KICAgICAgICAgICAgICAgICJhdXhpbGlhcnlfbWV0cmlj'
            'cyI6IHsNCiAgICAgICAgICAgICAgICAgICAgImNvbnRvdXJfYmFuZF9yYXRpbyI6IHNlbGYuQ09O'
            'VE9VUl9CQU5EX1JBVElPLA0KICAgICAgICAgICAgICAgICAgICAiY29udG91cl9zYW1wbGVzIjog'
            'c2VsZi5DT05UT1VSX1NBTVBMRVMsDQogICAgICAgICAgICAgICAgICAgICJzaXplX2JpbnMiOiBs'
            'aXN0KHNlbGYuU0laRV9CSU5TKSwNCiAgICAgICAgICAgICAgICAgICAgImRhdGFfdHlwZV9ncm91'
            'cHMiOiBkaWN0KERBVEFfVFlQRV9HUk9VUFMpLA0KICAgICAgICAgICAgICAgIH0sDQogICAgICAg'
            'ICAgICAgICAgInJlcGFpciI6IHsNCiAgICAgICAgICAgICAgICAgICAgImVuYWJsZWQiOiBzZWxm'
            'LlJFUEFJUl9FTkFCTEVELA0KICAgICAgICAgICAgICAgICAgICAibWF4X2F0dGVtcHRzIjogc2Vs'
            'Zi5SRVBBSVJfTUFYX0FUVEVNUFRTLA0KICAgICAgICAgICAgICAgICAgICAid2FpdF9zZWNvbmRz'
            'Ijogc2VsZi5SRVBBSVJfV0FJVF9TRUNPTkRTLA0KICAgICAgICAgICAgICAgICAgICAic3RhYmls'
            'aXR5X2NoZWNrcyI6IHNlbGYuUkVQQUlSX1NUQUJJTElUWV9DSEVDS1MsDQogICAgICAgICAgICAg'
            'ICAgfSwNCiAgICAgICAgICAgIH0NCiAgICAgICAgICAgIHBheWxvYWRbIndlaWdodHNfc3VtIl0g'
            'PSByb3VuZChzdW0ocGF5bG9hZFsiY2xhc3Nfd2VpZ2h0cyJdLnZhbHVlcygpKSwgNCkNCiAgICAg'
            'ICAgICAgIHBheWxvYWRbImNvbmZpZ19oYXNoIl0gPSBoYXNobGliLnNoYTI1NigNCiAgICAgICAg'
            'ICAgICAgICBqc29uLmR1bXBzKHBheWxvYWQsIHNvcnRfa2V5cz1UcnVlLCBlbnN1cmVfYXNjaWk9'
            'RmFsc2UpLmVuY29kZSgidXRmLTgiKQ0KICAgICAgICAgICAgKS5oZXhkaWdlc3QoKQ0KICAgICAg'
            'ICAgICAgcmV0dXJuIHBheWxvYWQNCg0KICAgIF9GQUxMQkFDS19DT05GSUcgPSBfRmFsbGJhY2tD'
            'b25maWcoKQ0KDQogICAgZGVmIGdldF9jb25maWcoKSAtPiBBbnk6DQogICAgICAgICIiItCh0LjQ'
            'vdCz0LvRgtC+0L0g0LfQsNC/0LDRgdC90L7Qs9C+INC60L7QvdGE0LjQs9CwICjQsNC90LDQu9C+'
            '0LMgY29uZmlnLmdldF9jb25maWcsINC00LXRhNC+0LvRgtGLIENvbmZpZykuIiIiDQogICAgICAg'
            'IHJldHVybiBfRkFMTEJBQ0tfQ09ORklHDQoNCl9NRVRSSUNfQ0ZHID0gZ2V0X2NvbmZpZygpDQoN'
            'Cg0KZGVmIHNwbGl0X21lcmdlZF9ieV9zb3VyY2VfYW5kX2NsYXNzKA0KICAgIGdlb2pzb25fZGF0'
            'YTogRGljdFtzdHIsIEFueV0sDQopIC0+IERpY3Rbc3RyLCBEaWN0W3N0ciwgRGljdF1dOg0KICAg'
            'ICIiItCg0LDQt9Cx0LjQstCw0LXRgiDQvtCx0YrQtdC00LjQvdC10L3QvdGL0LkgRmVhdHVyZUNv'
            'bGxlY3Rpb24g0L3QsCB7c291cmNlOiB7Y2xhc3M6IEZlYXR1cmVDb2xsZWN0aW9ufX0uIiIiDQog'
            'ICAgcmVzdWx0OiBEaWN0W3N0ciwgRGljdFtzdHIsIERpY3RdXSA9IHt9DQogICAgZmVhdHVyZXMg'
            'PSBnZW9qc29uX2RhdGEuZ2V0KCJmZWF0dXJlcyIsIFtdKQ0KICAgIGZvciBmZWF0dXJlIGluIGZl'
            'YXR1cmVzOg0KICAgICAgICBwcm9wcyA9IGZlYXR1cmUuZ2V0KCJwcm9wZXJ0aWVzIiwge30pIG9y'
            'IHt9DQogICAgICAgIHJlZ2lvbl9uYW1lID0gcHJvcHMuZ2V0KCJyZWdpb25fbmFtZSIsICJkZWZh'
            'dWx0X3NvdXJjZSIpDQogICAgICAgIHN1Yl9yZWdpb25fbmFtZSA9IHByb3BzLmdldCgic3ViX3Jl'
            'Z2lvbl9uYW1lIiwgIiIpDQogICAgICAgIGlmIHN1Yl9yZWdpb25fbmFtZSBhbmQgc3ViX3JlZ2lv'
            'bl9uYW1lLnN0cmlwKCk6DQogICAgICAgICAgICBzb3VyY2VfbmFtZSA9IGYie3JlZ2lvbl9uYW1l'
            'fV9fe3N1Yl9yZWdpb25fbmFtZX0iDQogICAgICAgIGVsc2U6DQogICAgICAgICAgICBzb3VyY2Vf'
            'bmFtZSA9IHJlZ2lvbl9uYW1lDQogICAgICAgIHJhd19jbGFzcyA9IHByb3BzLmdldCgiY2xhc3Nf'
            'bmFtZSIsICJ1bmtub3duIikNCiAgICAgICAgY2xhc3NfbmFtZSA9IG5vcm1hbGl6ZV9jbGFzc19u'
            'YW1lKHJhd19jbGFzcykNCiAgICAgICAgaWYgY2xhc3NfbmFtZSBpcyBOb25lOg0KICAgICAgICAg'
            'ICAgY29udGludWUgICMg0JjQs9C90L7RgNC40YDRg9C10LzRi9C5INC60LvQsNGB0YEgKGFydGVm'
            'YWt0eV9saWRhcmEg0Lgg0YIu0L8uKQ0KDQogICAgICAgIGlmIHNvdXJjZV9uYW1lIG5vdCBpbiBy'
            'ZXN1bHQ6DQogICAgICAgICAgICByZXN1bHRbc291cmNlX25hbWVdID0ge30NCiAgICAgICAgaWYg'
            'Y2xhc3NfbmFtZSBub3QgaW4gcmVzdWx0W3NvdXJjZV9uYW1lXToNCiAgICAgICAgICAgIHJlc3Vs'
            'dFtzb3VyY2VfbmFtZV1bY2xhc3NfbmFtZV0gPSB7InR5cGUiOiAiRmVhdHVyZUNvbGxlY3Rpb24i'
            'LCAiZmVhdHVyZXMiOiBbXX0NCiAgICAgICAgcmVzdWx0W3NvdXJjZV9uYW1lXVtjbGFzc19uYW1l'
            'XVsiZmVhdHVyZXMiXS5hcHBlbmQoZmVhdHVyZSkNCiAgICByZXR1cm4gcmVzdWx0DQoNCg0KZGVm'
            'IF9leHRyYWN0X2dlb21ldHJ5KGZlYXR1cmU6IERpY3QpIC0+IE9wdGlvbmFsW1VuaW9uW1BvbHln'
            'b24sIFBvaW50XV06DQogICAgIiIi0JjQt9Cy0LvQtdC60LDQtdGCIFNoYXBlbHkgUG9seWdvbiDQ'
            'uNC70LggUG9pbnQg0LjQtyBHZW9KU09OIGZlYXR1cmUuIiIiDQogICAgZ2VvbWV0cnkgPSBmZWF0'
            'dXJlLmdldCgiZ2VvbWV0cnkiKQ0KICAgIGlmIG5vdCBnZW9tZXRyeToNCiAgICAgICAgcmV0dXJu'
            'IE5vbmUNCiAgICBnZW9tX3R5cGUgPSBnZW9tZXRyeS5nZXQoInR5cGUiKQ0KICAgIGNvb3JkcyA9'
            'IGdlb21ldHJ5LmdldCgiY29vcmRpbmF0ZXMiLCBbXSkNCiAgICBpZiBub3QgY29vcmRzOg0KICAg'
            'ICAgICByZXR1cm4gTm9uZQ0KICAgIHRyeToNCiAgICAgICAgaWYgZ2VvbV90eXBlID09ICJQb2lu'
            'dCI6DQogICAgICAgICAgICBpZiBsZW4oY29vcmRzKSA+PSAyOg0KICAgICAgICAgICAgICAgIHJl'
            'dHVybiBQb2ludChjb29yZHNbMF0sIGNvb3Jkc1sxXSkNCiAgICAgICAgICAgIHJldHVybiBOb25l'
            'DQogICAgICAgIGVsaWYgZ2VvbV90eXBlID09ICJQb2x5Z29uIjoNCiAgICAgICAgICAgIHBvbHkg'
            'PSBQb2x5Z29uKGNvb3Jkc1swXSwgY29vcmRzWzE6XSBpZiBsZW4oY29vcmRzKSA+IDEgZWxzZSBO'
            'b25lKQ0KICAgICAgICBlbGlmIGdlb21fdHlwZSA9PSAiTXVsdGlQb2x5Z29uIjoNCiAgICAgICAg'
            'ICAgICMg0JHQtdGA0LXQvCDRgdCw0LzRi9C5INCx0L7Qu9GM0YjQvtC5INC/0L7Qu9C40LPQvtC9'
            'INC40Lcg0LzRg9C70YzRgtC40L/QvtC70LjQs9C+0L3QsA0KICAgICAgICAgICAgcG9seXMgPSBb'
            'XQ0KICAgICAgICAgICAgZm9yIHBvbHlfY29vcmRzIGluIGNvb3JkczoNCiAgICAgICAgICAgICAg'
            'ICBwID0gUG9seWdvbihwb2x5X2Nvb3Jkc1swXSwgcG9seV9jb29yZHNbMTpdIGlmIGxlbihwb2x5'
            'X2Nvb3JkcykgPiAxIGVsc2UgTm9uZSkNCiAgICAgICAgICAgICAgICBpZiBub3QgcC5pc192YWxp'
            'ZDoNCiAgICAgICAgICAgICAgICAgICAgcCA9IG1ha2VfdmFsaWQocCkNCiAgICAgICAgICAgICAg'
            'ICBpZiBwLmlzX3ZhbGlkIGFuZCBub3QgcC5pc19lbXB0eToNCiAgICAgICAgICAgICAgICAgICAg'
            'cG9seXMuYXBwZW5kKHApDQogICAgICAgICAgICBpZiBub3QgcG9seXM6DQogICAgICAgICAgICAg'
            'ICAgcmV0dXJuIE5vbmUNCiAgICAgICAgICAgIHBvbHkgPSBtYXgocG9seXMsIGtleT1sYW1iZGEg'
            'cDogcC5hcmVhKQ0KICAgICAgICBlbHNlOg0KICAgICAgICAgICAgcmV0dXJuIE5vbmUNCiAgICAg'
            'ICAgaWYgbm90IHBvbHkuaXNfdmFsaWQ6DQogICAgICAgICAgICBwb2x5ID0gbWFrZV92YWxpZChw'
            'b2x5KQ0KICAgICAgICAjIG1ha2VfdmFsaWQg0LzQvtC20LXRgiDQstC10YDQvdGD0YLRjCBNdWx0'
            'aVBvbHlnb24NCiAgICAgICAgaWYgaXNpbnN0YW5jZShwb2x5LCBNdWx0aVBvbHlnb24pOg0KICAg'
            'ICAgICAgICAgcG9seSA9IG1heChwb2x5Lmdlb21zLCBrZXk9bGFtYmRhIGc6IGcuYXJlYSkNCiAg'
            'ICAgICAgaWYgcG9seS5pc192YWxpZCBhbmQgbm90IHBvbHkuaXNfZW1wdHk6DQogICAgICAgICAg'
            'ICByZXR1cm4gcG9seQ0KICAgIGV4Y2VwdCBFeGNlcHRpb246DQogICAgICAgIHBhc3MNCiAgICBy'
            'ZXR1cm4gTm9uZQ0KDQoNCmRlZiBfZXh0cmFjdF9wb2x5Z29uKGZlYXR1cmU6IERpY3QpIC0+IE9w'
            'dGlvbmFsW1BvbHlnb25dOg0KICAgICIiItCe0LHRgNCw0YLQvdCw0Y8g0YHQvtCy0LzQtdGB0YLQ'
            'uNC80L7RgdGC0Yw6INC40LfQstC70LXQutCw0LXRgiDRgtC+0LvRjNC60L4gUG9seWdvbiDQuNC3'
            'IEdlb0pTT04gZmVhdHVyZS4iIiINCiAgICBnZW9tID0gX2V4dHJhY3RfZ2VvbWV0cnkoZmVhdHVy'
            'ZSkNCiAgICBpZiBpc2luc3RhbmNlKGdlb20sIFBvbHlnb24pOg0KICAgICAgICByZXR1cm4gZ2Vv'
            'bQ0KICAgIHJldHVybiBOb25lDQoNCg0KZGVmIF9nZXRfY29uZmlkZW5jZShmZWF0dXJlOiBEaWN0'
            'KSAtPiBmbG9hdDoNCiAgICAiIiLQmNC30LLQu9C10LrQsNC10YIgY29uZmlkZW5jZSDQuNC3IGZl'
            'YXR1cmUsINC/0L4g0YPQvNC+0LvRh9Cw0L3QuNGOIDEuMC4iIiINCiAgICBwcm9wcyA9IGZlYXR1'
            'cmUuZ2V0KCJwcm9wZXJ0aWVzIiwge30pIG9yIHt9DQogICAgY29uZiA9IHByb3BzLmdldCgiY29u'
            'ZmlkZW5jZSIsIDEuMCkNCiAgICB0cnk6DQogICAgICAgIHJldHVybiBmbG9hdChjb25mKQ0KICAg'
            'IGV4Y2VwdCAoVHlwZUVycm9yLCBWYWx1ZUVycm9yKToNCiAgICAgICAgcmV0dXJuIDEuMA0KDQoN'
            'CmRlZiBfcG9seWdvbl9pb3UoYTogUG9seWdvbiwgYjogUG9seWdvbikgLT4gZmxvYXQ6DQogICAg'
            'IiIi0JLRi9GH0LjRgdC70Y/QtdGCIElvVSDQvNC10LbQtNGDINC00LLRg9C80Y8g0L/QvtC70LjQ'
            's9C+0L3QsNC80LguIiIiDQogICAgdHJ5Og0KICAgICAgICBpZiBub3QgYS5pc192YWxpZCBvciBu'
            'b3QgYi5pc192YWxpZDoNCiAgICAgICAgICAgIHJldHVybiAwLjANCiAgICAgICAgaW50ZXIgPSBh'
            'LmludGVyc2VjdGlvbihiKQ0KICAgICAgICB1bmlvbiA9IGEudW5pb24oYikNCiAgICAgICAgaWYg'
            'dW5pb24uYXJlYSA9PSAwOg0KICAgICAgICAgICAgcmV0dXJuIDAuMA0KICAgICAgICByZXR1cm4g'
            'aW50ZXIuYXJlYSAvIHVuaW9uLmFyZWENCiAgICBleGNlcHQgRXhjZXB0aW9uOg0KICAgICAgICBy'
            'ZXR1cm4gMC4wDQoNCg0KZGVmIF9wb2x5Z29uX2NvdmVyYWdlKGE6IFBvbHlnb24sIGI6IFBvbHln'
            'b24pIC0+IGZsb2F0Og0KICAgICIiItCf0LXRgNC10LrRgNGL0YLQuNC1INC/0L4g0LzQtdC90YzR'
            'iNC10Lkg0YTQuNCz0YPRgNC1OiBjb3YgPSBhcmVhKEHiiKlCKSAvIG1pbihhcmVhKEEpLCBhcmVh'
            'KEIpKS4NCg0KICAgINCf0YDQuNC70L7QttC10L3QuNC1IDQ6INC60YDQuNGC0LXRgNC40Lkg0L3Q'
            'sNGF0L7QtNC60Lgg0LTQu9GPINC/0L7Qu9C40LPQvtC90LAg4oCUIGNvdiA+PSAwLjU7INCy0LvQ'
            'vtC20LXQvdC40LUNCiAgICDQu9GO0LHQvtC5INC40Lcg0YTQuNCz0YPRgCDQtNCw0ZHRgiBjb3Yg'
            'PSAxLjAuDQogICAgIiIiDQogICAgdHJ5Og0KICAgICAgICBpZiBub3QgYS5pc192YWxpZCBvciBu'
            'b3QgYi5pc192YWxpZDoNCiAgICAgICAgICAgIHJldHVybiAwLjANCiAgICAgICAgbWluX2FyZWEg'
            'PSBtaW4oYS5hcmVhLCBiLmFyZWEpDQogICAgICAgIGlmIG1pbl9hcmVhIDw9IDA6DQogICAgICAg'
            'ICAgICByZXR1cm4gMC4wDQogICAgICAgIHJldHVybiBhLmludGVyc2VjdGlvbihiKS5hcmVhIC8g'
            'bWluX2FyZWENCiAgICBleGNlcHQgRXhjZXB0aW9uOg0KICAgICAgICByZXR1cm4gMC4wDQoNCg0K'
            'ZGVmIF9wb2ludF9pbl9tYXNrKHBvaW50OiBQb2ludCwgbWFzazogUG9seWdvbikgLT4gYm9vbDoN'
            'CiAgICAiIiLQn9GA0L7QstC10YDRj9C10YIsINC90LDRhdC+0LTQuNGC0YHRjyDQu9C4INGC0L7R'
            'h9C60LAg0LLQvdGD0YLRgNC4INC/0L7Qu9C40LPQvtC90LAuDQoNCiAgICDQlNC+0L/Rg9GB0Log'
            '0LrQsNGB0LDQvdC40Y8g0LPRgNCw0L3QuNGG0Ysg0YLQvtGCINC20LUsINGH0YLQviDQsiBfaW5u'
            'ZXJfcG9pbnRfaW5fbWFzazog0YLQvtGH0LrQsCDQvdCwINCz0YDQsNC90LjRhtC1DQogICAg0YHR'
            'h9C40YLQsNC10YLRgdGPINC/0L7Qv9Cw0LLRiNC10LkuDQogICAgIiIiDQogICAgdHJ5Og0KICAg'
            'ICAgICByZXR1cm4gbWFzay5jb250YWlucyhwb2ludCkgb3IgbWFzay5ib3VuZGFyeS5kaXN0YW5j'
            'ZShwb2ludCkgPCAxZS0xMA0KICAgIGV4Y2VwdCBFeGNlcHRpb246DQogICAgICAgIHJldHVybiBG'
            'YWxzZQ0KDQoNCmRlZiBfaW5uZXJfcG9pbnRfaW5fbWFzayhwcmVkX3BvbHk6IFBvbHlnb24sIGd0'
            'X3BvbHk6IFBvbHlnb24pIC0+IGJvb2w6DQogICAgIiIi0J/RgNC+0LLQtdGA0Y/QtdGCLCDQvdCw'
            '0YXQvtC00LjRgtGB0Y8g0LvQuCDQstC90YPRgtGA0LXQvdC90Y/RjyAo0L/RgNC10LTRgdGC0LDQ'
            'stC40YLQtdC70YzQvdCw0Y8pINGC0L7Rh9C60LAg0L/RgNC10LTRgdC60LDQt9Cw0L3QuNGPDQog'
            'ICAg0LLQvdGD0YLRgNC4IEdULdC/0L7Qu9C40LPQvtC90LAuDQoNCiAgICDQkiDQvtGC0LvQuNGH'
            '0LjQtSDQvtGCINGG0LXQvdGC0YDQsCDQvNCw0YHRgSAoY2VudHJvaWQpLCDQutC+0YLQvtGA0YvQ'
            'uSDRgyDQvdC10LLRi9C/0YPQutC70YvRhSAo0LLQvtCz0L3Rg9GC0YvRhSwg0YPQs9C70L7QstGL'
            '0YUpDQogICAg0L7QsdGK0LXQutGC0L7QsiDQvNC+0LbQtdGCINC70LXQttCw0YLRjCDQstC90LUg'
            '0LrQvtC90YLRg9GA0LAsIHJlcHJlc2VudGF0aXZlX3BvaW50KCkgKHBvaW50LW9uLXN1cmZhY2Us'
            'DQogICAgR0VPUyBJbnRlcmlvclBvaW50KSDQv9C+INC/0L7RgdGC0YDQvtC10L3QuNGOINCy0YHQ'
            'tdCz0LTQsCDQu9C10LbQuNGCINCy0L3Rg9GC0YDQuCDRhNC40LPRg9GA0YsuINCa0YDQuNGC0LXR'
            'gNC40LkNCiAgICDQodGG0LXQvdCw0YDQuNGPINCQICjQn9GA0LjQu9C+0LbQtdC90LjQtSA0KSDQ'
            'v9C+0Y3RgtC+0LzRgyDRg9GB0YLQvtC50YfQuNCyINC6INGE0L7RgNC80LUg0L7QsdGK0LXQutGC'
            '0LAuDQogICAgIiIiDQogICAgdHJ5Og0KICAgICAgICBwb2ludCA9IHByZWRfcG9seS5yZXByZXNl'
            'bnRhdGl2ZV9wb2ludCgpDQogICAgICAgIHJldHVybiBndF9wb2x5LmNvbnRhaW5zKHBvaW50KSBv'
            'ciBndF9wb2x5LmJvdW5kYXJ5LmRpc3RhbmNlKHBvaW50KSA8IDFlLTEwDQogICAgZXhjZXB0IEV4'
            'Y2VwdGlvbjoNCiAgICAgICAgcmV0dXJuIEZhbHNlDQoNCg0KIyA9PT09PT09PT09PT09PT09PT09'
            'PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0N'
            'CiMgVVRNOiDQvNC10YLRgNC40YfQtdGB0LrQsNGPINC/0YDQvtC10LrRhtC40Y8g0LTQu9GPINGA'
            '0LDRgdGH0ZHRgtCwINC/0LvQvtGJ0LDQtNC10Lkg0Lgg0YDQsNGB0YHRgtC+0Y/QvdC40LkgKNCf'
            '0YDQuNC70L7QttC10L3QuNC1IDQpDQojID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09'
            'PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PQ0KDQpfVVRNX0FCU19Y'
            'X1RIUkVTSE9MRCA9IDFfMDAwXzAwMCAgIyDCq9Cx0L7Qu9GM0YjQuNC1wrsgeCA9INGA0LXQsNC7'
            '0YzQvdGL0LUg0LzQtdGC0YDRiyBFUFNHOjM4NTcNCl9UUkFOU0ZPUk1FUl8zODU3X1RPX1dHUzg0'
            'OiBPcHRpb25hbFtUcmFuc2Zvcm1lcl0gPSBOb25lDQpfVFJBTlNGT1JNRVJfMzg1N19UT19VVE06'
            'IERpY3RbaW50LCBUcmFuc2Zvcm1lcl0gPSB7fQ0KDQoNCmRlZiBfZ2VvbV9tYXhfYWJzX3goZ2Vv'
            'bTogQW55KSAtPiBmbG9hdDoNCiAgICAiIiLQnNCw0LrRgdC40LzRg9C8IHx4fCDQv9C+INCy0YHQ'
            'tdC8INC60L7QvtGA0LTQuNC90LDRgtCw0Lwg0LPQtdC+0LzQtdGC0YDQuNC4LiIiIg0KICAgIHRy'
            'eToNCiAgICAgICAgbWlueCwgXywgbWF4eCwgXyA9IGdlb20uYm91bmRzDQogICAgICAgIHJldHVy'
            'biBtYXgoYWJzKG1pbngpLCBhYnMobWF4eCkpDQogICAgZXhjZXB0IEV4Y2VwdGlvbjoNCiAgICAg'
            'ICAgcmV0dXJuIDAuMA0KDQoNCmRlZiBfdG9fdXRtX3pvbmVfZXBzZyhsb246IGZsb2F0LCBsYXQ6'
            'IGZsb2F0KSAtPiBpbnQ6DQogICAgIiIiRVBTRy3QutC+0LQgVVRNLdC30L7QvdGLOiAzMjZ4eCDi'
            'gJQg0YHQtdCy0LXRgNC90L7QtSDQv9C+0LvRg9GI0LDRgNC40LUsIDMyN3h4IOKAlCDRjtC20L3Q'
            'vtC1LiIiIg0KICAgIHpvbmUgPSBpbnQoKGxvbiArIDE4MCkgLy8gNikgKyAxDQogICAgem9uZSA9'
            'IG1pbihtYXgoem9uZSwgMSksIDYwKQ0KICAgIGJhc2UgPSAzMjYwMCBpZiBsYXQgPj0gMCBlbHNl'
            'IDMyNzAwDQogICAgcmV0dXJuIGJhc2UgKyB6b25lDQoNCg0KZGVmIF9nZXRfdHJhbnNmb3JtZXJf'
            'Mzg1N190b193Z3M4NCgpIC0+IFRyYW5zZm9ybWVyOg0KICAgICIiItCa0Y3RiNC40YDQvtCy0LDQ'
            'vdC90YvQuSDRgtGA0LDQvdGB0YTQvtGA0LzQtdGAIEVQU0c6Mzg1NyAtPiBFUFNHOjQzMjYgKNC0'
            '0LvRjyDQstGL0LHQvtGA0LAg0LfQvtC90YspLiIiIg0KICAgIGdsb2JhbCBfVFJBTlNGT1JNRVJf'
            'Mzg1N19UT19XR1M4NA0KICAgIGlmIF9UUkFOU0ZPUk1FUl8zODU3X1RPX1dHUzg0IGlzIE5vbmU6'
            'DQogICAgICAgIF9UUkFOU0ZPUk1FUl8zODU3X1RPX1dHUzg0ID0gVHJhbnNmb3JtZXIuZnJvbV9j'
            'cnMoMzg1NywgNDMyNiwgYWx3YXlzX3h5PVRydWUpDQogICAgcmV0dXJuIF9UUkFOU0ZPUk1FUl8z'
            'ODU3X1RPX1dHUzg0DQoNCg0KZGVmIF9nZXRfdHJhbnNmb3JtZXJfMzg1N190b191dG0oZXBzZzog'
            'aW50KSAtPiBUcmFuc2Zvcm1lcjoNCiAgICAiIiLQmtGN0YjQuNGA0L7QstCw0L3QvdGL0Lkg0YLR'
            'gNCw0L3RgdGE0L7RgNC80LXRgCBFUFNHOjM4NTcgLT4gRVBTRzozMjZ4eC8zMjd4eCDQv9C+INC3'
            '0L7QvdC1LiIiIg0KICAgIHRyYW5zZm9ybWVyID0gX1RSQU5TRk9STUVSXzM4NTdfVE9fVVRNLmdl'
            'dChlcHNnKQ0KICAgIGlmIHRyYW5zZm9ybWVyIGlzIE5vbmU6DQogICAgICAgIHRyYW5zZm9ybWVy'
            'ID0gVHJhbnNmb3JtZXIuZnJvbV9jcnMoMzg1NywgZXBzZywgYWx3YXlzX3h5PVRydWUpDQogICAg'
            'ICAgIF9UUkFOU0ZPUk1FUl8zODU3X1RPX1VUTVtlcHNnXSA9IHRyYW5zZm9ybWVyDQogICAgcmV0'
            'dXJuIHRyYW5zZm9ybWVyDQoNCg0KZGVmIF90b191dG0oZ2VvbTogQW55LCBtb2RlOiBzdHIpIC0+'
            'IEFueToNCiAgICAiIiLQn9GA0LjQstC+0LTQuNGCINCz0LXQvtC80LXRgtGA0LjRjiDQuNC3IEVQ'
            'U0c6Mzg1NyDQuiDQvNC10YLRgNC40YfQtdGB0LrQvtC5INC/0YDQvtC10LrRhtC40LggVVRNLg0K'
            'DQogICAg0KDQtdC20LjQvNGLIChJTlBVVF9DUlNfTU9ERSk6DQogICAgLSBhdXRvOiDQutC+0L3Q'
            'stC10YDRgtC40YDQvtCy0LDRgtGMLCDQtdGB0LvQuCDQutC+0L7RgNC00LjQvdCw0YLRiyDCq9Cx'
            '0L7Qu9GM0YjQuNC1wrsgKG1heCB8eHwgPiAxZTYsINGCLtC1LiDRjdGC0L4NCiAgICAgINGA0LXQ'
            'sNC70YzQvdGL0LUg0LzQtdGC0YDRiyBFUFNHOjM4NTcpOyDRhNC40LrRgdGC0YPRgNGLINGBINC8'
            '0LDQu9GL0LzQuCDRh9C40YHQu9Cw0LzQuCDQvdC1INGC0YDQvtCz0LDQtdC8Ow0KICAgIC0gd2Vi'
            'bWVyY2F0b3I6INC60L7QvdCy0LXRgNGC0LjRgNC+0LLQsNGC0Ywg0LLRgdC10LPQtNCwOw0KICAg'
            'IC0gdXRtOiDQtNCw0L3QvdGL0LUg0YPQttC1INCyINC80LXRgtGA0LjRh9C10YHQutC+0Lkg0L/R'
            'gNC+0LXQutGG0LjQuCDigJQgcGFzc3Rocm91Z2guDQoNCiAgICDQl9C+0L3QsCDQvtC/0YDQtdC0'
            '0LXQu9GP0LXRgtGB0Y8g0L/QviDRhtC10L3RgtGA0L7QuNC00YMg0LPQtdC+0LzQtdGC0YDQuNC4'
            'ICgzODU3IC0+IFdHUzg0IC0+IFVUTSBFUFNHKS4NCiAgICAiIiINCiAgICBpZiBtb2RlID09ICJ1'
            'dG0iOg0KICAgICAgICByZXR1cm4gZ2VvbQ0KICAgIGlmIG1vZGUgPT0gImF1dG8iIGFuZCBfZ2Vv'
            'bV9tYXhfYWJzX3goZ2VvbSkgPD0gX1VUTV9BQlNfWF9USFJFU0hPTEQ6DQogICAgICAgIHJldHVy'
            'biBnZW9tDQogICAgdHJ5Og0KICAgICAgICBjZW50cm9pZCA9IGdlb20uY2VudHJvaWQNCiAgICAg'
            'ICAgbG9uLCBsYXQgPSBfZ2V0X3RyYW5zZm9ybWVyXzM4NTdfdG9fd2dzODQoKS50cmFuc2Zvcm0o'
            'Y2VudHJvaWQueCwgY2VudHJvaWQueSkNCiAgICAgICAgZXBzZyA9IF90b191dG1fem9uZV9lcHNn'
            'KGxvbiwgbGF0KQ0KICAgICAgICByZXR1cm4gX3NoYXBlbHlfdHJhbnNmb3JtKF9nZXRfdHJhbnNm'
            'b3JtZXJfMzg1N190b191dG0oZXBzZykudHJhbnNmb3JtLCBnZW9tKQ0KICAgIGV4Y2VwdCBFeGNl'
            'cHRpb246DQogICAgICAgIHJldHVybiBnZW9tDQoNCg0KZGVmIF9wb2ludF90b19jaXJjbGUocG9p'
            'bnQ6IFBvaW50LCByYWRpdXM6IGZsb2F0LCBudW1fcG9pbnRzOiBpbnQgPSAzMikgLT4gUG9seWdv'
            'bjoNCiAgICAiIiLQmtC+0L3QstC10YDRgtC40YDRg9C10YIgUG9pbnQg0LIg0LrRgNGD0LPQvtCy'
            '0L7QuSBQb2x5Z29uICjQsNC/0L/RgNC+0LrRgdC40LzQsNGG0LjRjyDQvtC60YDRg9C20L3QvtGB'
            '0YLQuCkuDQoNCiAgICBBcmdzOg0KICAgICAgICBwb2ludDogU2hhcGVseSBQb2ludA0KICAgICAg'
            'ICByYWRpdXM6INCg0LDQtNC40YPRgSDQvtC60YDRg9C20L3QvtGB0YLQuCDQsiDQtdC00LjQvdC4'
            '0YbQsNGFINC60L7QvtGA0LTQuNC90LDRgiAo0LzQtdGC0YDRiyDQtNC70Y8gVVRNKQ0KICAgICAg'
            'ICBudW1fcG9pbnRzOiDQmtC+0LvQuNGH0LXRgdGC0LLQviDRgtC+0YfQtdC6INCw0L/Qv9GA0L7Q'
            'utGB0LjQvNCw0YbQuNC4DQogICAgIiIiDQogICAgY29vcmRzID0gW10NCiAgICBmb3IgaSBpbiBy'
            'YW5nZShudW1fcG9pbnRzKToNCiAgICAgICAgYW5nbGUgPSAyICogbWF0aC5waSAqIGkgLyBudW1f'
            'cG9pbnRzDQogICAgICAgIGNvb3Jkcy5hcHBlbmQoKHBvaW50LnggKyByYWRpdXMgKiBtYXRoLmNv'
            'cyhhbmdsZSksDQogICAgICAgICAgICAgICAgICAgICAgICBwb2ludC55ICsgcmFkaXVzICogbWF0'
            'aC5zaW4oYW5nbGUpKSkNCiAgICBjb29yZHMuYXBwZW5kKGNvb3Jkc1swXSkNCiAgICByZXR1cm4g'
            'UG9seWdvbihjb29yZHMpDQoNCg0KIyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09'
            'PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0NCiMg0KHQv9GA0LDQstC+'
            '0YfQvdGL0LUg0L/QvtC60LDQt9Cw0YLQtdC70Lgg0JrQntCXLTMgKNCf0YDQuNC70L7QttC10L3Q'
            'uNC1IDQpOiDQutCw0YfQtdGB0YLQstC+INC60L7QvdGC0YPRgNCwLg0KIyDQodGH0LjRgtCw0Y7R'
            'gtGB0Y8g0L/QviDQvNCw0YLRh9C90YPRgtGL0Lwg0L/QsNGA0LDQvCAoVFApINCyINC80LXRgtGA'
            '0LjRh9C10YHQutC+0Lkg0L/RgNC+0LXQutGG0LjQuCBVVE0g4oCUINCyINC80LXRgtGA0LDRhS4N'
            'CiMg0J3QsCBmaW5hbF9zY29yZSDQndCVINCy0LvQuNGP0Y7RgiAo0LfQsNC60YDQtdC/0LvQtdC9'
            '0L4g0YLQtdGB0YLQvtC8KS4NCiMgPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09'
            'PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09DQoNCmRlZiBfYnVtcChjb3Vu'
            'dGVyOiBEaWN0W3N0ciwgaW50XSwga2V5OiBzdHIpIC0+IE5vbmU6DQogICAgIiIi0JjQvdC60YDQ'
            'tdC80LXQvdGCINGB0YfRkdGC0YfQuNC60LAg0L/QviDQutC70Y7Rh9GDICjQvNC40LrRgNC+LdGF'
            '0LXQu9C/0LXRgCDRgdC/0YDQsNCy0L7Rh9C90YvRhSDRgdGH0ZHRgtGH0LjQutC+0LIpLiIiIg0K'
            'ICAgIGNvdW50ZXJba2V5XSA9IGNvdW50ZXIuZ2V0KGtleSwgMCkgKyAxDQoNCg0KZGVmIF9iYm94'
            'X2RpYWdvbmFsKGdlb206IEFueSkgLT4gZmxvYXQ6DQogICAgIiIi0JTQuNCw0LPQvtC90LDQu9GM'
            'IGJib3gg0LPQtdC+0LzQtdGC0YDQuNC4ICjQtdC00LjQvdC40YbRiyDQutC+0L7RgNC00LjQvdCw'
            '0YI7INC00LvRjyBVVE0g4oCUINC80LXRgtGA0YspLiIiIg0KICAgIG1pbngsIG1pbnksIG1heHgs'
            'IG1heHkgPSBnZW9tLmJvdW5kcw0KICAgIHJldHVybiBtYXRoLmh5cG90KG1heHggLSBtaW54LCBt'
            'YXh5IC0gbWlueSkNCg0KDQpkZWYgX2NvbnRvdXJfYmFuZChnZW9tOiBBbnksIHdpZHRoOiBmbG9h'
            'dCkgLT4gQW55Og0KICAgICIiItCf0L7Qu9C+0YHQsCDRiNC40YDQuNC90Ysgd2lkdGgg0LLQtNC+'
            '0LvRjCDQs9GA0LDQvdC40YbRiyDRhNC40LPRg9GA0Ys6IEIgPSBYIFxcIFguYnVmZmVyKC13aWR0'
            'aCkuDQoNCiAgICBqb2luX3N0eWxlPW1pdHJlIOKAlCDQv9GA0Y/QvNC+0YPQs9C+0LvRjNC90YvQ'
            'tSDRgdGC0YvQutC4LCDQutCw0Log0LIg0L7Qv9GA0LXQtNC10LvQtdC90LjQuCBCb3VuZGFyeSBJ'
            'b1UNCiAgICAoQ2hlbmcgZXQgYWwuLCAyMDIxKS4gRmFsbGJhY2sg0L3QsCDQstGL0YDQvtC20LTQ'
            'tdC90L3Ri9GFINGE0LjQs9GD0YDQsNGFOiDQtdGB0LvQuCDRjdGA0L7Qt9C40Y8g0L/Rg9GB0YLQ'
            'sA0KICAgICjRhNC40LPRg9GA0LAg0YLQvtC90YzRiNC1INC/0L7Qu9C+0YHRiyksINC/0L7Qu9C+'
            '0YHQvtC5INGB0YfQuNGC0LDQtdGC0YHRjyDQstGB0Y8g0YTQuNCz0YPRgNCwIOKAlCDQtNC70Y8g'
            '0LjQtNC10L3RgtC40YfQvdGL0YUNCiAgICDRhNC40LPRg9GAINGA0LXQt9GD0LvRjNGC0LDRgiDQ'
            'vtGB0YLQsNGR0YLRgdGPIDEuMCwg0LTQu9GPINGC0L7QvdC60LjRhSDQv9C+0LvQvtGB0LAg0L3Q'
            'tSDCq9GC0L7Qu9GJ0LXCuyDRgdCw0LzQvtC5INGE0LjQs9GD0YDRiy4NCiAgICAiIiINCiAgICBl'
            'cm9kZWQgPSBnZW9tLmJ1ZmZlcigtd2lkdGgsIGpvaW5fc3R5bGU9Im1pdHJlIikNCiAgICBpZiBl'
            'cm9kZWQuaXNfZW1wdHk6DQogICAgICAgIHJldHVybiBnZW9tDQogICAgYmFuZCA9IGdlb20uZGlm'
            'ZmVyZW5jZShlcm9kZWQpDQogICAgcmV0dXJuIGdlb20gaWYgYmFuZC5pc19lbXB0eSBlbHNlIGJh'
            'bmQNCg0KDQpkZWYgX2JvdW5kYXJ5X2lvdShwcmVkX2dlb206IEFueSwgZ3RfZ2VvbTogQW55LCBi'
            'YW5kX3JhdGlvOiBmbG9hdCkgLT4gT3B0aW9uYWxbZmxvYXRdOg0KICAgICIiIkJvdW5kYXJ5IElv'
            'VSDQv9Cw0YDRiyDCq9C/0L7Qu9C40LPQvtC9LdC/0L7Qu9C40LPQvtC9wrs7IE5vbmUg4oCUINC9'
            '0LXQstGL0YfQuNGB0LvQuNC80L4uDQoNCiAgICBkID0gYmFuZF9yYXRpbyDDlyBtYXgo0LTQuNCw'
            '0LPQvtC90LDQu9GMIGJib3ggcHJlZCwg0LTQuNCw0LPQvtC90LDQu9GMIGJib3ggZ3QpOw0KICAg'
            'IEJvdW5kYXJ5SW9VID0gYXJlYShCX1Ag4oipIEJfRykgLyBhcmVhKEJfUCDiiKogQl9HKS4g0JjQ'
            'tNC10L3RgtC40YfQvdGL0LUg0L/QvtC70LjQs9C+0L3RiyDihpIgMS4wLg0KICAgINCS0YvRgNC+'
            '0LbQtNC10L3QvdGL0LUg0YHQu9GD0YfQsNC4ICjQvdGD0LvQtdCy0LDRjyDRiNC40YDQuNC90LAg'
            '0L/QvtC70L7RgdGLLCDQvdGD0LvQtdCy0LDRjyDQv9C70L7RidCw0LTRjCDQvtCx0YrQtdC00LjQ'
            'vdC10L3QuNGPLA0KICAgINC40YHQutC70Y7Rh9C10L3QuNC1IHNoYXBlbHkpIOKGkiBOb25lOiDQ'
            'v9Cw0YDQsCDQv9GA0L7Qv9GD0YHQutCw0LXRgtGB0Y8g0LIg0YHRgNC10LTQvdC10LwgKNC90LUg'
            'TmFOKS4NCiAgICAiIiINCiAgICB0cnk6DQogICAgICAgIGQgPSBiYW5kX3JhdGlvICogbWF4KF9i'
            'Ym94X2RpYWdvbmFsKHByZWRfZ2VvbSksIF9iYm94X2RpYWdvbmFsKGd0X2dlb20pKQ0KICAgICAg'
            'ICBpZiBkIDw9IDA6DQogICAgICAgICAgICByZXR1cm4gTm9uZQ0KICAgICAgICBiYW5kX3AgPSBf'
            'Y29udG91cl9iYW5kKHByZWRfZ2VvbSwgZCkNCiAgICAgICAgYmFuZF9nID0gX2NvbnRvdXJfYmFu'
            'ZChndF9nZW9tLCBkKQ0KICAgICAgICB1bmlvbiA9IGJhbmRfcC51bmlvbihiYW5kX2cpDQogICAg'
            'ICAgIGlmIHVuaW9uLmFyZWEgPD0gMDoNCiAgICAgICAgICAgIHJldHVybiBOb25lDQogICAgICAg'
            'IHJldHVybiBiYW5kX3AuaW50ZXJzZWN0aW9uKGJhbmRfZykuYXJlYSAvIHVuaW9uLmFyZWENCiAg'
            'ICBleGNlcHQgRXhjZXB0aW9uOg0KICAgICAgICByZXR1cm4gTm9uZQ0KDQoNCmRlZiBfcGVyY2Vu'
            'dGlsZTk1KHZhbHVlczogTGlzdFtmbG9hdF0pIC0+IGZsb2F0Og0KICAgICIiIjk1LdC5INC/0LXR'
            'gNGG0LXQvdGC0LjQu9GMINGBINC70LjQvdC10LnQvdC+0Lkg0LjQvdGC0LXRgNC/0L7Qu9GP0YbQ'
            'uNC10LkgKNC60LDQuiBudW1weS5wZXJjZW50aWxlKS4iIiINCiAgICBvcmRlcmVkID0gc29ydGVk'
            'KHZhbHVlcykNCiAgICBuID0gbGVuKG9yZGVyZWQpDQogICAgaWYgbiA9PSAxOg0KICAgICAgICBy'
            'ZXR1cm4gb3JkZXJlZFswXQ0KICAgIHBvcyA9IDAuOTUgKiAobiAtIDEpDQogICAgbG93ID0gaW50'
            'KG1hdGguZmxvb3IocG9zKSkNCiAgICBoaWdoID0gaW50KG1hdGguY2VpbChwb3MpKQ0KICAgIGlm'
            'IGxvdyA9PSBoaWdoOg0KICAgICAgICByZXR1cm4gb3JkZXJlZFtsb3ddDQogICAgZnJhYyA9IHBv'
            'cyAtIGxvdw0KICAgIHJldHVybiBvcmRlcmVkW2xvd10gKiAoMSAtIGZyYWMpICsgb3JkZXJlZFto'
            'aWdoXSAqIGZyYWMNCg0KDQpkZWYgX3NhbXBsZV9ib3VuZGFyeShnZW9tOiBBbnksIHNhbXBsZXM6'
            'IGludCkgLT4gT3B0aW9uYWxbTGlzdFtQb2ludF1dOg0KICAgICIiInNhbXBsZXMg0YDQsNCy0L3Q'
            'vtGD0LTQsNC70ZHQvdC90YvRhSDRgtC+0YfQtdC6INCy0L3QtdGI0L3QtdCz0L4g0LrQvtC70YzR'
            'htCwOyBOb25lIOKAlCDQstGL0YDQvtC20LTQtdC90L3QsNGPINCz0YDQsNC90LjRhtCwLiIiIg0K'
            'ICAgIHJpbmcgPSBnZW9tLmV4dGVyaW9yDQogICAgaWYgcmluZy5sZW5ndGggPD0gMDoNCiAgICAg'
            'ICAgcmV0dXJuIE5vbmUNCiAgICByZXR1cm4gW3JpbmcuaW50ZXJwb2xhdGUoaSAvIHNhbXBsZXMs'
            'IG5vcm1hbGl6ZWQ9VHJ1ZSkgZm9yIGkgaW4gcmFuZ2Uoc2FtcGxlcyldDQoNCg0KZGVmIF9oYXVz'
            'ZG9yZmY5NShwcmVkX2dlb206IEFueSwgZ3RfZ2VvbTogQW55LCBzYW1wbGVzOiBpbnQpIC0+IE9w'
            'dGlvbmFsW2Zsb2F0XToNCiAgICAiIiJIYXVzZG9yZmY5NSDQv9Cw0YDRiyDQsiDQvNC10YLRgNCw'
            '0YU6IG1heChwOTUoQeKGkkIpLCBwOTUoQuKGkkEpKTsgTm9uZSDigJQg0L3QtdCy0YvRh9C40YHQ'
            'u9C40LzQvi4NCg0KICAgINCT0YDQsNC90LjRhtGLINGB0Y3QvNC/0LvQuNGA0YPRjtGC0YHRjyDR'
            'gNCw0LLQvdC+0YPQtNCw0LvRkdC90L3QviDQv9C+INCy0L3QtdGI0L3QtdC80YMg0LrQvtC70YzR'
            'htGDIChzYW1wbGVzINGC0L7Rh9C10LopOw0KICAgINC00LvRjyDQutCw0LbQtNC+0LPQviDRgdGN'
            '0LzQv9C70LAg4oCUINGA0LDRgdGB0YLQvtGP0L3QuNC1INC00L4g0LPRgNCw0L3QuNGG0Ysg0LTR'
            'gNGD0LPQvtC5INGE0LjQs9GD0YDRiyAo0LLQutC70Y7Rh9Cw0Y8NCiAgICDQstC90YPRgtGA0LXQ'
            'vdC90LjQtSDQutC+0LvRjNGG0LApLCDQt9Cw0YLQtdC8IDk1LdC5INC/0LXRgNGG0LXQvdGC0LjQ'
            'u9GMINC/0L4g0L3QsNC/0YDQsNCy0LvQtdC90LjRji4g0KHQuNC80LzQtdGC0YDQuNC30LDRhtC4'
            '0Y8NCiAgICDRh9C10YDQtdC3IG1heCDigJQg0YHRgtCw0L3QtNCw0YDRgtC90L7QtSDQvtC/0YDQ'
            'tdC00LXQu9C10L3QuNC1IEhEOTUuDQogICAgIiIiDQogICAgdHJ5Og0KICAgICAgICBwb2ludHNf'
            'YSA9IF9zYW1wbGVfYm91bmRhcnkocHJlZF9nZW9tLCBzYW1wbGVzKQ0KICAgICAgICBwb2ludHNf'
            'YiA9IF9zYW1wbGVfYm91bmRhcnkoZ3RfZ2VvbSwgc2FtcGxlcykNCiAgICAgICAgaWYgbm90IHBv'
            'aW50c19hIG9yIG5vdCBwb2ludHNfYjoNCiAgICAgICAgICAgIHJldHVybiBOb25lDQogICAgICAg'
            'IGJvdW5kYXJ5X2IgPSBndF9nZW9tLmJvdW5kYXJ5DQogICAgICAgIGJvdW5kYXJ5X2EgPSBwcmVk'
            'X2dlb20uYm91bmRhcnkNCiAgICAgICAgZm9yd2FyZCA9IF9wZXJjZW50aWxlOTUoW2JvdW5kYXJ5'
            'X2IuZGlzdGFuY2UocHQpIGZvciBwdCBpbiBwb2ludHNfYV0pDQogICAgICAgIGJhY2t3YXJkID0g'
            'X3BlcmNlbnRpbGU5NShbYm91bmRhcnlfYS5kaXN0YW5jZShwdCkgZm9yIHB0IGluIHBvaW50c19i'
            'XSkNCiAgICAgICAgcmV0dXJuIG1heChmb3J3YXJkLCBiYWNrd2FyZCkNCiAgICBleGNlcHQgRXhj'
            'ZXB0aW9uOg0KICAgICAgICByZXR1cm4gTm9uZQ0KDQoNCmRlZiBfcmluZ192ZXJ0aWNlcyhnZW9t'
            'OiBBbnkpIC0+IExpc3RbVHVwbGVbZmxvYXQsIGZsb2F0XV06DQogICAgIiIi0JLQtdGA0YjQuNC9'
            '0Ysg0LLQvdC10YjQvdC10LPQviDQutC+0LvRjNGG0LAg0LHQtdC3INC00YPQsdC70LjQutCw0YLQ'
            'sCDQt9Cw0LzRi9C60LDQvdC40Y8gKNC/0L7RgdC70LXQtNC90Y/RjyA9PSDQv9C10YDQstC+0Lkp'
            'LiIiIg0KICAgIGNvb3JkcyA9IGxpc3QoZ2VvbS5leHRlcmlvci5jb29yZHMpDQogICAgaWYgbGVu'
            'KGNvb3JkcykgPiAxIGFuZCBjb29yZHNbMF0gPT0gY29vcmRzWy0xXToNCiAgICAgICAgY29vcmRz'
            'ID0gY29vcmRzWzotMV0NCiAgICByZXR1cm4gY29vcmRzDQoNCg0KZGVmIF9wb2xpcyhwcmVkX2dl'
            'b206IEFueSwgZ3RfZ2VvbTogQW55KSAtPiBPcHRpb25hbFtmbG9hdF06DQogICAgIiIiUG9MaVMg'
            '0L/QsNGA0Ysg0LIg0LzQtdGC0YDQsNGFOyBOb25lIOKAlCDQvdC10LLRi9GH0LjRgdC70LjQvNC+'
            'Lg0KDQogICAg0KHRgNC10LTQvdC10LUg0L/QviBtK24g0LLQtdGA0YjQuNC90LDQvCDQstC90LXR'
            'iNC90LjRhSDQutC+0LvQtdGGINC+0LHQtdC40YUg0YTQuNCz0YPRgCDRgNCw0YHRgdGC0L7Rj9C9'
            '0LjQuSDQtNC+INCz0YDQsNC90LjRhtGLDQogICAg0LTRgNGD0LPQvtC5INGE0LjQs9GD0YDRizog'
            'W86jX3t24oiIUH0gZCh2LCDiiIJHKSArIM6jX3t24oiIR30gZCh2LCDiiIJQKV0gLyAobSArIG4p'
            'Lg0KICAgINCY0LTQtdC90YLQuNGH0L3Ri9C1INGE0LjQs9GD0YDRiyDihpIgMDsg0LfQsNC80YvQ'
            'utCw0Y7RidC40Lkg0LTRg9Cx0LvQuNC60LDRgiDQutC+0LvRjNGG0LAg0L3QtSDRgdGH0LjRgtCw'
            '0LXRgtGB0Y8g0LTQstCw0LbQtNGLLg0KICAgICIiIg0KICAgIHRyeToNCiAgICAgICAgdmVydHNf'
            'cCA9IF9yaW5nX3ZlcnRpY2VzKHByZWRfZ2VvbSkNCiAgICAgICAgdmVydHNfZyA9IF9yaW5nX3Zl'
            'cnRpY2VzKGd0X2dlb20pDQogICAgICAgIGlmIG5vdCB2ZXJ0c19wIG9yIG5vdCB2ZXJ0c19nOg0K'
            'ICAgICAgICAgICAgcmV0dXJuIE5vbmUNCiAgICAgICAgYm91bmRhcnlfZyA9IGd0X2dlb20uYm91'
            'bmRhcnkNCiAgICAgICAgYm91bmRhcnlfcCA9IHByZWRfZ2VvbS5ib3VuZGFyeQ0KICAgICAgICB0'
            'b3RhbCA9IHN1bShib3VuZGFyeV9nLmRpc3RhbmNlKFBvaW50KHYpKSBmb3IgdiBpbiB2ZXJ0c19w'
            'KQ0KICAgICAgICB0b3RhbCArPSBzdW0oYm91bmRhcnlfcC5kaXN0YW5jZShQb2ludCh2KSkgZm9y'
            'IHYgaW4gdmVydHNfZykNCiAgICAgICAgcmV0dXJuIHRvdGFsIC8gKGxlbih2ZXJ0c19wKSArIGxl'
            'bih2ZXJ0c19nKSkNCiAgICBleGNlcHQgRXhjZXB0aW9uOg0KICAgICAgICByZXR1cm4gTm9uZQ0K'
            'DQoNCmNsYXNzIEYxQ2FsY3VsYXRvcjoNCiAgICAiIiLQmtCw0LvRjNC60YPQu9GP0YLQvtGAIEFy'
            'Y2hhZW9sb2d5IEFzc2lzdGl2ZSBGMSDQtNC70Y8g0JrQntCXLTMgKNCf0YDQuNC70L7QttC10L3Q'
            'uNC1IDQpLiIiIg0KDQogICAgTUVUUklDX0NSUyA9ICJVVE0gKEVQU0c6MzI2eHgvMzI3eHgpIg0K'
            'DQogICAgZGVmIF9faW5pdF9fKA0KICAgICAgICBzZWxmLA0KICAgICAgICBjb3ZlcmFnZV90aHJl'
            'c2hvbGQ6IE9wdGlvbmFsW2Zsb2F0XSA9IE5vbmUsDQogICAgICAgIHJfdG9sZXJhbmNlOiBPcHRp'
            'b25hbFtmbG9hdF0gPSBOb25lLA0KICAgICAgICBpbnB1dF9jcnNfbW9kZTogT3B0aW9uYWxbc3Ry'
            'XSA9IE5vbmUsDQogICAgKToNCiAgICAgICAgIyDQl9C90LDRh9C10L3QuNGPINC/0L4g0YPQvNC+'
            '0LvRh9Cw0L3QuNGOIOKAlCDQuNC3INC10LTQuNC90L7Qs9C+INC60L7QvdGE0LjQs9CwICguZW52'
            'IC8gY29uZmlnLnB5KQ0KICAgICAgICBzZWxmLmNvdmVyYWdlX3RocmVzaG9sZCA9IChfTUVUUklD'
            'X0NGRy5DT1ZFUkFHRV9USFJFU0hPTEQNCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg'
            'ICAgaWYgY292ZXJhZ2VfdGhyZXNob2xkIGlzIE5vbmUgZWxzZSBjb3ZlcmFnZV90aHJlc2hvbGQp'
            'DQogICAgICAgIHNlbGYucl90b2xlcmFuY2UgPSAoX01FVFJJQ19DRkcuUE9JTlRfUl9UT0xFUkFO'
            'Q0UNCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiByX3RvbGVyYW5jZSBpcyBOb25lIGVs'
            'c2Ugcl90b2xlcmFuY2UpDQogICAgICAgIHNlbGYuaW5wdXRfY3JzX21vZGUgPSAoX01FVFJJQ19D'
            'RkcuSU5QVVRfQ1JTX01PREUNCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiBpbnB1'
            'dF9jcnNfbW9kZSBpcyBOb25lIGVsc2UgaW5wdXRfY3JzX21vZGUpDQogICAgICAgIHNlbGYuY2ly'
            'Y2xlX3BvaW50cyA9IF9NRVRSSUNfQ0ZHLkNJUkNMRV9QT0lOVFMNCiAgICAgICAgc2VsZi5lcHNp'
            'bG9uID0gX01FVFJJQ19DRkcuRVBTSUxPTg0KICAgICAgICBzZWxmLnRhcmdldF9jbGFzc2VzID0g'
            'bGlzdChBTExfVkFMSURfQ0xBU1NFUykNCiAgICAgICAgIyDQodC/0YDQsNCy0L7Rh9C90YvQtSDQ'
            'v9C+0LrQsNC30LDRgtC10LvQuCAo0J/RgNC40LvQvtC20LXQvdC40LUgNDsg0L3QsCBmaW5hbF9z'
            'Y29yZSDQvdC1INCy0LvQuNGP0Y7RgikNCiAgICAgICAgc2VsZi5jb250b3VyX2JhbmRfcmF0aW8g'
            'PSBfTUVUUklDX0NGRy5DT05UT1VSX0JBTkRfUkFUSU8NCiAgICAgICAgc2VsZi5jb250b3VyX3Nh'
            'bXBsZXMgPSBfTUVUUklDX0NGRy5DT05UT1VSX1NBTVBMRVMNCiAgICAgICAgc2VsZi5zaXplX2Jp'
            'bnMgPSBsaXN0KF9NRVRSSUNfQ0ZHLlNJWkVfQklOUykNCg0KICAgIGRlZiBjYWxjdWxhdGVfZjEo'
            'DQogICAgICAgIHNlbGYsDQogICAgICAgIHByZWRpY3Rpb25zX2dlb2pzb246IERpY3Rbc3RyLCBB'
            'bnldLA0KICAgICAgICBndF9nZW9qc29uOiBEaWN0W3N0ciwgQW55XSwNCiAgICApIC0+IERpY3Rb'
            'c3RyLCBBbnldOg0KICAgICAgICAiIiINCiAgICAgICAg0KDQsNGB0YHRh9C40YLRi9Cy0LDQtdGC'
            'IEYxLXNjb3JlINC/0L4g0YHQv9C10YbQuNGE0LjQutCw0YbQuNC4INCa0J7Qly0zLg0KDQogICAg'
            'ICAgIEFyZ3M6DQogICAgICAgICAgICBwcmVkaWN0aW9uc19nZW9qc29uOiBHZW9KU09OIEZlYXR1'
            'cmVDb2xsZWN0aW9uINC/0YDQtdC00YHQutCw0LfQsNC90LjQuQ0KICAgICAgICAgICAgZ3RfZ2Vv'
            'anNvbjogR2VvSlNPTiBGZWF0dXJlQ29sbGVjdGlvbiBncm91bmQgdHJ1dGgNCg0KICAgICAgICBS'
            'ZXR1cm5zOg0KICAgICAgICAgICAg0KHQu9C+0LLQsNGA0Ywg0YEg0YDQtdC30YPQu9GM0YLQsNGC'
            '0LDQvNC4OiBmaW5hbF9zY29yZSwgcGVyX2NsYXNzLCBhdXhpbGlhcnkNCiAgICAgICAgIiIiDQog'
            'ICAgICAgICMg0KHQvtCx0LjRgNCw0LXQvCDQv9C+0LvQuNCz0L7QvdGLINC/0L4g0LrQu9Cw0YHR'
            'gdCw0LwgKHBvb2xpbmcg0YfQtdGA0LXQtyDQstGB0LUg0LjRgdGC0L7Rh9C90LjQutC4KS4NCiAg'
            'ICAgICAgIyDQn9Cw0YDQsNC70LvQtdC70YzQvdC+IOKAlCDQutCw0YDRgtGLIGlkKNCz0LXQvtC8'
            '0LXRgtGA0LjRjykgLT4gbWFya3VwX3R5cGUg0LTQu9GPINGB0L/RgNCw0LLQvtGH0L3Ri9GFDQog'
            'ICAgICAgICMg0L/QvtC60LDQt9Cw0YLQtdC70LXQuSDQv9C+INGC0LjQv9Cw0Lwg0LTQsNC90L3R'
            'i9GFIChpZCDRgdGC0LDQsdC40LvRjNC90Ysg0LIg0L/RgNC10LTQtdC70LDRhSDQstGL0LfQvtCy'
            '0LApLg0KICAgICAgICBwcmVkX21hcmt1cDogRGljdFtpbnQsIE9wdGlvbmFsW3N0cl1dID0ge30N'
            'CiAgICAgICAgZ3RfbWFya3VwOiBEaWN0W2ludCwgT3B0aW9uYWxbc3RyXV0gPSB7fQ0KICAgICAg'
            'ICBwcmVkX2J5X2NsYXNzID0gc2VsZi5fY29sbGVjdF9ieV9jbGFzcyhwcmVkaWN0aW9uc19nZW9q'
            'c29uLCBtYXJrdXBfb3V0PXByZWRfbWFya3VwKQ0KICAgICAgICAjIEdULdGC0L7Rh9C60Lgg0YHQ'
            'vtGF0YDQsNC90Y/QtdC8INC60LDQuiBQb2ludCDigJQg0YHRhtC10L3QsNGA0LjQuSDQkSAo0YDQ'
            'sNGB0YHRgtC+0Y/QvdC40LUg0LTQviDRgtC+0YfQutC4KQ0KICAgICAgICBndF9ieV9jbGFzcyA9'
            'IHNlbGYuX2NvbGxlY3RfYnlfY2xhc3MoDQogICAgICAgICAgICBndF9nZW9qc29uLCBwcmVzZXJ2'
            'ZV9wb2ludHM9VHJ1ZSwgbWFya3VwX291dD1ndF9tYXJrdXApDQoNCiAgICAgICAgY2xhc3NfcmVz'
            'dWx0cyA9IHt9DQogICAgICAgIGFsbF90cF9wYWlycyA9IFtdICAjINCU0LvRjyBhdXhpbGlhcnkg'
            '0LzQtdGC0YDQuNC6OiAocHJlZF9jZW50cm9pZCwgZ3RfY2VudHJvaWQpDQogICAgICAgIHRvdGFs'
            'X3RwID0gMA0KICAgICAgICB0b3RhbF9mcCA9IDANCiAgICAgICAgdG90YWxfc29mdF9yZWNhbGxf'
            'aGl0cyA9IDANCiAgICAgICAgdG90YWxfZ3RfY291bnQgPSAwDQoNCiAgICAgICAgIyDQodC/0YDQ'
            'sNCy0L7Rh9C90YvQtSDQv9C+0LrQsNC30LDRgtC10LvQuCDQotCgINCa0J7Qly0zICjQn9GA0LjQ'
            'u9C+0LbQtdC90LjQtSA0OyDQvdCwIGZpbmFsX3Njb3JlINC90LUg0LLQu9C40Y/RjtGCKToNCiAg'
            'ICAgICAgIyDQutC+0L3RgtGD0YAg4oCUINC/0L4gVFAt0L/QsNGA0LDQvCDCq9C/0L7Qu9C40LPQ'
            'vtC9LdC/0L7Qu9C40LPQvtC9wrs7INGC0LjQv9GLINC00LDQvdC90YvRhSDQuCDRgNCw0LfQvNC1'
            '0YDQvdGL0LUNCiAgICAgICAgIyDQutCw0YLQtdCz0L7RgNC40Lgg4oCUINC/0L4gR1Qt0L7QsdGK'
            '0LXQutGC0LDQvCDQuCDQv9GA0LXQtNGB0LrQsNC30LDQvdC40Y/QvC4NCiAgICAgICAgY29udG91'
            'cl9wYWlyc19ieV9jbGFzczogRGljdFtzdHIsIExpc3RbVHVwbGVbQW55LCBBbnldXV0gPSB7fQ0K'
            'ICAgICAgICBhbGxfY29udG91cl9wYWlyczogTGlzdFtUdXBsZVtBbnksIEFueV1dID0gW10NCiAg'
            'ICAgICAgZ3RfcG9seWdvbnM6IExpc3RbQW55XSA9IFtdDQogICAgICAgIGR0eXBlX2d0X2NvdW50'
            'OiBEaWN0W3N0ciwgaW50XSA9IHt9DQogICAgICAgIGR0eXBlX3ByZWRfY291bnQ6IERpY3Rbc3Ry'
            'LCBpbnRdID0ge30NCiAgICAgICAgZHR5cGVfdHBfY291bnQ6IERpY3Rbc3RyLCBpbnRdID0ge30N'
            'CiAgICAgICAgZHR5cGVfbWF0Y2hlZF9wcmVkOiBEaWN0W3N0ciwgaW50XSA9IHt9DQoNCiAgICAg'
            'ICAgZm9yIGNscyBpbiBzZWxmLnRhcmdldF9jbGFzc2VzOg0KICAgICAgICAgICAgcHJlZHMgPSBw'
            'cmVkX2J5X2NsYXNzLmdldChjbHMsIFtdKQ0KICAgICAgICAgICAgZ3RzID0gZ3RfYnlfY2xhc3Mu'
            'Z2V0KGNscywgW10pDQoNCiAgICAgICAgICAgIHRwLCBmcCwgZm4sIG1hdGNoZXMgPSBzZWxmLl9z'
            'b2Z0X21hdGNoKHByZWRzLCBndHMpDQogICAgICAgICAgICAjINCb0LjRiNC90LXQtSDRgdGA0LDQ'
            'sdCw0YLRi9Cy0LDQvdC40LUg0YPRh9C40YLRi9Cy0LDQtdGC0YHRjyDRgSDQutC+0Y3RhNGE0LjR'
            'htC40LXQvdGC0L7QvCAwLjUgKNCf0YDQuNC70L7QttC10L3QuNC1IDQpDQogICAgICAgICAgICBw'
            'cmVjaXNpb24gPSB0cCAvICh0cCArIDAuNSAqIGZwICsgc2VsZi5lcHNpbG9uKQ0KICAgICAgICAg'
            'ICAgcmVjYWxsID0gdHAgLyAodHAgKyBmbiArIHNlbGYuZXBzaWxvbikNCiAgICAgICAgICAgIGYx'
            'ID0gMiAqIHByZWNpc2lvbiAqIHJlY2FsbCAvIChwcmVjaXNpb24gKyByZWNhbGwgKyBzZWxmLmVw'
            'c2lsb24pDQoNCiAgICAgICAgICAgICMgU29mdCByZWNhbGwgKElvVSA+IDApDQogICAgICAgICAg'
            'ICBzb2Z0X2hpdHMgPSBzZWxmLl9jb3VudF9zb2Z0X3JlY2FsbChwcmVkcywgZ3RzKQ0KDQogICAg'
            'ICAgICAgICBjbGFzc19yZXN1bHRzW2Nsc10gPSB7DQogICAgICAgICAgICAgICAgImYxIjogZjEs'
            'DQogICAgICAgICAgICAgICAgInByZWNpc2lvbiI6IHByZWNpc2lvbiwNCiAgICAgICAgICAgICAg'
            'ICAicmVjYWxsIjogcmVjYWxsLA0KICAgICAgICAgICAgICAgICJ0cCI6IHRwLA0KICAgICAgICAg'
            'ICAgICAgICJmcCI6IGZwLA0KICAgICAgICAgICAgICAgICJmbiI6IGZuLA0KICAgICAgICAgICAg'
            'ICAgICJudW1fcHJlZGljdGlvbnMiOiBsZW4ocHJlZHMpLA0KICAgICAgICAgICAgICAgICJudW1f'
            'Z3JvdW5kX3RydXRoIjogbGVuKGd0cyksDQogICAgICAgICAgICAgICAgInNvZnRfcmVjYWxsX2hp'
            'dHMiOiBzb2Z0X2hpdHMsDQogICAgICAgICAgICB9DQoNCiAgICAgICAgICAgIHRvdGFsX3RwICs9'
            'IHRwDQogICAgICAgICAgICB0b3RhbF9mcCArPSBmcA0KICAgICAgICAgICAgdG90YWxfc29mdF9y'
            'ZWNhbGxfaGl0cyArPSBzb2Z0X2hpdHMNCiAgICAgICAgICAgIHRvdGFsX2d0X2NvdW50ICs9IGxl'
            'bihndHMpDQogICAgICAgICAgICBhbGxfdHBfcGFpcnMuZXh0ZW5kKG1hdGNoZXMpDQoNCiAgICAg'
            'ICAgICAgICMgLS0tINCh0L/RgNCw0LLQvtGH0L3Ri9C1INC/0L7QutCw0LfQsNGC0LXQu9C4ICjQ'
            'n9GA0LjQu9C+0LbQtdC90LjQtSA0KTog0L3QtSDQstC70LjRj9GO0YIg0L3QsCBmaW5hbF9zY29y'
            'ZSAtLS0NCiAgICAgICAgICAgICMg0JrQvtC90YLRg9GAOiDRgtC+0LvRjNC60L4g0L/QsNGA0Ysg'
            'wqvQv9C+0LvQuNCz0L7QvS3Qv9C+0LvQuNCz0L7QvcK7INC40LcgVFAgKNGDIEdULdGC0L7Rh9C1'
            '0Log0LrQvtC90YLRg9GA0LAg0L3QtdGCKQ0KICAgICAgICAgICAgcG9seV9wYWlycyA9IFsNCiAg'
            'ICAgICAgICAgICAgICAocHJlZF9nZW9tLCBndF9nZW9tKSBmb3IgcHJlZF9nZW9tLCBndF9nZW9t'
            'IGluIG1hdGNoZXMNCiAgICAgICAgICAgICAgICBpZiBpc2luc3RhbmNlKHByZWRfZ2VvbSwgUG9s'
            'eWdvbikgYW5kIGlzaW5zdGFuY2UoZ3RfZ2VvbSwgUG9seWdvbikNCiAgICAgICAgICAgIF0NCiAg'
            'ICAgICAgICAgIGlmIHBvbHlfcGFpcnM6DQogICAgICAgICAgICAgICAgY29udG91cl9wYWlyc19i'
            'eV9jbGFzc1tjbHNdID0gcG9seV9wYWlycw0KICAgICAgICAgICAgICAgIGFsbF9jb250b3VyX3Bh'
            'aXJzLmV4dGVuZChwb2x5X3BhaXJzKQ0KDQogICAgICAgICAgICAjINCi0LjQv9GLINC00LDQvdC9'
            '0YvRhTogVFAvRk4g4oCUINC/0L4gbWFya3VwX3R5cGUgR1Qt0L7QsdGK0LXQutGC0LAsDQogICAg'
            'ICAgICAgICAjIEZQIOKAlCDQv9C+IG1hcmt1cF90eXBlINC/0YDQtdC00YHQutCw0LfQsNC90LjR'
            'jyAowqvQv9GA0L7Rh9C10LXCuyDQv9GA0Lgg0L7RgtGB0YPRgtGB0YLQstC40LgpDQogICAgICAg'
            'ICAgICBmb3IgcHJlZF9nZW9tLCBndF9nZW9tIGluIG1hdGNoZXM6DQogICAgICAgICAgICAgICAg'
            'X2J1bXAoZHR5cGVfdHBfY291bnQsIGRhdGFfdHlwZV9ncm91cChndF9tYXJrdXAuZ2V0KGlkKGd0'
            'X2dlb20pKSkpDQogICAgICAgICAgICAgICAgX2J1bXAoZHR5cGVfbWF0Y2hlZF9wcmVkLCBkYXRh'
            'X3R5cGVfZ3JvdXAocHJlZF9tYXJrdXAuZ2V0KGlkKHByZWRfZ2VvbSkpKSkNCiAgICAgICAgICAg'
            'IGZvciBndF9nZW9tLCBfIGluIGd0czoNCiAgICAgICAgICAgICAgICBfYnVtcChkdHlwZV9ndF9j'
            'b3VudCwgZGF0YV90eXBlX2dyb3VwKGd0X21hcmt1cC5nZXQoaWQoZ3RfZ2VvbSkpKSkNCiAgICAg'
            'ICAgICAgICAgICBpZiBpc2luc3RhbmNlKGd0X2dlb20sIFBvbHlnb24pOg0KICAgICAgICAgICAg'
            'ICAgICAgICBndF9wb2x5Z29ucy5hcHBlbmQoZ3RfZ2VvbSkNCiAgICAgICAgICAgIGZvciBwcmVk'
            'X2dlb20sIF8gaW4gcHJlZHM6DQogICAgICAgICAgICAgICAgX2J1bXAoZHR5cGVfcHJlZF9jb3Vu'
            'dCwgZGF0YV90eXBlX2dyb3VwKHByZWRfbWFya3VwLmdldChpZChwcmVkX2dlb20pKSkpDQoNCiAg'
            'ICAgICAgIyBXZWlnaHRlZCBGMSAo0LLQtdGB0LAg0L7RgiDQsNGA0YXQtdC+0LvQvtCz0L7Qsiku'
            'DQogICAgICAgICMg0J3QvtGA0LzQuNGA0L7QstC60LAg0L3QsCDQv9GA0LjRgdGD0YLRgdGC0LLR'
            'g9GO0YnQuNC1INC60LvQsNGB0YHRiyAo0YDQtdGI0LXQvdC40LUgMDguMTAuMjAyNik6INC10YHQ'
            'u9C4INCyIEdUDQogICAgICAgICMg0L3QsNCx0L7RgNCwINC90LXRgiDQvdC4INC+0LTQvdC+0LPQ'
            'viDQvtCx0YrQtdC60YLQsCDQutC70LDRgdGB0LAsINC10LPQviDQstC10YEg0LjRgdC60LvRjtGH'
            '0LDQtdGC0YHRjywg0LAg0LLQtdGB0LANCiAgICAgICAgIyDQvtGB0YLQsNC70YzQvdGL0YUg0LzQ'
            'sNGB0YjRgtCw0LHQuNGA0YPRjtGC0YHRjyDQtdC00LjQvdGL0Lwg0LrQvtGN0YTRhNC40YbQuNC1'
            '0L3RgtC+0LwgMTAwIC8g0YHRg9C80LzQsF/Qv9GA0LjRgdGD0YLRgdGC0LLRg9GO0YnQuNGFDQog'
            'ICAgICAgICMgKNC/0YDQvtC/0L7RgNGG0LjQuCDQv9GA0LjQvtGA0LjRgtC10YLQvtCyINC80LXQ'
            'ttC00YMg0LrQu9Cw0YHRgdCw0LzQuCDRgdC+0YXRgNCw0L3Rj9GO0YLRgdGPKS4g0JrQu9Cw0YHR'
            'gSwg0L/RgNC40YHRg9GC0YHRgtCy0YPRjtGJ0LjQuQ0KICAgICAgICAjINCyIEdULCDQvdC+INC/'
            '0L7Qu9C90L7RgdGC0YzRjiDQv9GA0L7Qv9GD0YnQtdC90L3Ri9C5INGD0YfQsNGB0YLQvdC40LrQ'
            'vtC8IChGMT0wKSwg0YHQvtGF0YDQsNC90Y/QtdGCINC/0L7Qu9C90YvQuSDQstC10YEg4oCUINGI'
            '0YLRgNCw0YQuDQogICAgICAgIHByZXNlbnRfY2xhc3NlcyA9IFsNCiAgICAgICAgICAgIGNscyBm'
            'b3IgY2xzIGluIHNlbGYudGFyZ2V0X2NsYXNzZXMNCiAgICAgICAgICAgIGlmIGNsYXNzX3Jlc3Vs'
            'dHNbY2xzXVsibnVtX2dyb3VuZF90cnV0aCJdID4gMA0KICAgICAgICBdDQogICAgICAgIHRvdGFs'
            'X3dlaWdodCA9IHN1bShDTEFTU19XRUlHSFRTLmdldChjbHMsIDAuMCkgZm9yIGNscyBpbiBwcmVz'
            'ZW50X2NsYXNzZXMpDQogICAgICAgIGZpbmFsX3Njb3JlID0gKA0KICAgICAgICAgICAgc3VtKGNs'
            'YXNzX3Jlc3VsdHNbY2xzXVsiZjEiXSAqIENMQVNTX1dFSUdIVFMuZ2V0KGNscywgMC4wKSBmb3Ig'
            'Y2xzIGluIHByZXNlbnRfY2xhc3NlcykNCiAgICAgICAgICAgIC8gdG90YWxfd2VpZ2h0DQogICAg'
            'ICAgICkgaWYgdG90YWxfd2VpZ2h0ID4gMCBlbHNlIDAuMA0KDQogICAgICAgICMgQXV4aWxpYXJ5'
            'IG1ldHJpY3MNCiAgICAgICAgc29mdF9yZWNhbGwgPSB0b3RhbF9zb2Z0X3JlY2FsbF9oaXRzIC8g'
            'KHRvdGFsX2d0X2NvdW50ICsgMWUtNikgaWYgdG90YWxfZ3RfY291bnQgPiAwIGVsc2UgMC4wDQog'
            'ICAgICAgIGNlbnRlcl9kZXZpYXRpb24gPSBzZWxmLl9jYWxjdWxhdGVfY2VudGVyX2RldmlhdGlv'
            'bihhbGxfdHBfcGFpcnMpDQogICAgICAgIGZwX3JhdGlvID0gdG90YWxfZnAgLyAodG90YWxfdHAg'
            'KyAxZS02KSBpZiB0b3RhbF90cCA+IDAgZWxzZSAwLjANCg0KICAgICAgICAjINCh0L/RgNCw0LLQ'
            'vtGH0L3Ri9C1INC/0L7QutCw0LfQsNGC0LXQu9C4INCi0KAg0JrQntCXLTMgKNCf0YDQuNC70L7Q'
            'ttC10L3QuNC1IDQpLiDQodGH0LjRgtCw0Y7RgtGB0Y8g0L/QvtCy0LXRgNGFDQogICAgICAgICMg'
            '0LzQsNGC0YfQuNC90LPQsCDQuCDQndCVINGD0YfQsNGB0YLQstGD0Y7RgiDQsiBmaW5hbF9zY29y'
            'ZSAo0LfQsNC60YDQtdC/0LvQtdC90L4g0YLQtdGB0YLQvtC8KS4NCiAgICAgICAgY29udG91ciA9'
            'IHNlbGYuX2FnZ3JlZ2F0ZV9jb250b3VyKGFsbF9jb250b3VyX3BhaXJzKQ0KICAgICAgICBjb250'
            'b3VyX2J5X2NsYXNzID0gew0KICAgICAgICAgICAgY2xzOiBzZWxmLl9hZ2dyZWdhdGVfY29udG91'
            'cihwYWlycykNCiAgICAgICAgICAgIGZvciBjbHMsIHBhaXJzIGluIGNvbnRvdXJfcGFpcnNfYnlf'
            'Y2xhc3MuaXRlbXMoKQ0KICAgICAgICB9DQogICAgICAgIGJ5X2RhdGFfdHlwZSA9IHNlbGYuX2J1'
            'aWxkX2J5X2RhdGFfdHlwZSgNCiAgICAgICAgICAgIGR0eXBlX2d0X2NvdW50LCBkdHlwZV9wcmVk'
            'X2NvdW50LCBkdHlwZV90cF9jb3VudCwgZHR5cGVfbWF0Y2hlZF9wcmVkKQ0KICAgICAgICBtYXRj'
            'aGVkX2d0X2lkcyA9IHtpZChndF9nZW9tKSBmb3IgXywgZ3RfZ2VvbSBpbiBhbGxfdHBfcGFpcnN9'
            'DQogICAgICAgIHNpemVfY2F0ZWdvcmllcyA9IHNlbGYuX2J1aWxkX3NpemVfY2F0ZWdvcmllcyhn'
            'dF9wb2x5Z29ucywgbWF0Y2hlZF9ndF9pZHMpDQoNCiAgICAgICAgcmV0dXJuIHsNCiAgICAgICAg'
            'ICAgICJmaW5hbF9zY29yZSI6IGZpbmFsX3Njb3JlLA0KICAgICAgICAgICAgInBlcl9jbGFzcyI6'
            'IGNsYXNzX3Jlc3VsdHMsDQogICAgICAgICAgICAiYXV4aWxpYXJ5Ijogew0KICAgICAgICAgICAg'
            'ICAgICJzb2Z0X3JlY2FsbCI6IHNvZnRfcmVjYWxsLA0KICAgICAgICAgICAgICAgICJjZW50ZXJf'
            'ZGV2aWF0aW9uIjogY2VudGVyX2RldmlhdGlvbiwNCiAgICAgICAgICAgICAgICAiZnBfcmF0aW8i'
            'OiBmcF9yYXRpbywNCiAgICAgICAgICAgICAgICAjINCh0L/RgNCw0LLQvtGH0L3Ri9C1INC/0L7Q'
            'utCw0LfQsNGC0LXQu9C4ICjQn9GA0LjQu9C+0LbQtdC90LjQtSA0KSDigJQg0L3QsCDQsdCw0LvQ'
            'uyDQvdC1INCy0LvQuNGP0Y7Rgg0KICAgICAgICAgICAgICAgICJjb250b3VyIjogY29udG91ciwN'
            'CiAgICAgICAgICAgICAgICAiY29udG91cl9ieV9jbGFzcyI6IGNvbnRvdXJfYnlfY2xhc3MsDQog'
            'ICAgICAgICAgICAgICAgImJ5X2RhdGFfdHlwZSI6IGJ5X2RhdGFfdHlwZSwNCiAgICAgICAgICAg'
            'ICAgICAic2l6ZV9jYXRlZ29yaWVzIjogc2l6ZV9jYXRlZ29yaWVzLA0KICAgICAgICAgICAgfSwN'
            'CiAgICAgICAgICAgICJwYXJhbWV0ZXJzIjogew0KICAgICAgICAgICAgICAgICJjb3ZlcmFnZV90'
            'aHJlc2hvbGQiOiBzZWxmLmNvdmVyYWdlX3RocmVzaG9sZCwNCiAgICAgICAgICAgICAgICAiaW5w'
            'dXRfY3JzX21vZGUiOiBzZWxmLmlucHV0X2Nyc19tb2RlLA0KICAgICAgICAgICAgICAgICJtZXRy'
            'aWNfY3JzIjogc2VsZi5NRVRSSUNfQ1JTLA0KICAgICAgICAgICAgICAgICJ0YXJnZXRfY2xhc3Nl'
            'cyI6IHNlbGYudGFyZ2V0X2NsYXNzZXMsDQogICAgICAgICAgICAgICAgImNsYXNzX3dlaWdodHMi'
            'OiBDTEFTU19XRUlHSFRTLA0KICAgICAgICAgICAgICAgICJtZXRyaWNfdmVyc2lvbiI6IF9NRVRS'
            'SUNfQ0ZHLk1FVFJJQ19WRVJTSU9OLA0KICAgICAgICAgICAgICAgICJjb25maWdfaGFzaCI6IF9N'
            'RVRSSUNfQ0ZHLm1ldHJpY19zdW1tYXJ5KClbImNvbmZpZ19oYXNoIl0sDQogICAgICAgICAgICAg'
            'ICAgImVwc2lsb24iOiBzZWxmLmVwc2lsb24sDQogICAgICAgICAgICAgICAgImNpcmNsZV9wb2lu'
            'dHMiOiBzZWxmLmNpcmNsZV9wb2ludHMsDQogICAgICAgICAgICAgICAgInJfdG9sZXJhbmNlIjog'
            'c2VsZi5yX3RvbGVyYW5jZSwNCiAgICAgICAgICAgICAgICAibW9kYWxpdHlfbW9kZSI6IF9NRVRS'
            'SUNfQ0ZHLk1PREFMSVRZX01PREUsDQogICAgICAgICAgICAgICAgImNvbnRvdXJfYmFuZF9yYXRp'
            'byI6IHNlbGYuY29udG91cl9iYW5kX3JhdGlvLA0KICAgICAgICAgICAgICAgICJjb250b3VyX3Nh'
            'bXBsZXMiOiBzZWxmLmNvbnRvdXJfc2FtcGxlcywNCiAgICAgICAgICAgICAgICAic2l6ZV9iaW5z'
            'IjogbGlzdChzZWxmLnNpemVfYmlucyksDQogICAgICAgICAgICB9LA0KICAgICAgICB9DQoNCiAg'
            'ICBkZWYgX2NvbGxlY3RfYnlfY2xhc3MoDQogICAgICAgIHNlbGYsDQogICAgICAgIGdlb2pzb25f'
            'ZGF0YTogRGljdFtzdHIsIEFueV0sDQogICAgICAgIHByZXNlcnZlX3BvaW50czogYm9vbCA9IEZh'
            'bHNlLA0KICAgICAgICBtYXJrdXBfb3V0OiBPcHRpb25hbFtEaWN0W2ludCwgT3B0aW9uYWxbc3Ry'
            'XV1dID0gTm9uZSwNCiAgICApIC0+IERpY3Rbc3RyLCBMaXN0W1R1cGxlW0FueSwgZmxvYXRdXV06'
            'DQogICAgICAgICIiItCh0L7QsdC40YDQsNC10YIg0LPQtdC+0LzQtdGC0YDQuNC4INGBIGNvbmZp'
            'ZGVuY2Ug0L/QviDRhtC10LvQtdCy0YvQvCDQutC70LDRgdGB0LDQvC4NCg0KICAgICAgICAtINC6'
            '0L7QvtGA0LTQuNC90LDRgtGLINC/0YDQuNCy0L7QtNGP0YLRgdGPINC6IFVUTSDQv9GA0Lgg0L3Q'
            'tdC+0LHRhdC+0LTQuNC80L7RgdGC0LggKElOUFVUX0NSU19NT0RFKTsNCiAgICAgICAgLSBQb2lu'
            'dC3Qs9C10L7QvNC10YLRgNC40Lgg0L/RgNC10LTRgdC60LDQt9Cw0L3QuNC5INC60L7QvdCy0LXR'
            'gNGC0LjRgNGD0Y7RgtGB0Y8g0LIg0LrRgNGD0LPQvtCy0YvQtSDQv9C+0LvQuNCz0L7QvdGLDQog'
            'ICAgICAgICAgKFIgPSByX3RvbGVyYW5jZSwg0LzQtdGC0YDRiyk7DQogICAgICAgIC0g0L/RgNC4'
            'IHByZXNlcnZlX3BvaW50cz1UcnVlIChHVCkgUG9pbnQt0LPQtdC+0LzQtdGC0YDQuNC4INGB0L7R'
            'hdGA0LDQvdGP0Y7RgtGB0Y8g0LrQsNC6INC10YHRgtGMIOKAlA0KICAgICAgICAgINC00LvRjyDQ'
            'vdC40YUg0YDQsNCx0L7RgtCw0LXRgiDRgdGG0LXQvdCw0YDQuNC5INCRINCi0JcgKNGA0LDRgdGB'
            '0YLQvtGP0L3QuNC1INGG0LXQvdGC0YDQvtC40LQt0YLQvtGH0LrQsCk7DQogICAgICAgIC0gbWFy'
            'a3VwX291dCAo0LXRgdC70Lgg0L/QtdGA0LXQtNCw0L0pINC30LDQv9C+0LvQvdGP0LXRgtGB0Y8g'
            '0LrQsNGA0YLQvtC5IGlkKNCz0LXQvtC80LXRgtGA0LjRjykgLT4NCiAgICAgICAgICBtYXJrdXBf'
            'dHlwZTog0YLQuNC/0Ysg0L/RgNC+0YLRj9Cz0LjQstCw0Y7RgtGB0Y8g0LLQsdC+0Log0LTQu9GP'
            'INGB0L/RgNCw0LLQvtGH0L3Ri9GFINC/0L7QutCw0LfQsNGC0LXQu9C10LkNCiAgICAgICAgICDQ'
            'v9C+INGC0LjQv9Cw0Lwg0LTQsNC90L3Ri9GFLCDRgdC40LPQvdCw0YLRg9GA0LAgX3NvZnRfbWF0'
            'Y2gg0L3QtSDQvNC10L3Rj9C10YLRgdGPLg0KICAgICAgICAiIiINCiAgICAgICAgcmVzdWx0OiBE'
            'aWN0W3N0ciwgTGlzdFtUdXBsZVtBbnksIGZsb2F0XV1dID0ge30NCiAgICAgICAgZm9yIGZlYXR1'
            'cmUgaW4gZ2VvanNvbl9kYXRhLmdldCgiZmVhdHVyZXMiLCBbXSk6DQogICAgICAgICAgICBwcm9w'
            'cyA9IGZlYXR1cmUuZ2V0KCJwcm9wZXJ0aWVzIiwge30pIG9yIHt9DQogICAgICAgICAgICByYXdf'
            'Y2xhc3MgPSBwcm9wcy5nZXQoImNsYXNzX25hbWUiLCAidW5rbm93biIpDQogICAgICAgICAgICBj'
            'bGFzc19uYW1lID0gbm9ybWFsaXplX2NsYXNzX25hbWUocmF3X2NsYXNzKQ0KICAgICAgICAgICAg'
            'aWYgY2xhc3NfbmFtZSBpcyBOb25lIG9yIGNsYXNzX25hbWUgbm90IGluIHNlbGYudGFyZ2V0X2Ns'
            'YXNzZXM6DQogICAgICAgICAgICAgICAgY29udGludWUNCiAgICAgICAgICAgIGdlb20gPSBfZXh0'
            'cmFjdF9nZW9tZXRyeShmZWF0dXJlKQ0KICAgICAgICAgICAgaWYgZ2VvbSBpcyBOb25lOg0KICAg'
            'ICAgICAgICAgICAgIGNvbnRpbnVlDQogICAgICAgICAgICAjINCc0LXRgtGA0LjRh9C10YHQutCw'
            '0Y8g0L/RgNC+0LXQutGG0LjRjzogMzg1NyAtPiBVVE0gKNC00L4g0LrQvtC90LLQtdGA0YLQsNGG'
            '0LjQuCBQb2ludCAtPiDQutGA0YPQsyEpDQogICAgICAgICAgICBnZW9tID0gX3RvX3V0bShnZW9t'
            'LCBzZWxmLmlucHV0X2Nyc19tb2RlKQ0KICAgICAgICAgICAgIyDQmtC+0L3QstC10YDRgtC40YDR'
            'g9C10LwgUG9pbnQg0L/RgNC10LTRgdC60LDQt9Cw0L3QuNC5INCyINC60YDRg9Cz0L7QstC+0Lkg'
            'UG9seWdvbg0KICAgICAgICAgICAgIyAo0YDQsNC00LjRg9GBINC4INGH0LjRgdC70L4g0YLQvtGH'
            '0LXQuiDigJQg0LjQtyDQutC+0L3RhNC40LPQsCk7IEdULdGC0L7Rh9C60Lgg0YHQvtGF0YDQsNC9'
            '0Y/QtdC8DQogICAgICAgICAgICBpZiBpc2luc3RhbmNlKGdlb20sIFBvaW50KSBhbmQgbm90IHBy'
            'ZXNlcnZlX3BvaW50czoNCiAgICAgICAgICAgICAgICBnZW9tID0gX3BvaW50X3RvX2NpcmNsZShn'
            'ZW9tLCBzZWxmLnJfdG9sZXJhbmNlLCBzZWxmLmNpcmNsZV9wb2ludHMpDQogICAgICAgICAgICBj'
            'b25maWRlbmNlID0gX2dldF9jb25maWRlbmNlKGZlYXR1cmUpDQogICAgICAgICAgICBpZiBtYXJr'
            'dXBfb3V0IGlzIG5vdCBOb25lOg0KICAgICAgICAgICAgICAgICMgbWFya3VwX3R5cGUg0L3Rg9C2'
            '0LXQvSDRgdC/0YDQsNCy0L7Rh9C90YvQvCDQv9C+0LrQsNC30LDRgtC10LvRj9C8INC/0L4g0YLQ'
            'uNC/0LDQvCDQtNCw0L3QvdGL0YUNCiAgICAgICAgICAgICAgICAjICjQn9GA0LjQu9C+0LbQtdC9'
            '0LjQtSA0KTsg0YMg0L/RgNC10LTRgdC60LDQt9Cw0L3QuNC5INC60LvRjtGH0LAg0LzQvtC20LXR'
            'giDQvdC1INCx0YvRgtGMIOKAlCBOb25lLg0KICAgICAgICAgICAgICAgIG1hcmt1cF9vdXRbaWQo'
            'Z2VvbSldID0gcHJvcHMuZ2V0KCJtYXJrdXBfdHlwZSIpDQogICAgICAgICAgICBpZiBjbGFzc19u'
            'YW1lIG5vdCBpbiByZXN1bHQ6DQogICAgICAgICAgICAgICAgcmVzdWx0W2NsYXNzX25hbWVdID0g'
            'W10NCiAgICAgICAgICAgIHJlc3VsdFtjbGFzc19uYW1lXS5hcHBlbmQoKGdlb20sIGNvbmZpZGVu'
            'Y2UpKQ0KICAgICAgICByZXR1cm4gcmVzdWx0DQoNCiAgICBkZWYgX3NvZnRfbWF0Y2goDQogICAg'
            'ICAgIHNlbGYsDQogICAgICAgIHByZWRzOiBMaXN0W1R1cGxlW0FueSwgZmxvYXRdXSwNCiAgICAg'
            'ICAgZ3RzOiBMaXN0W1R1cGxlW0FueSwgZmxvYXRdXSwNCiAgICApIC0+IFR1cGxlW2ludCwgaW50'
            'LCBpbnQsIExpc3RbVHVwbGVbQW55LCBBbnldXV06DQogICAgICAgICIiIg0KICAgICAgICDQltCw'
            '0LTQvdGL0Lkg0LzRj9Cz0LrQuNC5INC80LDRgtGH0LjQvdCzINC/0L4g0KLQoCDQmtCe0JctMyAo'
            '0J/RgNC40LvQvtC20LXQvdC40LUgNCkuDQoNCiAgICAgICAg0KHRhtC10L3QsNGA0LjQuSDQkCAo'
            'R1Qg4oCUINC/0L7Qu9C40LPQvtC9KTogaGl0LCDQtdGB0LvQuA0KICAgICAgICBjb3YgPSBhcmVh'
            'KFDiiKlHKSAvIG1pbihhcmVhKFApLCBhcmVhKEcpKSA+PSBjb3ZlcmFnZV90aHJlc2hvbGQNCiAg'
            'ICAgICAg0Jgg0LLQvdGD0YLRgNC10L3QvdGP0Y8gKNC/0YDQtdC00YHRgtCw0LLQuNGC0LXQu9GM'
            '0L3QsNGPKSDRgtC+0YfQutCwINC/0YDQtdC00YHQutCw0LfQsNC90LjRjyDQstC90YPRgtGA0Lgg'
            'R1QuDQogICAgICAgINCh0YbQtdC90LDRgNC40Lkg0JEgKEdUIOKAlCDRgtC+0YfQutCwKTogaGl0'
            'LCDQtdGB0LvQuCDRgNCw0YHRgdGC0L7Rj9C90LjQtSDQvtGCINGG0LXQvdGC0YDQvtC40LTQsCDQ'
            'v9GA0LXQtNGB0LrQsNC30LDQvdC40Y8NCiAgICAgICAg0LTQviBHVC3RgtC+0YfQutC4IDw9IHJf'
            'dG9sZXJhbmNlLg0KICAgICAgICDQodGA0LXQtNC4INC/0YDQvtGI0LXQtNGI0LjRhSDQv9GA0L7Q'
            'stC10YDQutGDIEdUINCy0YvQsdC40YDQsNC10YLRgdGPINC70YPRh9GI0LjQuTog0LTQu9GPIEdU'
            'LdC/0L7Qu9C40LPQvtC90L7QsiDigJQg0L/Qvg0KICAgICAgICDQvNCw0LrRgdC40LzQsNC70YzQ'
            'vdC+0LzRgyDQv9C10YDQtdC60YDRi9GC0LjRjiwg0LTQu9GPIEdULdGC0L7Rh9C10Log4oCUINC/'
            '0L4g0LzQuNC90LjQvNCw0LvRjNC90L7QvNGDINGA0LDRgdGB0YLQvtGP0L3QuNGOLg0KDQogICAg'
            'ICAgIFJldHVybnM6DQogICAgICAgICAgICAoVFAsIEZQLCBGTiwg0YHQv9C40YHQvtC6INGB0L7Q'
            'stC/0LDQstGI0LjRhSDQv9Cw0YAgKHByZWRfZ2VvbSwgZ3RfZ2VvbSkpDQogICAgICAgICIiIg0K'
            'ICAgICAgICBpZiBub3QgcHJlZHMgYW5kIG5vdCBndHM6DQogICAgICAgICAgICByZXR1cm4gMCwg'
            'MCwgMCwgW10NCiAgICAgICAgaWYgbm90IHByZWRzOg0KICAgICAgICAgICAgcmV0dXJuIDAsIDAs'
            'IGxlbihndHMpLCBbXQ0KICAgICAgICBpZiBub3QgZ3RzOg0KICAgICAgICAgICAgcmV0dXJuIDAs'
            'IGxlbihwcmVkcyksIDAsIFtdDQoNCiAgICAgICAgIyDQodC+0YDRgtC40YDRg9C10Lwg0L/RgNC1'
            '0LTRgdC60LDQt9Cw0L3QuNGPINC/0L4gY29uZmlkZW5jZSAo0L/QviDRg9Cx0YvQstCw0L3QuNGO'
            'KQ0KICAgICAgICBzb3J0ZWRfcHJlZHMgPSBzb3J0ZWQocHJlZHMsIGtleT1sYW1iZGEgeDogeFsx'
            'XSwgcmV2ZXJzZT1UcnVlKQ0KDQogICAgICAgIHRwID0gMA0KICAgICAgICBtYXRjaGVkX2d0X2lu'
            'ZGljZXMgPSBzZXQoKQ0KICAgICAgICBtYXRjaGVzID0gW10NCg0KICAgICAgICBmb3IgcHJlZF9n'
            'ZW9tLCBfIGluIHNvcnRlZF9wcmVkczoNCiAgICAgICAgICAgIGlmIG5vdCBpc2luc3RhbmNlKHBy'
            'ZWRfZ2VvbSwgUG9seWdvbik6DQogICAgICAgICAgICAgICAgY29udGludWUNCg0KICAgICAgICAg'
            'ICAgYmVzdF9zY29yZSA9IGZsb2F0KCItaW5mIikNCiAgICAgICAgICAgIGJlc3RfZ3RfaWR4ID0g'
            'LTENCg0KICAgICAgICAgICAgZm9yIGosIChndF9nZW9tLCBfKSBpbiBlbnVtZXJhdGUoZ3RzKToN'
            'CiAgICAgICAgICAgICAgICBpZiBqIGluIG1hdGNoZWRfZ3RfaW5kaWNlczoNCiAgICAgICAgICAg'
            'ICAgICAgICAgY29udGludWUNCg0KICAgICAgICAgICAgICAgIGhpdCwgc2NvcmUgPSBzZWxmLl9t'
            'YXRjaF9xdWFsaXR5KHByZWRfZ2VvbSwgZ3RfZ2VvbSkNCg0KICAgICAgICAgICAgICAgIGlmIGhp'
            'dCBhbmQgc2NvcmUgPiBiZXN0X3Njb3JlICsgMWUtOTogICMgdGllLWJyZWFrOiDRiNGD0Lwg0L/Q'
            'u9Cw0LLQsNGO0YnQtdC5INGC0L7Rh9C60Lgg0L3QtSDRgNC10YjQsNC10YIg0YDQsNCy0L3Ri9C1'
            'INC/0L7QutGA0YvRgtC40Y8NCiAgICAgICAgICAgICAgICAgICAgYmVzdF9zY29yZSA9IHNjb3Jl'
            'DQogICAgICAgICAgICAgICAgICAgIGJlc3RfZ3RfaWR4ID0gag0KDQogICAgICAgICAgICBpZiBi'
            'ZXN0X2d0X2lkeCAhPSAtMToNCiAgICAgICAgICAgICAgICB0cCArPSAxDQogICAgICAgICAgICAg'
            'ICAgbWF0Y2hlZF9ndF9pbmRpY2VzLmFkZChiZXN0X2d0X2lkeCkNCiAgICAgICAgICAgICAgICBt'
            'YXRjaGVzLmFwcGVuZCgocHJlZF9nZW9tLCBndHNbYmVzdF9ndF9pZHhdWzBdKSkNCg0KICAgICAg'
            'ICBmcCA9IGxlbihwcmVkcykgLSB0cA0KICAgICAgICBmbiA9IGxlbihndHMpIC0gdHANCiAgICAg'
            'ICAgcmV0dXJuIHRwLCBmcCwgZm4sIG1hdGNoZXMNCg0KICAgIGRlZiBfbWF0Y2hfcXVhbGl0eSgN'
            'CiAgICAgICAgc2VsZiwNCiAgICAgICAgcHJlZF9nZW9tOiBQb2x5Z29uLA0KICAgICAgICBndF9n'
            'ZW9tOiBBbnksDQogICAgKSAtPiBUdXBsZVtib29sLCBmbG9hdF06DQogICAgICAgICIiItCf0YDQ'
            'vtCy0LXRgNC60LAg0L/QsNGA0Ys6IChoaXQsINC+0YbQtdC90LrQsCDQtNC70Y8g0LLRi9Cx0L7R'
            'gNCwINC70YPRh9GI0LXQs9C+IEdUKS4NCg0KICAgICAgICDQlNC70Y8gR1Qt0L/QvtC70LjQs9C+'
            '0L3QsCDQvtGG0LXQvdC60LAg4oCUINC/0LXRgNC10LrRgNGL0YLQuNC1IGNvdiAo0LHQvtC70YzR'
            'iNC1IOKAlCDQu9GD0YfRiNC1KSwNCiAgICAgICAg0LTQu9GPIEdULdGC0L7Rh9C60Lgg4oCUINGA'
            '0LDRgdGB0YLQvtGP0L3QuNC1INGB0L4g0LfQvdCw0LrQvtC8INC80LjQvdGD0YEgKNCx0LvQuNC2'
            '0LUg4oCUINC70YPRh9GI0LUpLg0KICAgICAgICAiIiINCiAgICAgICAgaWYgaXNpbnN0YW5jZShn'
            'dF9nZW9tLCBQb2ludCk6DQogICAgICAgICAgICByZXR1cm4gc2VsZi5fbWF0Y2hfcG9pbnRfZ3Qo'
            'cHJlZF9nZW9tLCBndF9nZW9tKQ0KICAgICAgICByZXR1cm4gc2VsZi5fbWF0Y2hfcG9seWdvbl9n'
            'dChwcmVkX2dlb20sIGd0X2dlb20pDQoNCiAgICBkZWYgX21hdGNoX3BvbHlnb25fZ3Qoc2VsZiwg'
            'cHJlZF9nZW9tOiBQb2x5Z29uLCBndF9nZW9tOiBQb2x5Z29uKSAtPiBUdXBsZVtib29sLCBmbG9h'
            'dF06DQogICAgICAgICIiItCh0YbQtdC90LDRgNC40Lkg0JAgKNCi0KAg0JrQntCXLTMpOiBjb3Yg'
            'Pj0g0L/QvtGA0L7Qs9CwINCYINCy0L3Rg9GC0YDQtdC90L3Rj9GPICjQv9GA0LXQtNGB0YLQsNCy'
            '0LjRgtC10LvRjNC90LDRjykNCiAgICAgICAg0YLQvtGH0LrQsCDQv9GA0LXQtNGB0LrQsNC30LDQ'
            'vdC40Y8g0LLQvdGD0YLRgNC4INGN0YLQsNC70L7QvdCwLiIiIg0KICAgICAgICBjb3ZlcmFnZSA9'
            'IF9wb2x5Z29uX2NvdmVyYWdlKHByZWRfZ2VvbSwgZ3RfZ2VvbSkNCiAgICAgICAgaGl0ID0gY292'
            'ZXJhZ2UgPj0gc2VsZi5jb3ZlcmFnZV90aHJlc2hvbGQgYW5kIF9pbm5lcl9wb2ludF9pbl9tYXNr'
            'KHByZWRfZ2VvbSwgZ3RfZ2VvbSkNCiAgICAgICAgcmV0dXJuIGhpdCwgY292ZXJhZ2UNCg0KICAg'
            'IGRlZiBfbWF0Y2hfcG9pbnRfZ3Qoc2VsZiwgcHJlZF9nZW9tOiBQb2x5Z29uLCBndF9wb2ludDog'
            'UG9pbnQpIC0+IFR1cGxlW2Jvb2wsIGZsb2F0XToNCiAgICAgICAgIiIi0KHRhtC10L3QsNGA0LjQ'
            'uSDQkSAo0KLQoCDQmtCe0JctMyk6IEdULdGC0L7Rh9C60LAg0L7RhtC10L3QuNCy0LDQtdGC0YHR'
            'jyDQv9C+INGA0LDRgdGB0YLQvtGP0L3QuNGOIOKAlCDQvtCx0YrQtdC60YINCiAgICAgICAg0L3Q'
            'sNC50LTQtdC9LCDQtdGB0LvQuCDRgNCw0YHRgdGC0L7Rj9C90LjQtSDQvtGCINGG0LXQvdGC0YDQ'
            'vtC40LTQsCDQv9GA0LXQtNGB0LrQsNC30LDQvdC40Y8g0LTQviBHVC3RgtC+0YfQutC4INC90LUN'
            'CiAgICAgICAg0L/RgNC10LLRi9GI0LDQtdGCIHJfdG9sZXJhbmNlLiDQntGG0LXQvdC60LAg0LTQ'
            'u9GPINCy0YvQsdC+0YDQsCDQu9GD0YfRiNC10LPQviBHVCDigJQg0YDQsNGB0YHRgtC+0Y/QvdC4'
            '0LUNCiAgICAgICAgKNCy0L7Qt9Cy0YDQsNGJ0LDQtdGC0YHRjyDRgdC+INC30L3QsNC60L7QvCDQ'
            'vNC40L3Rg9GBOiDQsdC70LjQttC1IOKAlCDCq9Cx0L7Qu9GM0YjQtcK7KS4NCiAgICAgICAgIiIi'
            'DQogICAgICAgIHRyeToNCiAgICAgICAgICAgIGNlbnRyb2lkID0gcHJlZF9nZW9tLmNlbnRyb2lk'
            'DQogICAgICAgICAgICBkaXN0YW5jZSA9IG1hdGguaHlwb3QoY2VudHJvaWQueCAtIGd0X3BvaW50'
            'LngsDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY2VudHJvaWQueSAtIGd0X3Bv'
            'aW50LnkpDQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246DQogICAgICAgICAgICByZXR1cm4gRmFs'
            'c2UsIDAuMA0KICAgICAgICBpZiBkaXN0YW5jZSA8PSBzZWxmLnJfdG9sZXJhbmNlOg0KICAgICAg'
            'ICAgICAgcmV0dXJuIFRydWUsIC1kaXN0YW5jZQ0KICAgICAgICByZXR1cm4gRmFsc2UsIDAuMA0K'
            'DQogICAgZGVmIF9jb3VudF9zb2Z0X3JlY2FsbCgNCiAgICAgICAgc2VsZiwNCiAgICAgICAgcHJl'
            'ZHM6IExpc3RbVHVwbGVbQW55LCBmbG9hdF1dLA0KICAgICAgICBndHM6IExpc3RbVHVwbGVbQW55'
            'LCBmbG9hdF1dLA0KICAgICkgLT4gaW50Og0KICAgICAgICAiIiLQodGH0LjRgtCw0LXRgiDQutC+'
            '0LvQuNGH0LXRgdGC0LLQviBHVC3QvtCx0YrQtdC60YLQvtCyLCDQvtCx0L3QsNGA0YPQttC10L3Q'
            'vdGL0YUg0YXQvtGC0Y8g0LHRiyDQvtC00L3QuNC8INC/0YDQtdC00YHQutCw0LfQsNC90LjQtdC8'
            'Lg0KDQogICAgICAgINCa0YDQuNGC0LXRgNC40Lk6INC/0LXRgNC10YHQtdGH0LXQvdC40LUg0YEg'
            '0L3QtdC90YPQu9C10LLQvtC5INC/0LvQvtGJ0LDQtNGM0Y47INC00LvRjyBHVC3RgtC+0YfQtdC6'
            'IOKAlCDQv9C10YDQtdGB0LXRh9C10L3QuNC1DQogICAgICAgINGBINC60YDRg9Cz0L7QvCDRgNCw'
            '0LTQuNGD0YHQsCByX3RvbGVyYW5jZS4NCiAgICAgICAgIiIiDQogICAgICAgIGlmIG5vdCBwcmVk'
            'cyBvciBub3QgZ3RzOg0KICAgICAgICAgICAgcmV0dXJuIDANCiAgICAgICAgY291bnQgPSAwDQog'
            'ICAgICAgIGZvciBndF9nZW9tLCBfIGluIGd0czoNCiAgICAgICAgICAgIHRhcmdldCA9IGd0X2dl'
            'b20NCiAgICAgICAgICAgIGlmIGlzaW5zdGFuY2UoZ3RfZ2VvbSwgUG9pbnQpOg0KICAgICAgICAg'
            'ICAgICAgIHRhcmdldCA9IF9wb2ludF90b19jaXJjbGUoZ3RfZ2VvbSwgc2VsZi5yX3RvbGVyYW5j'
            'ZSwgc2VsZi5jaXJjbGVfcG9pbnRzKQ0KICAgICAgICAgICAgZm9yIHByZWRfZ2VvbSwgXyBpbiBw'
            'cmVkczoNCiAgICAgICAgICAgICAgICBpZiBub3QgaXNpbnN0YW5jZShwcmVkX2dlb20sIFBvbHln'
            'b24pOg0KICAgICAgICAgICAgICAgICAgICBjb250aW51ZQ0KICAgICAgICAgICAgICAgIHRyeToN'
            'CiAgICAgICAgICAgICAgICAgICAgaWYgcHJlZF9nZW9tLmludGVyc2VjdHModGFyZ2V0KSBhbmQg'
            'cHJlZF9nZW9tLmludGVyc2VjdGlvbih0YXJnZXQpLmFyZWEgPiAwOg0KICAgICAgICAgICAgICAg'
            'ICAgICAgICAgY291bnQgKz0gMQ0KICAgICAgICAgICAgICAgICAgICAgICAgYnJlYWsNCiAgICAg'
            'ICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uOg0KICAgICAgICAgICAgICAgICAgICBjb250aW51'
            'ZQ0KICAgICAgICByZXR1cm4gY291bnQNCg0KICAgIGRlZiBfY2FsY3VsYXRlX2NlbnRlcl9kZXZp'
            'YXRpb24oDQogICAgICAgIHNlbGYsIG1hdGNoZXM6IExpc3RbVHVwbGVbQW55LCBBbnldXQ0KICAg'
            'ICkgLT4gZmxvYXQ6DQogICAgICAgICIiItCh0YDQtdC00L3QtdC1INGA0LDRgdGB0YLQvtGP0L3Q'
            'uNC1INC80LXQttC00YMg0YbQtdC90YLRgNCw0LzQuCBUUC3Qv9Cw0YAgKNC00LvRjyBHVC3RgtC+'
            '0YfQtdC6IOKAlCDQtNC+INGB0LDQvNC+0Lkg0YLQvtGH0LrQuCkuIiIiDQogICAgICAgIGlmIG5v'
            'dCBtYXRjaGVzOg0KICAgICAgICAgICAgcmV0dXJuIDAuMA0KICAgICAgICB0b3RhbF9kaXN0ID0g'
            'MC4wDQogICAgICAgIGZvciBwcmVkX3BvbHksIGd0X3BvbHkgaW4gbWF0Y2hlczoNCiAgICAgICAg'
            'ICAgIHRyeToNCiAgICAgICAgICAgICAgICBwcmVkX2MgPSBwcmVkX3BvbHkuY2VudHJvaWQNCiAg'
            'ICAgICAgICAgICAgICBndF9jID0gZ3RfcG9seS5jZW50cm9pZA0KICAgICAgICAgICAgICAgIGRp'
            'c3QgPSBtYXRoLnNxcnQoKHByZWRfYy54IC0gZ3RfYy54KSAqKiAyICsgKHByZWRfYy55IC0gZ3Rf'
            'Yy55KSAqKiAyKQ0KICAgICAgICAgICAgICAgIHRvdGFsX2Rpc3QgKz0gZGlzdA0KICAgICAgICAg'
            'ICAgZXhjZXB0IEV4Y2VwdGlvbjoNCiAgICAgICAgICAgICAgICBjb250aW51ZQ0KICAgICAgICBy'
            'ZXR1cm4gdG90YWxfZGlzdCAvIGxlbihtYXRjaGVzKSBpZiBtYXRjaGVzIGVsc2UgMC4wDQoNCiAg'
            'ICBkZWYgX2FnZ3JlZ2F0ZV9jb250b3VyKA0KICAgICAgICBzZWxmLCBwYWlyczogTGlzdFtUdXBs'
            'ZVtBbnksIEFueV1dDQogICAgKSAtPiBEaWN0W3N0ciwgQW55XToNCiAgICAgICAgIiIi0KPRgdGA'
            '0LXQtNC90Y/QtdGCINC60L7QvdGC0YPRgNC90YvQtSDQvNC10YLRgNC40LrQuCDQv9C+INC/0LDR'
            'gNCw0LwgwqvQv9C+0LvQuNCz0L7QvS3Qv9C+0LvQuNCz0L7QvcK7ICjQn9GA0LjQu9C+0LbQtdC9'
            '0LjQtSA0KS4NCg0KICAgICAgICBCb3VuZGFyeSBJb1UsIEhhdXNkb3JmZjk1INC4IFBvTGlTINGB'
            '0YfQuNGC0LDRjtGC0YHRjyDQv9C+INC60LDQttC00L7QuSDQv9Cw0YDQtSDQsiDQvNC10YLRgNCw'
            '0YU7DQogICAgICAgINCw0LPRgNC10LPQsNGCIOKAlCDRgdGA0LXQtNC90LXQtSDQv9C+INC/0LDR'
            'gNCw0LwsINCz0LTQtSDQvNC10YLRgNC40LrQsCDQstGL0YfQuNGB0LvQuNC80LAgKE5vbmUg0L7R'
            'gtC00LXQu9GM0L3QvtC5INC/0LDRgNGLDQogICAgICAgINC40Lct0LfQsCDQstGL0YDQvtC20LTQ'
            'tdC90L3QvtC5INCz0LXQvtC80LXRgtGA0LjQuCDQvdC1INC70L7QvNCw0LXRgiDRgdGA0LXQtNC9'
            '0LXQtSkuINCd0LXRgiDQv9Cw0YAg0LjQu9C4INC90LXRgg0KICAgICAgICDQstGL0YfQuNGB0LvQ'
            'uNC80YvRhSDQt9C90LDRh9C10L3QuNC5IOKGkiBOb25lICjQvdC1IE5hTikuIGBgcGFpcnNgYCDi'
            'gJQg0YfQuNGB0LvQviDRgNCw0YHRgdC80L7RgtGA0LXQvdC90YvRhQ0KICAgICAgICDQv9Cw0YA7'
            'INC00LvRjyBHVC3RgtC+0YfQtdC6INC/0LDRgNGLINGB0Y7QtNCwINC90LUg0L/QvtC/0LDQtNCw'
            '0Y7RgiAo0YMg0YLQvtGH0LrQuCDQvdC10YIg0LrQvtC90YLRg9GA0LApLg0KICAgICAgICAiIiIN'
            'CiAgICAgICAgYm91bmRhcnlfaW91X3ZhbHVlczogTGlzdFtmbG9hdF0gPSBbXQ0KICAgICAgICBo'
            'YXVzZG9yZmZfdmFsdWVzOiBMaXN0W2Zsb2F0XSA9IFtdDQogICAgICAgIHBvbGlzX3ZhbHVlczog'
            'TGlzdFtmbG9hdF0gPSBbXQ0KICAgICAgICBmb3IgcHJlZF9nZW9tLCBndF9nZW9tIGluIHBhaXJz'
            'Og0KICAgICAgICAgICAgYm91bmRhcnlfaW91ID0gX2JvdW5kYXJ5X2lvdShwcmVkX2dlb20sIGd0'
            'X2dlb20sIHNlbGYuY29udG91cl9iYW5kX3JhdGlvKQ0KICAgICAgICAgICAgaGF1c2RvcmZmID0g'
            'X2hhdXNkb3JmZjk1KHByZWRfZ2VvbSwgZ3RfZ2VvbSwgc2VsZi5jb250b3VyX3NhbXBsZXMpDQog'
            'ICAgICAgICAgICBwb2xpcyA9IF9wb2xpcyhwcmVkX2dlb20sIGd0X2dlb20pDQogICAgICAgICAg'
            'ICBpZiBib3VuZGFyeV9pb3UgaXMgbm90IE5vbmU6DQogICAgICAgICAgICAgICAgYm91bmRhcnlf'
            'aW91X3ZhbHVlcy5hcHBlbmQoYm91bmRhcnlfaW91KQ0KICAgICAgICAgICAgaWYgaGF1c2RvcmZm'
            'IGlzIG5vdCBOb25lOg0KICAgICAgICAgICAgICAgIGhhdXNkb3JmZl92YWx1ZXMuYXBwZW5kKGhh'
            'dXNkb3JmZikNCiAgICAgICAgICAgIGlmIHBvbGlzIGlzIG5vdCBOb25lOg0KICAgICAgICAgICAg'
            'ICAgIHBvbGlzX3ZhbHVlcy5hcHBlbmQocG9saXMpDQogICAgICAgIHJldHVybiB7DQogICAgICAg'
            'ICAgICAiYm91bmRhcnlfaW91IjogKA0KICAgICAgICAgICAgICAgIHN1bShib3VuZGFyeV9pb3Vf'
            'dmFsdWVzKSAvIGxlbihib3VuZGFyeV9pb3VfdmFsdWVzKQ0KICAgICAgICAgICAgICAgIGlmIGJv'
            'dW5kYXJ5X2lvdV92YWx1ZXMgZWxzZSBOb25lDQogICAgICAgICAgICApLA0KICAgICAgICAgICAg'
            'ImhhdXNkb3JmZjk1X20iOiAoDQogICAgICAgICAgICAgICAgc3VtKGhhdXNkb3JmZl92YWx1ZXMp'
            'IC8gbGVuKGhhdXNkb3JmZl92YWx1ZXMpDQogICAgICAgICAgICAgICAgaWYgaGF1c2RvcmZmX3Zh'
            'bHVlcyBlbHNlIE5vbmUNCiAgICAgICAgICAgICksDQogICAgICAgICAgICAicG9saXNfbSI6ICgN'
            'CiAgICAgICAgICAgICAgICBzdW0ocG9saXNfdmFsdWVzKSAvIGxlbihwb2xpc192YWx1ZXMpIGlm'
            'IHBvbGlzX3ZhbHVlcyBlbHNlIE5vbmUNCiAgICAgICAgICAgICksDQogICAgICAgICAgICAicGFp'
            'cnMiOiBsZW4ocGFpcnMpLA0KICAgICAgICB9DQoNCiAgICBkZWYgX2J1aWxkX2J5X2RhdGFfdHlw'
            'ZSgNCiAgICAgICAgc2VsZiwNCiAgICAgICAgZ3RfY291bnRzOiBEaWN0W3N0ciwgaW50XSwNCiAg'
            'ICAgICAgcHJlZF9jb3VudHM6IERpY3Rbc3RyLCBpbnRdLA0KICAgICAgICB0cF9jb3VudHM6IERp'
            'Y3Rbc3RyLCBpbnRdLA0KICAgICAgICBtYXRjaGVkX3ByZWRfY291bnRzOiBEaWN0W3N0ciwgaW50'
            'XSwNCiAgICApIC0+IERpY3Rbc3RyLCBEaWN0W3N0ciwgQW55XV06DQogICAgICAgICIiItCh0L/R'
            'gNCw0LLQvtGH0L3Ri9C5INCx0LvQvtC6INC/0L4g0YLQuNC/0LDQvCDQtNCw0L3QvdGL0YUgKNCf'
            '0YDQuNC70L7QttC10L3QuNC1IDQpLg0KDQogICAgICAgIFRQINC4IEZOINC+0YLQvdC+0YHRj9GC'
            '0YHRjyDQv9C+IG1hcmt1cF90eXBlIEdULdC+0LHRitC10LrRgtCwLCBGUCDigJQg0L/QviBtYXJr'
            'dXBfdHlwZQ0KICAgICAgICDQv9GA0LXQtNGB0LrQsNC30LDQvdC40Y8gKNGDINC/0YDQtdC00YHQ'
            'utCw0LfQsNC90LjQuSDQutC70Y7Rh9CwINC80L7QttC10YIg0L3QtSDQsdGL0YLRjCDigJQg0LPR'
            'gNGD0L/Qv9CwIMKr0L/RgNC+0YfQtdC1wrspLg0KICAgICAgICBwcmVjaXNpb24gPSB0cCAvICh0'
            'cCArIDAuNSpmcCArIM61KSDigJQg0YLQvtGCINC20LUg0YjRgtGA0LDRhCDQt9CwINC70LjRiNC9'
            '0LXQtQ0KICAgICAgICDRgdGA0LDQsdCw0YLRi9Cy0LDQvdC40LUsINGH0YLQviDQuCDQsiDQvtGB'
            '0L3QvtCy0L3QvtC5INC80LXRgtGA0LjQutC1OyByZWNhbGwgPSB0cC8odHArZm4rzrUpLg0KICAg'
            'ICAgICDQmNGC0L7Qs9C+0LLRi9C5ICjRgdC80LXRiNCw0L3QvdGL0LkpIHRwINC80L7QttC10YIg'
            '0YPRh9C40YLRi9Cy0LDRgtGM0YHRjyDQsiDQs9GA0YPQv9C/0LUgR1QsINCwINGB0LLRj9C30LDQ'
            'vdC90YvQuQ0KICAgICAgICBGUCDigJQg0LIg0LPRgNGD0L/Qv9C1INC/0YDQtdC00YHQutCw0LfQ'
            'sNC90LjRjzsg0Y3RgtC+INC+0YHQvtC30L3QsNC90L3QsNGPINCw0YHQuNC80LzQtdGC0YDQuNGP'
            'INCi0JcuDQoNCiAgICAgICAg0JPRgNGD0L/Qv9CwINC/0L7Qv9Cw0LTQsNC10YIg0LIg0LLRi9Cy'
            '0L7QtCwg0LXRgdC70Lgg0LIg0L3QtdC5INC10YHRgtGMIEdULdC+0LHRitC10LrRgtGLICjQvtGB'
            '0L3QvtCy0LAg0L/QvtC70L3QvtGC0YspDQogICAgICAgINC40LvQuCDQvdC10LzQvtGC0YfQuNGA'
            '0L7QstCw0L3QvdGL0LUg0L/RgNC10LTRgdC60LDQt9Cw0L3QuNGPIChGUCwg0L7RgdC90L7QstCw'
            'INGC0L7Rh9C90L7RgdGC0LgpOyDQs9GA0YPQv9C/0YssINCz0LTQtQ0KICAgICAgICDQtdGB0YLR'
            'jCDRgtC+0LvRjNC60L4g0LzQsNGC0YfQvdGD0YLRi9C1INC/0YDQtdC00YHQutCw0LfQsNC90LjR'
            'jyDQsdC10LcgbWFya3VwX3R5cGUsINC90LUg0YHQvtC30LTQsNGO0YLRgdGPIOKAlA0KICAgICAg'
            'ICDQuNGFIFRQINGD0LbQtSDRg9GH0YLRkdC9INC/0L4g0LPRgNGD0L/Qv9C1IEdULdC+0LHRitC1'
            '0LrRgtCwLg0KICAgICAgICAiIiINCiAgICAgICAgZ3JvdXBzX3ByZXNlbnQgPSBzZXQoZ3RfY291'
            'bnRzKSB8IHNldChwcmVkX2NvdW50cykNCiAgICAgICAgb3JkZXIgPSBsaXN0KGRpY3QuZnJvbWtl'
            'eXMoREFUQV9UWVBFX0dST1VQUy52YWx1ZXMoKSkpDQogICAgICAgIGlmIERFRkFVTFRfREFUQV9U'
            'WVBFX0dST1VQIG5vdCBpbiBvcmRlcjoNCiAgICAgICAgICAgIG9yZGVyLmFwcGVuZChERUZBVUxU'
            'X0RBVEFfVFlQRV9HUk9VUCkNCiAgICAgICAgb3JkZXIgKz0gW2cgZm9yIGcgaW4gZ3JvdXBzX3By'
            'ZXNlbnQgaWYgZyBub3QgaW4gb3JkZXJdDQoNCiAgICAgICAgcmVzdWx0OiBEaWN0W3N0ciwgRGlj'
            'dFtzdHIsIEFueV1dID0ge30NCiAgICAgICAgZm9yIGdyb3VwIGluIG9yZGVyOg0KICAgICAgICAg'
            'ICAgZ3RfY291bnQgPSBndF9jb3VudHMuZ2V0KGdyb3VwLCAwKQ0KICAgICAgICAgICAgcHJlZF9j'
            'b3VudCA9IHByZWRfY291bnRzLmdldChncm91cCwgMCkNCiAgICAgICAgICAgIHRwX2NvdW50ID0g'
            'dHBfY291bnRzLmdldChncm91cCwgMCkNCiAgICAgICAgICAgICMgRlAg4oCUINC90LXQvNC+0YLR'
            'h9C40YDQvtCy0LDQvdC90YvQtSDQv9GA0LXQtNGB0LrQsNC30LDQvdC40Y8g0YHQstC+0LXQuSDQ'
            's9GA0YPQv9C/0YsNCiAgICAgICAgICAgIGZwX2NvdW50ID0gcHJlZF9jb3VudCAtIG1hdGNoZWRf'
            'cHJlZF9jb3VudHMuZ2V0KGdyb3VwLCAwKQ0KICAgICAgICAgICAgaWYgZ3RfY291bnQgPT0gMCBh'
            'bmQgZnBfY291bnQgPT0gMDoNCiAgICAgICAgICAgICAgICBjb250aW51ZQ0KICAgICAgICAgICAg'
            'Zm5fY291bnQgPSBndF9jb3VudCAtIHRwX2NvdW50DQogICAgICAgICAgICByZXN1bHRbZ3JvdXBd'
            'ID0gew0KICAgICAgICAgICAgICAgICJ0cCI6IHRwX2NvdW50LA0KICAgICAgICAgICAgICAgICJm'
            'cCI6IGZwX2NvdW50LA0KICAgICAgICAgICAgICAgICJmbiI6IGZuX2NvdW50LA0KICAgICAgICAg'
            'ICAgICAgICJwcmVjaXNpb24iOiB0cF9jb3VudCAvICh0cF9jb3VudCArIDAuNSAqIGZwX2NvdW50'
            'ICsgc2VsZi5lcHNpbG9uKSwNCiAgICAgICAgICAgICAgICAicmVjYWxsIjogdHBfY291bnQgLyAo'
            'dHBfY291bnQgKyBmbl9jb3VudCArIHNlbGYuZXBzaWxvbiksDQogICAgICAgICAgICB9DQogICAg'
            'ICAgIHJldHVybiByZXN1bHQNCg0KICAgIGRlZiBfYnVpbGRfc2l6ZV9jYXRlZ29yaWVzKA0KICAg'
            'ICAgICBzZWxmLCBndF9wb2x5Z29uczogTGlzdFtBbnldLCBtYXRjaGVkX2d0X2lkczogc2V0DQog'
            'ICAgKSAtPiBMaXN0W0RpY3Rbc3RyLCBBbnldXToNCiAgICAgICAgIiIi0KDQsNC30LzQtdGA0L3R'
            'i9C1INC60LDRgtC10LPQvtGA0LjQuCBHVC3QvtCx0YrQtdC60YLQvtCyICjQn9GA0LjQu9C+0LbQ'
            'tdC90LjQtSA0KS4NCg0KICAgICAgICDQmtCw0YLQtdCz0L7RgNC40Lgg4oCUINC40L3RgtC10YDQ'
            'stCw0LvRiyDQv9C+INC/0LvQvtGJ0LDQtNC4IEdULdC/0L7Qu9C40LPQvtC90L7QsiDQsiDQvMKy'
            'IChVVE0pOg0KICAgICAgICAoMCwgYjFdLCAoYjEsIGIyXSwgLi4uLCAoYk4sICviiJ4pLCDQs9GA'
            '0LDQvdC40YbRiyDigJQgU0laRV9CSU5TLiBHVC3RgtC+0YfQutC4DQogICAgICAgINC/0LvQvtGJ'
            '0LDQtNC4INC90LUg0LjQvNC10Y7RgiDQuCDQvdC1INGD0YfQuNGC0YvQstCw0Y7RgtGB0Y8uIHJl'
            'Y2FsbCA9IHRwL2d0INC/0L4g0LrQsNGC0LXQs9C+0YDQuNC4Ow0KICAgICAgICBOb25lIOKAlCDQ'
            'tdGB0LvQuCDQsiDQutCw0YLQtdCz0L7RgNC40Lgg0L3QtdGCIEdUICjQv9C+0LrQsNC30LDRgtC1'
            '0LvRjCDQvdC1INC+0L/RgNC10LTQtdC70ZHQvSwg0L3QtSBOYU4pLg0KICAgICAgICDQodC/0LjR'
            'gdC+0Log0YHQvtC00LXRgNC20LjRgiDQstGB0LUg0LrQsNGC0LXQs9C+0YDQuNC4LCDQstC60LvR'
            'jtGH0LDRjyDQv9GD0YHRgtGL0LUsIOKAlCDRhNC+0YDQvNCwINC+0YLRh9GR0YLQsCDRgdGC0LDQ'
            'sdC40LvRjNC90LAuDQogICAgICAgICIiIg0KICAgICAgICBiaW5zID0gc2VsZi5zaXplX2JpbnMN'
            'CiAgICAgICAgc3RhdHMgPSBbeyJndCI6IDAsICJ0cCI6IDB9IGZvciBfIGluIHJhbmdlKGxlbihi'
            'aW5zKSArIDEpXQ0KICAgICAgICBmb3IgZ2VvbSBpbiBndF9wb2x5Z29uczoNCiAgICAgICAgICAg'
            'IGFyZWEgPSBnZW9tLmFyZWENCiAgICAgICAgICAgIGlmIGFyZWEgPD0gMDoNCiAgICAgICAgICAg'
            'ICAgICBjb250aW51ZSAgIyDQstGL0YDQvtC20LTQtdC90L3QsNGPINCz0LXQvtC80LXRgtGA0LjR'
            'jyDQsdC10Lcg0L/Qu9C+0YnQsNC00Lgg4oCUINCy0L3QtSDQutCw0YLQtdCz0L7RgNC40LkNCiAg'
            'ICAgICAgICAgIGluZGV4ID0gbGVuKGJpbnMpDQogICAgICAgICAgICBmb3IgaSwgdXBwZXIgaW4g'
            'ZW51bWVyYXRlKGJpbnMpOg0KICAgICAgICAgICAgICAgIGlmIGFyZWEgPD0gdXBwZXI6DQogICAg'
            'ICAgICAgICAgICAgICAgIGluZGV4ID0gaQ0KICAgICAgICAgICAgICAgICAgICBicmVhaw0KICAg'
            'ICAgICAgICAgc3RhdHNbaW5kZXhdWyJndCJdICs9IDENCiAgICAgICAgICAgIGlmIGlkKGdlb20p'
            'IGluIG1hdGNoZWRfZ3RfaWRzOg0KICAgICAgICAgICAgICAgIHN0YXRzW2luZGV4XVsidHAiXSAr'
            'PSAxDQogICAgICAgIGNhdGVnb3JpZXM6IExpc3RbRGljdFtzdHIsIEFueV1dID0gW10NCiAgICAg'
            'ICAgZm9yIGksIHN0YXQgaW4gZW51bWVyYXRlKHN0YXRzKToNCiAgICAgICAgICAgIGd0X2NvdW50'
            'ID0gc3RhdFsiZ3QiXQ0KICAgICAgICAgICAgdHBfY291bnQgPSBzdGF0WyJ0cCJdDQogICAgICAg'
            'ICAgICBjYXRlZ29yaWVzLmFwcGVuZCh7DQogICAgICAgICAgICAgICAgIm1heF9hcmVhIjogYmlu'
            'c1tpXSBpZiBpIDwgbGVuKGJpbnMpIGVsc2UgTm9uZSwNCiAgICAgICAgICAgICAgICAiZ3QiOiBn'
            'dF9jb3VudCwNCiAgICAgICAgICAgICAgICAidHAiOiB0cF9jb3VudCwNCiAgICAgICAgICAgICAg'
            'ICAicmVjYWxsIjogKHRwX2NvdW50IC8gZ3RfY291bnQpIGlmIGd0X2NvdW50IGVsc2UgTm9uZSwN'
            'CiAgICAgICAgICAgIH0pDQogICAgICAgIHJldHVybiBjYXRlZ29yaWVzDQoNCg0KZGVmIG1haW4o'
            'KSAtPiBpbnQ6DQogICAgcGFyc2VyID0gYXJncGFyc2UuQXJndW1lbnRQYXJzZXIoDQogICAgICAg'
            'IGRlc2NyaXB0aW9uPSLQoNCw0YHRh9C10YIg0LzQtdGC0YDQuNC60LggQXJjaGFlb2xvZ3kgQXNz'
            'aXN0aXZlIEYxICjQmtCe0JctMykiLA0KICAgICAgICBmb3JtYXR0ZXJfY2xhc3M9YXJncGFyc2Uu'
            'QXJndW1lbnREZWZhdWx0c0hlbHBGb3JtYXR0ZXIsDQogICAgKQ0KICAgIHBhcnNlci5hZGRfYXJn'
            'dW1lbnQoIi0tcHJlZGljdGlvbnMiLCAiLXAiLCByZXF1aXJlZD1UcnVlLCBoZWxwPSJHZW9KU09O'
            'INGBINC/0YDQtdC00YHQutCw0LfQsNC90LjRj9C80LgiKQ0KICAgIHBhcnNlci5hZGRfYXJndW1l'
            'bnQoIi0tZ3JvdW5kLXRydXRoIiwgIi1nIiwgcmVxdWlyZWQ9VHJ1ZSwgaGVscD0iR2VvSlNPTiDR'
            'gSDRgNCw0LfQvNC10YLQutC+0LkgKEdUKSIpDQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgNCiAg'
            'ICAgICAgIi0tY292ZXJhZ2UtdGhyZXNob2xkIiwNCiAgICAgICAgdHlwZT1mbG9hdCwNCiAgICAg'
            'ICAgZGVmYXVsdD1Ob25lLA0KICAgICAgICBoZWxwPSLQn9C+0YDQvtCzINC/0LXRgNC10LrRgNGL'
            '0YLQuNGPIGNvdiA9IGFyZWEoUOKIqUcpL21pbihhcmVhKSDQtNC70Y8g0LzRj9Cz0LrQvtCz0L4g'
            '0LzQsNGC0YfQuNC90LPQsCAiDQogICAgICAgICAgICAgIijQv9C+INGD0LzQvtC70YfQsNC90LjR'
            'jiDQuNC3IGNvbmZpZzogMC41KSIsDQogICAgKQ0KICAgIHBhcnNlci5hZGRfYXJndW1lbnQoDQog'
            'ICAgICAgICItLXItdG9sZXJhbmNlIiwNCiAgICAgICAgdHlwZT1mbG9hdCwNCiAgICAgICAgZGVm'
            'YXVsdD0yLjAsDQogICAgICAgIGhlbHA9ItCg0LDQtNC40YPRgSDQtNC+0L/Rg9GB0LrQsCDQtNC7'
            '0Y8g0YLQvtGH0LXRh9C90L7QuSDRgNCw0LfQvNC10YLQutC4IEdUICjQsiDQvNC10YLRgNCw0YUs'
            'IFVUTSkiLA0KICAgICkNCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KA0KICAgICAgICAiLS1vdXRw'
            'dXQtanNvbiIsDQogICAgICAgIHR5cGU9c3RyLA0KICAgICAgICBkZWZhdWx0PU5vbmUsDQogICAg'
            'ICAgIGhlbHA9ItCf0YPRgtGMINC00LvRjyDRgdC+0YXRgNCw0L3QtdC90LjRjyDQtNC10YLQsNC7'
            '0YzQvdGL0YUg0YDQtdC30YPQu9GM0YLQsNGC0L7QsiDQsiBKU09OIiwNCiAgICApDQogICAgcGFy'
            'c2VyLmFkZF9hcmd1bWVudCgNCiAgICAgICAgIi0tZGlhZ25vc3RpY3MiLA0KICAgICAgICBhY3Rp'
            'b249InN0b3JlX3RydWUiLA0KICAgICAgICBoZWxwPSLQn9C10YfQsNGC0Ywg0LTQuNCw0LPQvdC+'
            '0YHRgtC40LrQuCDQv9GA0LjRh9C40L0gRjE9MDog0YfRgtC+INC/0YDQuNC90Y/RgtC+L9C+0YLQ'
            'sdGA0L7RiNC10L3QviwgQ1JTLCDQv9C+0LTRgdC60LDQt9C60LgiLA0KICAgICkNCg0KICAgIGFy'
            'Z3MgPSBwYXJzZXIucGFyc2VfYXJncygpDQoNCiAgICBsb2dnaW5nLmJhc2ljQ29uZmlnKGxldmVs'
            'PWxvZ2dpbmcuSU5GTywgZm9ybWF0PSIlKGxldmVsbmFtZSlzOiAlKG1lc3NhZ2UpcyIpDQoNCiAg'
            'ICBwcmVkX3BhdGggPSBQYXRoKGFyZ3MucHJlZGljdGlvbnMpDQogICAgZ3RfcGF0aCA9IFBhdGgo'
            'YXJncy5ncm91bmRfdHJ1dGgpDQogICAgaWYgbm90IHByZWRfcGF0aC5leGlzdHMoKToNCiAgICAg'
            'ICAgcmFpc2UgRmlsZU5vdEZvdW5kRXJyb3IoZiLQpNCw0LnQuyDQv9GA0LXQtNGB0LrQsNC30LDQ'
            'vdC40Lkg0L3QtSDQvdCw0LnQtNC10L06IHtwcmVkX3BhdGh9IikNCiAgICBpZiBub3QgZ3RfcGF0'
            'aC5leGlzdHMoKToNCiAgICAgICAgcmFpc2UgRmlsZU5vdEZvdW5kRXJyb3IoZiLQpNCw0LnQuyBn'
            'cm91bmQgdHJ1dGgg0L3QtSDQvdCw0LnQtNC10L06IHtndF9wYXRofSIpDQoNCiAgICB3aXRoIG9w'
            'ZW4ocHJlZF9wYXRoLCAiciIsIGVuY29kaW5nPSJ1dGYtOCIpIGFzIGY6DQogICAgICAgIHByZWRp'
            'Y3Rpb25zX2dlb2pzb24gPSBqc29uLmxvYWQoZikNCiAgICB3aXRoIG9wZW4oZ3RfcGF0aCwgInIi'
            'LCBlbmNvZGluZz0idXRmLTgiKSBhcyBmOg0KICAgICAgICBndF9nZW9qc29uID0ganNvbi5sb2Fk'
            'KGYpDQoNCiAgICBpZiBhcmdzLmRpYWdub3N0aWNzOg0KICAgICAgICB0cnk6DQogICAgICAgICAg'
            'ICBmcm9tIGRhdGFfdmFsaWRhdG9yIGltcG9ydCBidWlsZF9kaWFnbm9zdGljcw0KICAgICAgICBl'
            'eGNlcHQgSW1wb3J0RXJyb3I6DQogICAgICAgICAgICBsb2dnZXIuZXJyb3IoDQogICAgICAgICAg'
            'ICAgICAgIi0tZGlhZ25vc3RpY3Mg0LTQvtGB0YLRg9C/0L3QsCDRgtC+0LvRjNC60L4g0LIg0L/Q'
            'vtC70L3QvtC8INC60L7QvNC/0LvQtdC60YLQtSDRgdC10YDQstC40YHQsCAiDQogICAgICAgICAg'
            'ICAgICAgIijRgNGP0LTQvtC8INGB0L4g0YHQutGA0LjQv9GC0L7QvCDQvdGD0LbQtdC9INC80L7Q'
            'tNGD0LvRjCBkYXRhX3ZhbGlkYXRvci5weSkuIg0KICAgICAgICAgICAgKQ0KICAgICAgICAgICAg'
            'cmV0dXJuIDENCiAgICAgICAgZGlhZyA9IGJ1aWxkX2RpYWdub3N0aWNzKHByZWRpY3Rpb25zX2dl'
            'b2pzb24sIGd0X2dlb2pzb24sIGdldF9jb25maWcoKSkNCiAgICAgICAgcHJpbnQoIj09PSDQlNC4'
            '0LDQs9C90L7RgdGC0LjQutCwINC/0YDQtdC00YHQutCw0LfQsNC90LjQuSAo0L/RgNC40L3Rj9GC'
            '0L4v0L7RgtCx0YDQvtGI0LXQvdC+KSA9PT0iKQ0KICAgICAgICBwcmludChqc29uLmR1bXBzKGRp'
            'YWdbInByZWRpY3Rpb25zIl0sIGVuc3VyZV9hc2NpaT1GYWxzZSwgaW5kZW50PTIpKQ0KICAgICAg'
            'ICBwcmludCgiPT09INCh0LjRgdGC0LXQvNCwINC60L7QvtGA0LTQuNC90LDRgiA9PT0iKQ0KICAg'
            'ICAgICBwcmludChqc29uLmR1bXBzKGRpYWdbImNycyJdLCBlbnN1cmVfYXNjaWk9RmFsc2UsIGlu'
            'ZGVudD0yKSkNCiAgICAgICAgcHJpbnQoIj09PSDQn9C+0LTRgdC60LDQt9C60LggPT09IikNCiAg'
            'ICAgICAgZm9yIGhpbnQgaW4gZGlhZ1siaGludHMiXToNCiAgICAgICAgICAgIHByaW50KCIgLSIs'
            'IGhpbnQpDQogICAgICAgIHByaW50KCkNCg0KICAgIGNhbGN1bGF0b3IgPSBGMUNhbGN1bGF0b3Io'
            'DQogICAgICAgIGNvdmVyYWdlX3RocmVzaG9sZD1hcmdzLmNvdmVyYWdlX3RocmVzaG9sZCwNCiAg'
            'ICAgICAgcl90b2xlcmFuY2U9YXJncy5yX3RvbGVyYW5jZSwNCiAgICApDQogICAgcmVzdWx0cyA9'
            'IGNhbGN1bGF0b3IuY2FsY3VsYXRlX2YxKHByZWRpY3Rpb25zX2dlb2pzb24sIGd0X2dlb2pzb24p'
            'DQoNCiAgICAjINCS0YvQstC+0LQg0LjRgtC+0LPQvtCy0L7Qs9C+IHNjb3JlDQogICAgcHJpbnQo'
            'ZiJ7cmVzdWx0c1snZmluYWxfc2NvcmUnXTouMTBmfSIpDQoNCiAgICAjINCe0L/RhtC40L7QvdCw'
            '0LvRjNC90YvQuSDQstGL0LLQvtC0INCyIEpTT04NCiAgICBpZiBhcmdzLm91dHB1dF9qc29uOg0K'
            'ICAgICAgICBvdXRwdXRfcGF0aCA9IFBhdGgoYXJncy5vdXRwdXRfanNvbikNCiAgICAgICAgb3V0'
            'cHV0X3BhdGgucGFyZW50Lm1rZGlyKHBhcmVudHM9VHJ1ZSwgZXhpc3Rfb2s9VHJ1ZSkNCiAgICAg'
            'ICAgd2l0aCBvcGVuKG91dHB1dF9wYXRoLCAidyIsIGVuY29kaW5nPSJ1dGYtOCIpIGFzIGY6DQog'
            'ICAgICAgICAgICBqc29uLmR1bXAocmVzdWx0cywgZiwgaW5kZW50PTIsIGVuc3VyZV9hc2NpaT1G'
            'YWxzZSkNCiAgICAgICAgbG9nZ2VyLmluZm8oZiLQlNC10YLQsNC70YzQvdGL0LUg0YDQtdC30YPQ'
            'u9GM0YLQsNGC0Ysg0YHQvtGF0YDQsNC90LXQvdGLINCyIHtvdXRwdXRfcGF0aH0iKQ0KDQogICAg'
            'cmV0dXJuIDANCg0KDQppZiBfX25hbWVfXyA9PSAiX19tYWluX18iOg0KICAgIHJhaXNlIFN5c3Rl'
            'bUV4aXQobWFpbigpKQ0K'
        ),
    },
}
# --- END EMBEDDED_HELPERS ---



def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_status(state: str, **extra):
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"status": state, "ts": utc_now(), **extra}
    (OUT / "STATUS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [state, f"ts={payload['ts']}"]
    for k, v in extra.items():
        lines.append(f"{k}={v}")
    (OUT / "STATUS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_inputs() -> dict[str, Path]:
    root = INPUT_ROOT
    out: dict[str, Path] = {}
    if not root.is_dir():
        return out
    for p in sorted(root.iterdir()):
        if p.is_dir():
            out[p.name.lower()] = p
    return out


def find_named(inputs: dict[str, Path], *needles: str) -> Path | None:
    for n in needles:
        for k, p in inputs.items():
            if n in k:
                return p
    return None


def find_file(inputs: dict[str, Path], name: str) -> Path | None:
    for p in inputs.values():
        for hit in p.rglob(name):
            if hit.is_file():
                return hit
    return None



def _pkg_root() -> Path:
    return Path(os.environ.get("PIXGBM_CR_PKG", str(WORKING_ROOT / "_pixgbm_cr_pkg")))


def materialize_embedded_helpers(root: Path | None = None) -> Path:
    """Decode EMBEDDED helpers into root; verify sha256."""
    root = root or _pkg_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "helpers").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    report = []
    for rel, meta in _EMBEDDED_HELPERS.items():
        data = base64.b64decode(meta["b64"])
        got = hashlib.sha256(data).hexdigest()
        if got != meta["sha256"]:
            raise RuntimeError(f"embedded sha mismatch for {rel}: {got} != {meta['sha256']}")
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if (not dest.is_file()) or dest.read_bytes() != data:
            dest.write_bytes(data)
        report.append({"path": rel, "sha256": got, "bytes": len(data)})
    (root / "EMBEDDED_SHAS.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return root


def ensure_helpers() -> Path:
    """Materialize inlined helpers (Kaggle only ships code_file)."""
    root = materialize_embedded_helpers()
    helpers = root / "helpers"
    if not (helpers / "bot6_r3_gate.py").is_file():
        raise FileNotFoundError("helpers/bot6_r3_gate.py missing after materialize")
    return helpers


def load_rebuild():
    root = materialize_embedded_helpers()
    mod_path = root / "rebuild_w4_geo_tree.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("rebuild_w4_geo_tree", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod



def ensure_offline_wheels() -> list[str]:
    """Prefer vendored wheels; network pip allowed as fallback (enable_internet=true)."""
    installed: list[str] = []
    root = INPUT_ROOT
    wheel_dirs = [d for d in root.rglob("wheels") if d.is_dir()] if root.is_dir() else []

    def _need(mod: str) -> bool:
        try:
            __import__(mod)
            return False
        except ImportError:
            return True

    def _pip(pkgs: list[str], *, offline: bool) -> bool:
        cmd = [sys.executable, "-m", "pip", "install", "-q"]
        if offline:
            if not wheel_dirs:
                return False
            cmd += ["--no-index", "--no-deps"]
            for d in wheel_dirs:
                cmd.extend(["--find-links", str(d)])
        cmd += pkgs
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            log("WARN pip", pkgs, "offline=" + str(offline), (r.stderr or r.stdout or "")[:400])
            return False
        return True

    if _need("zstandard"):
        if _pip(["zstandard"], offline=True):
            installed.append("zstandard")
        elif _pip(["zstandard"], offline=False):
            installed.append("zstandard(net)")
    if _need("boto3"):
        if _pip(["boto3"], offline=True):
            installed.append("boto3")
        elif _pip(["boto3"], offline=False):
            installed.append("boto3(net)")
        else:
            log("WARN boto3 not installed — S3 rebuild will fail")
    return installed


def dry_run_syntax() -> int:
    import py_compile
    self_path = Path(__file__).resolve()
    py_compile.compile(str(self_path), doraise=True)
    print(f"OK compile {self_path.name}")

    root = materialize_embedded_helpers()
    print("OK materialize", root)
    for rel in sorted(_EMBEDDED_HELPERS):
        p = root / rel
        py_compile.compile(str(p), doraise=True)
        sha = _EMBEDDED_HELPERS[rel]["sha256"]
        print(f"OK compile embedded {rel} sha={sha[:16]}…")

    helpers = ensure_helpers()
    load_rebuild()
    print("OK helpers+rebuild load from", helpers)

    inputs = find_inputs()
    print("OK find_inputs", sorted(inputs) if inputs else "(none)")
    cand = find_cand_source(inputs)
    print(
        "OK find_cand_source",
        {k: str(v) if isinstance(v, Path) else v for k, v in (cand or {}).items()},
    )
    ref = find_ref_source(inputs)
    print(
        "OK find_ref_source",
        {k: str(v) if isinstance(v, Path) else v for k, v in (ref or {}).items()},
    )

    missing = []
    for mod in ("numpy", "scipy", "lightgbm", "rasterio", "shapely"):
        try:
            __import__(mod)
            print(f"OK import {mod}")
        except ImportError:
            missing.append(mod)
            print(f"MISSING import {mod} (needed at inference time; expect on Kaggle CPU)")

    meta = self_path.parent / "kernel-metadata.json"
    if meta.is_file():
        m = json.loads(meta.read_text(encoding="utf-8"))
        assert m.get("enable_internet") is True, "enable_internet must be true (S3)"
        assert m.get("enable_gpu") is False, "enable_gpu must be false"
        srcs = m.get("dataset_sources") or []
        assert "dmitrijkv/koz3-s3-cred" in srcs, "koz3-s3-cred required"
        assert "dmitrijkv/koz3-fr1h-chk-20261101" in srcs, "fr1h-chk REF required"
        assert "dmitrijkv/koz3-pixgbm-cr-ref" not in srcs, "legacy pixgbm-cr-ref must be dropped"
        assert not any("w4-geo" in s for s in srcs), "w4-geo must not be wired"
        print("OK kernel-metadata internet/gpu flags")
        print("dataset_sources:", srcs)
    else:
        print("OK skip kernel-metadata (not shipped with code_file alone)")

    sample = (
        "[post]\nkurgan_twins_only = true\n\n[post.frame_prior]\n"
        'groups = ["sp_or"]\n\n[rerank_arch]\nenabled = true\n'
    )
    stripped = _strip_frame_prior_section(sample)
    assert "[post.frame_prior]" not in stripped and "kurgan_twins_only" in stripped
    print("OK FR1h force-off strip helper")
    print("DRY_RUN_OK missing_optional=", missing)
    return 0


def _strip_frame_prior_section(text: str) -> str:
    """Remove [post.frame_prior] table (FR1h ON → OFF)."""
    return re.sub(
        r"(?ms)^\[post\.frame_prior\][^\n]*\n(?:(?!\[)[^\n]*\n)*",
        "",
        text,
    )


def force_fr1h_off(repo: Path) -> dict:
    """Force FR1h OFF in configs/default.toml (explicit, not relying on default).

    Flag name: TOML table [post.frame_prior] (read as cfg['post']['frame_prior']).
    Default in fr1h.zip.bin: ON (section present).
    OFF = section absent / falsy → watchdog/pipeline skip post.frame_prior().
    Also restore archive SpOr kurgan thresholds (FR1h re-sweep → taus_base).
    """
    toml = repo / "configs" / "default.toml"
    text = toml.read_text(encoding="utf-8")
    before_has = bool(re.search(r"(?m)^\[post\.frame_prior\]", text))
    text = _strip_frame_prior_section(text)
    # Ensure no leftover truthy frame_prior assignment under [post]
    text = re.sub(r"(?m)^frame_prior\s*=\s*.*\n", "", text)
    # Restore archive kurgan floors in [thresholds_by_markup.sp_or] / sp_focal2
    for group, taus in ARCHIVE_SP_TAUS.items():
        for cls, val in taus.items():
            # Replace only within that group's values — line-based for kurgan keys after a matching header is fragile;
            # do global class lines carefully scoped by rewriting the two known FR1h-swept numbers when under group.
            pat = rf"(?ms)(\[thresholds_by_markup\.{re.escape(group)}\][^\[]*?)(^{re.escape(cls)}\s*=\s*)[0-9.]+"
            text2, n = re.subn(pat, rf"\g<1>\g<2>{val}", text, count=1)
            if n:
                text = text2
    toml.write_text(text, encoding="utf-8")
    # Verify
    try:
        import tomllib
        cfg = tomllib.loads(text)
    except Exception:
        cfg = {}
    fp = (cfg.get("post") or {}).get("frame_prior")
    info = {
        "flag": "post.frame_prior",
        "default_in_zip": "ON (section present)",
        "forced": "OFF",
        "had_section_before": before_has,
        "frame_prior_after": fp,
        "archive_taus_restored": ARCHIVE_SP_TAUS,
    }
    if fp:
        raise RuntimeError(f"FR1h force-off failed; frame_prior still truthy: {fp!r}")
    return info


def apply_r3_overrides(repo: Path) -> dict:
    """budget_s=100000, watchdog=false; env applied at run_main."""
    toml = repo / "configs" / "default.toml"
    text = toml.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^budget_s\s*=\s*\d+", "budget_s = 100000", text)
    text = re.sub(r"(?m)^watchdog\s*=\s*true", "watchdog = false", text)
    # pixgbm section budget if present
    if re.search(r"(?ms)^\[geophys\.pixgbm\].*?^budget_s\s*=", text):
        text = re.sub(
            r"(?ms)(^\[geophys\.pixgbm\][^\[]*?)^budget_s\s*=\s*\S+",
            r"\1budget_s = 100000",
            text,
            count=1,
        )
    toml.write_text(text, encoding="utf-8")
    return {"budget_s": 100000, "watchdog": False}


def detect_pix_impl(repo: Path) -> dict:
    """Validity check: which pix implementation the tree will actually run."""
    info: dict = {"repo": str(repo)}
    gp = repo / "expds" / "geo_pixgbm.py"
    pc = list((repo / "models").rglob("finds.txt")) if (repo / "models").is_dir() else []
    pix_clean = any("pix_clean" in str(p) for p in pc)
    pix_all = any("pixgbm/all" in str(p).replace("\\", "/") or p.parent.name == "all" for p in pc)
    info["has_geo_pixgbm"] = gp.is_file()
    if gp.is_file():
        info["geo_pixgbm_sha"] = sha256_file(gp)
        info["geo_pixgbm_matches_v2"] = info["geo_pixgbm_sha"] == GEO_PIX_SHA
    info["has_pix_clean_models"] = pix_clean
    info["has_pixgbm_all_models"] = pix_all
    try:
        import tomllib
        cfg = tomllib.loads((repo / "configs" / "default.toml").read_text(encoding="utf-8"))
        pix = (cfg.get("geophys") or {}).get("pixgbm") or {}
        info["pixgbm_enabled"] = bool(pix.get("enabled"))
        info["pixgbm_model_dir"] = pix.get("model_dir")
        info["pixgbm_impl_toml"] = pix.get("impl")  # clean packs may set impl=clean
    except Exception as e:
        info["toml_error"] = type(e).__name__
    if pix_clean and not gp.is_file():
        info["pix_implementation"] = "pix_clean"
    elif gp.is_file() and info.get("geo_pixgbm_matches_v2"):
        info["pix_implementation"] = "geo_pixgbm (v2)"
    elif gp.is_file():
        info["pix_implementation"] = "geo_pixgbm"
    else:
        info["pix_implementation"] = "unknown"
    return info


def unpack_zip_like(src: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    # fr1h.zip.bin is a zip despite .bin suffix
    with zipfile.ZipFile(src) as zf:
        zf.extractall(dest)
    for cand in [dest, *dest.iterdir()]:
        if (cand / "inference" / "main.py").is_file():
            return cand
    raise FileNotFoundError(f"inference/main.py not found under {dest} from {src}")






def verify_fr1h_sha(bin_path: Path, sha_file: Path | None) -> dict:
    got = sha256_file(bin_path)
    exp = FR1H_ZIP_SHA
    if sha_file and sha_file.is_file():
        first = sha_file.read_text(encoding="utf-8").strip().split()[0]
        if re.fullmatch(r"[0-9a-fA-F]{64}", first):
            exp = first.lower()
    ok = got == exp
    return {"path": str(bin_path), "sha256": got, "expected": exp, "ok": ok, "sha_file": str(sha_file) if sha_file else None}


def _repo_root_from(path: Path) -> Path | None:
    """Return directory containing inference/main.py (path or ancestor)."""
    p = path
    if p.is_file():
        p = p.parent
    for cand in [p, *p.parents]:
        if (cand / "inference" / "main.py").is_file():
            return cand
        # stop at input root-ish
        if cand == INPUT_ROOT or cand.resolve() == INPUT_ROOT.resolve():
            break
        if cand.name == "input" and cand.parent.name == "kaggle":
            break
    # one-level children
    if p.is_dir():
        for ch in sorted(p.iterdir()):
            if ch.is_dir() and (ch / "inference" / "main.py").is_file():
                return ch
    return None


def record_tree_key_shas(repo: Path) -> dict:
    """Log sha256 of key files in an extracted submission tree (no FAIL)."""
    keys = [
        "inference/main.py",
        "configs/default.toml",
        "expds/geo_pixgbm.py",
        "metadata.json",
    ]
    out: dict = {"repo": str(repo), "files": {}}
    for rel in keys:
        fp = repo / rel
        if fp.is_file():
            out["files"][rel] = sha256_file(fp)
    # models fingerprints
    models = []
    for p in sorted((repo / "models").rglob("*")) if (repo / "models").is_dir() else []:
        if p.is_file() and p.suffix.lower() in {".txt", ".json", ".pt", ".bin", ".lgb"}:
            models.append({"rel": str(p.relative_to(repo)), "sha256": sha256_file(p), "bytes": p.stat().st_size})
            if len(models) >= 30:
                break
    out["models_sample"] = models
    # optional per-file manifest inside tree
    for name in ("SHA256SUMS", "SHA256", "sha256.txt", "MANIFEST.sha256"):
        man = repo / name
        if man.is_file():
            out["manifest"] = str(man)
            break
        # also parent dataset root
        man2 = repo.parent / name
        if man2.is_file():
            out["manifest"] = str(man2)
            break
    return out


def find_cand_source(inputs: dict[str, Path]) -> dict | None:
    """Locate CAND as zip OR Kaggle-auto-extracted tree. Never require zip sha."""
    if os.environ.get("CAND_ZIP") and Path(os.environ["CAND_ZIP"]).is_file():
        return {"kind": "zip", "path": Path(os.environ["CAND_ZIP"])}
    if os.environ.get("CAND_TREE"):
        t = Path(os.environ["CAND_TREE"])
        root = _repo_root_from(t)
        if root:
            return {"kind": "tree", "path": root}

    ds_list = []
    for n in ("pixgbm-cr-cand", "pixgbm_cr_cand", "pix-clean", "pix_clean", "pixgbm-clean"):
        ds = find_named(inputs, n)
        if ds is not None:
            ds_list.append(ds)
    if not ds_list:
        ds_list = list(inputs.values())

    # Prefer extracted tree with inference/main.py
    for ds in ds_list:
        # direct
        root = _repo_root_from(ds)
        if root:
            return {"kind": "tree", "path": root}
        # named submission_* dirs
        for ch in sorted(ds.glob("submission_pixgbm_clean*")):
            root = _repo_root_from(ch)
            if root:
                return {"kind": "tree", "path": root}
        for ch in sorted(ds.rglob("inference/main.py")):
            return {"kind": "tree", "path": ch.parent.parent}

    # Fallback: zip still present
    for ds in ds_list:
        zips = sorted(ds.rglob("*.zip"))
        prefer = [z for z in zips if "pixgbm_clean" in z.name.lower() or "clean" in z.name.lower()]
        if prefer:
            return {"kind": "zip", "path": prefer[0]}
        if zips:
            return {"kind": "zip", "path": zips[0]}
    return None


def find_ref_source(inputs: dict[str, Path]) -> dict | None:
    """Locate REF as fr1h.zip.bin and/or auto-extracted tree."""
    ds = find_named(inputs, "fr1h-chk", "fr1h_chk", "koz3-fr1h", "fr1h")
    bin_path = find_file(inputs, "fr1h.zip.bin")
    tree = None
    search_roots = [ds] if ds is not None else list(inputs.values())
    for root in search_roots:
        if root is None:
            continue
        # extracted pack often has inference/main.py under dataset or subdir
        t = _repo_root_from(root)
        if t:
            tree = t
            break
        for ch in sorted(root.glob("*")):
            t = _repo_root_from(ch)
            if t:
                tree = t
                break
        if tree:
            break
    if bin_path is None and ds is not None:
        for p in ds.rglob("fr1h.zip.bin"):
            bin_path = p
            break
    if bin_path is None and tree is None:
        return None
    out: dict = {}
    if bin_path is not None:
        out["kind_bin"] = "zip.bin"
        out["bin"] = bin_path
        sha_file = None
        for name in ("SHA256", "SHA256SUMS", "sha256.txt"):
            p = bin_path.parent / name
            if p.is_file():
                sha_file = p
                break
        out["sha_file"] = sha_file
    if tree is not None:
        out["kind_tree"] = "tree"
        out["tree"] = tree
    out["kind"] = "bin" if bin_path is not None else "tree"
    return out


def resolve_repo_from_source(src: dict, dest: Path, *, role: str) -> tuple[Path, dict]:
    """Unpack zip/bin or reuse extracted tree into dest; return (repo_root, notes)."""
    notes: dict = {"role": role}
    if src.get("kind") == "tree" or (role == "cand" and src.get("kind") == "tree"):
        tree = src["path"] if role == "cand" else src.get("tree")
        notes["source_kind"] = "tree"
        notes["source"] = str(tree)
        notes["key_shas"] = record_tree_key_shas(tree)
        # Copy into working so we can mutate toml (FR1h off / r3 overrides)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(tree, dest, symlinks=False)
        repo = _repo_root_from(dest) or dest
        notes["repo"] = str(repo)
        return repo, notes

    if role == "ref":
        # Prefer bin when present; else tree
        if src.get("bin") is not None:
            bin_path: Path = src["bin"]
            work_bin = WORKING_ROOT / "fr1h.zip.bin"
            try:
                if (not work_bin.exists()) or work_bin.stat().st_size != bin_path.stat().st_size:
                    shutil.copy2(bin_path, work_bin)
                ver = verify_fr1h_sha(work_bin, src.get("sha_file"))
                notes["fr1h_sha"] = ver
                if not ver["ok"]:
                    notes["fr1h_sha_warn"] = "sha_mismatch_logged_not_fatal_if_tree"
                    # still try unpack if zip readable
                repo = unpack_zip_like(work_bin, dest)
                notes["source_kind"] = "zip.bin"
                notes["source"] = str(bin_path)
                notes["repo"] = str(repo)
                notes["key_shas"] = record_tree_key_shas(repo)
                return repo, notes
            except Exception as e:
                notes["bin_error"] = type(e).__name__
                if src.get("tree") is None:
                    raise
        if src.get("tree") is not None:
            tree = src["tree"]
            notes["source_kind"] = "tree"
            notes["source"] = str(tree)
            notes["key_shas"] = record_tree_key_shas(tree)
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(tree, dest, symlinks=False)
            repo = _repo_root_from(dest) or dest
            notes["repo"] = str(repo)
            return repo, notes
        raise FileNotFoundError("REF source unresolved")

    # cand zip
    zpath: Path = src["path"]
    notes["source_kind"] = "zip"
    notes["source"] = str(zpath)
    # optional: compare to dataset SHA256SUMS if present — log only, never FAIL
    for man in (zpath.parent / "SHA256SUMS", zpath.parent / "SHA256"):
        if man.is_file():
            notes["manifest"] = str(man)
            try:
                got = sha256_file(zpath)
                notes["zip_sha256"] = got
                for line in man.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if parts and re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
                        notes["manifest_sha"] = parts[0].lower()
                        notes["manifest_match"] = got == parts[0].lower()
                        break
            except Exception as e:
                notes["manifest_error"] = type(e).__name__
            break
    repo = unpack_zip_like(zpath, dest)
    notes["repo"] = str(repo)
    notes["key_shas"] = record_tree_key_shas(repo)
    return repo, notes




def list_geo_regions(tree: Path) -> list[str]:
    if not tree.is_dir():
        return []
    return sorted(
        p.name for p in tree.iterdir()
        if p.is_dir() and any(p.name.startswith(pref) for pref in GEO_PREFIXES)
    )


def merge_geojsons(parts: list[Path], dest: Path) -> int:
    feats = []
    for p in parts:
        if not p.is_file():
            continue
        try:
            fc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        feats.extend(fc.get("features") or [])
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(feats)


def run_main_resumable(
    repo: Path,
    tree: Path,
    result: Path,
    log_path: Path,
    tag: str,
    env_extra: dict | None = None,
) -> int:
    """Run inference per geo region with checkpointing (12h / ~20GB safety)."""
    regions = list_geo_regions(tree)
    if not regions:
        log("no geo regions in", tree)
        return 2
    ckpt_dir = OUT / f"ckpt_{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    done_file = ckpt_dir / "DONE.txt"
    parts: list[Path] = []
    env = os.environ.copy()
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "EXPDS_PIXGBM_STRICT": os.environ.get("EXPDS_PIXGBM_STRICT", "1"),
        "PIXGBM_BUDGET_S": os.environ.get("PIXGBM_BUDGET_S", "100000"),
        "EXPDS_SOFT_CEILING_S": os.environ.get("EXPDS_SOFT_CEILING_S", "0"),
    })
    if env_extra:
        env.update(env_extra)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"\n=== resumable {tag} n_regions={len(regions)} ts={utc_now()} ===\n")
        lf.flush()
        for i, name in enumerate(regions):
            part = ckpt_dir / f"{name}.geojson"
            parts.append(part)
            if part.is_file() and part.stat().st_size > 50:
                log(f"{tag} skip-exists {name} ({i+1}/{len(regions)})")
                lf.write(f"SKIP {name}\n"); lf.flush()
                continue
            # Single-region input via temp dir of symlinks
            stage = ckpt_dir / f"_stage_{name}"
            if stage.exists():
                shutil.rmtree(stage)
            stage.mkdir(parents=True)
            os.symlink(tree / name, stage / name)
            # Also expose sibling non-region helpers if pipeline expects root files — none for geo.
            cmd = [sys.executable, str(repo / "inference" / "main.py"), str(stage), str(part)]
            log(f"{tag} run {name} ({i+1}/{len(regions)})")
            lf.write(f"RUN {name} cmd={' '.join(cmd)}\n"); lf.flush()
            t1 = time.time()
            proc = subprocess.run(cmd, cwd=str(repo), stdout=lf, stderr=subprocess.STDOUT, env=env, text=True)
            lf.write(f"EC {name}={proc.returncode} wall={time.time()-t1:.1f}s exists={part.is_file()}\n"); lf.flush()
            shutil.rmtree(stage, ignore_errors=True)
            if not part.is_file():
                # write empty so merge can proceed; mark failure
                part.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
                write_status("FAILED", phase=f"infer_{tag}", region=name, exit=proc.returncode)
                return proc.returncode or 4

    n = merge_geojsons(parts, result)
    done_file.write_text("\n".join(regions) + f"\n# features={n}\n", encoding="utf-8")
    log(f"{tag} merged features={n} → {result}")
    return 0


def run_gate(helpers: Path, scores: Path) -> int:
    gate = helpers / "bot6_r3_gate.py"
    cand = OUT / "out_cand"
    newp = OUT / "out_new"
    if cand.is_dir() and not newp.exists():
        try:
            newp.symlink_to(cand)
        except OSError:
            shutil.copytree(cand, newp)
    cmd = [
        sys.executable, str(gate),
        "--job", str(OUT),
        "--scores", str(scores),
        "--tol", str(TOL),
        "--log-ref", str(OUT / "out_ref" / "main.log"),
        "--log-new", str(OUT / "out_new" / "main.log"),
    ]
    log("gate", cmd)
    r = subprocess.run(cmd, capture_output=True, text=True)
    (OUT / "notes" / "gate_stdout.txt").write_text(r.stdout or "", encoding="utf-8")
    (OUT / "notes" / "gate_stderr.txt").write_text(r.stderr or "", encoding="utf-8")
    gj = OUT / "gate.json"
    if gj.is_file():
        shutil.copy2(gj, OUT / "gate_report.json")
    return r.returncode


def build_gt(helpers: Path, repo: Path, tree: Path, gt_out: Path) -> int:
    script = helpers / "bot6_r3_build_gt.py"
    if not script.is_file():
        return 18
    region_list = tree / "_region_list.txt"
    geos = list_geo_regions(tree)
    region_list.write_text("\n".join(geos) + "\n", encoding="utf-8")
    cmd = [sys.executable, str(script), "--repo", str(repo), "--root", str(tree),
           "--out", str(gt_out), "--regions-from", str(region_list)]
    log("build_gt", cmd)
    r = subprocess.run(cmd, capture_output=True, text=True)
    (OUT / "notes" / "build_gt.log").write_text((r.stdout or "") + "\n" + (r.stderr or ""), encoding="utf-8")
    return r.returncode


def find_score_one(repo: Path) -> Path | None:
    for cand in (
        repo / "score" / "score_one.py",
        repo / "materials" / "compute_metrics.py",
        repo / "scripts" / "e2e_score.py",
    ):
        if cand.is_file():
            return cand
    return None


def score_pair(repo: Path, gt: Path, ref_gj: Path, cand_gj: Path, scores_out: Path) -> int:
    score_one = find_score_one(repo)
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    if score_one and score_one.name == "score_one.py":
        trials = [
            [sys.executable, str(score_one), str(ref_gj), str(cand_gj), str(gt), str(scores_out)],
            [sys.executable, str(score_one), "--ref", str(ref_gj), "--new", str(cand_gj), "--gt", str(gt), "--out", str(scores_out)],
        ]
        for cmd in trials:
            log("score try", cmd)
            r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
            if r.returncode == 0 and scores_out.is_file():
                return 0
            (OUT / "notes" / "score_try.log").parent.mkdir(parents=True, exist_ok=True)
            with (OUT / "notes" / "score_try.log").open("a", encoding="utf-8") as f:
                f.write(f"cmd={cmd}\nec={r.returncode}\nstdout={r.stdout[:2000]}\nstderr={r.stderr[:2000]}\n\n")
        return 19
    # Fallback: embedded e2e_score.py + compute_metrics.py (same as PC r3 runner).
    sdir = helpers_root() / "scorer"
    e2e, cm = sdir / "e2e_score.py", sdir / "compute_metrics.py"
    if e2e.is_file() and cm.is_file():
        cmd = [sys.executable, str(e2e), "--metric", str(cm), "--gt", str(gt),
               "--a", f"pix_ref={ref_gj}", "--b", f"pix_new={cand_gj}", "--out", str(scores_out)]
        log("score e2e", cmd)
        r = subprocess.run(cmd, capture_output=True, text=True)
        (OUT / "notes" / "score_e2e.log").write_text((r.stdout or "")[-20000:] + "\n" + (r.stderr or "")[-20000:], encoding="utf-8")
        if r.returncode == 0 and scores_out.is_file():
            return 0
        log("score e2e failed", r.returncode, (r.stderr or "")[-1500:])
    for p in INPUT_ROOT.rglob("scores.json") if INPUT_ROOT.is_dir() else []:
        shutil.copy2(p, scores_out)
        return 0
    write_status("FAILED", reason="no_score_tool")
    return 19


def helpers_root() -> Path:
    return materialize_embedded_helpers()



def seed_prior_outputs() -> dict:
    """Resume without re-inference: if a previous run's output is attached under
    /kaggle/input (kernel output or a dataset made from it), copy
    pixgbm_cr/out_{ref,cand}/result.geojson (+main.log) and gt.geojson into OUT."""
    got = {}
    if os.environ.get("PIXGBM_CR_NO_SEED", "").strip() in ("1", "true", "yes") or not INPUT_ROOT.is_dir():
        return got
    for role in ("ref", "cand"):
        for src in sorted(INPUT_ROOT.rglob(f"out_{role}/result.geojson")):
            dst_dir = OUT / f"out_{role}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / "result.geojson")
            lg = src.parent / "main.log"
            if lg.is_file():
                shutil.copy2(lg, dst_dir / "main.log")
            got[role] = {"src": str(src), "sha256": sha256_file(src)}
            break
    for src in sorted(INPUT_ROOT.rglob("gt.geojson")):
        if "pixgbm_cr" in str(src):
            shutil.copy2(src, OUT / "gt.geojson")
            got["gt"] = {"src": str(src), "sha256": sha256_file(src)}
            break
    if got:
        log("seeded from prior outputs:", {k: v["src"] for k, v in got.items()})
    return got

def main() -> int:
    if os.environ.get("PIXGBM_CR_DRY_RUN", "").strip() in ("1", "true", "yes") or (
        len(sys.argv) > 1 and sys.argv[1] in ("--syntax", "--dry-run", "syntax", "dry-run")
    ):
        return dry_run_syntax()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "notes").mkdir(parents=True, exist_ok=True)
    (OUT / "out_ref").mkdir(parents=True, exist_ok=True)
    (OUT / "out_cand").mkdir(parents=True, exist_ok=True)
    write_status("RUNNING", phase="start")
    try:
        helpers = ensure_helpers()
        inputs = find_inputs()
        wheeled = ensure_offline_wheels()
        log("inputs", sorted(inputs), "wheels", wheeled)

        # --- S3 geo tree ---
        write_status("RUNNING", phase="rebuild")
        rebuild = load_rebuild()
        report = rebuild.rebuild(dest=TREE)
        (OUT / "notes" / "rebuild_report.json").write_text(
            json.dumps({k: v for k, v in report.items() if k != "keys"}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        log("rebuild", {k: report.get(k) for k in ("n_geo_dirs", "n_geo_raster_files_guess", "hollow")})
        if report.get("hollow"):
            write_status("FAILED", phase="rebuild", reason="hollow_geo_tree", exit=2)
            return 2

        # --- REF: fr1h.zip.bin and/or auto-extracted tree ---
        write_status("RUNNING", phase="prep_ref")
        ref_src = find_ref_source(inputs)
        if ref_src is None:
            write_status("FAILED", reason="INVALID_missing_ref_fr1h")
            return 3
        ref_repo, ref_notes = resolve_repo_from_source(ref_src, OUT / "unpack_ref", role="ref")
        (OUT / "notes" / "ref_source.json").write_text(
            json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in ref_notes.items()}, indent=2) + chr(10),
            encoding="utf-8",
        )
        ver = ref_notes.get("fr1h_sha") or {"ok": None, "note": "no_bin_or_sha_skipped"}
        if isinstance(ver, dict) and ver.get("ok") is True:
            log("fr1h sha OK", str(ver.get("sha256", ""))[:16] + "…")
        elif isinstance(ver, dict) and ver.get("ok") is False:
            log("WARN fr1h sha mismatch (continuing if tree usable)", ver)
        else:
            log("REF source", ref_notes.get("source_kind"), ref_notes.get("source"))

        fr1h_info = force_fr1h_off(ref_repo)
        r3_ref = apply_r3_overrides(ref_repo)
        ref_impl = detect_pix_impl(ref_repo)
        (OUT / "notes" / "fr1h_force_off.json").write_text(json.dumps(fr1h_info, indent=2) + chr(10), encoding="utf-8")
        (OUT / "notes" / "ref_impl.json").write_text(json.dumps(ref_impl, indent=2) + chr(10), encoding="utf-8")
        log("REF pix_implementation:", ref_impl.get("pix_implementation"), "FR1h:", fr1h_info["forced"])
        if not ref_impl.get("geo_pixgbm_matches_v2"):
            write_status("FAILED", reason="INVALID_ref_geo_pixgbm_not_v2", impl=ref_impl)
            return 3
        if ref_impl.get("pix_implementation") != "geo_pixgbm (v2)":
            write_status("FAILED", reason="INVALID_ref_impl", impl=ref_impl)
            return 3

        # --- CAND: zip OR Kaggle-auto-extracted tree ---
        write_status("RUNNING", phase="prep_cand")
        cand_src = find_cand_source(inputs)
        if cand_src is None:
            write_status("FAILED", reason="INVALID_missing_cand")
            return 3
        cand_repo, cand_notes = resolve_repo_from_source(cand_src, OUT / "unpack_cand", role="cand")
        (OUT / "notes" / "cand_source.json").write_text(
            json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in cand_notes.items()}, indent=2) + chr(10),
            encoding="utf-8",
        )
        log("CAND source", cand_notes.get("source_kind"), cand_notes.get("source"))
        r3_cand = apply_r3_overrides(cand_repo)
        # Prefer impl=clean if toml supports it
        toml = cand_repo / "configs" / "default.toml"
        if toml.is_file():
            t = toml.read_text(encoding="utf-8")
            if re.search(r"(?m)^impl\s*=", t):
                t = re.sub(r"(?m)^impl\s*=\s*\S+", 'impl = "clean"', t, count=1)
            else:
                t = re.sub(r"(?m)^(\[geophys\.pixgbm\]\s*\n)", r'\1impl = "clean"\n', t, count=1)
            # point model_dir at pix_clean if present
            if (cand_repo / "models" / "pix_clean").is_dir():
                if re.search(r"(?m)^model_dir\s*=", t):
                    t = re.sub(r'(?m)^model_dir\s*=\s*\S+', 'model_dir = "models/pix_clean"', t, count=1)
            toml.write_text(t, encoding="utf-8")
        cand_impl = detect_pix_impl(cand_repo)
        (OUT / "notes" / "cand_impl.json").write_text(json.dumps(cand_impl, indent=2) + "\n", encoding="utf-8")
        log("CAND pix_implementation:", cand_impl.get("pix_implementation"))
        if "pix_clean" not in str(cand_impl.get("pix_implementation")):
            # soft warn — clean zip may still vendor geo_pixgbm-less pix via impl
            log("WARN CAND impl detection:", cand_impl)

        meta = {
            "tol": TOL,
            "fr1h_sha": ver,
            "fr1h_force_off": fr1h_info,
            "r3_overrides": {"ref": r3_ref, "cand": r3_cand},
            "ref_impl": ref_impl,
            "cand_impl": cand_impl,
            "rebuild": {k: report.get(k) for k in ("n_geo_dirs", "n_geo_raster_files_guess", "hollow", "geo_region_dirs")},
            "runtime_estimate_note": (
                "Prior FR1h black-box geo 200–217 ≈ 267s on 6-cpu official image; "
                "Kaggle CPU ~4 vCPU ×2 runs (v2 geo_pixgbm + pix_clean ens6) ≈ 1–3 h typical, "
                "worst-case budget still ≪ 12 h with per-region checkpoints under pixgbm_cr/ckpt_*."
            ),
            "docker_ec": {},
            "t_start_epoch": {},
        }

        ref_gj = OUT / "out_ref" / "result.geojson"
        cand_gj = OUT / "out_cand" / "result.geojson"

        seeded = seed_prior_outputs()
        meta["seeded_from_prior"] = seeded
        write_status("RUNNING", phase="infer_ref")
        meta["t_start_epoch"]["ref"] = int(time.time())
        if "ref" in seeded and ref_gj.is_file():
            log("REF result seeded from prior run, skip inference")
            ec_ref = 0
        else:
            ec_ref = run_main_resumable(ref_repo, TREE, ref_gj, OUT / "out_ref" / "main.log", "ref")
        meta["docker_ec"]["ref"] = ec_ref
        if ec_ref != 0 or not ref_gj.is_file():
            write_status("FAILED", phase="infer_ref", exit=ec_ref)
            return 4

        write_status("RUNNING", phase="infer_cand")
        meta["t_start_epoch"]["new"] = int(time.time())
        if "cand" in seeded and cand_gj.is_file():
            log("CAND result seeded from prior run, skip inference")
            ec_cand = 0
        else:
            ec_cand = run_main_resumable(cand_repo, TREE, cand_gj, OUT / "out_cand" / "main.log", "cand")
        meta["docker_ec"]["new"] = ec_cand
        if ec_cand != 0 or not cand_gj.is_file():
            write_status("FAILED", phase="infer_cand", exit=ec_cand)
            return 5

        write_status("RUNNING", phase="gt_score_gate")
        repo_for_tools = None
        for cand in (OUT / "unpack_ref", OUT / "unpack_cand"):
            if (cand / "expds" / "catalog.py").is_file():
                repo_for_tools = cand
                break
        gt_out = OUT / "gt.geojson"
        if repo_for_tools is not None and not gt_out.is_file():
            gec = build_gt(helpers, repo_for_tools, TREE, gt_out)
            if gec != 0 or not gt_out.is_file():
                log("WARN build_gt failed", gec)

        scores = OUT / "scores.json"
        if repo_for_tools is not None and gt_out.is_file():
            sec = score_pair(repo_for_tools, gt_out, ref_gj, cand_gj, scores)
        else:
            sec = 19
            for p in INPUT_ROOT.rglob("scores.json") if INPUT_ROOT.is_dir() else []:
                shutil.copy2(p, scores)
                sec = 0
                break

        if ref_gj.is_file() and cand_gj.is_file():
            hr = sha256_file(ref_gj)
            hn = sha256_file(cand_gj)
            meta["result_sha"] = {"ref": hr, "new": hn}
            meta["result_identical"] = hr == hn

        (OUT / "notes" / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if sec != 0 or not scores.is_file():
            write_status("FAILED", phase="score", exit=sec, reason="scores_missing")
            return 19

        gec = run_gate(helpers, scores)
        if gec == 0:
            write_status("DONE", verdict="PASS", gate_exit=0, tol=TOL,
                         ref_impl=ref_impl.get("pix_implementation"),
                         cand_impl=cand_impl.get("pix_implementation"))
            return 0
        if gec == 9:
            write_status("DONE", verdict="FAIL_EQUIV", gate_exit=9, tol=TOL)
            return 9
        if gec in (11, 12, 13, 14):
            write_status("DONE", verdict="INVALID", gate_exit=gec, tol=TOL)
            return gec
        write_status("FAILED", verdict="GATE_ERROR", gate_exit=gec)
        return gec if gec else 1
    except Exception as e:
        write_status("FAILED", error=repr(e)[:300])
        (OUT / "notes" / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
