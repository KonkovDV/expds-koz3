"""Unit tests for kaggle/common/koz3_s3.py (mock boto3; no network).

Loaded by file path so the pip ``kaggle`` package does not shadow the repo tree.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
_MOD_PATH = ROOT / "kaggle" / "common" / "koz3_s3.py"


def _load():
    spec = importlib.util.spec_from_file_location("koz3_s3_under_test", _MOD_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["koz3_s3_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load()

def _fake_transfer(monkeypatch):
    import types
    transfer = types.ModuleType("boto3.s3.transfer")
    transfer.TransferConfig = lambda **kw: object()
    s3mod = types.ModuleType("boto3.s3")
    s3mod.transfer = transfer
    b3 = types.ModuleType("boto3")
    b3.s3 = s3mod
    monkeypatch.setitem(sys.modules, "boto3", b3)
    monkeypatch.setitem(sys.modules, "boto3.s3", s3mod)
    monkeypatch.setitem(sys.modules, "boto3.s3.transfer", transfer)

FAKE_ID = "AKIA_TEST_ACCESS_KEY_XYZ"
FAKE_SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch, tmp_path):
    for k in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        m.SECRET_ID,
        m.SECRET_KEY,
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(m, "_INPUT_GLOB", str(tmp_path / "nosuch" / "**" / "*"))


def test_secret_names_match_dfine_or():
    assert m.SECRET_ID == "KOZ3_S3_ACCESS_KEY_ID"
    assert m.SECRET_KEY == "KOZ3_S3_SECRET_ACCESS_KEY"
    assert m.BUCKET == "train-expds-3"
    assert "s3.ru-7.storage.selcloud.ru" in m.ENDPOINT
    assert m.REGION == "ru-7"


def test_cred_from_json_id_key(tmp_path, monkeypatch):
    d = tmp_path / "koz3-s3-cred"
    d.mkdir()
    (d / "cred.json").write_text(json.dumps({"id": FAKE_ID, "key": FAKE_SECRET}))
    monkeypatch.setattr(m, "_INPUT_GLOB", str(d / "**" / "*"))
    kid, sec = m._load_credentials()
    assert kid == FAKE_ID and sec == FAKE_SECRET


def test_cred_from_json_aws_names(tmp_path, monkeypatch):
    d = tmp_path / "inp"
    d.mkdir()
    (d / "creds.json").write_text(
        json.dumps(
            {
                "aws_access_key_id": FAKE_ID,
                "aws_secret_access_key": FAKE_SECRET,
            }
        )
    )
    monkeypatch.setattr(m, "_INPUT_GLOB", str(d / "**" / "*"))
    kid, sec = m._load_credentials()
    assert kid == FAKE_ID and sec == FAKE_SECRET


def test_cred_from_env_file(tmp_path, monkeypatch):
    d = tmp_path / "inp"
    d.mkdir()
    (d / "credentials.env").write_text(
        f"AWS_ACCESS_KEY_ID={FAKE_ID}\nAWS_SECRET_ACCESS_KEY={FAKE_SECRET}\n"
    )
    monkeypatch.setattr(m, "_INPUT_GLOB", str(d / "**" / "*"))
    kid, sec = m._load_credentials()
    assert kid == FAKE_ID and sec == FAKE_SECRET


def test_cred_from_ini(tmp_path, monkeypatch):
    d = tmp_path / "inp"
    d.mkdir()
    (d / "credentials").write_text(
        "[default]\n"
        f"aws_access_key_id = {FAKE_ID}\n"
        f"aws_secret_access_key = {FAKE_SECRET}\n"
    )
    monkeypatch.setattr(m, "_INPUT_GLOB", str(d / "**" / "*"))
    kid, sec = m._load_credentials()
    assert kid == FAKE_ID and sec == FAKE_SECRET


def test_cred_from_environ(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", FAKE_ID)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", FAKE_SECRET)
    kid, sec = m._load_credentials()
    assert kid == FAKE_ID and sec == FAKE_SECRET


def test_missing_cred_raises_without_values():
    with pytest.raises(m.S3CredentialsError) as ei:
        m._load_credentials()
    msg = str(ei.value)
    assert FAKE_ID not in msg and FAKE_SECRET not in msg
    assert m.SECRET_ID in msg


def test_scrub_strips_secret_from_exception():
    class Boom(Exception):
        pass

    err = m._wrap(
        Boom(f"InvalidAccessKeyId {FAKE_ID} secret={FAKE_SECRET}"), FAKE_ID, FAKE_SECRET
    )
    assert isinstance(err, m.S3TransferError)
    assert FAKE_ID not in str(err)
    assert FAKE_SECRET not in str(err)
    assert "***" in str(err)


def test_list_keys_paginated():
    cli = mock.Mock()
    cli.list_objects_v2.side_effect = [
        {
            "Contents": [{"Key": "a/1.tif", "Size": 10}],
            "IsTruncated": True,
            "NextContinuationToken": "t2",
        },
        {
            "Contents": [{"Key": "a/2.tif", "Size": 20}],
            "IsTruncated": False,
        },
    ]
    got = m.list_keys("a/", client=cli)
    assert got == [("a/1.tif", 10), ("a/2.tif", 20)]
    assert cli.list_objects_v2.call_count == 2
    assert cli.list_objects_v2.call_args_list[1][1]["ContinuationToken"] == "t2"


def test_list_keys_failure_message_has_no_secrets():
    cli = mock.Mock()
    cli.list_objects_v2.side_effect = RuntimeError(
        f"403 SignatureDoesNotMatch key={FAKE_ID} secret={FAKE_SECRET}"
    )
    with pytest.raises(m.S3TransferError) as ei:
        m.list_keys("x/", client=cli)
    msg = str(ei.value)
    assert FAKE_SECRET not in msg
    assert FAKE_ID not in msg


def test_download_skips_existing(tmp_path):
    cli = mock.Mock()
    dst = tmp_path / "f.tif"
    dst.write_bytes(b"12345")
    out = m.download("p/f.tif", dst, client=cli, expected_size=5)
    assert out == dst
    cli.download_file.assert_not_called()


def test_download_retries_then_ok(tmp_path, monkeypatch):
    _fake_transfer(monkeypatch)
    monkeypatch.setattr(m.time, "sleep", lambda *_: None)
    cli = mock.Mock()
    calls = {"n": 0}

    def _dl(bucket, key, path, Config=None):  # noqa: N803
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError(f"timeout secret={FAKE_SECRET}")
        Path(path).write_bytes(b"abcd")

    cli.download_file.side_effect = _dl
    dst = tmp_path / "out.tif"
    out = m.download("p/out.tif", dst, client=cli, expected_size=4, retries=3)
    assert out.read_bytes() == b"abcd"
    assert calls["n"] == 2


def test_download_failure_strips_secrets(tmp_path, monkeypatch):
    _fake_transfer(monkeypatch)
    monkeypatch.setattr(m.time, "sleep", lambda *_: None)
    cli = mock.Mock()
    cli.download_file.side_effect = RuntimeError(
        f"AccessDenied id={FAKE_ID} key={FAKE_SECRET}"
    )
    with pytest.raises(m.S3TransferError) as ei:
        m.download(
            "p/x.tif", tmp_path / "x.tif", client=cli, expected_size=1, retries=2
        )
    assert FAKE_ID not in str(ei.value)
    assert FAKE_SECRET not in str(ei.value)


def test_download_prefix_preserves_rel(tmp_path, monkeypatch):
    _fake_transfer(monkeypatch)
    cli = mock.Mock()
    keys = [("geo/MagGr/200.tif", 3), ("geo/MagGr/201.tif", 4)]

    def _dl(bucket, key, path, Config=None):  # noqa: N803
        Path(path).write_bytes(b"x" * (3 if key.endswith("200.tif") else 4))

    cli.download_file.side_effect = _dl
    out = m.download_prefix("geo/MagGr/", tmp_path / "dst", client=cli, keys=keys)
    assert len(out) == 2
    assert (tmp_path / "dst" / "200.tif").stat().st_size == 3
    assert (tmp_path / "dst" / "201.tif").stat().st_size == 4


def test_s3_client_passes_endpoint(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", FAKE_ID)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", FAKE_SECRET)
    fake_client = object()
    fake_boto3 = mock.Mock()
    fake_boto3.client.return_value = fake_client
    fake_bc = mock.Mock()
    fake_bc.Config = mock.Mock(return_value=object())
    with mock.patch.dict(
        sys.modules, {"boto3": fake_boto3, "botocore": mock.Mock(), "botocore.config": fake_bc}
    ):
        cli = m.s3_client()
    assert cli is fake_client
    kwargs = fake_boto3.client.call_args.kwargs
    assert kwargs["endpoint_url"] == m.ENDPOINT
    assert kwargs["region_name"] == m.REGION
    assert kwargs["aws_access_key_id"] == FAKE_ID
    # secret passed to boto but must not leak via our API errors (covered elsewhere)
    assert kwargs["aws_secret_access_key"] == FAKE_SECRET
