"""Письмо организаторов 06.11: веса DINOv3 в архиве только вместе с копией лицензии."""
import importlib.util
import zipfile

from conftest import ROOT


def _load():
    spec = importlib.util.spec_from_file_location("check_submission_dino", ROOT / "scripts" / "check_submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _zip(path, names):
    with zipfile.ZipFile(path, "w") as z:
        for name in names:
            z.writestr(name, b"x")
    return zipfile.ZipFile(path)


def test_archive_without_dinov3_passes(tmp_path):
    mod = _load()
    mod.RESULTS.clear()
    with _zip(tmp_path / "a.zip", ["models/dinov2-small/model.safetensors", "expds/gor_gate.py"]) as z:
        mod.check_dinov3_license(z)
    assert mod.RESULTS[-1][0] == "PASS"


def test_dinov3_weights_without_license_fail(tmp_path):
    mod = _load()
    mod.RESULTS.clear()
    with _zip(tmp_path / "b.zip", ["models/dinov3/vitl.safetensors"]) as z:
        mod.check_dinov3_license(z)
    assert mod.RESULTS[-1][0] == "FAIL"


def test_comment_about_dinov3_is_not_a_loader(tmp_path):
    mod = _load()
    mod.RESULTS.clear()
    note = "x = 1  # письмо 06.11: DINOv3 в эту сдачу не входит\n"
    with zipfile.ZipFile(tmp_path / "d.zip", "w") as z:
        z.writestr("expds/gor_gate.py", note)
    with zipfile.ZipFile(tmp_path / "d.zip") as z:
        mod.check_dinov3_license(z)
    assert mod.RESULTS[-1][0] == "PASS"


def test_loader_call_without_license_fails(tmp_path):
    mod = _load()
    mod.RESULTS.clear()
    src = 'm = net.from_pretrained("facebook/dinov3-vitl16")\n'
    with zipfile.ZipFile(tmp_path / "e.zip", "w") as z:
        z.writestr("expds/load_dino.py", src)
    with zipfile.ZipFile(tmp_path / "e.zip") as z:
        mod.check_dinov3_license(z)
    assert mod.RESULTS[-1][0] == "FAIL"


def test_dinov3_weights_with_license_pass(tmp_path):
    mod = _load()
    mod.RESULTS.clear()
    with _zip(tmp_path / "c.zip", ["models/dinov3/vitl.pt", "third_party/DINOv3_LICENSE.txt"]) as z:
        mod.check_dinov3_license(z)
    assert mod.RESULTS[-1][0] == "PASS"
