import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(1)

from centerhead.d8 import d8_array, d8_vec, torch_untta_offsets      # noqa: E402
from centerhead.head import CenterHead, UnetCenterNet                 # noqa: E402
from centerhead.losses import center_loss, focal_loss_logits, valid_to_stride          # noqa: E402
from centerhead.targets import render_targets                         # noqa: E402


def _naive_focal(x, g):
    p = torch.sigmoid(x.double())
    pos = (g >= 1).double()
    l = pos * (1 - p) ** 2 * torch.log(p) + (1 - pos) * (1 - g) ** 4 * p ** 2 * torch.log(1 - p)
    return -l.sum() / pos.sum().clamp(min=1)


def test_focal_matches_naive_and_is_stable():
    torch.manual_seed(0)
    g = torch.from_numpy(render_targets(np.array([[8.2, 9.7, 10, 0], [20.5, 3.3, 6, 1]]), 32, 32)["hm"])[None]
    x = torch.randn(1, 2, 32, 32) * 3
    assert torch.allclose(focal_loss_logits(x, g).double(), _naive_focal(x, g), rtol=1e-4)
    x = torch.full((1, 2, 32, 32), 80.0)
    x[0, 0, 9, 8] = -80.0
    l = focal_loss_logits(x.half(), g)
    assert torch.isfinite(l) and l > 0
    xx = torch.zeros(1, 2, 32, 32, requires_grad=True)
    (focal_loss_logits(xx * 1e4, g)).backward()
    assert torch.isfinite(xx.grad).all()


def test_valid_mask_and_no_positives():
    g = torch.zeros(1, 2, 8, 8)
    x = torch.zeros(1, 2, 8, 8)
    v = torch.zeros(1, 8, 8)
    assert focal_loss_logits(x, g, v) == 0
    assert focal_loss_logits(x, g) > 0                               # норма clamp(min=1)


def test_head_shapes_and_bias():
    h = CenterHead(16)
    o = h(torch.zeros(2, 16, 24, 24))
    assert torch.allclose(h.hm[-1].bias, torch.full((2,), -2.19))
    assert o["hm"].shape == (2, 2, 24, 24) and o["wh"].shape == (2, 1, 24, 24) and o["off"].shape == (2, 2, 24, 24)


def test_toy_overfit_loss_decreases():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    S = 32
    objs = np.column_stack([rng.uniform(4, S - 4, (4, 2)), rng.uniform(4, 8, 4), [0, 1, 0, 1]])
    t = render_targets(objs, S, S)
    batch = {k: torch.from_numpy(np.ascontiguousarray(v))[None] for k, v in t.items()}
    batch["valid"] = torch.ones(1, S, S)
    feat = torch.zeros(1, 4, S, S)
    feat[0, :2] = torch.from_numpy(t["hm"])                          # «идеальные» признаки: задача выучиваема
    feat[0, 2:] = torch.from_numpy(t["off"]) * 4
    net = torch.nn.Sequential(torch.nn.Conv2d(4, 16, 3, padding=1), torch.nn.ReLU())
    head = CenterHead(16, head_ch=16)
    opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), 1e-2)
    hist = []
    for _ in range(150):
        res = center_loss(head(net(feat)), batch)
        loss = res["loss_center"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(float(loss.detach()))
    assert hist[-1] < 0.25 * hist[0], (hist[0], hist[-1])
    assert set(res) == {"loss_center", "loss_hm", "loss_wh", "loss_off"}


@pytest.mark.parametrize("k", range(4))
@pytest.mark.parametrize("fl", [False, True])
def test_tta_offset_inverse(k, fl):
    rng = np.random.default_rng(k)
    off = rng.normal(size=(2, 8, 8)).astype(np.float32)
    # «сеть-эквивариант»: на входе rot+flip выдаёт карту смещений в повернутой системе
    fwd = d8_array(off, k, fl)
    fwd = np.moveaxis(d8_vec(np.moveaxis(fwd, 0, -1), k, fl), -1, 0)
    back = torch_untta_offsets(torch.from_numpy(np.ascontiguousarray(fwd, np.float32))[None], k, fl)[0].numpy()
    np.testing.assert_allclose(back, off, atol=1e-6)


@pytest.mark.parametrize("ch,stride", [(3, 4), (5, 4), (5, 1), (12, 4)])
def test_unet_wrapper_smp(ch, stride):
    smp = pytest.importorskip("segmentation_models_pytorch")
    pytest.importorskip("timm")
    unet = smp.Unet("tu-convnext_tiny", encoder_weights=None, in_channels=ch, classes=3)
    m = UnetCenterNet(unet, ch, stride=stride).eval()
    n = 128
    with torch.no_grad():
        o = m(torch.zeros(1, ch, n, n))
    h = n // stride
    assert o["hm"].shape == (1, 2, h, h) and o["mask"].shape == (1, 3, n, n)
    assert o["wh"].shape == (1, 1, h, h) and o["off"].shape == (1, 2, h, h)
    assert torch.allclose(m.center.hm[-1].bias, torch.full((2,), -2.19))
    print("feat_ch", ch, stride, m.feat_ch)


def test_valid_to_stride():
    v = torch.zeros(1, 8, 8)
    v[0, 0, 0] = 1
    assert valid_to_stride(v, 4).tolist() == [[[1.0, 0.0], [0.0, 0.0]]]
