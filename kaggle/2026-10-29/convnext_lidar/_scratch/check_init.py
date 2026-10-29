# проверка: smp tu-convnext_tiny.fb_in1k + encoder_weights=True грузит веса timm; scratch — нет; голова seg 5 каналов
import torch, timm, segmentation_models_pytorch as smp, warnings
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
ref = timm.create_model("convnext_tiny.fb_in1k", pretrained=True, features_only=True).state_dict()
for ch in (3, 5):
    for w, name in ((True, "tu-convnext_tiny.fb_in1k"), (None, "tu-convnext_tiny")):
        m = smp.Unet(name, encoder_weights=w, in_channels=ch, classes=5); sd = m.encoder.model.state_dict()
        common = [k for k in ref if k in sd and ref[k].shape == sd[k].shape]
        eq = sum(torch.equal(ref[k], sd[k]) for k in common)
        stem = [k for k in sd if "stem_0.weight" in k or k.endswith("stem.0.weight")]
        print(f"ch={ch} weights={w}: совпало с timm {eq}/{len(common)} тензоров; stem {stem[:1]} {tuple(sd[stem[0]].shape) if stem else ''};"
              f" out {tuple(m(torch.zeros(1, ch, 64, 64)).shape)}")
