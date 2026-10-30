"""Класс без источника не входит в Dice. Обычный вызов без веса прежний."""
import torch

from expds.train import absent_class_weight, dice_loss


def test_absent_class_drops_out_of_dice():
    logits = torch.zeros(2, 2, 4, 4)
    target = torch.zeros(2, 2, 4, 4)
    target[0, 0, 1, 1] = 1
    valid = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    weight = absent_class_weight(target, valid)
    loss = dice_loss(logits, target, weight=weight)
    only_present = dice_loss(logits[:, :1], target[:, :1])
    assert torch.allclose(loss, only_present)


def test_dice_without_weight_is_unchanged():
    logits = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 0, 0] = 1
    assert torch.allclose(dice_loss(logits, target), dice_loss(logits, target, weight=None))
