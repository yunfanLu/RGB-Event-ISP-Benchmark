import torch
from absl.logging import info
from torch.nn.modules.loss import _Loss

from ev_rgb_isp.losses.event_loss import GlobalShutterDifferentialReconstructedLoss
from ev_rgb_isp.losses.image_loss import (
    GlobalShutterReconstructedLoss,
    L1CharbonnierLossColor,
    RGBEISPLoss,
    RollingShutterBlurReconstructedLoss,
)


def get_single_loss(config):
    if "rgbe_isp" in config.NAME:
        return RGBEISPLoss(config)
    else:
        raise ValueError(f"Unknown loss: {config.NAME}")


class EmptyLoss(_Loss):
    def __init__(self, config=None):
        super().__init__()
        info(f"Empty Loss:")
        info(f"  config:{config}")

    def forward(self, batch):
        return torch.tensor(0.0, requires_grad=True)


class MixedLoss(_Loss):
    def __init__(self, configs):
        super(MixedLoss, self).__init__()
        self.loss = []
        self.weight = []
        self.criterion = []
        for item in configs:
            self.loss.append(item.NAME)
            self.weight.append(item.WEIGHT)
            self.criterion.append(get_single_loss(item))
        info(f"Init Mixed Loss: {configs}")

    def forward(self, batch):
        name_to_loss = []
        total = 0
        for n, w, fun in zip(self.loss, self.weight, self.criterion):
            tmp = fun(batch)
            name_to_loss.append((n, tmp))
            total = total + tmp * w
        return total, name_to_loss
