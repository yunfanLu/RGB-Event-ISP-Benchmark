import torch
from absl.logging import info
from torch import nn

from ev_rgb_isp.losses.image_loss import RGBEISPLoss
from ev_rgb_isp.losses.lpips import LPIPS
from ev_rgb_isp.losses.psnr import _PSNR
from ev_rgb_isp.losses.ssim import SSIM


def get_single_metric(config):
    if config.NAME == "empty":
        return lambda x: 0.0
    # vfi + sr
    elif "rgbe_isp" in config.NAME:
        return RGBEISPLoss(config)
    # other
    elif config.NAME == "empty":
        return EmptyMetric(config)
    else:
        raise ValueError(f"Unknown metric: {config.NAME}")


class EmptyMetric(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        info(f"EmptyMetric:")
        info(f"  config: {config}")

    def forward(self, batch):
        return torch.tensor(0.0, requires_grad=True)


class MixedMetric(nn.Module):
    def __init__(self, configs):
        super(MixedMetric, self).__init__()
        self.metric = []
        self.eval = []
        for config in configs:
            self.metric.append(config.NAME)
            self.eval.append(get_single_metric(config))
        info(f"Init Mixed Metric: {configs}")

    def forward(self, batch):
        r = []
        for m, e in zip(self.metric, self.eval):
            r.append((m, e(batch)))
        return r
