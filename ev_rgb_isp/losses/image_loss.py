import torch
from torch.nn.modules.loss import _Loss

from ev_rgb_isp.losses.lpips import LPIPS
from ev_rgb_isp.losses.psnr import _PSNR
from ev_rgb_isp.losses.ssim import SSIM


class L1CharbonnierLossColor(_Loss):
    def __init__(self):
        super(L1CharbonnierLossColor, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        diff = torch.add(x, -y)
        diff_sq = diff * diff
        error = torch.sqrt(diff_sq + self.eps)
        loss = torch.mean(error)
        return loss


class RGBEISPLoss(_Loss):
    def __init__(self, config):
        super(RGBEISPLoss, self).__init__()
        loss_type = config.NAME.split("-")[-1]
        if loss_type == "l1":
            self.loss = torch.nn.L1Loss()
        elif loss_type == "l2":
            self.loss = torch.nn.MSELoss()
        elif loss_type == "charbonnier":
            self.loss = L1CharbonnierLossColor()
        elif loss_type == "PSNR":
            self.loss = _PSNR()
        elif loss_type == "SSIM":
            self.loss = SSIM()
        elif loss_type == "LPIPS":
            self.loss = LPIPS()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        from ev_rgb_isp.datasets.basic_batch import EventRAWISPBatch

        self.BC = EventRAWISPBatch

    def forward(self, batch):
        # B, N, C, H, W
        good_rgb = batch[self.BC.GROUND_TRUTH]
        # B, N, C, H, W
        pred_frames = batch[self.BC.PREDICTION]
        loss = self.loss(good_rgb, pred_frames)
        return loss
