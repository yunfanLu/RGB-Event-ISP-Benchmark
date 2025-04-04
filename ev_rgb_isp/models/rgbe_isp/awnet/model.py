import torch
import torch.nn as nn
import torch.nn.functional as F

from ev_rgb_isp.datasets.basic_batch import EventRAWISPBatch as EBC
from ev_rgb_isp.functions.raw_tools import quad_bayes_to_bayes
from ev_rgb_isp.models.rgbe_isp.awnet.model_3channel import AWNet3
from ev_rgb_isp.models.rgbe_isp.awnet.model_4channel import AWNet4


def convert_img(raws):
    ch_Gb = raws[:, 0, 1::2, 1::2]
    ch_R = raws[:, 0, 0::2, 1::2]
    ch_Gr = raws[:, 0, 0::2, 0::2]
    ch_B = raws[:, 0, 1::2, 0::2]

    img1 = torch.stack((ch_B, ch_Gb, ch_R, ch_Gr), dim=1)
    # print(f"img1:{img1.shape}")

    scale_factor = (2, 2)
    de_B = F.interpolate(ch_B.unsqueeze(1), scale_factor=scale_factor, mode="nearest")
    de_Gb = F.interpolate(ch_Gb.unsqueeze(1), scale_factor=scale_factor, mode="nearest")
    de_R = F.interpolate(ch_R.unsqueeze(1), scale_factor=scale_factor, mode="nearest")
    de_Gr = F.interpolate(ch_Gr.unsqueeze(1), scale_factor=scale_factor, mode="nearest")

    # de_B = F.interpolate(ch_B, scale_factor=scale_factor, mode='nearest')
    # de_Gb = F.interpolate(ch_Gb, scale_factor=scale_factor, mode='nearest')
    # de_R = F.interpolate(ch_R, scale_factor=scale_factor, mode='nearest')
    # de_Gr = F.interpolate(ch_Gr, scale_factor=scale_factor, mode='nearest')
    # print(f"de_B:{de_B.shape}")
    de_G = (de_Gb + de_Gr) / 2

    img2 = torch.cat((de_R, de_G, de_B), dim=1)
    # print(f"img2:{img2.shape}")

    return img1, img2


class AWNet(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.awnet3 = AWNet3(3, 3, block=[3, 3, 3, 4, 4])
        self.awnet4 = AWNet4(4, 3, block=[3, 3, 3, 4, 4])

    def forward(self, batch):
        raw = batch[EBC.RAW_TENSORS]  # N, 1, H, W
        x = quad_bayes_to_bayes(raw)
        img1, img2 = convert_img(x)
        out1, _ = self.awnet4(img1)
        out2, _ = self.awnet3(img2)
        out = torch.zeros_like(out1[0])
        out = (out1[0] + out2[0]) / 2

        batch[EBC.PREDICTION] = out
        return batch
