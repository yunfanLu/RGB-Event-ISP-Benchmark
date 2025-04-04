import torch
import torch.nn as nn
import torch.nn.functional as F

from ev_rgb_isp.datasets.basic_batch import EventRAWISPBatch as EBC
from ev_rgb_isp.models.demosaic.raw_to_rggb import quad_raw_to_rggb


class _SCN(nn.Module):
    def __init__(self):
        super(_SCN, self).__init__()
        self.W1 = nn.Conv2d(40, 128, 3, 1, 1, bias=False)
        self.S1 = nn.Conv2d(128, 40, 3, 1, 1, groups=1, bias=False)
        self.S2 = nn.Conv2d(40, 128, 3, 1, 1, groups=1, bias=False)
        self.shlu = nn.ReLU(True)

    def forward(self, input):
        x1 = input[:, range(0, 40), :, :]
        event_input = input[:, range(40, 80), :, :]

        x1 = torch.mul(x1, event_input)
        z = self.W1(x1)
        tmp = z
        for i in range(25):
            ttmp = self.shlu(tmp)
            x = self.S1(ttmp)
            x = torch.mul(x, event_input)
            x = torch.mul(x, event_input)
            x = self.S2(x)
            x = ttmp - x
            tmp = torch.add(x, z)
        c = self.shlu(tmp)
        return c


class ISPESLNet(nn.Module):
    def __init__(self, moments):
        super(ISPESLNet, self).__init__()
        self.scale = 2

        self.in_channel = 12
        self.moments = moments
        self.scn = nn.Sequential(_SCN())
        self.image_d = nn.Conv2d(
            in_channels=self.in_channel,
            out_channels=40,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.event_c1 = nn.Conv2d(
            in_channels=moments,
            out_channels=40,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.event_c2 = nn.Conv2d(
            in_channels=40,
            out_channels=40,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)
        if self.scale >= 2:
            self.shu1 = nn.Conv2d(
                in_channels=128,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.ps1 = nn.PixelShuffle(2)
        if self.scale == 4:
            self.shu2 = nn.Conv2d(
                in_channels=128,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.ps2 = nn.PixelShuffle(2)
        self.end_conv = nn.Conv2d(
            in_channels=128,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    # # 创建插值函数
    # def interpolate_along_axis(self, data, axis, new_size):
    #     if axis != 1:
    #         raise ValueError("Only channel (C) dimension interpolation is supported.")

    #     B, C, H, W = data.shape

    #     # # 交换轴顺序，把待插值的轴放到最后
    #     # data = data.permute(0, 2, 3, 1)

    #     # 插值
    #     data = F.interpolate(data, size=(new_size, H, W), mode='linear', align_corners=False)

    #     # # 恢复原来的轴顺序
    #     # data = data.permute(0, 3, 1, 2)

    #     return data

    def forward(self, batch):
        # Get a RGB E batch
        raw = batch[EBC.RAW_TENSORS]  # 1, 3, H, W
        raw = quad_raw_to_rggb(raw)  # 1, 12, H/2, W/2
        # info(f"PYNET raw shape: {raw.shape}")
        # B, 4, 3, H, W -> B, 12, H, W
        B, L, N, H, W = raw.shape
        raw = raw.reshape(B, L * N, H, W)
        event = batch[EBC.EVENTS_VOXEL_GRID]
        # event = self.interpolate_along_axis(event, axis=1, new_size=self.moments)
        # for inference
        x1 = raw
        # print(f"raw.shape:{x1.shape}")
        # exit()
        x1 = self.image_d(x1)

        event_out = self.event_c1(event)
        event_out = torch.sigmoid(event_out)
        event_out = self.event_c2(event_out)
        event_out = torch.sigmoid(event_out)
        scn_input = torch.cat([x1, event_out], 1)
        out = self.scn(scn_input)

        if self.scale >= 2:
            out = self.shu1(out)
            out = self.ps1(out)
        if self.scale == 4:
            out = self.shu2(out)
            out = self.ps2(out)

        out = self.end_conv(out)
        #
        batch[EBC.PREDICTION] = out
        return batch
