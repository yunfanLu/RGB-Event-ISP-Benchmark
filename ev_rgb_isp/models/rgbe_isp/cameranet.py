import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color import rgb_to_xyz, xyz_to_rgb

from ev_rgb_isp.datasets.basic_batch import EventRAWISPBatch as EBC
from ev_rgb_isp.functions.raw_tools import quad_bayes_to_bayes


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs**2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs**2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs**2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


class RestoreNetBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1, n=3):
        super(RestoreNetBlock, self).__init__()

        self.block = nn.Sequential()

        for _ in range(n):
            self.block.append(nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding))
            in_c = out_c
            self.block.append(nn.ReLU())

    def forward(self, x):
        return self.block(x)

    def to(self, device):
        for d in self.block:
            d.to(device)


class EnhanceNetBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1, n=3):
        super(EnhanceNetBlock, self).__init__()

        self.block = nn.Sequential()

        for _ in range(n):
            self.block.append(nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding))
            in_c = out_c
            # need to add some AdaptiveBatchNorm
            # self.block.append(ABN())
            self.block.append(nn.ReLU())

    def forward(self, x):
        # skip connections
        x2 = self.block[0](x)
        return self.block[1:](x2) + x

    def to(self, device):
        for d in self.block:
            d.to(device)


class GlobalComponent(nn.Module):
    def __init__(self, n=512):
        super(GlobalComponent, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.lin = nn.Sequential()
        self.lin.append(nn.Linear(n, n))
        self.lin.append(nn.LeakyReLU())
        self.lin.append(nn.Linear(n, n))
        self.lin.append(nn.LeakyReLU())

    #         self.lin.append(nn.Linear(n, n))

    def forward(self, x):
        y = self.pool(x)
        b, c, _, _ = y.shape
        y = self.lin(y.view(b, c))
        r = torch.einsum("bcij,bc->bcij", x, y.view(b, c))
        r = self.lin[-1](r)
        return r

    def to(self, device):
        self.pool.to(device)
        for p in self.lin:
            p.to(device)


class UNetBlock(nn.Module):
    def __init__(self, first_block, second_block, dilation=1):
        super(UNetBlock, self).__init__()

        self.fb = first_block
        self.sb = second_block

    def forward(self, x):
        return self.sb(self.fb(x))

    def to(self, device):
        self.fb.to(device)
        self.sb.to(device)


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block=RestoreNetBlock):
        super(UNetDownBlock, self).__init__()

        self.conv_pool = nn.Sequential(
            block(in_channels, out_channels),
            nn.MaxPool2d(2),
            #             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #             nn.ReLU()
        )

    def forward(self, x):
        return self.conv_pool(x)

    def to(self, device):
        for m in self.conv_pool:
            m.to(device)


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block=RestoreNetBlock):
        super(UNetUpBlock, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.block = block(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is BCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x11 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x11], dim=1)
        return self.block(x1)

    def to(self, device):
        self.up.to(device)
        self.block.to(device)


class UNet(nn.Module):
    def __init__(self, block):
        super(UNet, self).__init__()
        dims = [32, 64, 128, 256, 512]
        dils = [1, 2, 2, 4, 8]

        self.std = SpaceToDepth(2)
        self.dts = DepthToSpace(2)

        self.down = nn.Sequential(nn.Conv2d(12, 32, kernel_size=3, padding=1), nn.ReLU())
        for i, j in zip(dims[:-1], dims[1:]):
            #             self.down.append(UNetBlock(first_block=block(i, j),
            #                                        second_block=nn.MaxPool2d(2)))
            self.down.append(UNetDownBlock(i, j))

        # GlobalComponent
        self.global_component = GlobalComponent(n=dims[-1])

        self.up = nn.Sequential()
        for i, j, k in zip(dims[:0:-1], dims[-2::-1], dils):
            #             self.up.append(UNetBlock(first_block=nn.ConvTranspose2d(i, i, kernel_size=2, stride=2),
            #                                      second_block=block(i, j)))
            self.up.append(UNetUpBlock(i, j))

        self.end = nn.Sequential()
        self.end.append(nn.Conv2d(32, 12, kernel_size=1, padding=0))
        self.end.append(nn.ReLU())

    def forward(self, x):
        # weird system with lists
        x = self.std(x)
        d = [x]
        for i in self.down:
            d.append(i(d[-1]))
        #             print(f'down {d[-1].shape=}')

        m = self.global_component(d[-1])
        #         m = d[-1]

        u = [m]
        for idx, i in enumerate(self.up):
            # skip connections
            #             tmp = i(u[-1])
            #             print(f'up s {u[-1].shape} + {d[-idx - 2].shape}')
            #             u.append(tmp + d[-idx - 2])
            u.append(i(u[-1], d[-idx - 2]))
        #             print(f'up {u[-1].shape=}')

        r = self.end(u[-1])

        return self.dts(r)

    def to(self, device):
        self.global_component.to(device)
        for p in self.down:
            p.to(device)
        for p in self.up:
            p.to(device)


class Wrapper(nn.Module):
    def __init__(self, f):
        super(Wrapper, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class CameraNet(nn.Module):
    """
    input shape: (B, 1, H, W)
    output shape: (B, 3, H, W)
    """

    def __init__(self):
        super(CameraNet, self).__init__()

        # rgb pattern
        # red_pattern = torch.tensor([[0, 1], [0, 0]])
        # green_pattern = torch.tensor([[1, 0], [0, 1]])
        # blue_pattern = torch.tensor([[0, 0], [1, 0]])
        # self.quad_pixel = torch.stack((red_pattern, green_pattern, blue_pattern)).unsqueeze(0)

        # RGB to XYZ transform
        #         self.rgb2xyz = Wrapper(rgb_to_xyz)
        self.restore = nn.Sequential(
            # Wrapper(rgb_to_xyz),
            UNet(block=RestoreNetBlock)
        )
        # XYZ to RGB transform
        #         self.xyz2rgb = Wrapper(xyz_to_rgb)
        #         self.enhance = UNet(block=RestoreNetBlock)
        self.enhance = nn.Sequential(
            # Wrapper(xyz_to_rgb),
            UNet(block=EnhanceNetBlock)
        )

    def forward(self, batch):
        # Get a RGB E batch
        #
        raw = batch[EBC.RAW_TENSORS]  # N, 1, H, W
        x = quad_bayes_to_bayes(raw)
        x = x.expand(-1, 3, -1, -1)

        # print(f"x: {x.shape}")
        # exit()
        # assign the color to each pixel by converting (1, H, W) to (3, H, W)
        # |G|R|G|R|G|R|G|R|G|R|
        # |B|G|B|G|B|G|B|G|B|G|
        # |G|R|G|R|G|R|G|R|G|R|
        # |B|G|B|G|B|G|B|G|B|G|

        # B, C, H, W = x.shape
        # convert_mtx = self.quad_pixel.repeat(1, 1, H // 2, W // 2)
        # convert_mtx = convert_mtx.to(x.device)
        # x = x * convert_mtx
        print(f"x: {x[0, 0, :, :]}")
        print(f"x.shape: {x.shape}")

        x = self.restore(x)
        x = self.enhance(x)

        batch[EBC.PREDICTION] = x
        return batch

    # def to(self, device):
    #     self.quad_pixel.to(device)
    #     print(f"self.quad_pixel: {self.quad_pixel.device}")
    #     exit()
    #     self.restore[0].to(device)
    #     self.enhance[0].to(device)


if __name__ == "__main__":
    net = CameraNet()

    print(net)
