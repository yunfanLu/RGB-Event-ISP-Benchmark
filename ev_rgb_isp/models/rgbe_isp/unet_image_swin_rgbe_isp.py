from os.path import abspath, dirname, join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from ev_rgb_isp.datasets.basic_batch import EventRAWISPBatch as EBC
from ev_rgb_isp.models.demosaic.raw_to_rggb import quad_raw_to_rggb
from ev_rgb_isp.models.rstt.layers import DecoderLayer, Downsample, EncoderLayer, InputProj, Upsample


def get_unet_image_swin_rgbe_isp(config):
    in_chans = config.in_chans
    using_rggb_4_channel = config.using_rggb_4_channel

    if config.TYPE == "costom":
        model = UNetImageSwin(
            using_rggb_4_channel=using_rggb_4_channel,
            in_chans=in_chans,
            embed_dim=config.embed_dim,
            depths=config.depths,
            num_heads=config.num_heads,
            window_sizes=config.window_sizes,
        )
    elif config.TYPE == "large":
        model = get_unet_image_swin_rgbe_isp_large(using_rggb_4_channel, in_chans)
    elif config.TYPE == "medium":
        model = get_unet_image_swin_rgbe_isp_medium(using_rggb_4_channel, in_chans)
    elif config.TYPE == "small":
        model = get_unet_image_swin_rgbe_isp_small(using_rggb_4_channel, in_chans)
    elif config.TYPE == "tiny":
        model = get_unet_image_swin_rgbe_isp_tiny(using_rggb_4_channel, in_chans)
    return model


def get_unet_image_swin_rgbe_isp_large(using_rggb_4_channel, in_chans):
    model = UNetImageSwinRGBEISP(
        using_rggb_4_channel=using_rggb_4_channel,
        in_chans=in_chans,
        embed_dim=96,
        depths=[8, 8, 8, 8, 8, 8, 8, 8],
        num_heads=[2, 4, 8, 16, 16, 8, 4, 2],
        window_sizes=[[4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]],
    )
    return model


def get_unet_image_swin_rgbe_isp_medium(using_rggb_4_channel, in_chans):
    model = UNetImageSwinRGBEISP(
        using_rggb_4_channel=using_rggb_4_channel,
        in_chans=in_chans,
        embed_dim=96,
        depths=[6, 6, 6, 6, 6, 6, 6, 6],
        num_heads=[2, 4, 8, 16, 16, 8, 4, 2],
        window_sizes=[[4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]],
    )
    return model


def get_unet_image_swin_rgbe_isp_small(using_rggb_4_channel, in_chans):
    model = UNetImageSwinRGBEISP(
        using_rggb_4_channel=using_rggb_4_channel,
        in_chans=in_chans,
        embed_dim=96,
        depths=[4, 4, 4, 4, 4, 4, 4, 4],
        num_heads=[2, 4, 8, 16, 16, 8, 4, 2],
        window_sizes=[[4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]],
    )
    return model


def get_unet_image_swin_rgbe_isp_tiny(using_rggb_4_channel, in_chans):
    model = UNetImageSwinRGBEISP(
        using_rggb_4_channel=using_rggb_4_channel,
        in_chans=in_chans,
        embed_dim=96,
        depths=[2, 2, 2, 2, 2, 2, 2, 2],
        num_heads=[2, 4, 8, 16, 16, 8, 4, 2],
        window_sizes=[[4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]],
    )
    return model


class UNetImageSwinRGBEISP(nn.Module):
    def __init__(
        self,
        using_rggb_4_channel,
        in_chans=1,
        embed_dim=96,
        depths=[8, 8, 8, 8, 8, 8, 8, 8],
        num_heads=[2, 4, 8, 16, 16, 8, 4, 2],
        num_frames=4,
        window_sizes=[(4, 4), (4, 4), (4, 4), (4, 4), (4, 4), (4, 4), (4, 4), (4, 4)],
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    ):
        """

        Args:
            in_chans (int, optional): Number of input image channels. Defaults to 3.
            embed_dim (int, optional): Number of projection output channels. Defaults to 32.
            depths (list[int], optional): Depths of each Transformer stage. Defaults to [2, 2, 2, 2, 2, 2, 2, 2].
            num_heads (list[int], optional): Number of attention head of each stage. Defaults to [2, 4, 8, 16, 16, 8, 4, 2].
            num_frames (int, optional): Number of input frames. Defaults to 4.
            window_size (tuple[int], optional): Window size. Defaults to (8, 8).
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4..
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop_rate (float, optional): Dropout rate. Defaults to 0.
            attn_drop_rate (float, optional): Attention dropout rate. Defaults to 0.
            drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0.1.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
            patch_norm (bool, optional): If True, add normalization after patch embedding. Defaults to True.
        """
        super().__init__()

        # global config
        self.using_rggb_4_channel = using_rggb_4_channel

        self.num_layers = len(depths)
        self.num_enc_layers = self.num_layers // 2
        self.num_dec_layers = self.num_layers // 2
        self.scale = 2 ** (self.num_enc_layers - 1)
        dec_depths = depths[self.num_enc_layers :]
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_in_frames = num_frames
        self.num_out_frames = 2 * num_frames - 1

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[: self.num_enc_layers]))]
        dec_dpr = enc_dpr[::-1]

        self.positional_embedding = nn.Conv2d(1, embed_dim, 1, 1, 0)

        self.input_proj = InputProj(
            in_channels=in_chans, embed_dim=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU
        )

        # Encoder
        self.encoder_layers = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for i_layer in range(self.num_enc_layers):
            encoder_layer = EncoderLayer(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                num_frames=num_frames,
                window_size=window_sizes[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=enc_dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
            )
            downsample = Downsample(embed_dim, embed_dim)
            self.encoder_layers.append(encoder_layer)
            self.downsample.append(downsample)

        # Decoder
        self.decoder_layers = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i_layer in range(self.num_dec_layers):
            decoder_layer = DecoderLayer(
                dim=embed_dim,
                depth=depths[i_layer + self.num_enc_layers],
                num_heads=num_heads[i_layer + self.num_enc_layers],
                num_frames=num_frames,
                window_size=window_sizes[i_layer + self.num_enc_layers],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dec_dpr[sum(dec_depths[:i_layer]) : sum(dec_depths[: i_layer + 1])],
                norm_layer=norm_layer,
            )
            self.decoder_layers.append(decoder_layer)
            if i_layer != self.num_dec_layers - 1:
                upsample = Upsample(embed_dim, embed_dim)
                self.upsample.append(upsample)

        # Upsampling
        if self.using_rggb_4_channel:
            self.high_dim_to_rgb = nn.Sequential(
                nn.Conv2d(embed_dim * 7, embed_dim * 4, 3, 1, 1, bias=True),
                nn.PixelShuffle(2),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(embed_dim, 3, 3, 1, 1, bias=True),
            )
        else:
            self.high_dim_to_rgb = nn.Sequential(
                nn.Conv2d(embed_dim * 7, embed_dim * 4, 3, 1, 1, bias=True),
                nn.Conv2d(embed_dim * 4, 3, 3, 1, 1, bias=True),
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, batch):
        x = batch[EBC.RAW_TENSORS]  # B, C, H, W

        if self.using_rggb_4_channel:
            # B, 1, H, W -> B, 4, H/2, W/2
            x = quad_raw_to_rggb(x)
            position = 0
        else:
            # B C H W -> B D C H W
            x = x.unsqueeze(1).expand(-1, self.num_in_frames, -1, -1, -1)
            # B, 1, H, W -> B, D, H, W
            position = self.positional_embedding(position)

        # D input video frames
        B, D, C, H, W = x.size()
        # B, D, C, H, W
        x = self.input_proj(x)
        x = x + position

        # Start Swin Transformer
        Hp = int(np.ceil(H / self.scale)) * self.scale
        Wp = int(np.ceil(W / self.scale)) * self.scale
        x = F.pad(x, (0, Wp - W, 0, Hp - H))

        encoder_features = []
        for i_layer in range(self.num_enc_layers):
            x = self.encoder_layers[i_layer](x)
            encoder_features.append(x)
            if i_layer != self.num_enc_layers - 1:
                x = self.downsample[i_layer](x)

        _, _, C, h, w = x.size()
        # TODO: Use interpolation for queries
        y = torch.zeros((B, self.num_out_frames, C, h, w), device=x.device)
        for i in range(self.num_out_frames):
            if i % 2 == 0:
                y[:, i, :, :, :] = x[:, i // 2]
            else:
                y[:, i, :, :, :] = (x[:, i // 2] + x[:, i // 2 + 1]) / 2

        for i_layer in range(self.num_dec_layers):
            y = self.decoder_layers[i_layer](y, encoder_features[-i_layer - 1])
            if i_layer != self.num_dec_layers - 1:
                y = self.upsample[i_layer](y)

        y = y[:, :, :, :H, :W].contiguous()
        # Reconstruction
        B, D, C, H, W = y.size()
        out = y.view(B, D * C, H, W)

        # High dim to rgb
        out = self.high_dim_to_rgb(out)

        batch[EBC.PREDICTION] = out
        return batch
