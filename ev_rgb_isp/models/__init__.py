from ev_rgb_isp.models.demosaic.unet_image_swin import get_unet_image_swin
from ev_rgb_isp.models.els_net.els_net import get_elsnet
from ev_rgb_isp.models.pynet.pynet import PyNET
from ev_rgb_isp.models.rgbe_isp.awnet.model import AWNet
from ev_rgb_isp.models.rgbe_isp.cameranet import CameraNet
from ev_rgb_isp.models.rgbe_isp.ispenet.ispe_net import ISPESLNet
from ev_rgb_isp.models.rgbe_isp.unet_2d_events_rgbe_isp import RGBEventISPUNet
from ev_rgb_isp.models.rgbe_isp.unet_2d_rgbe_isp import RGBEISPUNet
from ev_rgb_isp.models.rgbe_isp.unet_image_swin_rgbe_isp import get_unet_image_swin_rgbe_isp


def get_model(config):
    if config.NAME == "CameraNet":
        return CameraNet()
    elif config.NAME == "ISPESLNet":
        return ISPESLNet(moments=config.moments)
    elif config.NAME == "RGBEventISPUNet":
        return RGBEventISPUNet(in_channels=config.in_channels, moments=config.moments, with_events=config.with_events)
    elif config.NAME == "AWNet":
        return AWNet()
    elif config.NAME == "unet_image_swin_rgbe_isp":
        return get_unet_image_swin_rgbe_isp(config)
    elif config.NAME == "RGBEISPUNet":
        return RGBEISPUNet()
    elif config.NAME == "pynet":
        return PyNET(
            level=config.level, instance_norm=config.instance_norm, instance_norm_level_1=config.instance_norm_level_1
        )
    elif config.NAME == "els_net":
        return get_elsnet(scale=config.scale, is_color=config.is_color)
    # vfi + sr
    elif config.NAME == "unet_image_swin":
        return get_unet_image_swin(config)
    else:
        raise ValueError(f"Model {config.NAME} is not supported.")
