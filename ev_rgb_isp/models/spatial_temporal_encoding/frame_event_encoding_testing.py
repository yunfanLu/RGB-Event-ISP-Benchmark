import time

import pudb
import torch
import torchstat
from absl.logging import info
from absl.testing import absltest
from thop import profile

from ev_rgb_isp.models.spatial_temporal_encoding.frame_event_encoding import FramesEventSpatialTemporalEncoding


class FramesEventSpatialTemporalEncodingTest(absltest.TestCase):
    def setUp(self):
        B, C, H, W = 1, 32, 32, 64
        self.frame = torch.rand(B, 2, 3, H, W).cuda()
        self.events = torch.rand(B, 32, H, W).cuda()

        spatial_temporal_attention_loop = 2

        frame_resolution = H, W
        event_channels = 32
        frame_channels = 3
        in_frames = 2
        feature_channels = 32
        transformer_resolution_downsample = False
        transformer_patch_size = 5, 5
        transformer_dim = 32
        transformer_drop = 0.2
        transformer_depth = 2

        temporal_encoding_depth = 2
        temporal_encoding_loop = 2
        temporal_encoding_cal_kernel_size = 1
        self.model = FramesEventSpatialTemporalEncoding(
            spatial_temporal_attention_loop,
            frame_resolution,
            frame_channels,
            event_channels,
            in_frames,
            feature_channels,
            transformer_resolution_downsample,
            transformer_patch_size,
            transformer_dim,
            transformer_drop,
            transformer_depth,
            temporal_encoding_depth,
            temporal_encoding_loop,
            temporal_encoding_cal_kernel_size,
        )
        self.model = self.model.cuda()

    def test_inference_time(self):
        pudb.set_trace()

        N = 20
        # preheat
        torch.cuda.synchronize()
        for i in range(N):
            batch = self.model(self.events, self.frame)
            torch.cuda.synchronize()
        # test time
        torch.cuda.synchronize()
        start = time.time()
        for i in range(N):
            batch = self.model(self.events, self.frame)
            torch.cuda.synchronize()
        end = time.time()
        info(f"Inference with loding model time: {(end - start)*1000 / N}ms")

    def test_params_flops(self):
        model = self.model.cpu()
        # torchstat.stat(model, (32, 128, 128))
        flops, params = profile(model, inputs=(self.events.cpu(), self.frame.cpu()))
        info(f"flops: {flops}")
        info(f"params: {params / 1000000} M")


if __name__ == "__main__":
    absltest.main()
