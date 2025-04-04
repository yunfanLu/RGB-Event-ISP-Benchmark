import torch
import torch.nn as nn

from ev_rgb_isp.datasets.basic_batch import EventRAWISPBatch as EBC
from ev_rgb_isp.models.demosaic.raw_to_rggb import quad_raw_to_rggb


class RGBEventISPUNet(nn.Module):
    def __init__(
        self,
        in_channels=12,
        moments=50,
        out_channels=3,
        with_events=False,
        with_exposure_time_embedding=False,
        bias=True,
    ):
        super(RGBEventISPUNet, self).__init__()

        self.with_events = with_events
        self.with_exposure_time_embedding = with_exposure_time_embedding

        if self.with_exposure_time_embedding:
            self.exposure_time_embedding = nn.Conv2d(3, 32, 1, 1, 0, bias=bias)

        self.conv1 = nn.Conv2d(in_channels, 32, 7, 1, 3, bias=bias)
        self.conv2 = nn.Conv2d(32, 32, 7, 1, 3, bias=bias)
        self.relu = nn.LeakyReLU(0.1, True)

        self.conv1_e = nn.Conv2d(moments, 32, 1, 1, 0, bias=bias)
        self.conv2_e = nn.Conv2d(32, 32, 7, 1, 3, bias=bias)
        # Down 1
        self.avgpool1 = nn.AvgPool2d(kernel_size=7, stride=2, padding=3)
        self.conv3 = nn.Conv2d(32, 64, 5, 1, 2, bias=bias)
        self.conv4 = nn.Conv2d(64, 64, 5, 1, 2, bias=bias)
        self.avgpool1_e = nn.AvgPool2d(kernel_size=7, stride=2, padding=3)
        self.conv3_e = nn.Conv2d(32, 64, 5, 1, 2, bias=bias)
        self.conv4_e = nn.Conv2d(64, 64, 5, 1, 2, bias=bias)
        # Down 2
        self.avgpool2 = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        self.conv5 = nn.Conv2d(64, 128, 3, 1, 1, bias=bias)
        self.conv6 = nn.Conv2d(128, 128, 3, 1, 1, bias=bias)
        self.avgpool2_e = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        self.conv5_e = nn.Conv2d(64, 128, 3, 1, 1, bias=bias)
        self.conv6_e = nn.Conv2d(128, 128, 3, 1, 1, bias=bias)
        # Down 3
        self.avgpool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(128, 256, 3, 1, 1, bias=bias)
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 1, bias=bias)
        self.avgpool3_e = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv7_e = nn.Conv2d(128, 256, 3, 1, 1, bias=bias)
        self.conv8_e = nn.Conv2d(256, 256, 3, 1, 1, bias=bias)
        # Down 4
        self.avgpool4 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv9 = nn.Conv2d(256, 512, 3, 1, 1, bias=bias)
        self.conv10 = nn.Conv2d(512, 512, 3, 1, 1, bias=bias)
        self.avgpool4_e = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv9_e = nn.Conv2d(256, 512, 3, 1, 1, bias=bias)
        self.conv10_e = nn.Conv2d(512, 512, 3, 1, 1, bias=bias)
        # Down 5
        self.avgpool5 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv11 = nn.Conv2d(512, 512, 3, 1, 1, bias=bias)
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 1, bias=bias)
        self.avgpool5_e = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv11_e = nn.Conv2d(512, 512, 3, 1, 1, bias=bias)
        self.conv12_e = nn.Conv2d(512, 512, 3, 1, 1, bias=bias)

        # Decoder
        self.upsample2D = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 1, bias=bias)
        self.conv14 = nn.Conv2d(512, 512, 3, 1, 1, bias=bias)
        self.conv15 = nn.Conv2d(512, 256, 3, 1, 1, bias=bias)
        self.conv16 = nn.Conv2d(256, 256, 3, 1, 1, bias=bias)
        self.conv17 = nn.Conv2d(256, 128, 3, 1, 1, bias=bias)
        self.conv18 = nn.Conv2d(128, 128, 3, 1, 1, bias=bias)
        self.conv19 = nn.Conv2d(128, 64, 3, 1, 1, bias=bias)
        self.conv20 = nn.Conv2d(64, 64, 3, 1, 1, bias=bias)
        self.conv21 = nn.Conv2d(64, 32, 3, 1, 1, bias=bias)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 1, bias=bias)
        self.conv23 = nn.Conv2d(32, out_channels, 3, 1, 1, bias=bias)

    def forward(self, batch):
        # Get a RGB E batch
        #
        raw = batch[EBC.RAW_TENSORS]  # 1, 1, H, W
        raw = quad_raw_to_rggb(raw)  # 1, 4, 1, H, W
        # remove B, 4, 1, H, W -> B, 4, H, W
        B, L, N, H, W = raw.shape
        X = raw.reshape(B, L * N, H, W)

        events = batch[EBC.EVENTS_VOXEL_GRID]
        if self.with_events:
            events = torch.zeros_like(events) + 1.0

        if self.with_exposure_time_embedding:
            image_timestamps = batch[EBC.IMAGE_TIMESTAMPS]
            first_event_timestamp = batch[EBC.EVENTS_VOXEL_GRID_TIMESTAMPS_START]
            last_event_timestamp = batch[EBC.EVENTS_VOXEL_GRID_TIMESTAMPS_END]
            exposure_times = torch.tensor(
                [image_timestamps, first_event_timestamp, last_event_timestamp], dtype=torch.float32
            )
            exposure_times = exposure_times.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            # shape to B, 3, H, W
            exposure_times = exposure_times.expand(B, 3, H, W).to(X.device)
            exposure_times = self.exposure_time_embedding(exposure_times)
        else:
            exposure_times = 0

        events_sources = []
        sources = []
        ## Encoder
        X = self.conv1(X)
        X = self.relu(X)
        X = self.conv2(X)
        X = self.relu(X)
        X = X + exposure_times
        sources.append(X)

        E = self.relu(self.conv1_e(events))
        E = self.relu(self.conv2_e(E))
        E = E + exposure_times
        events_sources.append(E)

        # # print(f"X: {X.shape}, E: {E.shape}")

        X = self.avgpool1(X)
        X = self.conv3(X)
        X = self.relu(X)
        X = self.conv4(X)
        X = self.relu(X)
        sources.append(X)

        E = self.avgpool1_e(E)
        E = self.relu(self.conv3_e(E))
        E = self.relu(self.conv4_e(E))
        events_sources.append(E)

        # print(f"X: {X.shape}, E: {E.shape}")

        X = self.avgpool2(X)
        X = self.conv5(X)
        X = self.relu(X)
        X = self.conv6(X)
        X = self.relu(X)
        sources.append(X)

        E = self.avgpool2_e(E)
        E = self.relu(self.conv5_e(E))
        E = self.relu(self.conv6_e(E))
        events_sources.append(E)

        # print(f"X: {X.shape}, E: {E.shape}")

        X = self.avgpool3(X)
        X = self.conv7(X)
        X = self.relu(X)
        X = self.conv8(X)
        X = self.relu(X)
        sources.append(X)

        E = self.avgpool3_e(E)
        E = self.relu(self.conv7_e(E))
        E = self.relu(self.conv8_e(E))
        events_sources.append(E)

        # print(f"X: {X.shape}, E: {E.shape}")

        X = self.avgpool4(X)
        X = self.conv9(X)
        X = self.relu(X)
        X = self.conv10(X)
        X = self.relu(X)
        sources.append(X)

        E = self.avgpool4_e(E)
        E = self.relu(self.conv9_e(E))
        E = self.relu(self.conv10_e(E))
        events_sources.append(E)

        # print(f"X: {X.shape}, E: {E.shape}")

        X = self.avgpool5(X)
        X = self.conv11(X)
        X = self.relu(X)
        X = self.conv12(X)
        X = self.relu(X)

        ## Decoder
        X = self.upsample2D(X)
        X = self.conv13(X)
        X = self.relu(X)
        X = X + sources[-1] + events_sources[-1]
        X = self.conv14(X)
        X = self.relu(X)

        X = self.upsample2D(X)
        X = self.conv15(X)
        X = self.relu(X)
        X = X + sources[-2] + events_sources[-2]
        X = self.conv16(X)
        X = self.relu(X)

        X = self.upsample2D(X)
        X = self.conv17(X)
        X = self.relu(X)
        X = X + sources[-3] + events_sources[-3]
        X = self.conv18(X)
        X = self.relu(X)

        X = self.upsample2D(X)
        X = self.conv19(X)
        X = self.relu(X)
        X = X + sources[-4] + events_sources[-4]
        X = self.conv20(X)
        X = self.relu(X)

        X = self.upsample2D(X)
        X = self.conv21(X)
        X = self.relu(X)
        X = X + sources[-5] + events_sources[-5]
        X = self.conv22(X)
        X = self.relu(X)

        X = self.upsample2D(X)
        X = self.conv23(X)
        out = X

        batch[EBC.PREDICTION] = out
        return batch
