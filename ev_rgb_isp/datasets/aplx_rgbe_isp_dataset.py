#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2022/11/2 11:12
import logging
from os import listdir
from os.path import isdir, join

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from absl.logging import info
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

from ev_rgb_isp.datasets.basic_batch import EventRAWISPBatch as EBC
from ev_rgb_isp.datasets.basic_batch import get_rgbe_isp_batch

logging.getLogger("PIL").setLevel(logging.WARNING)

"""
Video Names:
20240419105126903  20240421162059899  20240421165503036  20240424143540185  20240424150007213  20240424150624272  20240424152127640  20240516152255520
20240421161715658  20240421164709698  20240421165711996  20240424143633558  20240424150334757  20240424151256816  20240424152550671  20240516152425517
20240421161926310  20240421164920988  20240421172952069  20240424145127802  20240424150445496  20240424151559855  20240424152829967  20240516152445031
"""


def get_alpx_rgbe_isp_dataset(
    alpx_rgbe_isp_root,
    moments,
    in_frame,
    future_frame,
    past_frame,
    using_events,
    using_interpolate,
    random_crop_resolution,
    evaluation_visualization,
):
    all_videos = sorted(listdir(alpx_rgbe_isp_root))
    test_videos = [
        "20240421172952069",
        "20240424150334757",
        "20240424152127640",
        "20240516152255520",
        "20240421162059899",
        "20240516152445031",
    ]
    train_videos = []
    for v in all_videos:
        if v in test_videos:
            continue
        if not isdir(join(alpx_rgbe_isp_root, v)):
            continue
        train_videos.append(v)

    info(f"train_videos ({len(train_videos)}): {train_videos}")
    info(f"test_videos  ({len(test_videos)}): {test_videos}")

    train_dataset = AlpxEventRGBISPDataset(
        alpx_rgbe_isp_root,
        train_videos,
        moments,
        in_frame,
        future_frame,
        past_frame,
        using_events,
        using_interpolate,
        random_crop_resolution,
        evaluation_visualization,
    )
    test_dataset = AlpxEventRGBISPDataset(
        alpx_rgbe_isp_root,
        test_videos,
        moments,
        in_frame,
        future_frame,
        past_frame,
        using_events,
        using_interpolate,
        random_crop_resolution,
        evaluation_visualization,
    )
    return train_dataset, test_dataset


class AlpxEventRGBISPDataset(Dataset):
    def __init__(
        self,
        alpx_rgbe_isp_root,
        videos,
        moments,
        in_frame,
        future_frame,
        past_frame,
        using_events,
        using_interpolate,
        random_crop_resolution,
        evaluation_visualization,
    ):
        super(AlpxEventRGBISPDataset, self).__init__()
        self.moments = moments
        self.alpx_rgbe_isp_root = alpx_rgbe_isp_root
        self.videos = videos
        self.in_frame = in_frame
        self.future_frame = future_frame
        self.past_frame = past_frame
        self.random_crop_resolution = random_crop_resolution
        # static values
        self.raw_resolution = (2448, 3264)
        self.event_resolution = (1224, 1632)
        self.positive = 2
        self.negative = 1
        self.using_events = using_events
        self.using_interpolate = using_interpolate
        self.evaluation_visualization = evaluation_visualization
        self.items = self._generate_items()

    # # 创建插值函数
    # def _interpolate_along_axis(self, data, axis, new_size):
    #     original_indices = np.arange(data.shape[axis])
    #     new_indices = np.linspace(0, data.shape[axis] - 1, new_size)

    #     # 创建插值函数
    #     interp_function = interp1d(original_indices, data, axis=axis, kind='linear')

    #     # 对指定轴进行插值
    #     interpolated_data = interp_function(new_indices)

    #     return interpolated_data

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_paths, raw_paths, event_paths = self.items[index]
        h, w = self.event_resolution
        images = []
        # for path in image_paths:
        for i in range(self.past_frame, self.in_frame - self.future_frame):
            # to float32
            path = image_paths[i]
            image = cv2.imread(path).astype(np.float32)
            if image is None:
                raise ValueError(f"image is None: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        images = np.stack(images, axis=0) / 255.0
        # (N, H, W, C) -> (N, C, H, W)
        images = images.transpose((0, 3, 1, 2))

        raws = []
        for path in raw_paths:
            raw = np.load(path)["aps_raw"].astype(np.float32)
            raws.append(raw)  # raw.shape = (H, W)
        # (H, W) -> (N, H, W)
        raws = np.stack(raws, axis=0) / 255.0

        if self.using_events:
            events = []
            for path in event_paths:
                event = np.load(path)["event_raw"].astype(np.float32)
                event[event == self.negative] = -1
                event[event == self.positive] = 1
                events.append(event)
            events = np.stack(events, axis=0)
            if self.using_interpolate:
                events = self._interpolate_along_axis(events, axis=0, new_size=self.moments)
            else:
                N, H, W = events.shape
                # get the middle channels with mometns
                if N < self.moments:
                    start = 0
                    new_events = np.zeros((self.moments, H, W))
                    new_events[start : start + N] = events
                else:
                    start = (N - self.moments) // 2
                    new_events = events[start : start + self.moments]

            events = new_events.astype(np.float32)
        else:
            events = 0

        crop_images, crop_raws, crop_events = self._random_crop(images, raws, events)
        #
        keyframe_name = image_paths[self.past_frame].split("/")[-1]
        video_name = image_paths[self.past_frame].split("/")[-2]
        batch = get_rgbe_isp_batch()
        batch[EBC.INPUT_FRAME_COUNT] = self.in_frame
        batch[EBC.FRAME_NAME] = keyframe_name
        batch[EBC.VIDEO_NAME] = video_name
        batch[EBC.HEIGHT] = h
        batch[EBC.WIDTH] = w
        batch[EBC.RAW_TENSORS] = crop_raws
        batch[EBC.RAW_TIMESTAMPS] = 0
        batch[EBC.GROUND_TRUTH] = crop_images
        batch[EBC.GROUND_TRUTH_TIMESTAMPS] = 0
        batch[EBC.EVENTS_VOXEL_GRID] = crop_events

        image_timestamps, first_event_timestamp, last_event_timestamp = self._get_timestamps(
            image_paths, raw_paths, event_paths
        )
        # print(f"image_timestamps: {image_timestamps}, first_event_timestamp: {first_event_timestamp}, last_event_timestamp: {last_event_timestamp}")
        batch[EBC.GROUND_TRUTH_TIMESTAMPS] = image_timestamps
        batch[EBC.EVENTS_VOXEL_GRID_TIMESTAMPS_START] = first_event_timestamp
        batch[EBC.EVENTS_VOXEL_GRID_TIMESTAMPS_END] = last_event_timestamp
        return batch

    def _get_timestamps(self, image_paths, raw_paths, event_paths):
        image_timestamps = float(image_paths[self.past_frame].split("/")[-1].split("_")[0])
        first_event_timestamp = float(event_paths[0].split("/")[-1].split("_")[0])
        last_event_timestamp = float(event_paths[-1].split("/")[-1].split("_")[0])
        min_timestamp = min(image_timestamps, first_event_timestamp, last_event_timestamp)
        max_timestamp = max(image_timestamps, first_event_timestamp, last_event_timestamp)
        length = max_timestamp - min_timestamp
        image_timestamps = (image_timestamps - min_timestamp) / length
        first_event_timestamp = (first_event_timestamp - min_timestamp) / length
        last_event_timestamp = (last_event_timestamp - min_timestamp) / length
        return image_timestamps, first_event_timestamp, last_event_timestamp

    def _random_crop(self, images, raws, events):
        h, w = self.event_resolution  # (1224, 1632)
        crop_h, crop_w = self.random_crop_resolution
        if self.evaluation_visualization:
            x, y = 0, 0
        else:
            x = (np.random.randint(0, w - crop_h - 8) // 4) * 4
            y = (np.random.randint(0, h - crop_w - 8) // 4) * 4
        if self.using_events:
            crop_events = events[:, x : x + crop_h, y : y + crop_w]
        else:
            crop_events = 0
        # the image resolution is 2448 x 3264
        x = x * 2
        y = y * 2
        crop_h = crop_h * 2
        crop_w = crop_w * 2
        crop_images = images[:, :, x : x + crop_h, y : y + crop_w]
        crop_raws = raws[:, x : x + crop_h, y : y + crop_w]
        return crop_images, crop_raws, crop_events

    def _generate_items(self):
        samples = []
        for video in self.videos:
            video_path = join(self.alpx_rgbe_isp_root, video)
            samples += self._generate_video_items(video_path)
        return samples

    def _generate_video_items(self, video_path):
        items = []
        files = sorted(listdir(video_path))
        for i in range(len(files)):
            if not files[i].endswith("good_rgb.png"):
                continue
            frame_indexes = []
            for j in range(i + 1, len(files)):
                if not files[j].endswith("good_rgb.png"):
                    continue
                frame_indexes.append(j)
                if len(frame_indexes) == self.in_frame:
                    break
            # 1. Check has enough frames.
            if len(frame_indexes) != self.in_frame:
                break
            # 2. Check the event in neighbor frames is average.

            for k in range(self.in_frame - 1):
                index_step = frame_indexes[k + 1] - frame_indexes[k]

            # 3. Generate items.
            good_rgb = []
            raw = []
            events = []
            for k in range(min(frame_indexes), max(frame_indexes) + 1):
                frame_name = files[k]
                if frame_name.endswith("good_rgb.png"):
                    good_rgb_path = join(video_path, frame_name)
                    good_rgb.append(good_rgb_path)
                    raw_name = frame_name.replace("good_rgb.png", "aps_raw.npz")
                    raw.append(join(video_path, raw_name))
                elif frame_name.endswith(".npz") and (not frame_name.endswith("raw.npz")):
                    events.append(join(video_path, frame_name))
            item = [good_rgb, raw, events]
            items.append(item)
        return items
