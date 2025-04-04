#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2022/12/24 15:57
from enum import Enum, unique


@unique
class EventRAWISPBatch(Enum):
    INPUT_FRAME_COUNT = "input_frame_count"
    VIDEO_NAME = "video_name"
    FRAME_NAME = "frame_name"
    HEIGHT = "height"
    WIDTH = "width"
    RAW_TENSORS = "raw_tensors"
    RAW_TIMESTAMPS = "raw_timestamps"

    GROUND_TRUTH = "ground_truth"
    GROUND_TRUTH_TIMESTAMPS = "ground_truth_timestamps"
    PREDICTION = "prediction"

    EVENTS_VOXEL_GRID = "events_voxel_grid"
    EVENTS_VOXEL_GRID_TIMESTAMPS_START = "events_voxel_grid_timestamps_start"
    EVENTS_VOXEL_GRID_TIMESTAMPS_END = "events_voxel_grid_timestamps_end"


def get_rgbe_isp_batch():
    batch = {}
    for item in EventRAWISPBatch:
        batch[item] = "NONE(str)"
    return batch
