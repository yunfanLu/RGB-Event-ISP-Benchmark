import logging
import os
from os.path import join

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pudb
import torch
from absl.logging import debug, flags, info

from ev_rgb_isp.datasets.basic_batch import DemosaicHybridevsBatch as BC

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

FLAGS = flags.FLAGS


def _tensor_to_image(tensor, path):
    image = tensor.permute(1, 2, 0).numpy().astype(np.float32) * 255
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)


def raw_to_raw_image(quad):
    h, w = quad.shape
    raw = np.zeros([h, w, 3], dtype=np.float32)
    # 0, 1, 2 B G R
    # red 00,01,10
    raw[0::4, 0::4, 2] = quad[0::4, 0::4]
    raw[0::4, 1::4, 2] = quad[0::4, 1::4]
    raw[1::4, 0::4, 2] = quad[1::4, 0::4]
    # green 02, 03, 12, 13
    raw[0::4, 2::4, 1] = quad[0::4, 2::4]
    raw[0::4, 3::4, 1] = quad[0::4, 3::4]
    raw[1::4, 2::4, 1] = quad[1::4, 2::4]
    raw[1::4, 3::4, 1] = quad[1::4, 3::4]
    # green 20, 21, 30, 31
    raw[2::4, 0::4, 1] = quad[2::4, 0::4]
    raw[2::4, 1::4, 1] = quad[2::4, 1::4]
    raw[3::4, 0::4, 1] = quad[3::4, 0::4]
    raw[3::4, 1::4, 1] = quad[3::4, 1::4]
    # blue 22, 23, 32
    raw[2::4, 2::4, 0] = quad[2::4, 2::4]
    raw[2::4, 3::4, 0] = quad[2::4, 3::4]
    raw[3::4, 2::4, 0] = quad[3::4, 2::4]
    return raw


class DemosaicHybridevsBatchVisualization:
    def __init__(self, config):
        self.saving_folder = join(FLAGS.log_dir, config.folder)
        os.makedirs(self.saving_folder, exist_ok=True)
        self.count = 0
        self.intermediate_visualization = config.intermediate_visualization
        info("Init Visualization:")
        info(f"  saving_folder: {self.saving_folder}")
        info(f"  intermediate_visualization: {self.intermediate_visualization}")

    def visualize(self, batch):
        image_names = batch[BC.IMAGE_NAME]
        raw_tensors = batch[BC.RAW_TENSOR]
        outputs = batch[BC.PREDICTION]
        batch_size = len(outputs)
        for b in range(batch_size):
            image_name = image_names[b]
            output = outputs[b].cpu()
            _tensor_to_image(output, join(self.saving_folder, f"{image_name}.png"))

            # raw_tensor = raw_tensors[b].cpu().
            # raw_image = raw_to_raw_image(raw_tensor)
