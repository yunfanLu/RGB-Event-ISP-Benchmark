import random
from glob import glob
from os.path import exists, join

import cv2
import numpy as np
import torch
from absl.logging import info, warning
from torch.utils.data import Dataset

from ev_rgb_isp.datasets.basic_batch import DemosaicHybridevsBatch as B
from ev_rgb_isp.datasets.basic_batch import get_demosaic_batch


def get_demosaic_hybridevs_dataset(config):
    root = config.root
    assert config.training_set_rate >= 0.1 and config.training_set_rate <= 0.99

    raw_root = join(root, "input")
    gt_root = join(root, "gt")
    raws = sorted(glob(join(raw_root, "*.bin")))
    gts = sorted(glob(join(gt_root, "*.png")))
    assert len(raws) == len(gts)

    training_length = int(len(raws) * config.training_set_rate)
    train_raws = raws[:training_length]
    train_gts = gts[:training_length]
    test_raws = raws[training_length:]
    test_gts = gts[training_length:]
    train = DemosaicHybridevsDataset(
        is_train=True,
        is_random_crop=config.is_random_crop,
        is_flip=config.is_flip,
        is_rotation=config.is_rotation,
        is_adding_noise=config.is_adding_noise,
        noise_threshold=config.noise_threshold,
        is_adding_defect_pixel=config.is_adding_defect_pixel,
        positive_defect_rate=config.positive_defect_rate,
        negative_defect_rate=config.negative_defect_rate,
        raws=train_raws,
        gts=train_gts,
        random_crop_resolution=config.random_crop_resolution,
        is_raw_from_vis=config.is_raw_from_vis,
    )
    test = DemosaicHybridevsDataset(
        is_train=False,
        is_random_crop=True,
        is_flip=False,
        is_rotation=False,
        is_adding_noise=False,
        noise_threshold=0,
        is_adding_defect_pixel=False,
        positive_defect_rate=1,
        negative_defect_rate=0,
        raws=test_raws,
        gts=test_gts,
        random_crop_resolution=config.random_crop_resolution,
        is_raw_from_vis=config.is_raw_from_vis,
    )
    return train, test


def get_demosaic_hybridevs_val_dataset(config):
    root = config.root
    is_raw_from_vis = config.is_raw_from_vis
    is_random_crop = config.is_random_crop
    random_crop_resolution = config.random_crop_resolution

    raw_root = join(root, "input")
    raws = sorted(glob(join(raw_root, "*.bin")))
    fake_gt = sorted(glob(join(raw_root, "*-raw.png")))
    test = DemosaicHybridevsDataset(
        is_train=False,
        is_random_crop=is_random_crop,
        is_flip=False,
        is_rotation=False,
        is_adding_noise=False,
        noise_threshold=0,
        is_adding_defect_pixel=False,
        positive_defect_rate=1,
        negative_defect_rate=0,
        raws=raws,
        gts=fake_gt,
        random_crop_resolution=random_crop_resolution,
        is_raw_from_vis=is_raw_from_vis,
    )
    return test, test


def read_bin(bin_path):
    with open(bin_path, "rb") as f:
        img_data = np.fromfile(f, dtype=np.uint16)
        w = int(img_data[0])
        h = int(img_data[1])
        assert w * h == img_data.size - 2
    quad = img_data[2:].reshape([h, w]).astype(np.float32)
    quad = np.clip(quad, 0, 1023)
    quad = quad / 1024.0
    return quad


def read_image(image_path):
    img = cv2.imread(image_path)
    return img


class DemosaicHybridevsDataset(Dataset):
    def __init__(
        self,
        is_train,
        is_random_crop,
        is_flip,
        is_rotation,
        is_adding_noise,
        noise_threshold,
        is_adding_defect_pixel,
        positive_defect_rate,
        negative_defect_rate,
        raws,
        gts,
        random_crop_resolution,
        is_raw_from_vis,
    ):
        super(DemosaicHybridevsDataset, self).__init__()
        assert len(raws) == len(gts)
        self.is_train = is_train
        # data augmentation
        self.is_random_crop = is_random_crop
        self.is_flip = is_flip
        self.is_rotation = is_rotation
        self.is_adding_noise = is_adding_noise
        self.noise_threshold = noise_threshold
        self.is_adding_defect_pixel = is_adding_defect_pixel
        self.positive_defect_rate = positive_defect_rate
        self.negative_defect_rate = negative_defect_rate
        # data
        self.raws = raws
        self.gts = gts
        self.is_raw_from_vis = is_raw_from_vis
        self.rc_h, self.rc_w = random_crop_resolution

        assert self._all_files_exist(), "Some files do not exist"

    def __len__(self):
        return len(self.gts)

    def __getitem__(self, index):
        raw = self.raws[index]
        gt = self.gts[index]
        name = gt.split("/")[-1].split(".")[0]

        if self.is_raw_from_vis:
            raw = raw.replace(".bin", "-raw.png")
            raw = read_image(raw)
            raw = torch.from_numpy(raw)
            raw = raw.permute(2, 0, 1) / 255.0
        else:
            raw = read_bin(raw)
            raw = torch.from_numpy(raw)
            raw = raw.unsqueeze(0)

        rgb_position = self._get_rgb_position(raw)
        gt = read_image(gt)
        gt = torch.from_numpy(gt).permute(2, 0, 1) / 255.0

        _, w, h = raw.shape
        if self.is_random_crop:
            raw, gt, rgb_position, w, h = self._random_crop(raw, gt, rgb_position)

        if self.is_train:
            raw, gt, rgb_position = self._data_augmentation(raw, gt, rgb_position)

        # create batch
        batch = get_demosaic_batch()
        batch[B.IMAGE_NAME] = name
        batch[B.HEIGHT] = h
        batch[B.WIDTH] = w
        batch[B.RAW_TENSOR] = raw
        batch[B.GROUND_TRUTH] = gt
        batch[B.RAW_RGB_POSITION] = rgb_position
        return batch

    def _data_augmentation(self, raw, gt, position):
        if self.is_flip and random.random() > 0.5:
            raw = torch.flip(raw, [2])
            gt = torch.flip(gt, [2])
            position = torch.flip(position, [2])

        if self.is_rotation and random.random() > 0.5:
            raw = torch.rot90(raw, 1, [1, 2])
            gt = torch.rot90(gt, 1, [1, 2])
            position = torch.rot90(position, 1, [1, 2])

        if self.is_adding_noise and random.random() > 0.5:
            noise = torch.randn_like(raw) * 0.01
            raw = raw + noise
            raw = torch.clamp(raw, 0, 1)

        if self.is_adding_defect_pixel and random.random() > 0.5:
            # random select some pixel and to set the value to zero or one
            rand_matrix = torch.rand_like(raw)
            mask = torch.zeros_like(raw)
            mask[rand_matrix > self.positive_defect_rate] = 1
            mask[rand_matrix < self.negative_defect_rate] = -1
            raw = raw + mask
            raw = torch.clamp(raw, 0, 1)

        return raw, gt, position

    def _get_rgb_position(self, raw):
        c, w, h = raw.shape
        position = torch.zeros([w, h], dtype=torch.float32)
        RED = 1
        GREEN = 2
        BLUE = 3
        EVENT = 4
        # red 00,01,10
        position[0::4, 0::4] = RED
        position[0::4, 1::4] = RED
        position[1::4, 0::4] = RED
        # green 02, 03, 12, 13
        position[0::4, 2::4] = GREEN
        position[0::4, 3::4] = GREEN
        position[1::4, 2::4] = GREEN
        position[1::4, 3::4] = GREEN
        # green 20, 21, 30, 31
        position[2::4, 0::4] = GREEN
        position[2::4, 1::4] = GREEN
        position[3::4, 0::4] = GREEN
        position[3::4, 1::4] = GREEN
        # blue 22, 23, 32
        position[2::4, 2::4] = BLUE
        position[2::4, 3::4] = BLUE
        position[3::4, 2::4] = BLUE
        # 11, 33
        position[1::4, 1::4] = EVENT
        position[3::4, 3::4] = EVENT
        #
        position = position[np.newaxis, :, :]
        return position / 5.0

    def _random_crop(self, raw, gt, position):
        c, w, h = raw.shape
        x1, x2 = random.randint(0, w - self.rc_w), random.randint(0, h - self.rc_h)
        x1 = (x1 // 8) * 8
        x2 = (x2 // 8) * 8
        raw = raw[:, x1 : x1 + self.rc_w, x2 : x2 + self.rc_h]
        gt = gt[:, x1 : x1 + self.rc_w, x2 : x2 + self.rc_h]
        position = position[:, x1 : x1 + self.rc_w, x2 : x2 + self.rc_h]
        return raw, gt, position, self.rc_w, self.rc_h

    def _all_files_exist(self):
        for raw, gt in zip(self.raws, self.gts):
            if not exists(raw) or not exists(gt):
                warning(f"File not found: {raw} or {gt}")
                return False
        return True
