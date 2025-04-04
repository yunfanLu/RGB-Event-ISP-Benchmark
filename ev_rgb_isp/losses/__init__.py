#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project :EG-VFI
@Author  :Yunfan Lu (yunfanlu@ust.hk)
@Date    :9/12/2022 15:03
"""
from ev_rgb_isp.losses.loss_factory import MixedLoss
from ev_rgb_isp.losses.metric_factory import MixedMetric


def get_loss(config):
    return MixedLoss(config)


def get_metric(configs):
    return MixedMetric(configs)
