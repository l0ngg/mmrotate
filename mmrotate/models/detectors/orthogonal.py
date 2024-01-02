# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_DETECTORS
from .single_stage import RotatedSingleStageDetector


@ROTATED_DETECTORS.register_module()
class Orthogonal(RotatedSingleStageDetector):
    """Implementation of Orthogonal Detector."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Orthogonal, self).__init__(backbone, neck, bbox_head,
                                               train_cfg, test_cfg, pretrained)
