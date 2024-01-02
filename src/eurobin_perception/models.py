#!/usr/bin/env python3

"""
Contains functions for loading trained Pytorch detection models.
"""

import torchvision

from eurobin_perception.dataset import TaskboardDataset

tb_dataset_path_ = '/home/ahmed/tum/workspace/euRobin/detection_task/dataset'

def get_tb_cnn_model():
    """
    Initializes and returns a FastRCNN CNN model that is configured for the
    number of classes in the taskboard dataset.

    Parameters
    ----------
    None

    Returns
    -------
    model: torchvision.models.detection.faster_rcnn.FasterRCNN
        FasterRCNN model object
    """
    tb_dataset = TaskboardDataset(root=tb_dataset_path_)
    num_classes = tb_dataset.get_num_classes()

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model

# TODO: Create model wrapper that maintains labels list.
