#!/usr/bin/env python3

"""
Contains functions for loading trained Pytorch detection models.
"""

import torchvision

from eurobin_perception.dataset import TaskboardDataset

def get_tb_cnn_model(num_classes):
    """
    Initializes and returns a FastRCNN CNN model that is configured for the
    number of classes in the taskboard dataset.

    Parameters
    ----------
    num_classes: int
        Number of classes that the model should expect

    Returns
    -------
    model: torchvision.models.detection.faster_rcnn.FasterRCNN
        FasterRCNN model object
    """

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model

# TODO: Create model wrapper that maintains labels list.
