#!/usr/bin/env python3

"""
Contains the class that encapsulates the image object detection model.
"""

import time

import cv2
import torch

from torchvision import tv_tensors

from eurobin_perception.models import get_tb_cnn_model
from eurobin_perception.dataset import TaskboardDataset, apply_transforms
from eurobin_perception.utils import get_bbox_dicts, filter_preds
from eurobin_perception.visualization import load_class_color_map, annotate_image


class ImageDetector(object):
    """
    RGB Image-based CNN object detection model.

    Note: this class was designed and is currently used for the taskboard
    challenge and dataset, but could be made more general in the future.
    """

    def __init__(self, dataset_dir_path, model_weights_file_path, 
                 class_colors_file_path, confidence_threshold,
                 device='cpu'):
        self.model_weights_file_path = model_weights_file_path
        self.confidence_threshold = confidence_threshold
        self.device = device

        self.dataset = TaskboardDataset(root=dataset_dir_path)
        self.labels_list = self.dataset.get_labels_list()
        self.class_colors_dict = load_class_color_map(class_colors_file_path)
        self.tensor_transforms = apply_transforms()

        self.name = self.__class__.__name__
        self.load_model()

    def load_model(self):
        """
        Loads the pre-trained CNN model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        print(f'[INFO] [{self.name}] Loading pretrained Faster R-CNN model' + \
              f' from: {self.model_weights_file_path}')
        self.model = get_tb_cnn_model()
        self.model.load_state_dict(torch.load(self.model_weights_file_path, 
                                              map_location=self.device))
        self.model.eval()

    def detect_objects(self, image_cv, return_annotated_image=True):
        """
        Runs the detection model on the given image, processes the output,
        and returns the bounding box information, the model inference time,
        and (optionally) the input annotated with the detected objects
        (the color-coded bounding boxes, label names, and confidence scores).

        Parameters
        ----------
        image_cv: ndarray
            Input image array (in BGR format)
        return_annotated_image: bool
            Whether to annotate the input image and return the result

        Returns
        -------
        bboxes: list
            Dicts containing info on each bbox:
            (class, xmin, ymin, xmax, ymax, confidence)
        model_inference_time: float
            Time taken by the model to run on the input image
        annotated_image_cv: float
            Annotated image array (in BGR format)
        """
        # Preprocess input image:
        image_torch = tv_tensors.Image(image_cv)
        
        image_torch = image_torch.permute((2, 0, 1))                        # Source: (H, W, C). Desired(C, H, W)
        image_torch = self.tensor_transforms(image_torch)
        model_inputs = [image_torch.to(self.device)]

        # Run NN model on input image:
        model_start_time = time.time()
        outputs = self.model(model_inputs)
        model_inference_time = time.time() - model_start_time 

        # Process NN outputs:
        bboxes = get_bbox_dicts(outputs[0], 
                                [item for item in self.labels_list if item != 'background'])
        bboxes = filter_preds(bboxes, score_threshold=self.confidence_threshold)

        annotated_image_cv = None
        if return_annotated_image:
            # Annotate image with BBs:
            annotated_image_cv = annotate_image(image_cv, bboxes, 
                                                self.class_colors_dict,
                                                input_encoding='bgr')

        return bboxes, model_inference_time, annotated_image_cv 
