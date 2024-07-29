#!/usr/bin/env python3

"""
Contains functions and classes that are used to load and manipulate the data
for training and testing the object detection model.
"""

import os
import torch

import numpy as np
import xml.etree.ElementTree as ET

from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T


## ----------------------------------------------------------------------
## Helper functions:
## ----------------------------------------------------------------------

def load_xml_labels(filepath):
    """
    Loads the XML label information from a given file and returns
    the image ID and a list of dicts containing bbox information.

    Parameters
    ----------
    filepath: str
        Absolute path to an XML file.

    Returns
    -------
    image_id: str
        Unique Image ID (filename without extension, e.g. jpg).
    bboxes: list
        Dicts containing info on each bbox (class, xmin, ymin, xmax, ymax)
    """
    xml_root = ET.parse(filepath).getroot()

    image_id = None
    bboxes = []

    for xml_element in xml_root:
        if 'filename' in xml_element.tag:
            image_id = xml_element.text.split('.')[0]
        if 'object' in xml_element.tag:
            bboxes.append({'class': None, 'confidence': None,
                           'xmin': None, 'ymin': None, 
                           'xmax': None, 'ymax': None})
            for xml_object_element in xml_element:
                if xml_object_element.tag == 'name':
                    bboxes[-1]['class'] = xml_object_element.text
                if xml_object_element.tag == 'bndbox':
                    bboxes[-1]['xmin'] = int(xml_object_element.find('xmin').text)
                    bboxes[-1]['ymin'] = int(xml_object_element.find('ymin').text)
                    bboxes[-1]['xmax'] = int(xml_object_element.find('xmax').text)
                    bboxes[-1]['ymax'] = int(xml_object_element.find('ymax').text)

    return image_id, bboxes

def get_bboxes_array(bboxes_list):
    """
    Aggregates bbox coordinates in a single numpy array.
    Each row contains (xmin, ymin, xmax, ymax)

    Parameters
    ----------
    bboxes_list: list
        Dicts containing info on each bbox (class, xmin, ymin, xmax, ymax, confidence)

    Returns
    -------
    bboxes_array: numpy.ndarray
        Array containing bbox coordinates.
    """
    ## TODO: Switch to list comprehension:
    bbox_list = []
    for bbox_dict in bboxes_list:
        bbox_list.append(np.array([bbox_dict['xmin'], 
                                  bbox_dict['ymin'],
                                  bbox_dict['xmax'], 
                                  bbox_dict['ymax']]))
    return np.array(bbox_list)

def get_labels_array(bboxes_list, gt_labels_list):
    """
    Aggregates bbox labels in a single numpy array.
    Each row contains the index of the label in the 
    gt_labels_list.
    
    Note: index 0 is reserved for background.

    Parameters
    ----------
    bboxes_list: list
        Dicts containing info on each bbox:
        (class, xmin, ymin, xmax, ymax, confidence)
    gt_labels_list: list
        Strings representing each class.

    Returns
    -------
    labels_array: numpy.ndarray
        Array containing label indices.
    """
    ## TODO: Switch to list comprehension:
    labels_list = []
    for bbox_dict in bboxes_list:
        # labels_list.append(gt_labels_list.index(bbox_dict['class']) + 1)
        labels_list.append(gt_labels_list.index(bbox_dict['class']))

    return np.array(labels_list)


## Note: Extra data augmentation steps; possibly unnecessary
## -----------------------------------------------------------
def apply_transforms(training=False):
    """
    Returns a function that composes and applies tensor transformations
    from the torchvision library:
      - Changing data type to float
      - Converting TVTensor objects to "pure tensors" (removing metadata)
      - If tensors are for training, applies random horizontal flips
        (augmentation)

    Inspired by the following resource:
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

    Parameters
    ----------
    None

    Returns
    -------
    transforms: torchvision.transforms.v2._container.Compose
        Function that composes and applies tensor transformations
    """
    transforms = []
    if training:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)
## -----------------------------------------------------------


## ----------------------------------------------------------------------
## Dataset definition:
## ----------------------------------------------------------------------

class TaskboardDataset(torch.utils.data.Dataset):
    """
    Dataset for detecting the taskboard and its components from RGB images.

    Files should be located under root in sub-directories images/ and annotations/.

    The images are divided into categories according to the conditions under
    which they were taken (useful for later analyses). image_dirs_dict defines
    each category and the sub-directory containing the image files.
    The locations of corresponding label files under annotations/ are defined
    in label_dirs_dict.

    The root directory should contain a labels.txt containing the names of each
    class in the order of their indices in the trained model.
    (Note: the first i.e. 0th class should be "background".)
    """

    def __init__(self, root, training=False):
        self.root = root
        self.training = training
        self.image_dirs_dict = {'low_light': 'realsense/dark_low_light', 
                                'med_light': 'realsense/near_window', 
                                'high_light': 'realsense/light_on'}
        self.label_dirs_dict = {'low_light': 'dark_low_light', 
                                'med_light': 'near_window', 
                                'high_light': 'light_on'}
        self.imgs = []
        self.labels = []
        self.gt_labels_list = None
        self.num_classes = None

        self.load_img_and_label_filenames()
        self.load_labels()

        self.transforms = apply_transforms(training)
        
    def load_img_and_label_filenames(self):
        """
        Loads and stores all image and corresponding label (XML) filenames.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        assert list(self.image_dirs_dict.keys()) == list(self.label_dirs_dict.keys()), \
                '[ERROR] Non-identical data categories in image_dirs_dict' + \
                ' and label_dirs_dict'
        
        for data_category in self.image_dirs_dict.keys():
            images_path = os.path.join(self.root, 'images', 
                                       self.image_dirs_dict[data_category])
            annotations_path = os.path.join(self.root, 'annotations', 
                                            self.label_dirs_dict[data_category])
            img_filenames = [os.path.join(images_path, filename) \
                             for filename in \
                             list(sorted(os.listdir(images_path)))]
            label_filenames = [os.path.join(annotations_path, filename) \
                               for filename in \
                               list(sorted(os.listdir(annotations_path)))]
            
            self.imgs.extend(img_filenames)
            self.labels.extend(label_filenames)
        
    def load_labels(self, remove_bg=False):
        """
        Loads and stores the list of labels.

        Parameters
        ----------
        remove_bg: bool
            Whether to exclude the "background" class from the list.

        Returns
        -------
        None
        """
        labels_file_path = os.path.join(self.root, 'labels.txt')
        with open(labels_file_path, 'r') as filehandle:
            self.gt_labels_list = filehandle.read().splitlines()

        if remove_bg:
            self.gt_labels_list.remove('background')

        self.num_classes = len(self.gt_labels_list)
        
    def get_labels_list(self):
        """
        Returns the labels list.

        Parameters
        ----------
        None

        Returns
        -------
        gt_labels_list: list
            Strings defining class names (indices the same as in trained model)
        """
        return self.gt_labels_list
    
    def get_num_classes(self):
        """
        Returns the number of classes in the dataset.

        Parameters
        ----------
        None

        Returns
        -------
        num_classes: int
            Number of unique classes in the dataset.
        """
        return self.num_classes
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label_path = self.labels[idx]
        img = read_image(img_path)
        img_id, bboxes_list = load_xml_labels(label_path)

        bboxes_tensor = torch.Tensor(get_bboxes_array(bboxes_list))
        labels_tensor = torch.Tensor(get_labels_array(bboxes_list, 
                                                      self.gt_labels_list)).long()

        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(bboxes_tensor, format="XYXY", 
                                                   canvas_size=F.get_size(img))
        target["labels"] = labels_tensor
        target["image_id"] = img_id

        img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
