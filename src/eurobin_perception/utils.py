#!/usr/bin/env python3

"""
Contains various, general utility functions that are used in different
components.
"""

import numpy as np

from scipy.spatial import ConvexHull
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation


def get_bbox_dicts(torch_result, image_id, gt_labels_list):
    """
    Converts a torch model result for a single image to a format that is 
    more convenient for downstream processing.

    The output is a list of bbox dicts, each containing:
      - xmin
      - xmax
      - ymin
      - ymax
      - class
      - confidence

    Parameters
    ----------
    torch_result: dict
        Output of torch model, mapping image_id to a dict containing that
        image's bbox, labels, and score tensors.
    image_id: str
        ID of image (e.g. filename)
    gt_labels_list: list
        Strings defining class names (indices the same as in trained model)

    Returns
    -------
    bboxes: list
        Dicts containing each detected object's information.
    """
    bboxes = []

    bbox_tensor = torch_result[image_id]['boxes']
    label_tensor = torch_result[image_id]['labels']
    score_tensor = torch_result[image_id]['scores']

    for i, row in enumerate(bbox_tensor):
        bboxes.append({'class': None, 'confidence': None,
                       'xmin': None, 'ymin': None, 
                       'xmax': None, 'ymax': None})
        bb_coords = row.tolist()
        bboxes[-1]['xmin'] = int(row[0])
        bboxes[-1]['ymin'] = int(row[1])
        bboxes[-1]['xmax'] = int(row[2])
        bboxes[-1]['ymax'] = int(row[3])

        bboxes[-1]['class'] = gt_labels_list[label_tensor[i] - 1]
        bboxes[-1]['confidence'] = float(score_tensor[i])
        
    return bboxes

def filter_preds(bboxes, score_threshold=None, label_ignore_list=[]):
    """
    Filters out detections whose:
      - scores are below score_threshold
      - labels are in the label_ignore_list

    Parameters
    ----------
    bboxes: list
        Dicts containing each detected object's information.
    score_threshold: float
        Score (i.e. confidence) filtering threshold
    label_ignore_list: list
        Labels whose detections are to be ignored

    Returns
    -------
    filtered_bboxes: list
        Dicts containing each detected object's information, after filtering.
    """
    filtered_bboxes = bboxes[:]

    if score_threshold is not None:
        filtered_bboxes = [bbox for bbox in filtered_bboxes \
                            if bbox['confidence'] > score_threshold]
        
    if label_ignore_list:
        filtered_bboxes = [bbox for bbox in filtered_bboxes \
                            if bbox['class'] not in label_ignore_list]
        
    return filtered_bboxes

def get_intersect_box(bbox_1_dict, bbox_2_dict):
    """
    Returns a box that represents the intersection between two bounding boxes.

    Parameters
    ----------
    bbox_1_dict: dict
        Dict containing the first object's box coordinates.
    bbox_2_dict: dict
        Dict containing the second object's box coordinates.

    Returns
    -------
    intersect_box_dict: dict
        Dict containing the coordinates of the intersection box.
    """
    intersect_box_dict = {}

    if bbox_1_dict['xmin'] < bbox_2_dict['xmin']:
        intersect_box_dict['xmin'] = bbox_2_dict['xmin']
    else:
        intersect_box_dict['xmin'] = bbox_1_dict['xmin']
    if bbox_1_dict['ymin'] < bbox_2_dict['ymin']:
        intersect_box_dict['ymin'] = bbox_2_dict['ymin']
    else:
        intersect_box_dict['ymin'] = bbox_1_dict['ymin']
    if bbox_1_dict['xmax'] < bbox_2_dict['xmax']:
        intersect_box_dict['xmax'] = bbox_1_dict['xmax']
    else:
        intersect_box_dict['xmax'] = bbox_2_dict['xmax']
    if bbox_1_dict['ymax'] < bbox_2_dict['ymax']:
        intersect_box_dict['ymax'] = bbox_1_dict['ymax']
    else:
        intersect_box_dict['ymax'] = bbox_2_dict['ymax']

    return intersect_box_dict

def get_iou_score(bbox_1_dict, bbox_2_dict):
    """
    Computes the Intersection over Union (IoU) score between two bounding boxes,
    which indicates their alignment (0., 1.).

    Parameters
    ----------
    bbox_1_dict: dict
        Dict containing the first object's box coordinates.
    bbox_2_dict: dict
        Dict containing the second object's box coordinates.

    Returns
    -------
    iou_score: float
        Computed IoU score
    """
    intersect_dict = get_intersect_box(bbox_1_dict, bbox_2_dict)
    if (intersect_dict['xmax'] - intersect_dict['xmin']) < 0 or \
        (intersect_dict['ymax'] - intersect_dict['ymin']) < 0:
        intersect_area = 0
    else:
        intersect_area = abs(intersect_dict['xmax'] - intersect_dict['xmin']) * \
                           abs(intersect_dict['ymax'] - intersect_dict['ymin'])

    bbox_1_dict_area = abs(bbox_1_dict['xmax'] - bbox_1_dict['xmin']) * \
                         abs(bbox_1_dict['ymax'] - bbox_1_dict['ymin'])
    bbox_2_dict_area = abs(bbox_2_dict['xmax'] - bbox_2_dict['xmin']) * \
                         abs(bbox_2_dict['ymax'] - bbox_2_dict['ymin'])
    union_area = (bbox_1_dict_area + bbox_2_dict_area) - intersect_area

    return intersect_area / union_area

## ----------------------------------------------------------------------
## Orientation Estimation Helper Functions:
## ----------------------------------------------------------------------

# The following classes and functions enable arrows and text annotations in 3D plots.
# Source: https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)

class Annotation3D(Annotation):

    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """Add an 3d arrow to an `Axes3D` instance."""

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

def _annotate3D(ax, text, xyz, *args, **kwargs):
    """Add anotation `text` to an `Axes3d` instance."""

    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)

# The following functions fit a minimum bounding rectangle on a set of 2D points.
# Source: https://gis.stackexchange.com/a/169633

def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    Parameters
    ----------
    points: ndarray
        2D coordinates of all points (num_points, 2)

    Returns
    -------
    rval: ndarray
        2D coordinates of corners of fit rectangle.
    """
    from scipy.ndimage import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval

# The following function finds the rotation matrix that aligns two input vectors:
# Source: https://stackoverflow.com/a/59204638

def rotation_matrix_from_vectors(vec_1, vec_2):
    """
    Find the rotation matrix that aligns vec_1 to vec_2.

    Parameters
    ----------
    vec_1: ndarray
        3D source vector (3,)
    vec_2: ndarray
        3D destination vector (3,)

    Returns
    -------
    rotation_matrix: ndarray
        3D matrix that rotates vec_1 to align with vec_2 (3, 3)
    """
    a = (vec_1 / np.linalg.norm(vec_1)).reshape(3) 
    b = (vec_2 / np.linalg.norm(vec_2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
