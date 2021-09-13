import numpy as np


def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    # calculate the overlap proportion between the anchor and all bbox for supervise signal,
    # the length of the anchor is 0.01
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    # print inter_len,union_len
    jaccard = np.divide(inter_len, union_len)
    return jaccard

def inter_len_anchors(anchors_min, anchors_max, box_min, box_max):
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    return inter_len

# xmin = 0.5
# xmax = 0.7
# box_min = [0.3, 0.4, 0.7]
# box_max = [0.5, 0.6, 0.9]
# len_anchors = xmax - xmin
# int_xmin = np.maximum(xmin, box_min)
# int_xmax = np.minimum(xmax, box_max)
# inter_len = np.maximum(int_xmax - int_xmin, 0.)
# union_len = len_anchors - inter_len + box_max - box_min
# jaccard = np.divide(inter_len, union_len)
# print(jaccard)
