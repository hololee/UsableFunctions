import numpy as np


# ============================== <binary semantic segmentation> ============================= #

def binary_cal_iou(img, target):
    """
    :param img: binary image (ingredients are 0 or 1)
    :param target: binary image (ingredients are 0 or 1)
    :return: iou
    """
    img_region = list(zip(np.where(img > 0)[0], np.where(img > 0)[1]))
    target_region = list(zip(np.where(target > 0)[0], np.where(target > 0)[1]))

    intersection = len(set(img_region).intersection(target_region))
    union = len(img_region) + len(target_region) - intersection

    iou = intersection / union

    return iou


def binary_cal_precision_and_recall(img, target):
    """
    :param img: binary image (ingredients are 0 or 1)
    :param target: binary image (ingredients are 0 or 1)
    :return:
    """
    # area interesting regions.
    img_region = list(zip(np.where(img > 0)[0], np.where(img > 0)[1]))
    target_region = list(zip(np.where(target > 0)[0], np.where(target > 0)[1]))

    TP = list(set(img_region).intersection(target_region))
    FN = set(img_region).difference(TP)
    FP = set(target_region).difference(TP)

    intersection = len(set(img_region).intersection(target_region))
    union = len(img_region) + len(target_region) - intersection

    TN_val = (img.shape[0] * img.shape[1]) - union

    precision = len(TP) / (len(TP) + len(FP))
    recall = len(TP) / (len(TP) + len(FN))

    return precision, recall


# ============================== <semantic segmentation> ============================= #
def semantic_cal_iou(img1, target):
    pass


# ============================== <scores> ============================= #

def cal_F1_score(precision, recall):
    return 2 * (precision * recall / (precision + recall))
