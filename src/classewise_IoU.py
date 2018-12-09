#!/usr/bin/env python3
import numpy as np


def classwise_IoU(feature_map, ground_truth):
    """
    sample usage:
    ious, occurs = np.zeros(n_classes), np.zeros(n_classes)

    for feature_map, ground_truth in dataloader:
        new_ious, new_occurs = class_wise_IoU(feature_map, ground_truth)
        iou += new_ious
        occurs += new_occurs

    iou /= occurs    # (C,) numpy array, weighted classwise IoU

    :param feature_map: [N x C x H x W] numpy array, could either be output of last layer or logits
    :param ground_truth: [N x H x W] numpy array
    :return: (C,) numpy array: sum of iou for every class
             (C,) numpy array: number of occurences for every class
    """
    n_samples = feature_map.shape[0]  # N
    n_classes = feature_map.shape[1]  # C
    h = feature_map.shape[2]  # H
    w = feature_map.shape[3]  # W
    preds = np.argmax(feature_map, axis=1)  # N x H x W

    ious = [0. for i in range(n_classes)]
    occurences = [0 for i in range(n_classes)]

    for idx in range(n_samples):
        pred = preds[idx, :, :]
        gt = ground_truth[idx, :, :]

        classes = np.unique([pred, gt])
        for c in classes:
            iou, occur = single_class_iou(c, pred, gt)
            ious[c] += iou
            occurences[c] += occur

    return np.array(ious), np.array(occurences)


def single_class_iou(class_id, pred, gt):
    """
    :param class_id: scalar, the idx of the class for which to calculate iou
    :param pred: [H x W] numpy array
    :param gt: [H x W] numpy array
    :return: the sum of IoU value of given class, the number of occurence of given class in the ground truth
    """
    pred_mask = (pred == class_id)
    gt_mask = (gt == class_id)

    if np.sum(gt_mask) == 0:
        return 0, 0

    intersection = np.sum(np.logical_and(pred_mask, gt_mask))
    union = np.sum(gt_mask) + np.sum(pred_mask) - intersection

    occur = np.sum(gt_mask)
    iou = occur * (intersection / union)  # the IoU summed over occurences of the class

    del pred_mask, gt_mask

    return iou, occur


if __name__ == '__main__':
    # test case
    a = np.random.random((2,3,5,5))   # N x C x H x W
    iou, occur = classwise_IoU(a, a.argmax(axis=1))
    print(iou)
    print(occur)
    assert sum(iou) == 2 * 5 * 5
    assert sum(occur) == 2 * 5 * 5
