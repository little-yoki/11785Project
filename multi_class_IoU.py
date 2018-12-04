import numpy as np

def mean_IoU(feature_map, ground_truth):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))

    n_cl : number of classes included in ground truth segmentation
    n_ij : number of pixels of class i predicted to belong to class j
    t_i : total number of pixels of class i in ground truth segmentation

    eval_segm: shape N * C * H * W
    ground_truth: N * H * W

    output list of size N, each one is a mean IoU of a image

    '''
    feature_map = np.argmax(feature_map,axis=1)  # N * H * W
    mean_IU_s = []

    for img_ind in range(len(feature_map)):

        eval_segm = feature_map[img_ind, :, :]
        gt_segm = ground_truth[img_ind, :, :]

        cl, n_cl = union_classes(eval_segm, gt_segm)
        _, n_cl_gt = extract_classes(gt_segm)

        eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

        IU = list([0]) * n_cl

        for i, c in enumerate(cl):
            curr_eval_mask = eval_mask[i, :, :]
            curr_gt_mask = gt_mask[i, :, :]
            if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
                continue
            n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))  # intersection

            t_i  = np.sum(curr_gt_mask)
            n_ij = np.sum(curr_eval_mask)

            IU[i] = n_ii / (t_i + n_ij - n_ii)

        mean_IU_s.append(np.sum(IU) / n_cl_gt)
    return mean_IU_s

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _   = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl

def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl

def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask   = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_masks(segm, cl, n_cl):

    h, w  = segm.shape[0], segm.shape[1]
    masks = np.zeros((n_cl, h, w))
    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c
    return masks


# a = np.arange(150).reshape(2,3,5,5)
# print(mean_IoU(a, a.argmax(axis=1)))



