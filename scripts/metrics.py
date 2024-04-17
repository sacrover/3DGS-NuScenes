import numpy as np
from scripts.utils import normalize_depth


def compute_iou(model_seg, gt_seg):
    """Calculate IoU for each class."""
    all_iou = []

    for seg_class in np.unique(gt_seg):

        # ignore void, traffic.cone and flat.other (for COCO and gt comparison)
        if seg_class == 0 | seg_class == 8 | seg_class == 12:
            continue
        # Find indices belonging to the current class
        class_indices = model_seg == seg_class
        not_class_indices = np.logical_not(class_indices)

        # Calculate TP, FP, FN for the current class
        TP = np.sum(gt_seg[class_indices] == seg_class)
        FP = np.sum(gt_seg[class_indices] != seg_class)
        FN = np.sum(gt_seg[not_class_indices] == seg_class)

        # Calculate IoU for the current class
        if TP + FP + FN < 10:
            # class_iou = 0  # Avoid division by zero
            continue
        else:
            class_iou = TP / (TP + FP + FN)
        print(f"seg class: {seg_class}, iou: {class_iou:.3f}, TP|FP|FN, {TP}  {FP} {FN}")

        all_iou.append(class_iou)

    mean_iou = sum(all_iou) / (len(all_iou) - 1)
    print("mean IOU: ", mean_iou)

    return all_iou, mean_iou


def compute_batch_iou():
    # TODO
    pass


def mse(image1, image2):
    # Convert images to grayscale
    image1_gray = image1.convert('L')
    image2_gray = image2.convert('L')

    # Convert grayscale images to NumPy arrays
    norm_array1 = normalize_depth(image1_gray)
    norm_array2 = normalize_depth(image2_gray)

    # Compute squared difference
    squared_difference = (norm_array1 - norm_array2) ** 2

    # Compute mean squared difference for the entire image
    mse = np.mean(squared_difference)

    return mse
