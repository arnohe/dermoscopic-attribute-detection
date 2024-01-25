from skimage.util import img_as_float32, img_as_ubyte, img_as_float
from skimage.io import imread, imsave
import numpy as np
import cv2

from dermo_attributes.io.paths import Splits, Datatypes, isic_path, processed_path
from dermo_attributes.config import NUM_ATTRIBUTES

"""
functions to read/write images in the folder structure
"""


def get_isic_input(idx, split=Splits.TRAIN):
    """
    read isic input
    return type float32, range [0,1], shape NxMx3 (RGB)
    """
    section = Datatypes.IN
    return img_as_float32(imread(isic_path(split, section, idx), False))


def get_isic_truth(idx, split=Splits.TRAIN, attributes=None):
    """
    read isic attribute ground truth
    return type float32, range [0,1], shape NxMxK
    """
    if attributes is None:
        attributes = list(range(NUM_ATTRIBUTES))
    section = Datatypes.GTX
    files = isic_path(split, section, idx)
    gt_list = [img_as_float32(imread(files[a], False)) for a in attributes]
    return np.stack(gt_list, axis=2)


def get_isic_segmentation(idx, split=Splits.TRAIN):
    """
    read isic segmentation mask
    return type float32, range [0,1], shape NxM
    """
    section = Datatypes.SEG
    return img_as_float32(imread(isic_path(split, section, idx), True))


def save_processed_images(dataset_name, isic_idx, img, sgm, gt, split):
    """
    save processed images from processed.py to folder processed_name in the relevant folder structure
    """
    im_file = processed_path(dataset_name, split, Datatypes.IN, isic_idx)
    seg_file = processed_path(dataset_name, split, Datatypes.SEG, isic_idx)
    gt_files = processed_path(dataset_name, split, Datatypes.GTX, isic_idx)

    imsave(im_file, img_as_ubyte(img), check_contrast=False)
    imsave(seg_file, img_as_ubyte(sgm), check_contrast=False)
    for i, file in enumerate(gt_files):
        imsave(file, img_as_ubyte(gt[:, :, i]), check_contrast=False)


def convert_dtype(image, dtype_str):
    converter = img_as_float if "float" in dtype_str else img_as_ubyte
    image_conv = converter(image)
    if dtype_str == "float32":
        return image_conv.astype(np.float32)
    if dtype_str == "float16":
        return image_conv.astype(np.float16)
    return image_conv


def reorder_list(the_list, order):
    if order is None:
        order = list(range(len(the_list)))
    return [the_list[i] for i in order]


def get_processed_truth(isic_idx, attributes, dataset_name, split=Splits.TRAIN, dtype_str="float32"):
    """
    read processed ground truth
    return dtype float32, range [0,1], shape NxNxK
    """
    file_names = processed_path(dataset_name, split, Datatypes.GTX, isic_idx)
    file_names = reorder_list(file_names, attributes)

    truths = [imread(file, as_gray=True) for file in file_names]
    truths = [convert_dtype(t, dtype_str) for t in truths]
    return np.stack(truths, axis=2)


def get_processed_input(isic_idx, dataset_name, split=Splits.TRAIN, dtype_str="float32"):
    """
    read processed isic input
    return dtype float 32, range [0-1], shape NxNx3
    """
    file = processed_path(dataset_name, split, Datatypes.IN, isic_idx)
    image = imread(file, as_gray=False)
    return convert_dtype(image, dtype_str)


def get_processed_both(isic_idx, attributes, dataset_name, test=Splits.TRAIN, dtype="float32"):
    """
        read processed input and ground truth
        return dtype float32, range [0,1], shape NxNx(3+K)
        """
    image = get_processed_input(isic_idx, dataset_name, test, dtype)
    truth = get_processed_truth(isic_idx, attributes, dataset_name, test, dtype)
    return np.dstack((image, truth))


def grey_to_rgb(x):
    """
    convert NxMx1 -> NxMx3
    """
    return np.concatenate((x,) * 3, axis=2)


def flatten_gray(x):
    """
    convert NxMx1 -> NxM
    """
    return x.reshape((x.shape[0], x.shape[1]))


def rgba_to_bgra(im):
    return im[:, :, [2, 1, 0, 3]]


def rgb_to_bgr(im):
    return im[:, :, [2, 1, 0]]


def threshold(im):
    if "uint8" in str(im.dtype):
        greater = im > 127
        return 255 * greater.astype(np.uint8)
    else:
        return im.round()


def visualize_matrix(im, truth, prediction):
    """
    Create image showing confusion matrix
    TP in green
    FP in red
    FN in blue
    TN in black
    """
    # im = convert_dtype(im, "uint8")
    truth_c = convert_dtype(truth, "uint8")
    prediction_c = threshold(convert_dtype(prediction, "uint8"))
    true_positive = np.logical_and(truth_c == 255, prediction_c == 255)
    false_positive = np.logical_and(truth_c == 0, prediction_c == 255)
    false_negative = np.logical_and(truth_c == 255, prediction_c == 0)
    confusion_image = np.zeros([truth_c.shape[0], truth_c.shape[1], 3], dtype=np.uint8)
    confusion_image[true_positive] = [0, 255, 0]
    confusion_image[false_positive] = [255, 0, 0]
    confusion_image[false_negative] = [0, 0, 255]
    # im_overlay = cv2.addWeighted(im, 0.5, confusion_image, 0.5, 0)
    return confusion_image


def visualize_contours(im, truth, prediction, thickness=3):
    """
    Create an image overlaying contours of prediction and ground truth on the input image
    """
    im_draw = convert_dtype(im, "uint8").copy()
    truth_c = convert_dtype(truth, "uint8")
    prediction_c = threshold(convert_dtype(prediction, "uint8"))
    truth_cont, _ = cv2.findContours(truth_c, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    im_draw = cv2.drawContours(im_draw, truth_cont, -1, (0, 255, 0), thickness=thickness)
    prediction_cont, _ = cv2.findContours(prediction_c, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    im_draw = cv2.drawContours(im_draw, prediction_cont, -1, (0, 0, 255), thickness=thickness)
    return im_draw


def visualize_probability(im, prob):
    """
    Create image overlaying predicted probability on the original image (as a heatmap)
    """
    y_pred = convert_dtype(prob, "uint8")
    overlay = cv2.cvtColor(cv2.applyColorMap(y_pred, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)  # color hot?
    im = convert_dtype(im, "uint8")
    im_overlay = cv2.addWeighted(im, 0.5, overlay, 0.5, 0)
    return im_overlay
