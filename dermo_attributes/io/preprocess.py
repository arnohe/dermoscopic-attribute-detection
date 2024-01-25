from tqdm.contrib.concurrent import process_map
from skimage.util import img_as_ubyte
from itertools import repeat
import numpy as np
import cv2

from dermo_attributes.io.paths import Splits, processed_path, load_json, save_json
from dermo_attributes.io.images import get_isic_input, get_isic_segmentation, get_isic_truth, save_processed_images
from dermo_attributes.io.class_id import get_isic_split

"""
process images from ISIC dataset
"""


def process_all(dataset_name, size):
    """
    process ISIC dataset: images are cropped to square shape without stretching and resized to square shape: size x size

    results are saved to data/dataset_name
    data to undo processing is saved as crop.json
    training-validation split is saved as split.json
    """
    crop_data = []
    for split in list(Splits):
        ids = get_isic_split(split)
        crop_data += process_map(process_one_crop, ids, repeat(size), repeat(dataset_name), repeat(split), chunksize=1)

    crop_data = {idx: data for idx, data in crop_data}
    save_crop_data(crop_data, dataset_name)


def read_crop_data(dataset_name):
    """
    read crop.json data for undo_process function - data format dict {idx: {"property": value}}
    """
    read_file = processed_path(dataset_name) + '/crop.json'
    return load_json(read_file)


def save_crop_data(crop_data, dataset_name):
    """
    Write crop.json with data for undo_process function - data format: dict {"idx": {"property": value}}
    """
    save_file = processed_path(dataset_name) + "/crop.json"
    save_json(crop_data, save_file)


def process_one_crop(isic_id, size, dataset_name, split):
    """
    process and save one image for process_all_crop
    """
    im = get_isic_input(isic_id, split)
    segm = get_isic_segmentation(isic_id, split)
    gt = get_isic_truth(isic_id, split)
    im, segm, gt, crop_data = crop_to_square(im, segm, gt)
    im, segm, gt = scale(im, segm, gt, size)
    save_processed_images(dataset_name, isic_id, im, segm, gt, split)
    return isic_id, crop_data


def scale(img, sgm, gt, size):
    """
    scales all images to shape [size x size]
    """
    h, w, _ = img.shape
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    sgm = cv2.resize(sgm, (size, size), interpolation=cv2.INTER_NEAREST)
    gt = [cv2.resize(gt[:, :, j], (size, size), interpolation=cv2.INTER_NEAREST) for j in range(gt.shape[2])]

    return img, sgm, np.stack(gt, axis=2)


def crop_to_square(img, segm, gt):
    """
    crop images to a square shape around the lesion without stretching - adds black bars if needed
    returns img, segm, gt, crop_data
    """
    # find bounding box of lesion
    contours, hierarchy = cv2.findContours(img_as_ubyte(segm), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])

    # corners of bounding rectangle around lesion with extra padding if possible
    yr1 = np.max([0, y - h // 20])
    yr2 = np.min([y + h + h // 20, img.shape[0]])
    xr1 = np.max([0, x - w // 20])
    xr2 = np.min([x + w + w // 20, img.shape[1]])

    # make bounding square
    w = xr2 - xr1
    h = yr2 - yr1
    z = max(w, h)
    xs1 = (z - w) // 2
    ys1 = (z - h) // 2
    xs2 = z - w - xs1
    ys2 = z - h - ys1

    # coordinates in original image
    yo1 = np.max([0, yr1 - ys1])
    yo2 = np.min([yr2 + ys2, img.shape[0]])
    xo1 = np.max([0, xr1 - xs1])
    xo2 = np.min([xr2 + xs2, img.shape[1]])

    # coordinates in new image
    yn1 = yo1 - (yr1 - ys1)
    yn2 = z + (yo2 - (yr2 + ys2))
    xn1 = xo1 - (xr1 - xs1)
    xn2 = z + (xo2 - (xr2 + xs2))

    # create empty images for cropping
    crop_img = np.zeros([z, z, 3], img.dtype)
    crop_segm = np.zeros([z, z], segm.dtype)
    crop_gt = np.zeros([z, z, 5], gt.dtype)

    # do cropping
    crop_img[yn1:yn2, xn1:xn2] = img[yo1:yo2, xo1:xo2]
    crop_segm[yn1:yn2, xn1:xn2] = segm[yo1:yo2, xo1:xo2]
    crop_gt[yn1:yn2, xn1:xn2] = gt[yo1:yo2, xo1:xo2]

    # dict with extra data for uncropping
    crop_data = {"x_shape": int(img.shape[1]),
                 "y_shape": int(img.shape[0]),
                 "Z": int(z),
                 "yo1": int(yo1),
                 "yo2": int(yo2),
                 "xo1": int(xo1),
                 "xo2": int(xo2),
                 "yn1": int(yn1),
                 "yn2": int(yn2),
                 "xn1": int(xn1),
                 "xn2": int(xn2)}
    return crop_img, crop_segm, crop_gt, crop_data


def undo_process(isic_id, im, dict_crop_data, nearest=True):
    """
    undo cropping and scaling done in processing (A x B x K --> N x M x K)
    """
    if nearest:
        mode = cv2.INTER_NEAREST
    else:
        mode = cv2.INTER_LINEAR
    data = dict_crop_data[isic_id]

    # undo scale
    scaled = [cv2.resize(im[:, :, k], (data["Z"], data["Z"]), interpolation=mode) for k in range(im.shape[2])]
    scaled = np.stack(scaled, axis=2)
    # undo crop
    undo_crop = np.zeros([data["shape"][1], data["shape"][0]], dtype=im.dtype)
    undo_crop[data["yo1"]:data["yo2"], data["xo1"]:data["xo2"]] = \
        scaled[data["yn1"]:data["yn2"], data["xn1"]:data["xn2"]]
    return undo_crop
