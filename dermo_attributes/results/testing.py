import os
import cv2
import wandb
import pickle
import numpy as np
from itertools import repeat
import multiprocessing as mp

from tqdm.contrib.concurrent import process_map

from dermo_attributes.io.paths import Splits
from dermo_attributes.io.preprocess import read_crop_data
from dermo_attributes.io.dataset import load_numpy_data, get_ids
from dermo_attributes.io.images import threshold, get_isic_truth

from dermo_attributes.config import ATTRIBUTE_NAMES, WANDB_PROJECT, WANDB_USER
from dermo_attributes.learning.training import load_model
from dermo_attributes.results.figures import visualize_horizontal

"""
functions to make predictions on the test set
"""


def get_run_name_by_id(run_ids):
    run_names = []
    attributes = []
    api = wandb.Api()
    for run_id in run_ids:
        run = api.run(WANDB_USER + "/" + WANDB_PROJECT + "/" + run_id)
        attributes.append(*run.config["attributes"])
        run_names.append(run.name)
    return run_names, attributes


def run_tests(run_ids, dataset_name="crop_512"):
    """
    Calculate test scores to compare to ISIC competition
    Scoring is done after uncropping and standardizing size
    Models for each attribute are given by their run id
    Returns test scores for each attribute and appends the mean
    """
    run_names, attributes = get_run_name_by_id(run_ids)

    id_list = get_ids(dataset_name, Splits.TEST, [0])
    dataset = load_numpy_data(dataset_name, attributes, id_list, Splits.TEST)

    predictions = []
    for i, run_name in enumerate(run_names):
        model = load_model(run_name)
        predictions.append(model.predict(dataset[..., 0:3]))

    predictions = np.array(predictions, dtype=np.float32)
    predictions = predictions.squeeze(axis=4)
    predictions = np.moveaxis(predictions, 0, 3)
    crop_data = read_crop_data(dataset_name)

    pred_list = process_map(prepare_pred, predictions, [crop_data[str(idx)] for idx in id_list], chunksize=1,
                            max_workers=mp.cpu_count() - 2)
    gt_list = process_map(prepare_gt, id_list, repeat(attributes), chunksize=1, max_workers=mp.cpu_count() - 2)

    pred_list = np.array(pred_list)
    gt_list = np.array(gt_list)

    jaccard = cumulative_jaccard(gt_list, pred_list)
    return jaccard


def prepare_pred(prediction, crop_data):
    """
    function to apply processing of prediction in parallel
    """
    return scale(uncrop(threshold(prediction), crop_data), size=512)


def prepare_gt(isic_id, attributes):
    """
    function to apply processing of ground truth in parallel
    """
    return scale(get_isic_truth(isic_id, Splits.TEST, attributes), size=512)


def cumulative_jaccard(gt, pred):
    """
    calculate cumulative jaccard metric for each of the attributes and mean
    """
    scores = []
    for i in range(gt.shape[-1]):
        tp = np.count_nonzero(np.logical_and(gt[..., i] == 1, pred[..., i] == 1))
        fn = np.count_nonzero(np.logical_and(gt[..., i] == 1, pred[..., i] == 0))
        fp = np.count_nonzero(np.logical_and(gt[..., i] == 0, pred[..., i] == 1))
        scores.append(tp / (tp + fn + fp))
    scores.append(np.array(scores).mean())
    return scores


def scale(masks, size):
    """
    function to rescale the predictions to shape: (size , size , M)
    """
    scaled = [cv2.resize(masks[:, :, j], (size, size), interpolation=cv2.INTER_NEAREST) for j in range(masks.shape[2])]
    return np.stack(scaled, axis=2)


def uncrop(pred, crop_data):
    """
    undoes the effects of preprocessing on the prediction
    """
    canvas = np.zeros((crop_data["y_shape"], crop_data["x_shape"], pred.shape[2]), dtype=pred.dtype)
    up_scaled = scale(pred, crop_data["Z"])
    yo1, yo2, xo1, xo2 = crop_data["yo1"], crop_data["yo2"], crop_data["xo1"], crop_data["xo2"]
    yn1, yn2, xn1, xn2 = crop_data["yn1"], crop_data["yn2"], crop_data["xn1"], crop_data["xn2"]
    canvas[yo1:yo2, xo1:xo2] = up_scaled[yn1:yn2, xn1:xn2]
    return canvas


def make_test_images(run_ids, dataset_name="crop_512"):
    """
    create a visualisation showing a prediction for each model
    models are given by their run id
    """
    isic_examples = {0: 36236, 1: 36237, 2: 36291, 3: 15492, 4: 22039}

    run_names, attributes = get_run_name_by_id(run_ids)

    id_list = [isic_examples[a] for a in attributes]
    dataset = load_numpy_data(dataset_name, attributes, id_list, Splits.VALIDATION)

    predictions = []
    for i, run_name in enumerate(run_names):
        predictions.append(load_model(run_name).predict(dataset[i:i + 1, :, :, 0:3]))

    im = dataset[:, :, :, 0:3]
    gt = np.array([dataset[i, :, :, 3 + k] for i, k in enumerate(attributes)])

    predictions = np.array(predictions)
    predictions = predictions.reshape([len(attributes), predictions.shape[-3], predictions.shape[-2]])

    viz = visualize_horizontal(im, gt, predictions, [ATTRIBUTE_NAMES[k].replace("_", " ") for k in attributes])

    return viz


def make_test_predictions(run_name, dataset):
    """
    used to read and write predictions to disk
    """
    if os.path.exists("data/results/" + run_name + ".pkl"):
        with open("data/results/" + run_name + ".pkl", "rb") as pickle_file:
            return pickle.load(pickle_file)

    model = load_model(run_name)
    predictions = model.predict(dataset)
    with open("data/results/" + run_name + ".pkl", 'wb') as pickle_file:
        pickle.dump(predictions, pickle_file)

    return predictions
