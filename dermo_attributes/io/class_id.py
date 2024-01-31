from itertools import repeat
from glob import glob

from tqdm.contrib.concurrent import process_map
import numpy as np
import multiprocessing as mp

from dermo_attributes.io.paths import processed_path, load_json, save_json, Splits, isic_path, Datatypes, str_to_id
from dermo_attributes.io.images import get_isic_truth

"""
functions to manage distribution of classes over the dataset
"""


def read_training_splits(dataset_name):
    """
    get list of isic ids in 5 splits of 20 %
    format dict: {"split_k": [idx]}
    """
    file_path = processed_path(dataset_name) + "/split.json"
    data_split = load_json(file_path)
    if data_split is not None:
        return data_split
    return create_data_split(dataset_name)


def create_data_split(dataset_name):
    """
    split isic ids into 5 splits of 20 % split
    attempts to distribute classes evently over splits
    format dict: {"split_k": [idx]}
    """
    class_labels_dict = get_class_labels_dict()
    splits = divide_dict_of_lists(class_labels_dict, 5)
    splits_dict = {int(n): splits[n] for n in range(5)}
    new_file = processed_path(dataset_name) + "/split.json"
    save_json(splits_dict, new_file)

    return splits


def divide_dict_of_lists(dict_of_lists, partition_count=5):
    """
    split dict {keys: "list"} into equal parts
    attempts to distribute keys evently over parts
    """
    keys = list(dict_of_lists.keys())
    extras = []
    divided_lists = [[] for i in range(partition_count)]
    rng = np.random.default_rng(seed=12345)
    rng.shuffle(keys)
    for key in keys:
        idx_list = np.array(dict_of_lists[key])
        rng.shuffle(idx_list)
        split_end = idx_list.shape[0] // partition_count * partition_count
        if split_end >= partition_count:
            division = np.split(idx_list[: split_end], partition_count)
            for n in range(partition_count):
                divided_lists[n] += division[n].tolist()
        extras.append(idx_list[split_end:])
    extras = np.concatenate(extras, axis=0).tolist()
    for n in range(min(len(extras), partition_count)):
        divided_lists[n] += extras[n::partition_count]
    return divided_lists


def get_class_labels_dict(split=Splits.TRAIN):
    """
    split isic ids into 5 splits of 20 % split
    attempts to distribute classes evently over splits
    format dict: {"split_k": [idx]}
    """
    file_path = isic_path(split) + "/class_id.json"
    class_labels_dict = load_json(file_path)
    if class_labels_dict is not None:
        return class_labels_dict
    return make_class_labels_dict()


def make_class_labels_dict(split=Splits.TRAIN):
    """
    create dict mapping class_labels to idx
    class label example: "23" for negative_network and pigment_network
    format dict: {"classes": [idx]}
    """
    id_list = get_isic_split(split)
    class_dict = {}
    class_data = process_map(find_class_labels, id_list, repeat(split), chunksize=1, max_workers=mp.cpu_count() - 2)
    for class_str, idx in class_data:
        if class_str not in class_dict.keys():
            class_dict[class_str] = [idx]
        else:
            class_dict[class_str] += [idx]

    new_file = isic_path(split) + "/class_id.json"
    save_json(class_dict, new_file)

    return class_dict


def find_class_labels(isic_idx, split=Splits.TRAIN):
    """
    reads isic image by idx
    returns string of numbers representing present classes ("23" for negative_network and pigment_network)
    """
    gt = get_isic_truth(isic_idx, split)
    found = [str(k) if np.any(gt[:, :, k]) else "" for k in range(gt.shape[2])]
    return "".join(found), isic_idx


def get_isic_split(split):
    """
    returns list of ids that are part of a split
    """
    file_extension = ".jpg"
    path = isic_path(split, Datatypes.IN) + "/*" + file_extension
    files = glob(path)
    return [str_to_id(t) for t in files]
