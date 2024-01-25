from itertools import product
from re import compile
from enum import Enum
import json
import os

from dermo_attributes.config import BASE_FOLDER, ATTRIBUTE_NAMES, NUM_ATTRIBUTES

"""
these functions return or create the folders where images are stored
"""


class Splits(Enum):
    TRAIN = "training"
    VALIDATION = "validation"
    TEST = "test"


Datatypes = Enum("Datatypes",
                 {"IN": "input",
                  "SEG": "segmentation",
                  "GTX": "ground_truth",
                  **{"GT" + str(i): name for i, name in enumerate(ATTRIBUTE_NAMES)}})


def id_to_prefix(isic_id):
    """
    returns prefix for file with isic_id
    """
    return "ISIC_" + str(isic_id).zfill(7)


def str_to_id(name):
    """
    returns id from a file name as int
    (last set of digits in string: *ISIC_X...*)
    """
    regex = compile(r'\d+')
    return int(regex.findall(name)[-1])


def id_to_file(isic_idx, datatype):
    """
    returns the full isic file name
    """

    prefix = id_to_prefix(isic_idx)
    match datatype:
        case Datatypes.IN:
            return prefix + ".jpg"
        case Datatypes.SEG:
            return prefix + "_" + datatype.value + ".png"
        case Datatypes.GTX:
            return [prefix + "_attribute_" + at + ".png" for at in ATTRIBUTE_NAMES]
        case _:
            return prefix + "_attribute_" + datatype.value + ".png"


def isic_path(split, datatype=None, isic_idx=None):
    """
    returns path to isic files
    """
    if datatype in ATTRIBUTE_NAMES:
        datatype = Datatypes.GTX

    folder = BASE_FOLDER + "/ISIC/" + split.value
    if datatype is None:
        return folder
    folder += "/" + datatype.value
    if isic_idx is None:
        return folder

    if datatype != Datatypes.GTX:
        return folder + "/" + id_to_file(isic_idx, datatype)
    else:
        return [folder + '/' + f for f in id_to_file(isic_idx, datatype)]


def processed_path(dataset_name, split=None, section=None, isic_idx=None):
    """
    returns path to isic files
    """
    if section == Datatypes.GTX:
        return [processed_path(dataset_name, split, k, isic_idx) for k in list(Datatypes)[-NUM_ATTRIBUTES:]]
    folder = BASE_FOLDER + "/processed/" + dataset_name
    if split is None:
        return folder
    folder += "/" + split.value
    if section is None:
        return folder
    folder += "/" + section.value
    if isic_idx is None:
        return folder
    else:
        return folder + "/" + id_to_file(isic_idx, section)


def clear_folder(folder):
    """
    remove folder contents
    """
    path = os.path.normpath(folder)
    for root, dirs, files in os.walk(path, topdown=False):
        for f in files:
            os.remove(os.path.join(root, f))
        for d in dirs:
            os.rmdir(os.path.join(root, d))


def create_new_processed_folders(dataset_name):
    """
    make folder structure for storing processed files
    """
    clear_folder(processed_path(dataset_name))
    split_list = list(Splits)
    datatype_list = list(Datatypes)
    datatype_list.remove(Datatypes.GTX)
    for split, datatype in product(split_list, datatype_list):
        os.makedirs(processed_path(dataset_name, split, datatype))


def save_json(dict_to_save, file_path):
    with open(file_path, 'w') as write_file:
        json.dump(dict_to_save, write_file)


def load_json(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as read_file:
        return json.load(read_file)
