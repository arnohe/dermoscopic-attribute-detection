from itertools import chain, repeat
from glob import glob

from tqdm.contrib.concurrent import process_map
import imgaug.augmenters as aug
import tensorflow as tf
import numpy as np
import imgaug

from dermo_attributes.io.class_id import read_training_splits, get_class_labels_dict, get_isic_split
from dermo_attributes.io.images import get_processed_both, get_processed_input
from dermo_attributes.io.paths import Splits

"""
functions that create a tf.data.dataset object to use in training
"""


#
def get_dataset(dataset_name, attribute_list, batch_size, backbone, isic_split=Splits.TRAIN, train_splits=None,
                balanced=False, augmentation_value=0, drop_remainder=False):
    """
    create a tf.data.dataset pipeline
    attribute_list is a numeric list 0-4
    train_splits is numeric list to contain training splits 0-4
    """

    # index lists to sample
    id_list = get_ids(dataset_name, isic_split, train_splits)
    num_steps = get_num_steps(len(id_list), batch_size, drop_remainder)
    if balanced:
        data_set = balanced_dataset_ids(attribute_list, id_list, isic_split == Splits.TRAIN)
    else:
        data_set = unbalanced_dataset_ids(id_list, isic_split == Splits.TRAIN)

    # batch indexes
    data_set = data_set.batch(batch_size, drop_remainder, tf.data.AUTOTUNE, True)

    # get numpy image data for index
    numpy_data = load_numpy_data(dataset_name, attribute_list, id_list, isic_split)
    data_set = data_set.map(func_data_from_ids(numpy_data), tf.data.AUTOTUNE, True)

    # add augmentation and preprocessing
    aug_function = get_augmentation_function(augmentation_value, backbone)
    data_set = data_set.map(aug_function, tf.data.AUTOTUNE, True)

    # prefetch batches
    data_set = data_set.prefetch(tf.data.AUTOTUNE)

    return data_set, num_steps


def get_ids(dataset_name, isic_split, train_splits):
    """
    get isic ids for dataset
    """
    if isic_split == Splits.TRAIN and train_splits is not None:
        all_ids = [read_training_splits(dataset_name)[str(k)] for k in train_splits]
        return list(chain(*all_ids))

    return get_isic_split(isic_split)


def get_num_steps(num_elements, batch_size, drop_remainder):
    num_steps = num_elements // batch_size
    if not drop_remainder:
        num_steps += (1 if num_elements % batch_size != 0 else 0)
    return num_steps


def load_numpy_data(dataset_name, attribute_list, id_list, isic_split):
    """
    load image and ground_truth data into ndarray
    """
    if attribute_list is not None:
        data = process_map(get_processed_both, id_list, repeat(attribute_list),
                           repeat(dataset_name), repeat(isic_split), repeat("uint8"), chunksize=1)
    else:
        data = process_map(get_processed_input, id_list,
                           repeat(dataset_name), repeat(isic_split), repeat("uint8"), chunksize=1)
    return np.stack(data, axis=0)


def balanced_dataset_ids(attribute_list, id_list, shuffle):
    """
    cycle the different attributes using oversampling
    returns a tf.Dataset
    """
    class_split_ids = split_train_ids_by_class(id_list, attribute_list)
    class_split_pos = [[id_list.index(i) for i in s] for s in class_split_ids]
    data_sets = [tf.data.Dataset.from_tensor_slices(pos) for pos in class_split_pos]
    if shuffle:
        data_sets = [d.shuffle(len(pos), None, True) for pos, d in zip(class_split_pos, data_sets)]
    data_sets = [d.repeat() for d in data_sets]

    return tf.data.Dataset.sample_from_datasets(data_sets, len(data_sets) * [1 / len(data_sets)], None, False)


def unbalanced_dataset_ids(id_list, shuffle):
    """
    returns a (shuffled) tf.Dataset from ids
    """
    data_set = tf.data.Dataset.range(len(id_list))
    if shuffle:
        data_set = data_set.shuffle(len(id_list), None, True)
    return data_set


def get_augmentation_function(aug_value, backbone, dtype=tf.float32):
    """
    returns the image augmentation function
    """
    x_aug, xy_aug, preprocess, mode = augmentation_setup(aug_value, backbone)

    def augmentation(im_gt):
        if 0 < aug_value < 1:
            xy = tf.numpy_function(
                lambda t: xy_aug(images=np.concatenate([x_aug(images=t[:, :, :, 0:3]), t[:, :, :, 3:]], axis=3)),
                [im_gt], tf.uint8, False)
            _, w, h, d = im_gt.shape
            xy.set_shape((None, w, h, d))
        else:
            xy = im_gt
        xy = tf.cast(xy, dtype=dtype)

        x = tf.slice(xy, [0, 0, 0, 0], [-1, -1, -1, 3])
        x = preprocess(x, mode=mode)
        if im_gt.shape[3] <= 3:
            return x

        y = tf.slice(xy, [0, 0, 0, 3], [-1, -1, -1, -1])
        y = tf.round(tf.clip_by_value(y / 255.0, 0., 1.))
        return x, y

    return augmentation


def augmentation_setup(aug_value, backbone):
    """
    setup for the augmentation function
    """
    imgaug.seed(1)
    x_aug = aug.Sequential([
        aug.Multiply((1 - aug_value, 1 + aug_value)),
    ])
    xy_aug = aug.Sequential([
        aug.Fliplr(0.5),
        aug.Flipud(0.5),
        aug.Rot90([0, 1]),
        aug.Affine(scale={"x": (1 - aug_value, 1 + aug_value), "y": (1 - aug_value, 1 + aug_value)},
                   translate_percent={"x": (-aug_value, aug_value), "y": (-aug_value, aug_value)},
                   rotate=(-90 * aug_value, 90 * aug_value),
                   shear=(-45 * aug_value, 45 * aug_value))
    ])
    if "EfficientNet" in backbone:
        preprocess = (lambda x, **y: x)
        mode = None
    else:
        preprocess = tf.keras.applications.imagenet_utils.preprocess_input
        mode = "torch" if "DenseNet" in backbone else "tf" if ("ResNet" in backbone and "V2" in backbone) else "caffe"
    return x_aug, xy_aug, preprocess, mode


def func_data_from_ids(data):
    """
    returns a function used to slice the data-array
    """
    def data_from_ids(idx_list):
        res = tf.numpy_function(lambda x: data[x.flatten()], [idx_list], tf.uint8, False)
        _, w, h, d = data.shape
        res.set_shape((None, w, h, d))
        return res

    return data_from_ids


def split_train_ids_by_class(split_ids, attributes_list):
    """
    returns list of train ids by attribute clas
    last list contains no attributes
    if multiple classes it goes to the rarest class
    """
    sorted_attributes = [k for k in [3, 1, 0, 2, 4] if k in attributes_list]  # sorted by frequency
    class_data = get_class_labels_dict(Splits.TRAIN)
    class_ids = [[] for k in range(len(sorted_attributes) + 1)]
    for key, id_list in class_data.items():
        matches = [i for i, a in enumerate(sorted_attributes) if str(a) in key]
        prio_match = matches[0] if len(matches) > 0 else len(sorted_attributes)
        class_ids[prio_match] += list(np.intersect1d(split_ids, id_list))

    return class_ids
