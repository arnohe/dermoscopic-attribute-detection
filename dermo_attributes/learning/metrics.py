import tensorflow.python.keras.backend as bk
from keras.metrics import Metric  # do not use tf.keras.metrics.Metric as it won't reset the state
import tensorflow as tf
import numpy as np

"""
Functions to calculate extra metrics
debug with tf.config.experimental_run_functions_eagerly(True)
"""


def iou_score(confusion):
    """
    calculates iou from confusion = [tp, fn, fp, tn]
    iou can return nan if all are zero (ground truth and prediction both empty)
    """
    acc = bk.constant(0., dtype="float32")
    for i in range(confusion.shape[0]):
        acc += confusion[i][0] / (confusion[i][0] + confusion[i][1] + confusion[i][2])
    return acc / confusion.shape[0]


def precision(confusion):
    """
    calculates precision from confusion = [tp, fn, fp, tn]
    precision can return nan if tp and fp are zero (prediction empty)
    """
    # matrix = tp, fn, fp, tn
    acc = bk.constant(0., dtype="float32")
    for i in range(confusion.shape[0]):
        acc += confusion[i][0] / (confusion[i][0] + confusion[i][2])
    return acc / confusion.shape[0]


def recall(confusion):
    """
    calculates recall from confusion = [tp, fn, fp, tn]
    recall can return nan if fp and fn are zero (ground truth empty)
    """
    acc = bk.constant(0., dtype="float32")
    for i in range(confusion.shape[0]):
        acc += confusion[i][0] / (confusion[i][0] + confusion[i][1])
    return acc / confusion.shape[0]


metric_functions = {"recall": recall, "precision": precision, "iou": iou_score}


class SegmentationMetric(Metric):
    """
    class to produce stateful metrics
    accumulates values over whole epoch for the metric
    reset between epochs
    """

    def __init__(self, channels, use_logits=False, variation="fuzzy", metric="iou", **kwargs):
        super(SegmentationMetric, self).__init__(name=variation + "_" + metric, **kwargs)
        self.binarize = variation == "crisp"
        self.classify_image = variation == "class"
        self.stateful = True
        self.channels = channels
        self.use_logits = use_logits
        self.metric_name = variation + "_" + metric
        self.metric_func = metric_functions[metric]
        self.accumulator = self.add_weight(name="accumulator",
                                           initializer="zeros",
                                           shape=[channels, 4],
                                           dtype="float32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.debug_code(y_pred)
        y_t, y_p = prepare_input(y_true, y_pred, self.binarize, self.classify_image, self.use_logits)
        self.accumulator.assign_add(confusion_matrix(y_t, y_p))

    def result(self):
        return self.metric_func(self.accumulator)

    def reset_state(self):
        tf.keras.backend.set_value(self.accumulator, np.zeros((self.channels, 4), dtype=np.float32))

    def debug_code(self, y_pred):
        if self.metric_name == "fuzzy_iou":
            nan_frq = tf.math.zero_fraction(tf.math.is_nan(y_pred))
            if nan_frq < 1:
                tf.print("nan values in prediction:", 100.0 * (1 - nan_frq), " %")


def prepare_input(y_true, y_pred_log, binarize=False, classify_image=False, use_logits=False):
    """
    setup depending on metric variation
    """
    y_t = bk.round(bk.clip(y_true, 0., 1.))
    y_p = bk.sigmoid(bk.cast(y_pred_log, dtype="float32")) if use_logits else y_pred_log
    if binarize or classify_image:
        y_p = bk.round(bk.clip(y_p, 0., 1.))
    if classify_image:
        y_t = tf.cast(bk.any(y_t, axis=[1, 2], keepdims=True), dtype="float32")
        y_p = tf.cast(bk.any(y_p, axis=[1, 2], keepdims=True), dtype="float32")
    return y_t, y_p


def confusion_matrix(y_true, y_pred):
    """
    calculates confusion matrix values from ground truth y_true and prediction y_pred
    return as [tp, fn, fp, tn]
    """
    y_t_pos = bk.cast(y_true, dtype="float32")
    y_p_pos = bk.cast(y_pred, dtype="float32")
    y_t_neg = 1 - y_t_pos
    y_p_neg = 1 - y_p_pos
    tp = bk.sum(y_t_pos * y_p_pos, axis=[0, 1, 2], keepdims=False)
    fp = bk.sum(y_t_neg * y_p_pos, axis=[0, 1, 2], keepdims=False)
    fn = bk.sum(y_t_pos * y_p_neg, axis=[0, 1, 2], keepdims=False)
    tn = bk.sum(y_t_neg * y_p_neg, axis=[0, 1, 2], keepdims=False)
    return bk.stack([tp, fn, fp, tn], axis=1)
