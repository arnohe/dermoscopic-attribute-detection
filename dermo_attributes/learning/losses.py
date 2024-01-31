import tensorflow as tf
import tensorflow.python.keras.backend as bk

"""
source: https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions
paper: https://doi.org/10.1109/CIBCB48159.2020.9277638

source: https://github.com/JunMa11/SegLoss
paper: https://doi.org/10.1016/j.media.2021.102035

Note the parameter gamma has been modified after writing my thesis to now match: 
paper:  Unified Focal loss (https://doi.org/10.1016/j.compmedimag.2021.102026)
"""


def binary_focal_loss(alpha=0.25, gamma=2.):
    a = tf.constant(alpha, dtype="float32")
    b = tf.constant(gamma, dtype="float32")

    def loss(y_true, y_pred):
        y_t = tf.round(tf.clip_by_value(y_true, 0., 1.))
        y_t = bk.cast(bk.flatten(y_t), dtype="float32")
        y_p = bk.cast(bk.flatten(y_pred), dtype="float32")
        y_p = bk.clip(y_p, bk.epsilon(), 1.0 - bk.epsilon())

        loss_0 = - y_t * (a * bk.pow(1 - y_p, (1 - b)) * bk.log(y_p))
        loss_1 = - (1 - y_t) * ((1 - a) * bk.pow(y_p, (1 - b)) * bk.log(1 - y_p))
        return bk.mean(loss_0 + loss_1)

    return loss


def tversky_index(y_true, y_pred, a, s):
    y_t = bk.cast(bk.flatten(y_true), dtype=tf.float32)
    y_p = bk.cast(bk.flatten(y_pred), dtype=tf.float32)
    tp = bk.sum(y_t * y_p)
    fn = bk.sum(y_t * (1 - y_p))
    fp = bk.sum((1 - y_t) * y_p)
    return (tp + s) / (tp + a * fn + (1 - a) * fp + s)


def focal_tversky_loss(alpha=0.7, gamma=0.75, smooth=1.):
    a = tf.constant(alpha, dtype="float32")
    b = tf.constant(gamma, dtype="float32")
    s = tf.constant(smooth, dtype="float32")

    def loss(y_true, y_pred):
        # calculate average across all slices
        acc = tf.constant(0., dtype="float32")
        count = y_true.shape[3]
        y_true_bin = tf.round(tf.clip_by_value(y_true, 0., 1.))
        for i in range(count):
            y_t = tf.slice(y_true_bin, [0, 0, 0, i], [-1, -1, -1, 1])
            y_p = tf.slice(y_pred, [0, 0, 0, i], [-1, -1, -1, 1])
            tvi = tversky_index(y_t, y_p, a, s)
            acc += bk.pow((1 - tvi), b)
        return acc / count

    return loss


"""
Log Cosh Dice / Tversky Loss
source: https://doi.org/10.1109/CIBCB48159.2020.9277638
"""


def log_cosh_tversky_loss(alpha=0.7, smooth=1.):
    a = tf.constant(alpha, dtype="float32")
    s = tf.constant(smooth, dtype="float32")

    def loss(y_true, y_pred):
        acc = tf.constant(0., dtype="float32")
        count = y_true.shape[3]
        y_true_bin = tf.round(tf.clip_by_value(y_true, 0., 1.))
        for i in range(count):
            y_t = tf.slice(y_true_bin, [0, 0, 0, i], [-1, -1, -1, 1])
            y_p = tf.slice(y_pred, [0, 0, 0, i], [-1, -1, -1, 1])
            x = 1 - tversky_index(y_t, y_p, a, s)
            acc += tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)
        return acc / count

    return loss


"""
Lovasz Loss:
source: https://github.com/bermanmaxim/LovaszSoftmax
paper; https://doi.org/10.48550/arXiv.1705.08790
(you have to use LINEAR ACTIVATION instead of sigmoid)
y_logit between -inf and +inf
y_true 0 or 1
"""


def lovasz_hinge_loss(per_image=False):
    if not per_image:
        return calc_lovasz_loss

    def loss(y_labels, y_logits):
        losses = tf.map_fn(lambda x: calc_lovasz_loss(tf.expand_dims(x[0], 0), tf.expand_dims(x[1], 0)),
                           (y_labels, y_logits), dtype=tf.float32)
        return tf.reduce_mean(losses)

    return loss


def calc_lovasz_loss(y_labels, y_logits):
    y_lab = tf.round(tf.clip_by_value(y_labels, 0., 1.))
    y_log = tf.cast(tf.reshape(y_logits, (-1,)), dtype="float32")
    y_lab = tf.cast(tf.reshape(y_lab, (-1,)), dtype="float32")
    signs = 2. * y_lab - 1.
    errors = 1. - y_log * tf.stop_gradient(signs)
    errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
    gt_sorted = tf.gather(y_lab, perm)
    grad = lovasz_grad(gt_sorted)
    loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss")
    return loss


def lovasz_grad(gt_sorted):
    # Computes gradient of the Lovasz extension w.r.t sorted errors
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard
