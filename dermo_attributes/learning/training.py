from keras_unet_collection.models import unet_2d
from keras_unet_collection import _backbone_zoo
import tensorflow as tf
import numpy as np
import wandb
import os

from dermo_attributes.io.paths import Splits
from dermo_attributes.io.dataset import get_dataset
from dermo_attributes.learning.metrics import SegmentationMetric
from dermo_attributes.learning.losses import binary_focal_loss, focal_tversky_loss, log_cosh_tversky_loss
from dermo_attributes.config import WANDB_USER, WANDB_PROJECT


def train_unet(config, fine_tune_run_name=None):
    """
    Train UNet model
    Integrated with wandb. Learning for dashboard and archiving
    """

    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("No GPU available")
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    if fine_tune_run_name is None:
        job_type = "learning"
        name = None
        model = create_unet_model(config)
    else:
        job_type = "fine_tuning"
        name = "fine-" + fine_tune_run_name
        config["unfreeze_backbone"] = True
        config["unfreeze_batch_norm"] = True
        config["patience"] = 20
        model = load_model(fine_tune_run_name)

    train_set, train_steps = create_train_dataset(config)
    valid_set, valid_steps = create_validation_dataset(config)

    model.compile(optimizer=get_optimizer(config), loss=get_loss_function(config), metrics=get_metrics(config), )

    # train model
    if not config["disable_wandb"]:
        run = wandb.init(mode="online",
                         project=WANDB_PROJECT,
                         config=config,
                         save_code=True,
                         job_type=job_type,
                         name=name)
        # wandb.use_artifact(config["dataset_name"] + ":latest")

    model.fit(
        train_set,
        epochs=config['max_epochs'],
        verbose=2,
        callbacks=get_callbacks(config),
        validation_data=valid_set,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps
    )

    if not config["disable_wandb"]:
        name = "model_" + "".join(str(a) for a in config["attributes"])
        artifact = wandb.Artifact(run.name, type=name)
        artifact.add_file(config["model_filename"])
        wandb.log_artifact(artifact)
        run.finish()


def create_unet_model(config):
    """
    create keras model using keras_unet_collection
    """
    edit_backbone_zoo(config["backbone"])
    model = unet_2d(
        input_size=(config["input_size"], config["input_size"], 3),
        filter_num=get_num_filters(config["unet_depth"], config["decoder_scale"]),
        n_labels=len(config["attributes"]),
        stack_num_up=config["decoder_depth"],
        activation='ReLU',
        output_activation=None if config["use_logits"] else "Sigmoid",
        batch_norm=config["batch_norm"],
        unpool=not config["transposed_conv"],
        backbone=config["backbone"],
        weights='imagenet',
        freeze_backbone=not config["unfreeze_backbone"],
        freeze_batch_norm=not config["unfreeze_batch_norm"],
        name='unet_2d'
    )
    return model


def edit_backbone_zoo(backbone):
    """
    Add support for EfficientNetV2 to keras_unet_collection
    """
    if backbone in _backbone_zoo.layer_cadidates.keys():
        return

    new_backbones = {
        'EfficientNetV2B0': (
            'block1a_project_activation', 'block2b_expand_activation', 'block4a_expand_activation',
            'block6a_expand_activation', 'top_activation'),
        'EfficientNetV2B1': (
            'block1b_project_activation', 'block2c_expand_activation', 'block4a_expand_activation',
            'block6a_expand_activation', 'top_activation'),
        'EfficientNetV2B2': (
            'block1b_project_activation', 'block2c_expand_activation', 'block4a_expand_activation',
            'block6a_expand_activation', 'top_activation'),
        'EfficientNetV2B3': (
            'block1b_project_activation', 'block2c_expand_activation', 'block4a_expand_activation',
            'block6a_expand_activation', 'top_activation'),
        'EfficientNetV2S': (
            'block1b_project_activation', 'block2d_expand_activation', 'block4a_expand_activation',
            'block6a_expand_activation', 'top_activation'),
        'EfficientNetV2M': (
            'block1c_project_activation', 'block2e_expand_activation', 'block4a_expand_activation',
            'block6a_expand_activation', 'top_activation'),
        'EfficientNetV2L': (
            'block1d_project_activation', 'block2g_expand_activation', 'block4a_expand_activation',
            'block6a_expand_activation', 'top_activation')}
    _backbone_zoo.layer_cadidates = _backbone_zoo.layer_cadidates | new_backbones


def create_train_dataset(config):
    """
    creates tf.data.dataset for training and validation
    """
    train_split = list([0, 1, 2, 3, 4])
    train_split.remove(config["validation_split"])
    if config['fast_epochs']:
        train_split = [train_split[0]]

    train_set, train_steps = get_dataset(
        config["dataset_name"], config["attributes"], config["batch_size"], config["backbone"],
        isic_split=Splits.TRAIN, train_splits=train_split,
        balanced=config["balanced_batches"], augmentation_value=config["augmentation"], drop_remainder=True)

    return train_set, train_steps


def create_validation_dataset(config):
    valid_split = [config["validation_split"]]

    valid_set, valid_steps = get_dataset(
        config["dataset_name"], config["attributes"], config["batch_size"], config["backbone"],
        isic_split=Splits.TRAIN, train_splits=valid_split,
        balanced=False, augmentation_value=0, drop_remainder=False)
    return valid_set, valid_steps


def get_optimizer(config):
    """
    get optimizer object matching config string
    """
    if config["optimizer"].lower() == "rmsprop":
        return tf.keras.optimizers.RMSprop(config["learning_rate"])
    if config["optimizer"].lower() == "adam":
        return tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    return None


def get_num_filters(depth, scale=1):
    """
    filter/feature map counts that match original unet architecture
    """
    num_filters = np.array([16, 32, 64, 128, 256, 0]) * scale
    return num_filters[6 - depth:]


def get_callbacks(config):
    """
    generate list with useful callback functions using config-dict
    """
    callbacks = []
    tensorboard_logs_dir = "logs"

    # callback for saving best model
    callbacks += [tf.keras.callbacks.ModelCheckpoint(
        config["model_filename"],
        verbose=0,
        monitor=config['monitor'],
        mode=config['monitor_mode']
    )]

    if config["tensorboard"]:
        # callback tensorboard
        callbacks += [tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_logs_dir,
            histogram_freq=1,
            profile_batch='10, 15',

        )]

    if not config["disable_wandb"]:
        # callback for wandb
        callbacks += [wandb.keras.WandbCallback(
            save_model=False,
            monitor=config['monitor'],
            mode=config['monitor_mode'],
            log_best_prefix="best_",
        )]

    # callback for early stopping
    callbacks += [tf.keras.callbacks.EarlyStopping(
        monitor=config['monitor'],
        patience=config['patience'],
        min_delta=config["min_delta"],
        verbose=1,
        mode=config['monitor_mode'],
        restore_best_weights=True
    )]
    return callbacks


def get_loss_function(config):
    """
    get loss function matching config string
    """
    loss = config["loss"]
    if loss == "binary_focal_loss":
        return binary_focal_loss(config["alpha"], config["gamma"])
    if loss == "focal_tversky_loss":
        return focal_tversky_loss(config["alpha"], config["gamma"], config["smooth"])
    if loss == "log_cosh_tversky_loss":
        return log_cosh_tversky_loss(config["alpha"], config["smooth"])
    return None


def get_metrics(config):
    """
    set up segmentation metrics
    """
    chan = len(config["attributes"])
    metrics = [SegmentationMetric(chan, config["use_logits"], "fuzzy", "iou"),
               SegmentationMetric(chan, config["use_logits"], "fuzzy", "precision"),
               SegmentationMetric(chan, config["use_logits"], "fuzzy", "recall"),
               SegmentationMetric(chan, config["use_logits"], "crisp", "iou"),
               SegmentationMetric(chan, config["use_logits"], "crisp", "precision"),
               SegmentationMetric(chan, config["use_logits"], "crisp", "recall"),
               SegmentationMetric(chan, config["use_logits"], "class", "iou"),
               SegmentationMetric(chan, config["use_logits"], "class", "precision"),
               SegmentationMetric(chan, config["use_logits"], "class", "recall")]

    return metrics


def load_model(wandb_model_name):
    path = 'data/models'
    file_name = path + '/' + wandb_model_name + '.h5'
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(file_name):
        api = wandb.Api()
        artifact = api.artifact(WANDB_USER + "/" + WANDB_PROJECT + "/" + wandb_model_name + ":v0")
        artifact.download(".")
        os.rename("model.h5", file_name)
    model = tf.keras.models.load_model(file_name,
                                       custom_objects={"loss": lambda x, y: 0,
                                                       "SegmentationMetric": lambda name, dtype: 0})
    # model.summary()
    return model
