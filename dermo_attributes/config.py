import os
from argparse import ArgumentParser

BASE_FOLDER = os.getenv('BASE_FOLDER', "data")  # points to where data is stored (if in a different location)
ATTRIBUTE_NAMES = ["globules", "milia_like_cyst", "negative_network", "pigment_network", "streaks"]
NUM_ATTRIBUTES = len(ATTRIBUTE_NAMES)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

"""
arguments for preprocessing and training can be set from the command line
or by changing the default values here
"""


def preprocess_arguments():
    parser = ArgumentParser(description='Process images from ISIC dataset')
    parser.add_argument('--size', type=int, help='Size for resizing images', default=512)
    parser.add_argument('--dataset_name', type=str, help='Folder name for processed dataset', default="crop_512")
    return vars(parser.parse_args())


def training_arguments():
    parser = ArgumentParser(description='Train UNet model')

    parser.add_argument('-d', '--disable_wandb',
                        action='store_true', help='override wandb usage')
    parser.add_argument('-t', '--tensorboard',
                        action='store_true', help='toggle tensorboard logging and profiling')
    parser.add_argument('-f', '--fast_epochs',
                        action='store_true', help='use smaller train set for validation')

    parser.add_argument('-a', '--attributes',
                        nargs='+', type=int, default=[3],
                        help='list with indexes of attributes to predict'
                             '(0-globules, 1-milia_like_cyst, 2-negative_network, 3-pigment_network, 4-streaks)')
    parser.add_argument('-i', '--input_size',
                        type=int, default=512,
                        help='size of square input images')
    parser.add_argument('-b', '--batch_size',
                        type=int, default=28,
                        help="batch size for learning network")

    parser.add_argument("--backbone",
                        type=str, default="EfficientNetV2B0",
                        help="name of backbone to use as encoder in UNet (see keras unet collection)")
    parser.add_argument('--decoder_scale',
                        type=int, default=1,
                        help="scale up the number of filters in decoder at each stage")
    parser.add_argument('--decoder_depth',
                        type=int, default=2,
                        help="number of conv layers at each decoder stage")
    parser.add_argument('--unet_depth',
                        type=int, default=6,
                        help="depth of the unet model")
    parser.add_argument("--transposed_conv",
                        action='store_true',
                        help="use transposed convolutions instead of upsampling in model")

    parser.add_argument('-e', "--max_epochs",
                        type=int, default=100,
                        help="maximum number of epochs")
    parser.add_argument("--patience",
                        type=int, default=10,
                        help="early stopping patience")
    parser.add_argument("--min_delta",
                        type=float, default=0.,
                        help="minimum delta for early stopping")

    parser.add_argument("--learning_rate",
                        type=float, default=0.001,
                        help="learning rate for optimizer")
    parser.add_argument("--loss",
                        type=str, default="focal_tversky_loss",
                        help="name of the loss function")
    parser.add_argument('--alpha',
                        type=float, default=-1,
                        help='parameter for losses')
    parser.add_argument('--gamma',
                        type=float, default=-1,
                        help='parameter for losses')
    parser.add_argument('--smooth',
                        type=float, default=-1,
                        help="parameter for losses")

    parser.add_argument('--augmentation',
                        type=float, default=0.20,
                        help="augmentation value range between 0 and 1")
    parser.add_argument('--unfreeze_backbone',
                        action='store_true',
                        help="true to freeze backbone weights")
    parser.add_argument('--unfreeze_batch_norm',
                        action='store_true',
                        help="true to freeze backbone batch normalization layers")

    parser.add_argument('--validation_split',
                        type=int, default=0,
                        help="[part of split to use for validation: [0-4]")
    parser.add_argument('--balanced_batches',
                        action='store_true',
                        help="create balanced batches from dataset")
    return parser.parse_args()


def training_config():
    """
    create config dict with parameters
    """
    args = training_arguments()
    # setup aliases and default values
    loss = args.loss
    match loss:
        case "binary_crossentropy":
            loss = "binary_focal_loss"
            alpha = 0.5
            gamma = 0.
        case "binary_focal_loss":
            alpha = args.alpha if 0 <= args.alpha <= 1 else 0.25
            gamma = args.gamma if 0 <= args.gamma else 2.
        case "dice_loss":
            loss = "focal_tversky_loss"
            alpha = 0.5
            gamma = 1.
        case "focal_tversky_loss":
            alpha = args.alpha if 0 <= args.alpha <= 1 else 0.7
            gamma = args.gamma if 0 <= args.gamma else 0.75
        case "log_cosh_dice_loss":
            loss = "log_cosh_tversky_loss"
            alpha = 0.5
            gamma = -1
        case "log_cosh_tversky_loss":
            alpha = args.alpha if 0 <= args.alpha <= 1 else 0.7
            gamma = -1
        case "lovasz_hinge_loss":
            alpha = 1 if 0 < args.alpha else -1  # lazy implementation to save per_image param for loss
            gamma = -1
        case _:
            raise ValueError("bad input for loss function")

    if "tversky" in loss:
        smooth = args.smooth if 0 < args.smooth else 1.
    else:
        smooth = -1

    # settings dict
    config = dict(
        tensorboard=args.tensorboard,
        disable_wandb=args.disable_wandb,
        fast_epochs=args.fast_epochs,
        attributes=args.attributes,
        attribute_names=[ATTRIBUTE_NAMES[i] for i in args.attributes],
        model_filename="model.h5",
        dataset_name="crop_" + str(args.input_size),
        input_size=args.input_size,
        batch_size=args.batch_size,
        backbone=args.backbone,
        decoder_scale=args.decoder_scale,
        decoder_depth=args.decoder_depth,
        unet_depth=args.unet_depth if "VGG" not in args.backbone else args.unet_depth - 1,
        batch_norm=("VGG" not in args.backbone),
        transposed_conv=args.transposed_conv,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=0,
        learning_rate=args.learning_rate,
        loss=loss,
        alpha=alpha,
        gamma=gamma,
        smooth=smooth,
        augmentation=args.augmentation,
        unfreeze_backbone=args.unfreeze_backbone,
        unfreeze_batch_norm=args.unfreeze_batch_norm,
        validation_split=args.validation_split,
        balanced_batches=args.balanced_batches,
        monitor='val_fuzzy_iou',
        monitor_mode="max",
        optimizer="adam",
        use_logits=(loss == "lovasz_hinge_loss"),
    )
    return config
