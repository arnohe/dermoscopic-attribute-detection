import sys

from dermo_attributes.config import preprocess_arguments, training_config, sweep_config, test_arguments, \
    validation_arguments
from dermo_attributes.io.class_id import read_training_splits
from dermo_attributes.io.paths import create_new_processed_folders
from dermo_attributes.io.preprocess import process_all
from dermo_attributes.io.download_dataset import download_dataset as download_dataset_function
from dermo_attributes.learning.sweep import search
from dermo_attributes.learning.training import train_unet
from dermo_attributes.results.figures import make_all_heat_plots, make_bar_plots
from dermo_attributes.results.tables import print_formatted_table
from dermo_attributes.results.testing import run_tests, make_test_images


def download_dataset():
    download_dataset_function()


def preprocess_dataset():
    args = preprocess_arguments()
    dataset_name = "crop_" + str(args["size"])
    create_new_processed_folders(dataset_name)
    read_training_splits(dataset_name)
    process_all(dataset_name, args["size"])


def train_model():
    config = training_config()
    train_unet(config)


def sweep_gridsearch():
    config = sweep_config()
    search(config)


def validation_results():
    args = validation_arguments()
    make_all_heat_plots(metric=args.metric)
    make_bar_plots(metric=args.metric)
    print_formatted_table()


def isic_test_results():
    args = test_arguments()
    print("Jaccard scores:", run_tests(args.idx))
    make_test_images(args.idx, "data/results/output_examples.png")


if __name__ == "__main__":
    main_methods = {"download": download_dataset,
                    "preprocess": preprocess_dataset,
                    "train": train_model,
                    "sweep": sweep_gridsearch,
                    "validation": validation_results,
                    "test": isic_test_results}
    if len(sys.argv) > 1 and sys.argv[1] in main_methods.keys():
        main_methods[sys.argv[1]]()
    else:
        methods_string = ", ".join(main_methods.keys())
        print("Use one of the following arguments to choose method: " + methods_string)
