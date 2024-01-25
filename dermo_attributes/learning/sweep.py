from subprocess import PIPE, Popen
from itertools import product
from time import time
import wandb

from dermo_attributes.learning.training import train_unet

"""
functions to run the main experiment
iterating over losses / alpha / gamma / balancing
"""


def search(input_size, batch_size, loss, list_alpha_gamma):
    """
    iterate over the models to be trained
    """
    list_attributes = [[k] for k in range(5)]
    list_balanced_batches = [False, True]
    total = "/" + str(len(list_alpha_gamma) * len(list_attributes) * len(list_balanced_batches))
    for i, params in enumerate(product(list_attributes, list_alpha_gamma, list_balanced_batches)):
        attributes, alpha_gamma, balanced_batches = params
        alpha, gamma = alpha_gamma
        start_run(i, total, attributes, balanced_batches, loss, input_size, batch_size, alpha, gamma)


def start_run(i, total, attributes, balanced_batches, loss, input_size, batch_size, alpha=None, gamma=None):
    """
    runs a process to train a single model
    """
    alpha_gamma_str = ""
    if alpha is not None:
        alpha_gamma_str += "   " + format(alpha, '.2f')
    if gamma is not None:
        alpha_gamma_str += "   " + format(gamma, '.2f')

    print((str(i + 1) + total).rjust(7) + "  >> ",
          str(attributes).rjust(25) + "  ",
          str(balanced_batches).ljust(5) + "  ",
          loss.ljust(21), alpha_gamma_str,
          end="  ", flush=True)

    args = ['python', 'sweep.py']
    args += ["--input_size", str(input_size)]
    args += ["--batch_size", str(batch_size)]
    args += ["--attributes", *[str(a) for a in attributes]]
    args += ["--loss", loss]
    args += ["--balanced_batches"] if balanced_batches else []
    args += ["--alpha", str(alpha)] if alpha is not None else []
    args += ["--gamma", str(gamma)] if gamma is not None else []

    start = time()
    process = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = process.communicate()
    exit_code = process.poll()

    time_spent = time() - start
    if exit_code == 0:
        print(">>  success [ " + str(int(round(time_spent / 60, 0))).rjust(2) + "m ]")
    else:
        print(">>   failed [ " + str(int(round(time_spent / 60, 0))).rjust(2) + "m ]")
        # print(out.decode('UTF-8'))
        print(err.decode('UTF-8'))


def fine_tuning_runs():
    """
    used to train models a second time with the encoder unfrozen
    """
    api = wandb.Api()
    runs = api.runs(path="arno/lesion-attributes")
    # filters={"$and": [{"config.loss": "binary_focal_loss"},
    #                   {"config.attributes": [0]}]})
    for run in runs:
        train_unet(run.config, run.name)
