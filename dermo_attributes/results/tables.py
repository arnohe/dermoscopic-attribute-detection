from itertools import product
import pandas as pd
import numpy as np
import wandb
import os

from dermo_attributes.config import ATTRIBUTE_NAMES, WANDB_USER, WANDB_PROJECT

"""
functions to tabulate and process the results
"""


def set_display_size():
    """
    increase size of printed tables
    """
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)


def get_best_runs(metric_name):
    """
    returns a table containing the best results for the metric given as argument
    """
    df = get_raw_results()
    df = df[df["metric"] == metric_name]
    best = [df[df["attribute"] == a].max()["id"] for a in ATTRIBUTE_NAMES]
    return best


def add_averages(df, metric_name="crisp_iou"):
    """
    adds average column to the table
    """
    metric = df.loc[:, (slice(None), metric_name)]
    average = pd.DataFrame(metric.sum(axis=1) / 5)
    average.columns = pd.MultiIndex.from_tuples([("average", metric_name)], names=["attritube", ""])
    df = pd.concat([df, average], axis=1)
    return df


def table_all(df, metric_name="crisp_iou"):
    """
    returns the table containing an entry for the best in each subtype of loss function and overall best models
    """
    base_df = table_for_overview(df, metric_name)
    best_df = table_for_best(df, metric_name)
    rdf = pd.concat([best_df, base_df]).sort_index()
    return rdf


def table_for_overview(df, metric_name="crisp_iou"):
    """
    returns the table containing an entry for the best in each subtype of loss function
    """
    bce = table_for_selection(df, 0.5, 0, "binary focal loss", metric_name, "cross entropy")
    dl = table_for_selection(df, 0.5, 1, "focal tversky loss", metric_name, "dice loss")
    lcdl = table_for_selection(df, 0.5, -1, "log cosh tversky loss", metric_name, "log cosh dice loss")
    wce = table_for_selection(df, None, 0, "binary focal loss", metric_name, "weighted cross entropy")
    tl = table_for_selection(df, None, 1, "focal tversky loss", metric_name, "tversky loss")
    ufl = table_for_selection(df, 0.5, None, "binary focal loss", metric_name, "unweighted focal loss")
    fdl = table_for_selection(df, 0.5, None, "focal tversky loss", metric_name, "focal dice loss")

    rdf = pd.concat([bce, dl, lcdl, wce, tl, ufl, fdl])
    rdf = rdf.sort_index()
    return rdf


def table_for_best(df, metric_name="crisp_iou"):
    """
    returns the table containing the best results grouped by loss function and oversampling
    """
    metric = df.groupby(["balanced batches", "loss"]).max().stack()
    idx = df.groupby(["balanced batches", "loss"]).idxmax()
    alpha = idx.applymap(lambda k: k[-2]).stack()
    gamma = idx.applymap(lambda k: k[-1] if k[-1] != -1 else np.nan).stack()

    rdf = pd.concat([alpha, gamma, metric], axis=1)
    rdf.columns = [["alpha", "gamma", metric_name]]
    rdf = pd.pivot_table(rdf, index=["balanced batches", "loss"], columns=["attribute"],
                         values=["alpha", "gamma", metric_name])
    rdf = rdf.swaplevel(axis='columns').sort_index(axis=1, level=0)
    return rdf


def table_for_selection(df, alpha, gamma, loss_name, metric_name, new_loss_name=None):
    """
    returns the table containing best results for a subtype of loss function
    """
    if alpha is None:
        alpha = slice(None)
    if gamma is None:
        gamma = slice(None)
    if new_loss_name is None:
        new_loss_name = loss_name

    zdf = df.fillna(0, inplace=False)
    metric = zdf.loc[(slice(None), loss_name, alpha, gamma), :].groupby(["balanced batches", "loss"]).max()
    idx = zdf.loc[(slice(None), loss_name, alpha, gamma), :].groupby(["balanced batches", "loss"]).idxmax()

    alpha = idx.applymap(lambda k: k[-2])
    gamma = idx.applymap(lambda k: k[-1] if k[-1] != -1 else np.nan)
    rdf = pd.concat([alpha, gamma, metric], axis=1, keys=["alpha", "gamma", metric_name])
    rdf.columns = rdf.columns.droplevel(1)
    rdf = rdf.swaplevel(axis='columns').sort_index(axis=1, level=0)
    rdf.index = rdf.index.map(lambda k: (k[0], new_loss_name))
    return rdf


def get_raw_results():
    """
    reads raw results from experiments from disk (data/diagrams/raw_results_dataframe.pkl)
    if not present retrieve them from wandb and save them to disk
    """
    if os.path.exists("data/results/raw_results_dataframe.pkl"):
        return pd.read_pickle("data/results/raw_results_dataframe.pkl")

    api = wandb.Api()
    runs = api.runs(WANDB_USER + "/" + WANDB_PROJECT)
    metrics = ["val_" + j + "_" + i for i, j in product(["iou", "recall", "precision"], ["fuzzy", "crisp", "class"])]
    config = ["balanced_batches", "loss", "alpha", "gamma", "attribute_names"]

    rows = []
    for run in runs:
        print(run.id)
        if run.id == "1d5do82w":
            best = 46
        elif run.id == "jx62n5vf":
            best = 25
        else:
            best = run.summary["best_epoch"]
        run_results = run.history().loc[[best]].to_dict(orient="list")
        row_conf = [run.config[conf] for conf in config]
        row_conf[-1] = row_conf[-1][0]
        for metr in metrics:
            rows.append((run.id, *row_conf, metr[4:], float(run_results[metr][0])))
    df = pd.DataFrame.from_records(data=rows, columns=["id", *config[:-1], "attribute", "metric", "value"])

    df.to_pickle("data/results/raw_results_dataframe.pkl")
    return df


def get_multi_index_results(metric="crisp_iou"):
    """
    process raw results into a multi index table
    """
    df = get_raw_results()
    df["loss"] = df["loss"].apply(
        lambda k: k.replace("_", " "))
    df["attribute"] = df["attribute"].apply(lambda k: k.replace("_", " "))
    df.columns = df.columns.map(lambda k: k.replace("_", " "))
    df = df.sort_values(by=["balanced batches", "loss", "attribute", "value"])
    df = df[df["metric"] == metric]
    df = pd.pivot_table(df, index=["balanced batches", "loss", "alpha", "gamma"], columns=["attribute"],
                        values=["value"])
    return df


def print_formatted_table():
    set_display_size()
    df = add_averages(table_for_best(get_multi_index_results())).round(5)
    loss_order = {"cross entropy": 0,
                  "weighted cross entropy": 1,
                  "binary focal loss": 3,
                  "unweighted focal loss": 2,
                  "dice loss": 4,
                  "tversky loss": 5,
                  "focal dice loss": 6,
                  "focal tversky loss": 7,
                  "log cosh dice loss": 8,
                  "log cosh tversky loss": 9}
    value_order = {"iou": 0,
                   "alpha": 1,
                   "gamma": 2}
    class_order = {"globules": 0,
                   "milia like cyst": 1,
                   "negative network": 2,
                   "pigment network": 3,
                   "streaks": 4,
                   "average": 5}

    df = df.swaplevel()
    df = df.sort_index(key=lambda x: x.map(loss_order))
    df = df.transpose()
    df = df.sort_index(level=1, key=lambda x: x.map(value_order))
    df = df.sort_index(level=0, sort_remaining=False, key=lambda x: x.map(class_order))
    df = df.transpose()
    print(df)
