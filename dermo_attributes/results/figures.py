import io
from itertools import product

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from matplotlib import cm, colors
from matplotlib.patches import Patch
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from dermo_attributes.results.tables import get_multi_index_results, add_averages, table_all
from dermo_attributes.io.images import visualize_contours, visualize_matrix, visualize_probability

"""
Functions used for visualizing model predictions
"""


def visualize_horizontal(ims, truth, pred, attribute_names):
    """
    creates a visualisation containing 3 images per attribute
    contour - heatmap overlay - overlap
    layout is horizontal
    """
    print(truth.shape)
    ncols = truth.shape[0]
    fig, ax = plt.subplots(nrows=3, ncols=ncols, figsize=(12, 6))
    fig.tight_layout()
    if ncols == 1:
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        ax[0].imshow(visualize_contours(ims[0], truth[0, :, :], pred[0, :, :]))
        ax[1].imshow(visualize_probability(ims[0], pred[0, :, :]))
        ax[2].imshow(visualize_matrix(truth[0, :, :, 0], pred[0, :, :]))
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax[0].text(25, -75, attribute_names[0], fontsize=10, verticalalignment='top', bbox=props)
        axzero = ax[0]
        axone = ax[1]
        axtwo = ax[2]

    else:
        for i in range(ncols):
            ax[0, i].axis('off')
            ax[1, i].axis('off')
            ax[2, i].axis('off')
            ax[0, i].imshow(visualize_contours(ims[i], truth[i, :, :], pred[i, :, :]))
            ax[1, i].imshow(visualize_probability(ims[i], pred[i, :, :]))
            ax[2, i].imshow(visualize_matrix(ims[i], truth[i, :, :], pred[i, :, :]))

        for i in range(ncols):
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax[0, i].text(25, -75, attribute_names[i], fontsize=10, verticalalignment='top', bbox=props)
            axzero = ax[0, 0]
            axone = ax[1, 0]
            axtwo = ax[2, 0]

    patches0 = [Patch(color=[0, 1, 0], label="Ground truth"),
                Patch(color=[0, 0, 1], label="Prediction")]
    patches2 = [Patch(color=[0, 1, 0], label="True positive"),
                Patch(color=[1, 0, 0], label="False positive"),
                Patch(color=[0, 0, 0], label="True negative"),
                Patch(color=[0, 0, 1], label="False negative")]

    fig.subplots_adjust(top=0.9, left=0.13, right=0.9, bottom=0.1, wspace=0.05, hspace=0.05)
    axzero.legend(handles=patches0, loc='lower right', bbox_to_anchor=(0, 0.35), fancybox=True, shadow=False)
    axtwo.legend(handles=patches2, loc='lower right', bbox_to_anchor=(0, 0.2), fancybox=True, shadow=False)

    axins = inset_axes(axone,
                       width="5%",
                       height="85%",
                       loc='lower left',
                       bbox_to_anchor=(-0.15, 0.075, 1, 1),
                       bbox_transform=axone.transAxes,
                       borderpad=0,
                       )

    fig.colorbar(cm.ScalarMappable(colors.Normalize(0, 1), cmap="jet"), cax=axins, ticks=[0, 0.25, 0.5, 0.75, 1],
                 ticklocation="left")

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    final_image = Image.open(buf)
    plt.close()
    return np.array(final_image)


def visualize_vertical(ims, truth, pred, attribute_names):
    """
    creates a visualisation containing 3 images per attribute
    contour - heatmap overlay - overlap
    layout is vertical
    """
    nrows = truth.shape[0]
    fig, ax = plt.subplots(nrows=nrows, ncols=3, figsize=(6, 11))
    fig.tight_layout()

    for i in range(nrows):
        ax[i, 0].axis('off')
        ax[i, 1].axis('off')
        ax[i, 2].axis('off')
        ax[i, 0].imshow(visualize_contours(ims[i], truth[i, :, :], pred[i, :, :]))
        ax[i, 1].imshow(visualize_probability(ims[i], pred[i, :, :]))
        ax[i, 2].imshow(visualize_matrix(ims[i], truth[i, :, :], pred[i, :, :]))
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax[i, 0].text(15, -25, attribute_names[i], fontsize=10, verticalalignment='bottom', bbox=props)

    axzero = ax[nrows - 1, 0]
    axone = ax[nrows - 1, 1]
    axtwo = ax[nrows - 1, 2]

    patches0 = [Patch(color=[0, 1, 0], label="Ground truth"),
                Patch(color=[0, 0, 1], label="Prediction")]
    patches2 = [Patch(color=[0, 1, 0], label="True positive"),
                Patch(color=[1, 0, 0], label="False positive"),
                Patch(color=[0, 0, 0], label="True negative"),
                Patch(color=[0, 0, 1], label="False negative")]
    #
    fig.subplots_adjust(top=0.9, left=0.13, right=0.9, bottom=0.1, wspace=0.05, hspace=0.05)
    axzero.legend(handles=patches0, loc='upper left', bbox_to_anchor=(-0.02, 0), fancybox=True, shadow=False)
    axtwo.legend(handles=patches2, loc='upper left', bbox_to_anchor=(-0.05, 0), fancybox=True, shadow=False)
    #
    axins = inset_axes(axone,
                       width="90%",
                       height="5%",
                       loc='lower center',
                       bbox_to_anchor=(0, -0.1, 1, 1),
                       bbox_transform=axone.transAxes,
                       borderpad=0,
                       )
    fig.colorbar(cm.ScalarMappable(colors.Normalize(0, 1), cmap="jet"), cax=axins, ticks=[0, 0.25, 0.5, 0.75, 1],
                 orientation="horizontal")

    # plt.gca().add_artist(legend1)
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    final_image = Image.open(buf)
    plt.close()
    return np.array(final_image)


def make_all_heat_plots(metric="crisp_iou", vertical=True):
    """

    """
    df = get_multi_index_results(metric)
    df.columns = df.columns.droplevel(0)
    loss_list = df.index.get_level_values("loss").unique().values
    attribute_name_list = df.columns.values

    results_to_plot = {}
    for loss_name, attribute_name in product(loss_list, attribute_name_list):
        sdf = df.loc[(slice(None), loss_name, slice(None), slice(None)), [attribute_name]]
        sdf = sdf.pivot_table(sdf, index=["balanced batches", "loss", "alpha"], columns=["gamma"])
        results_to_plot[(loss_name, attribute_name)] = sdf
    for loss_name in loss_list:
        full_heat_plot(df, loss_name, results_to_plot, metric, vertical)


def full_heat_plot(df, loss_name, results_to_plot, metric="crisp_iou", vertical=False):
    """
    create a figure to show the gridsearch data using 2d heat plots
    figure saved in "data/results/" as a pdf
    """
    attribute_name_list = df.columns.values
    fig_size = 2.4, 2
    width, height = len(attribute_name_list), 2
    if vertical:
        width, height = height, width

    fig, ax = plt.subplots(height, width, figsize=(width * fig_size[0], height * fig_size[1]), constrained_layout=True)

    min_values = [0, 0, 0, 0, 0]
    max_values = [0.363, 0.178, 0.315, 0.553, 0.192]
    for i, attribute_name in enumerate(attribute_name_list):
        data = results_to_plot[(loss_name, attribute_name)]

        if vertical:
            ax1, ax2 = ax[i, 0], ax[i, 1]
        else:
            ax1, ax2 = ax[0, i], ax[1, i]

        im1 = ax_heat_plot(ax1, data[0:4], min_values[i], max_values[i], invert_x=loss_name == "focal tversky loss")
        im2 = ax_heat_plot(ax2, data[4:8], min_values[i], max_values[i], invert_x=loss_name == "focal tversky loss")

        def get_text_coords(num_gamma):
            if num_gamma == 5:
                return -3.5, 1.5
            if num_gamma == 4:
                return -3, 1.5
            if num_gamma == 1:
                return -1, 0

        if not vertical:
            ax1.set_title(attribute_name)
            if i == 0:
                textx, texty = get_text_coords(len(data.columns))
                ax1.text(x=textx, y=texty, s="unbalanced", ha='center', va='center', rotation=90, size="large")
                ax2.text(x=textx, y=texty, s="balanced", ha='center', va='center', rotation=90, size="large")
            clb1 = fig.colorbar(im1, ax=ax1, shrink=0.9, orientation="horizontal", location="bottom")
            clb1.ax.set_title("$J_{C}$", x=-0.12, y=-1)
        else:
            textx, texty = get_text_coords(len(data.columns))
            ax1.text(x=textx, y=texty, s=attribute_name, ha='center', va='center', rotation=90, size="large")
            clb2 = fig.colorbar(im2, ax=ax2, orientation="vertical")
            clb2.ax.set_title("$J_{C}$")
            if i == 0:
                ax1.set_title("unbalanced")
                ax2.set_title("balanced")

    orientation_string = "vertical" if vertical else "horizontal"
    plt.savefig("data/results/plot_" + orientation_string + "_heatmap_" + loss_name + "_" + metric + ".pdf",
                bbox_inches="tight")


def ax_heat_plot(ax, data, min_val, max_val, invert_x):
    """
    create a heatmap of the gridsearch data for one loss/oversampling combination
    x-axis: gamma
    y-axis: alpha
    """
    x_labs = [str(round(k, 2)) for k in data.columns.droplevel(0)]
    if invert_x:
        x_labs = [str(round(1.0 / k, 2)) for k in data.columns.droplevel(0)]
        x_labs = x_labs[::-1]
        data = data.iloc[:, ::-1]

    y_labs = [str(round(k, 3)) for k in data.index.droplevel([0, 1])]
    y_labs = y_labs[::-1]
    data = data.iloc[::-1]

    x_sym, y_sym = "γ", "α"

    if len(data.columns) == 1:
        data = data.transpose()
        x_labs, y_labs = y_labs, [""]
        x_sym, y_sym = y_sym, ""
        ax.tick_params("y", left=False)

    fig = ax.imshow(data, cmap="viridis", vmin=min_val, vmax=max_val)

    ax.set_yticks(np.arange(len(y_labs)), labels=y_labs)
    ax.set_xticks(np.arange(len(x_labs)), labels=x_labs)
    ax.set_ylabel(y_sym, rotation=0)
    ax.set_xlabel(x_sym)

    idx_flat = data.to_numpy().ravel().argmax()
    idx = np.unravel_index(idx_flat, data.to_numpy().shape)
    fig.axes.text(idx[1], idx[0], "□", ha="center", va="center", color="black", size="x-large")

    return fig


def make_bar_plots(metric="crisp_iou"):
    """
    create a bar plot comparing best scores for different combinations of loss / oversampling
    """
    df = add_averages(table_all(get_multi_index_results(metric), metric), metric)
    df = df.loc[:, (slice(None), metric)]
    df.columns = df.columns.droplevel(1)

    df = df.rename(index={'binary focal loss': 'focal cross-entropy loss'})

    fig_size = 5, 4
    fig, ax = plt.subplots(figsize=(fig_size[0], fig_size[1]), constrained_layout=True)

    width = 0.2
    offset = width
    hues = [40, 220]
    f_sat, f_val = [1, 0.75]
    t_sat, t_val = [0.5, 0.9]

    colors = [hsv_to_rgb([hues[0] / 360, f_sat, f_val]), hsv_to_rgb([hues[0] / 360, t_sat, t_val]),
              hsv_to_rgb([hues[1] / 360, f_sat, f_val]), hsv_to_rgb([hues[1] / 360, t_sat, t_val])]

    for i, values in enumerate(product(["focal tversky loss", "focal cross-entropy loss"], [False, True])):
        loss_func, balanced = values
        series = df.loc[(balanced, loss_func)]
        x = np.arange(0, len(series))
        ax.bar(x + (i - 1.5) * offset, series, width, color=colors[i])  # , color=f_colors[i])

    ax.set_axisbelow(True)  # gridlines under bars
    ax.grid(axis="y", which='both', alpha=0.5, linewidth=1)
    range_max = np.round(df.to_numpy().max() + 0.05, 1)
    y_range = [0, range_max]
    y_ticks_delta = 0.1
    ax.set_ylim([y_range[0], y_range[1]])
    if "recall" in metric:
        label = "recall"
        rot = 90
    elif "precision" in metric:
        label = "precision"
        rot = 90
    else:
        label = "$J_{C}$"
        rot = 0

    ax.set_ylabel(label, rotation=rot, labelpad=0, y=0.51)
    ax.set_yticks(np.arange(y_range[0], y_range[1], y_ticks_delta))
    ax.set_yticks(np.arange(y_range[0], y_range[1], y_ticks_delta / 2), minor=True)

    x_ticks = list(df.columns)
    x_ticks.remove("average")
    x_ticks.append("mean")
    ax.set_xticks(np.arange(0, len(df.columns)), x_ticks, rotation=45, rotation_mode="anchor", ha="right", va="top")
    #
    patches = [Patch(facecolor=k, edgecolor='black') for k in [colors[k] for k in [0, 2, 1, 3]] + colors]
    legend_labels = ['', '', 'focal tversky loss              ', 'focal cross-entropy loss',
                     '', '', 'unbalanced', 'balanced']
    ax.legend(handles=patches, labels=legend_labels,
              frameon=False, bbox_to_anchor=(0.1, 1.2), ncol=4, handletextpad=0.5,
              handlelength=1.0, columnspacing=-0.5, loc="upper left", fontsize=10)

    plt.savefig('data/results/plot_bars_' + metric + '.pdf', bbox_inches="tight")


def get_loss_names():
    """
    creates a dictionary containing names of loss function variations
    key: shortened name
    value: full name
    """
    bce_losses = ['cross entropy', 'weighted cross entropy', 'unweighted focal loss', 'binary focal loss']
    bce_short = ['CE', 'WCE', 'FL', 'WFL']
    dice_losses = ['dice loss', 'tversky loss', 'focal dice loss', 'focal tversky loss']
    dice_short = ['DL', 'TL', 'FDL', 'FTL']
    log_cosh_losses = ['log cosh dice loss', 'log cosh tversky loss']
    log_cosh_short = ['LDL', 'LTL']
    labels = bce_losses + dice_losses + log_cosh_losses
    labels_short = bce_short + dice_short + log_cosh_short
    return {key: lab for key, lab in zip(labels_short, labels)}


def plot_results():
    """
    create bar plot comparing validation scores with literature
    """
    labels = ["globules", "milia like cyst", "negative network", "pigment network", "streaks", "mean"]
    literature_one = [0.341, 0.171, 0.228, 0.563, 0.156, 0.292]
    literature_two = [0.379, 0.172, 0.283, 0.584, 0.254, 0.334]

    this_work_old = [0.370, 0.170, 0.187, 0.580, 0.0740, 0.276]
    this_work_tversky = [0.376, 0.188, 0.288, 0.537, 0.345, 0.347]
    this_work_cross_entropy = [0.377, 0.198, 0.278, 0.533, 0.143, 0.306]

    x = np.arange(len(labels))  # the label locations
    width = 0.25

    fig, ax = plt.subplots()

    ax.bar(x - width, literature_one, width, label='ISIC 2018 winner', color="goldenrod")
    ax.bar(x, literature_two, width, label='TATL 2022', color="gold")
    ax.bar(x + width, this_work_tversky, width, label='Proposed method', color="royalblue")

    ax.legend()

    ax.set_axisbelow(True)  # gridlines under bars
    ax.grid(axis="y", which='both', alpha=0.5, linewidth=1)

    ax.set_ylabel("$J_{C}$", weight="demi", rotation=0)

    ax.set_xticks(np.arange(0, len(labels)), labels, rotation=45, rotation_mode="anchor", ha="right", va="top")

    fig.tight_layout()

    plt.show()
