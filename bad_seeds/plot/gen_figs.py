"""
Plotting functions for Figures in bad seed paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pathlib import Path


def smooth(scalars, weight):  # Weight between 0 and 1
    """Apply an exponential smoothing window

    Parameters
    ----------
    scalars : (N, ) array
        The time series to smooth.

    weight : float [0, 1]
        The smoothing weight.

    Returns
    -------
    smoothed : (N, ) array
        The smoothed data, same size as input data.
    """
    # First value in the plot (first timestep)
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        # Calculate smoothed value
        smoothed_val = last * weight + (1 - weight) * point
        # Save it
        smoothed.append(smoothed_val)
        # Anchor the last smoothed value
        last = smoothed_val

    return np.array(smoothed)


def general_axis_adjustments(ax, x_max):
    """Set standard axis labels and limits

    Parameters
    ----------
    ax : matplotlib.axes.Axes
       The Axes to adjust the axis of

    x_max : number
       The x-max for this Axes

    Returns
    -------
    ax : matplotlib.axes.Axes
       Same object passed in
    """
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 110, 10))
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Normalized score")
    return ax


def plot_timelimit_learning(df, timelimit, *, ax, label_loc="baseline", **kwargs):
    """Draw a single line for the time-limited figure.

    Parameters
    ----------
    df : pd.DataFrame

    timelimit : int
        The timelimit of this experiment in cycles

    ax : matplotlib.ax.Axes
        The axes to plot to.

    label_loc : {'baseline', 'data'}, optional
        Place the label at the baseline of the data or where it hits the
        right axes

    Other Parameters
    ----------------
    **kwargs
       Additional keyword arguments are passed through to ax.plot

    Returns
    -------
    ln, axline : matplotlib.lines.Line2D
    ann : matplotlib.text.Annotation
    ax : matplotlib.axes.Axes
    """
    values = smooth(df.val, 0.997)
    (ln,) = ax.plot(df.step, values, label=f"Fixed time = {timelimit}", **kwargs)

    if label_loc == "data":
        end_y = np.mean(values[-15:])
    elif label_loc == "baseline":
        end_y = 90 * timelimit / 100
    else:
        raise ValueError("label_loc must be one of {'data', 'baseline'}")

    axline = ax.axhline(end_y, ls="--", label=f"Sequential, t = {timelimit}", **kwargs)
    ann = ax.annotate(
        f"{timelimit} turns",
        (1, end_y),
        xycoords=ax.get_yaxis_transform(),
        xytext=(3, 0),
        ha="left",
        va="center",
        textcoords="offset points",
        weight="bold",
        **kwargs,
    )
    ax = general_axis_adjustments(ax, np.max(df.step))
    return ln, axline, ann, ax


def plot_all_timelimit(
    timelimits,
    ax,
    *,
    l_alpha=0.9,
    data_path=Path("../published_results"),
    score="default",
    batch_size=512,
):
    """Make the time-limited figure

    This expect that there will be CSV files in data_path with names ::

       {timelimit}_{score}_{batch_size}.csv

    Parameters
    ----------
    timelimits : List[int]
        The timelimits of the data to be plotted.

    ax : matplotlib.axes.Axes
        The axes to plot to

    l_alpha : float [0, 1], optional
        The alpha ot use drawing the lines
    data_path : Path, optional
        The location of the data files
    score : str, optional
        The scoring mode used.  Second value in name template.
    batch_size : int, optional
        The training batch size.  Third value in name template.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """

    cmap = plt.get_cmap("Dark2")

    plotted_data = []
    for j, timelimit in enumerate(sorted(timelimits, reverse=True)):
        path = data_path / Path(f"{timelimit}_{score}_{batch_size}.csv")
        df = pd.read_csv(str(path))
        plotted_data.append(
            plot_timelimit_learning(df, timelimit, ax=ax, alpha=l_alpha, color=cmap(j))
        )

    ax.legend(
        handles=(
            plt.Line2D([], [], color="k", ls="--", label="Sequential", alpha=l_alpha),
            plt.Line2D([], [], color="k", ls="-", label="Agent", alpha=l_alpha),
        ),
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=3,
        mode="expand",
        borderaxespad=0.0,
        frameon=False,
    )
    return ax


def plot_ideal_learning(batch_size, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    path = Path("../published_results") / Path(f"batch_{batch_size}.csv")
    df = pd.read_csv(str(path))
    ax.plot(
        df.step, smooth(df.val, 0.997), label=f"Batch size = {batch_size}", **kwargs
    )
    ax = general_axis_adjustments(ax, np.max(df.step))


def plot_all_ideal(batch_sizes=None, ax=None, l_alpha=0.9):
    if ax is None:
        fig, ax = plt.subplots()
    if batch_sizes is None:
        batch_sizes = [1, 8, 16, 32, 64, 128, 256, 512]
    norm = mpl.colors.LogNorm(vmin=1, vmax=512)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "Dan hates Yellow", plt.get_cmap("viridis_r")(np.linspace(0.2, 1, 256))
    )
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    for batch_size in batch_sizes:
        plot_ideal_learning(
            batch_size, ax=ax, alpha=l_alpha, color=cmap(norm(batch_size))
        )

    ideal_p = ax.axhline(90, color="k", ls=":", label="Ideal")
    seq_score = 90 * np.mean([i / 10 for i in range(1, 11)])
    seq_p = ax.axhline(seq_score, color="k", ls="--", label="Sequential")

    ax.figure.colorbar(sm, ax=ax, label="Batch Size")
    ax.legend(
        handles=(
            ideal_p,
            seq_p,
            plt.Line2D([], [], color="k", ls="-", label="Agent", alpha=l_alpha),
        ),
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=3,
        mode="expand",
        borderaxespad=0.0,
        frameon=False,
    )
    # TODO Add sequential policy expectation value
    # Legend
    # ax.legend(bbox_to_anchor=(.01, 1.11), loc='upper left', ncol=3)
    return ax


# ## Defaults for everything

# In[13]:


def make_figure_1(figsize=(8.5 / 2.54, 5), out_file="all.png"):
    mpl.rcParams["font.size"] = 7
    with mpl.rc_context({"font.size": 7}):
        fig, axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)
        plot_all_timelimit(ax=axes[1], timelimits=[10, 20, 30, 40, 50, 70, 100])
        plot_all_ideal(ax=axes[0])
        fig.savefig(out_file, dpi=300)
        return fig, axes


# ## PMM Suggestions
# Preferring the one with 4, even though the blue line is somewhat confusing. Clarification in the text.

# In[7]:


def make_figure_2(figsize=(8.5 / 2.54, 5), out_file="pmm_option.png"):
    mpl.rcParams["font.size"] = 7
    with mpl.rc_context({"font.size": 7}):
        fig, axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)
        plot_all_timelimit(timelimits=[10, 30, 70, 100], ax=axes[1])
        plot_all_ideal(batch_sizes=[32, 64, 128, 512], ax=axes[0])
        fig.savefig(out_file, dpi=300)


# In[8]:


def make_figure_3(figsize=(8.5 / 2.54, 5)):
    with mpl.rc_context({"font.size": 7}):
        fig, axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)
        plot_all_timelimit(timelimits=[10, 30, 70], ax=axes[1])
        plot_all_ideal(batch_sizes=[32, 64, 512], ax=axes[0])
