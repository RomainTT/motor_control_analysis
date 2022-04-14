#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from tqdm import tqdm_notebook
import pandas
import numpy
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

from . import signal as sigutils
from . import image as imutils
from . import data as datautils


def display_step_responses(
    processed_steps, joint_name, show=True, get_pil_image=False, title=None
):
    """Display all normalized step commands and responses on a single chart.

    Args:
        processed_steps (list): ProcessingResult objects.
        joint_name (str): the name of the joint
        show (bool) (optional): Display charts with matplotlib if True (default)
          If False, do not display anything.
        title (str) (optional): The title of the chart. A default one
          is dynamically created if None is given (default).
        get_pil_image (bool) (optional): True to return the PIL image of the figure.
          Default is False

    Return
       fig, img where:
           * fig: the matplotlib figure
           * img: the PIL image if get_pil_image is True
    """
    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(18, 10))

    for step in tqdm_notebook(processed_steps, desc="Plot steps", leave=False):
        norm_df = sigutils.get_normalized_position_timeseries(
            dataframe=step.dataframe,
            command_offset=step.kpis["InitialCommandValue"],
            command_amplitude=step.kpis["CommandStepAmplitude"],
        )
        sns.lineplot(
            x="Time",
            y=joint_name + "PositionSensorValue",
            data=norm_df,
            ax=ax,
            color=sns.color_palette()[0],
            alpha=0.1,
        )
        sns.lineplot(
            x="Time",
            y=joint_name + "PositionActuatorValue",
            data=norm_df,
            ax=ax,
            color=sns.color_palette()[3],
            alpha=0.1,
        )

    # Add custom legend
    custom_lines = [
        matplotlib.lines.Line2D([0], [0], color=sns.color_palette()[3], lw=8),
        matplotlib.lines.Line2D([0], [0], color=sns.color_palette()[0], lw=8),
    ]
    ax.legend(custom_lines, ["Inputs", "Responses"], prop={"size": 28})
    # Custom labels and title
    ax.set_xlabel("Time (s)", fontsize=26)
    ax.set_ylabel("{} position (normalized)".format(joint_name), fontsize=26)
    ax.tick_params(axis="both", which="major", labelsize=26)
    if title:
        ax.set_title(title, fontsize=30)
    else:
        ax.set_title("Step response of {}".format(joint_name), fontsize=30)

    # Create the PIL image if asked
    if get_pil_image:
        img = imutils.ax2img(fig, ax, expand=(1.2, 0, 0, 1.0))
    else:
        img = False

    if not show:
        plt.close(fig)

    return fig, img


def display_bode(
    processed_sines, joint_name, show=True, get_pil_image=False, title=None
):
    """Display the Bode diagram based en processed sine waves.

    Args:
        processed_sines (dict): keys are step unique IDs and values
          are ProcessingResult objects.
        joint_name (str): the name of the joint
        show (bool) (optional): Display charts with matplotlib if True (default)
          If False, do not display anything.
        title (str) (optional): The title of the chart. A default one
          is dynamically created if None is given (default).
        get_pil_image (bool) (optional): True to return the PIL image of the figure.
          Default is False

    Return
       fig, img where:
           * fig: the matplotlib figure
           * img: the PIL image if get_pil_image is True
    """
    # Collect data
    data = {"Frequency": [], "PhaseShiftAngle": [], "Gain": []}
    for sine in tqdm_notebook(processed_sines, desc="Plot sine waves", leave=False):
        data["Frequency"].append(sine.kpis["Frequency"])
        data["PhaseShiftAngle"].append(sine.kpis["PhaseShiftAngle"])
        data["Gain"].append(sine.kpis["Gain"])

    # Make a DataFrame to facilitate calculations
    df = pandas.DataFrame(data)

    sns.set(style="darkgrid")
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))

    # Set common settings for both axes
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Freq (Hz)")
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
        ax.minorticks_on()
        ax.grid(b=True, which="major", linewidth=3, linestyle="-")
        ax.grid(b=True, which="minor", linewidth=1, linestyle="-")

    # Gain
    # Conversion in dB is made by 20*log(Gain)
    axes[0].set_ylabel("Gain (dB)")
    axes[0].scatter(data["Frequency"], 20 * numpy.log10(df["Gain"]))

    # Phase
    # Given in multiple of pi
    # Display phase as negative in the diagram
    axes[1].set_ylabel("Phase (rad)")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g $\pi$"))
    axes[1].scatter(df["Frequency"], -df["PhaseShiftAngle"] / numpy.pi)

    # Set title
    if title:
        fig.suptitle(title, fontsize=30)
    else:
        fig.suptitle("Bode diagram", fontsize=30)

    # Create the PIL image if asked
    if get_pil_image:
        # Crop blank margins
        extent = ax.get_window_extent()
        extent = extent.transformed(fig.dpi_scale_trans.inverted())
        extent.x0 += 0.1
        img = imutils.fig2img(fig, bbox_inches="tight", dpi=100)
    else:
        img = False

    if not show:
        plt.close(fig)

    return fig, img


def display_kpi_over_step_speed(
    processed_steps,
    kpi_name,
    figure_title=None,
    label_text=None,
    fit_function=None,
    apply_func=None,
    show=True,
    get_pil_image=False,
):
    """Display a chart of a KPI in function of normalized step speed.

    What is called "normalized step speed" is the speed of the step for a
    normalized amplitude (amplitude = 1). It means that the step signal has been
    divided by its real amplitude, modifying the initial speed.

    Args:
        processed_steps (list): ProcessingResult objects.
        kpi_name (str): the name of the KPI to display in function of step speed.
        figure_title (str) (optional): Title of the figure. Default is None.
        label_text (str) (optional): the text of the y-label.
        fit_function (str) (optional): If not None, fit data with a function.
          Can be 'inverse' or 'first_order'. Default is None.
        apply_func (callable) (optional): a function to apply to the KPI
          before displaying it.
        show (bool) (optional): Display figure with matplotlib if True.
          If False, do not display anything. Default is True.
        get_pil_image (bool) (optional): Save the figure in the form of an
          image in the returned dictionary. Default is False.

    Returns:
        (matplotlib.figure.Figure, Image, Dict)
        * The pyplot figure containing the chart.
        * The PIL image of the figure if get_pil_image is True.
        * The result of get_data_fit()
        Objects can be None if data is empty or option is False.

    Raises:
        ValueError: if the KPI name is unknown.
    """
    # Initialize the dictionary where data will be collected from the processed_steps
    data = {k: [] for k in ["norm_step_speed", kpi_name]}

    # Collect data
    for step in tqdm_notebook(processed_steps, desc="Collect KPIs", leave=False):
        # Compute normalized step speed
        data["norm_step_speed"].append(
            abs(step.kpis["CommandSpeed"] / step.kpis["CommandStepAmplitude"])
        )
        if apply_func:
            kpi_value = apply_func(step.kpis[kpi_name])
        else:
            kpi_value = step.kpis[kpi_name]
        data[kpi_name].append(kpi_value)

    # Build a dataframe to make plotting and processing easier
    df = pandas.DataFrame(data)

    # Remove NA values
    df.dropna(axis="index", inplace=True)

    if df.size == 0:
        # No data left for this value ! Cannot display anything
        im = Image.new("RGB", (300, 200), color="white")
        imutils.add_txt_on_img(im, "No data", (100, 25), 60)
        return None, im, None

    # Fit data with a function if asked
    if fit_function:
        fit_res = sigutils.get_data_fit(
            df["norm_step_speed"], df[kpi_name], fit_function
        )
    else:
        fit_res = None

    # Make the figure if asked
    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot raw points
    ax.scatter(x=df["norm_step_speed"], y=df[kpi_name], s=170, c="b", edgecolors="w")

    if fit_function:
        # Plot the fitting curve on larger range than the input data
        max_speed = df["norm_step_speed"].max()
        min_speed = df["norm_step_speed"].min()
        speed_range = max_speed - min_speed
        if speed_range > 0:
            extended_max_speed = max_speed + 0.1 * speed_range
            # Do not exceed 2 on the left side to avoid infinite curve to zoom
            # out the chart. 2 s^-1 is a very low speed and is not likely to be
            # displayed.
            extended_min_speed = max(2, min_speed - 0.1 * speed_range)
        else:
            # max and min speed are equal. In this case, do not compute the displayed
            # range in function of the speed range. Just extend to 10% of the value
            # on each side.
            extended_max_speed = max_speed + 0.1 * max_speed
            extended_min_speed = max(2, min_speed - 0.1 * min_speed)
        extended_range = extended_max_speed - extended_min_speed
        x = numpy.arange(extended_min_speed, extended_max_speed, extended_range / 100)
        fit_data = fit_res["fit_func_call"](x)
        # Plot the fitting curve
        ax.plot(x, fit_data, linewidth=5, c=sns.color_palette()[1])
        # Calculate every limit curve
        limit_curves = {}
        limit_curves["upper_inner_limit"] = fit_data + fit_res["disp_inner"] * fit_data
        limit_curves["lower_inner_limit"] = fit_data - fit_res["disp_inner"] * fit_data
        limit_curves["upper_outer_limit"] = fit_data + fit_res["disp_outer"] * fit_data
        limit_curves["lower_outer_limit"] = fit_data - fit_res["disp_outer"] * fit_data

        for limit in limit_curves.values():
            ax.plot(x, limit, linewidth=5, c=sns.color_palette()[1], linestyle=":")

        # Plot an area between 50% limits
        ax.fill_between(
            x,
            limit_curves["lower_inner_limit"],
            limit_curves["upper_inner_limit"],
            facecolor=sns.color_palette()[1],
            alpha=0.25,
        )

        # Extend X limits to see all the fitting curve
        ax.set_xlim(left=extended_min_speed, right=extended_max_speed)

        # Display text below the chart
        xlim1, xlim2 = ax.get_xlim()
        ylim1, ylim2 = ax.get_ylim()
        xmid = xlim1 + (xlim2 - xlim1) / 2
        ypos = ylim1 - (ylim2 - ylim1) / 2.5  # value found after tests
        txt = (
            "Fitting function: {}\n" "inner limits: ±{:.0f}%  " "outer limits: ±{:.0f}%"
        ).format(
            fit_res["fit_func_str"],
            fit_res["disp_inner"] * 100,
            fit_res["disp_outer"] * 100,
        )
        ax.text(
            xmid,
            ypos,
            txt,
            horizontalalignment="center",
            fontsize=26,
            multialignment="left",
            bbox=dict(
                boxstyle="round",
                facecolor="#D8D8D8",
                edgecolor="0.5",
                pad=0.5,
                alpha=0.25,
            ),
            fontweight="bold",
        )

    # Set title and labels
    ax.tick_params(axis="both", which="major", labelsize=26)
    ax.set_xlabel("Normalized step speed (s⁻¹)", fontsize=26)
    if label_text:
        ax.set_ylabel(label_text, fontsize=26)
    else:
        ax.set_ylabel(kpi_name, fontsize=26)
    if figure_title:
        ax.set_title(figure_title, fontsize=30)

    if get_pil_image:
        # Save this subplot in the result as an image
        im = imutils.ax2img(fig, ax, expand=(1.5, 0, 1.0, 2.25))
    else:
        im = None

    # Remove the figure if not wanted
    if not show:
        plt.close(fig)

    return fig, im, fit_res


def display_kpi_distribution(
    processing_results,
    kpi_name,
    apply_func=None,
    figure_title=None,
    show=True,
    get_pil_image=False,
):
    """Get and display the distribution of a KPI for given processing results.

    Display a boxplot of the given KPI for the given processing results.

    Args:
        processing_results (list): ProcessingResult objects.
        kpi_name (str): the name of the KPI to display.
        apply_func (callable) (optional): function to modify data before
          creating the boxplot.
        show (bool) (optional): Display charts with matplotlib if True (default)
          If False, do not display anything.
        figure_title (str) (optional): Title of the whole figure. Default is None.
        display (bool) (optional): Display charts with matplotlib if True (default)
          If False, do not display anything.
        get_pil_image (bool) (optional): Save the figure in the form of an
          image in the returned dictionary. Default is False.

    Returns:
        (matplotlib.figure.Figure, Image, Dict)
        * The pyplot figure containing the chart.
        * The PIL image of the figure if get_subplots_as_img is True.
        * The following dictionary:
            {
                'q1': <float>,
                'q3': <float>,
                'median': <float>,
                'mean': <float>,
                'wmin': <float>,
                'wmax': <float>,
            }

    Raises:
        ValueError: if the KPI name is unknown.
    """
    # Collect data
    data = []
    for result in tqdm_notebook(
        processing_results, desc="Collect results", leave=False
    ):
        data.append(result.kpis[kpi_name])

    # Remove NaN from data
    data = numpy.array(data)
    data = data[~numpy.isnan(data)]

    # Avoid empty data
    # return None for figure and result, and a 'N/A' text for the image.
    if len(data) == 0:
        im = Image.new("RGB", (300, 200), color="white")
        imutils.add_txt_on_img(im, "No data", (100, 25), 60)
        return None, im, None

    # Apply optional function
    if apply_func:
        apply_func = numpy.vectorize(apply_func)
        data = apply_func(data)

    # Make the figure
    fig, ax = plt.subplots(figsize=(3, 5))
    if figure_title:
        ax.set_title(figure_title, weight="bold")
    # Make a single boxplot with a custom design
    bp_dict = ax.boxplot(
        x=data,
        widths=0.25,
        showmeans=True,
        patch_artist=True,
        flierprops=dict(markerfacecolor="grey", marker="o", alpha=0.25),
        boxprops=dict(linewidth=2, color="black", facecolor=sns.color_palette()[0]),
        medianprops=dict(linewidth=2, color="black"),
        meanprops=dict(
            marker="X",
            markeredgewidth=0,
            markerfacecolor=sns.color_palette()[1],
            markersize=10,
        ),
        whiskerprops=dict(linewidth=2),
        capprops=dict(linewidth=2),
    )
    ax.axis("off")

    # Display important values on the chart
    # --- Compute all important values
    q1, median, q3 = numpy.percentile(data, [25, 50, 75])
    mean = numpy.mean(data)
    irq = q3 - q1
    wmin = q1 - 1.5 * irq
    # Lower whisker cannot be below minimum value of data
    if wmin < numpy.min(data):
        wmin = numpy.min(data)
    wmax = q3 + 1.5 * irq
    # Upper whisker cannot be over maximum value of data
    if wmax > numpy.max(data):
        wmax = numpy.max(data)

    # Short functions to display right or left
    left_text = lambda t, v: ax.text(
        0.15, v, "{}: {:.2f}".format(t, v), verticalalignment="center"
    )
    right_text = lambda t, v: ax.text(
        1.15, v, "{}: {:.2f}".format(t, v), verticalalignment="center"
    )
    # --- Display median
    left_text("median", median)
    # --- Display mean
    right_text("mean", mean)
    # --- Display quartiles
    right_text("1st quartile", q1)
    right_text("3rd quartile", q3)
    # --- Display whiskers
    left_text("min whisker", wmin)
    left_text("max whisker", wmax)

    # Create result object
    res = {
        "q1": q1,
        "q3": q3,
        "median": median,
        "mean": mean,
        "wmin": wmin,
        "wmax": wmax,
    }

    if get_pil_image:
        # Save this subplot in the result as an image
        img = imutils.ax2img(fig, ax, expand=(0.75, 1, 0.20, 0), dpi=150)
    else:
        img = None

    if not show:
        plt.close(fig)

    return fig, img, res


def display_op_groups(op_groups, data_type):
    """Display the data distribution among operating groups."""

    # Build a dataframe with operating groups parameters and data size for each
    # I wanted to do a list comprehension here but it failed for an unknown reason.
    keys_with_size = []

    for k, v in op_groups.items():
        tmp = list(k)
        tmp.append(len(v))
        keys_with_size.append(tmp)

    df = pandas.DataFrame(
        keys_with_size,
        columns=["Current", "Temperature", "Load direction", "Data size"],
    )

    # Transform data to make them displayable on a chart
    df["Current"] = df["Current"].apply(str)
    df["Temperature"] = df["Temperature"].apply(str)
    load_dir_translate = {0: "No load", 1: "Same than movement", -1: "Against movement"}
    df["Load direction"] = df["Load direction"].apply(lambda x: load_dir_translate[x])
    col_order = numpy.sort(df["Current"].unique())
    row_order = numpy.sort(df["Temperature"].unique())

    # Display the chart
    if data_type == "step":
        sns.relplot(
            data=df,
            x="Current",
            y="Temperature",
            hue="Load direction",
            size="Data size",
            sizes=(100, 1000),
            alpha=0.5,
            row_order=row_order,
            col_order=col_order,
        )
    elif data_type == "sine":
        sns.relplot(
            data=df,
            x="Current",
            y="Temperature",
            size="Data size",
            sizes=(100, 1000),
            alpha=0.5,
            row_order=row_order,
            col_order=col_order,
        )


def display_step_synthesis_pages(
    processed_steps, op_groups, joint_name, show=True, save_dir="./build/pepper"
):
    """Build, display and save KPI synthesis for steps.

    If save_dir is not None, synthesis data is saved in a
    pickle files. These files contain the following dictionary:
        {
            'processing_results': <List[ProcessingResult]>,
            'operating_point':{
                'joint_name': <str>,
                'current_range': <2-tuple>,
                'temperature_range': <2-tuple>,
                'load_direction': <int>,
            }
            'rising_time_10_on_speed': {
                'fit_function': <str>,
                'fit_params': <List[float]>,
                'inner_disp': <float>,
                'outer_disp': <float>
            },
            'rising_time_90_on_speed': …,
            'overshoot_time_on_speed': …,
            'settling_time_on_speed': …,
            'overshoot_on_speed': …,
            'overshoot_dist': {
                'q1': <float>,
                'q3': <float>,
                'median': <float>,
                'mean': <float>,
                'wmin': <float>,
                'wmax': <float>,
            },
            'steady_state_error_dist': …,
            'steady_state_current_dist': …,
            'maximum_current_dist': …,
        }

    Args:
        processed_steps (list): ProcessingResult objects.
        op_groups (dict): operating groups.
        joint_name (str): the name of the joint
        show (bool) (optional): Display images in the notebook. Default is True.
        save_dir (str) (optional): Directory where to write output files.
            Output files are synthesis images and pickle files.
            If set to None, data won't be saved on the hard drive.
            Default is ./build/pepper

    Returns:
        (List[PIL.Images]) A list of Images.
    """
    all_img = []
    for group_keys, group_ids in tqdm_notebook(
        op_groups.items(), leave=False, desc="Groups"
    ):
        # No copy, only references to save memory
        group_results = [v for k, v in enumerate(processed_steps) if k in group_ids]

        # Get the responses in a single chart
        fig, img_step_responses = display_step_responses(
            processed_steps=group_results,
            joint_name=joint_name,
            show=False,
            get_pil_image=True,
        )

        # Get chart of 10% rising time
        fig, rt10_img, rt10_values = display_kpi_over_step_speed(
            processed_steps=group_results,
            kpi_name="RisingTime_0.1",
            figure_title=("10% Rising time\n" "in function of normalized step speed"),
            label_text="10% Rising time (ms)",
            fit_function="inverse",
            apply_func=lambda x: x * 1000.0,  # To convert into ms
            show=False,
            get_pil_image=True,
        )

        # Get chart of 90% rising time
        fig, rt90_img, rt90_values = display_kpi_over_step_speed(
            processed_steps=group_results,
            kpi_name="RisingTime_0.9",
            figure_title=("90% Rising time\n" "in function of normalized step speed"),
            label_text="90% Rising time (ms)",
            fit_function="inverse",
            apply_func=lambda x: x * 1000.0,  # To convert into ms
            show=False,
            get_pil_image=True,
        )

        # Get chart of overshoot time
        fig, ost_img, ost_values = display_kpi_over_step_speed(
            processed_steps=group_results,
            kpi_name="OvershootTime",
            figure_title=(
                "1st overshoot time\n" "in function of normalized step speed"
            ),
            label_text="1st overshoot time (ms)",
            fit_function="inverse",
            show=False,
            get_pil_image=True,
        )
        # Get chart of settling time
        fig, st_img, st_values = display_kpi_over_step_speed(
            processed_steps=group_results,
            kpi_name="SettlingTime",
            figure_title=("Settling time\n" "in function of normalized step speed"),
            label_text="Settling time (ms)",
            fit_function="inverse",
            apply_func=lambda x: x * 1000.0,  # To convert into ms
            show=False,
            get_pil_image=True,
        )

        # Get chart of overshoot percentage
        fig, osp_img, osp_values = display_kpi_over_step_speed(
            processed_steps=group_results,
            kpi_name="OvershootPercentage",
            figure_title=(
                "Overshoot percentage\n" "in function of normalized step speed"
            ),
            label_text="Overshoot (%)",
            fit_function="first_order",
            show=False,
            get_pil_image=True,
        )

        # Get overshoot distribution
        fig, osd_img, osd_values = display_kpi_distribution(
            processing_results=group_results,
            kpi_name="OvershootPercentage",
            figure_title="Overshoot (%) distribution",
            show=False,
            get_pil_image=True,
        )

        # Get SteadyStateError distribution
        fig, sse_img, sse_values = display_kpi_distribution(
            processing_results=group_results,
            kpi_name="SteadyStateErrorPercentage",
            figure_title="Steady state error (%) distribution",
            show=False,
            get_pil_image=True,
        )

        # Get max current distribution
        fig, maxcur_img, maxcur_values = display_kpi_distribution(
            processing_results=group_results,
            kpi_name="MaximumCurrent",
            figure_title="Maximum current (A) distribution",
            show=False,
            get_pil_image=True,
        )

        # Get ss current distribution
        fig, sscur_img, sscur_values = display_kpi_distribution(
            processing_results=group_results,
            kpi_name="SteadyStateCurrent",
            figure_title="Steady state current (A) distribution",
            show=False,
            get_pil_image=True,
        )

        # Open the template to create the new image
        synthesis_im = Image.open("resources/template_step_synthesis.png")
        # The following dict is used to put words on the load direction param.
        load_translation = {
            0: "Friction only",
            -1: "Opposite to movement",
            +1: "Same than movement",
        }
        # The following dict contains all the text that is written on the image
        text_info = {
            "joint": {"text": joint_name, "pos": (490, 220), "size": 50},
            "current_r": {
                "text": "[ {}, {} [  A".format(
                    str(group_keys[0][0]), str(group_keys[0][1])
                ),
                "pos": (490, 315),
                "size": 46,
            },
            "temp_r": {
                "text": "[ {}, {} [  °C".format(
                    str(group_keys[1][0]), str(group_keys[1][1])
                ),
                "pos": (490, 415),
                "size": 46,
            },
            "load_d": {
                "text": "{}".format(load_translation[group_keys[2]]),
                "pos": (490, 530),
                "size": 38,
            },
            "data_size": {
                "text": "{} steps".format(len(group_ids)),
                "pos": (490, 630),
                "size": 46,
            },
        }
        # The following dict contains all the images that are pasted on the image
        img_info = {
            "step_responses": {
                "img": img_step_responses,
                "pos": (220, 780),
                "resize": 0.80,
            },
            "t10": {"img": rt10_img, "pos": (220, 1290), "resize": 0.80},
            "t90": {"img": rt90_img, "pos": (220, 1750), "resize": 0.80},
            "ost": {"img": ost_img, "pos": (220, 2235), "resize": 0.80},
            "osp": {"img": osp_img, "pos": (1250, 1275), "resize": 0.80},
            "st": {"img": st_img, "pos": (1250, 785), "resize": 0.80},
            "osd": {"img": osd_img, "pos": (970, 1745), "resize": 0.68},
            "sse": {"img": sse_img, "pos": (1460, 1745), "resize": 0.68},
            "max_current": {"img": maxcur_img, "pos": (970, 2240), "resize": 0.68},
            "ss_current": {"img": sscur_img, "pos": (1460, 2240), "resize": 0.68},
        }

        # Add each piece of text on the image
        for item in text_info.values():
            imutils.add_txt_on_img(
                synthesis_im, item["text"], item["pos"], item["size"]
            )
        # Add each sub-image on the image
        for item in img_info.values():
            imutils.add_img_on_img(
                synthesis_im, item["img"], item["pos"], item["resize"]
            )
        # Save synthesis in files
        if save_dir:
            base_dir = os.path.join(save_dir, joint_name)
            # Create the directories if they do not exist
            os.makedirs(base_dir, exist_ok=True)

            base_name = (
                "synthesis_step_{joint}_{current[0]}-{current[1]}_"
                "{temp[0]}-{temp[1]}_{load}"
            ).format(
                joint=joint_name,
                current=group_keys[0],
                temp=group_keys[1],
                load=group_keys[2],
            )
            # Save the image
            filename = "{}.png".format(base_name)
            file_path = os.path.join(base_dir, filename)
            synthesis_im.save(file_path, format="png")

            # Save the pickle file
            filename = "{}.pickle".format(base_name)
            file_path = os.path.join(base_dir, filename)
            # Drop lambdas as they cannot be pickled
            for val_dict in [
                rt10_values,
                rt90_values,
                ost_values,
                st_values,
                osp_values,
            ]:
                if val_dict is not None:
                    val_dict.pop("fit_func_call", None)
            # Store data in pickle file
            datautils.store_op_data(
                file_path,
                "step",
                joint_name,
                group_results,
                group_keys,
                rising_time_10_on_speed=rt10_values,
                rising_time_90_on_speed=rt90_values,
                overshoot_time_on_speed=ost_values,
                settling_time_on_speed=st_values,
                overshoot_on_speed=osp_values,
                overshoot_dist=osd_values,
                steady_state_error_dist=sse_values,
                steady_state_current_dist=sscur_values,
                maximum_current_dist=maxcur_values,
            )

        # Add the image to the returned list
        all_img.append(synthesis_im)

    # Display the image if it is asked for
    if show:
        for img in all_img:
            display(img)

    return all_img


def display_sine_synthesis_pages(
    processed_sines, op_groups, joint_name, show=True, save_dir="./build/pepper"
):
    """Build, display and save KPI synthesis for sine waves.

    If save_dir is not None, synthesis data is saved in a
    pickle files. These files contain the following dictionary:
        {
            'processing_results': <List[ProcessingResult]>,
            'operating_point':{
                'joint_name': <str>,
                'current_range': <2-tuple>,
                'temperature_range': <2-tuple>,
            }
            'offset_dist': {
                'q1': <float>,
                'q3': <float>,
                'median': <float>,
                'mean': <float>,
                'wmin': <float>,
                'wmax': <float>,
            },
            'maximum_current_dist': …,
        }


    Args:
        processed_sines (list): ProcessingResult objects.
        op_groups (dict): operating groups.
        joint_name (str): the name of the joint
        show (bool) (optional): Display images in the notebook. Default is True.
        save_dir (str) (optional): Directory where to write output files.
          Output files are synthesis images and pickle files.
          If set to None, data won't be saved on the hard drive.
          Default is ./build/pepper

    Returns:
        (List[PIL.Images]) A list of Images.
    """
    all_img = []
    for group_keys, group_ids in tqdm_notebook(
        op_groups.items(), leave=False, desc="Groups"
    ):
        # No copy, only references to save memory
        group_results = [v for k, v in enumerate(processed_sines) if k in group_ids]

        # Get bode diagram
        fig, bode_img = display_bode(
            processed_sines=group_results,
            joint_name=joint_name,
            show=False,
            get_pil_image=True,
        )

        # Get max current distribution
        fig, maxcur_img, maxcur_values = display_kpi_distribution(
            processing_results=group_results,
            kpi_name="MaximumCurrent",
            figure_title="Maximum current (A) distribution",
            show=False,
            get_pil_image=True,
        )

        # Get offset distribution
        fig, offset_img, offset_values = display_kpi_distribution(
            processing_results=group_results,
            kpi_name="Offset",
            figure_title="Offset (mrad) distribution",
            apply_func=lambda x: numpy.abs(x) * 1000,
            show=False,
            get_pil_image=True,
        )

        # Open the template to create the new image
        synthesis_im = Image.open("resources/template_sine_synthesis.png")

        # The following dict contains all the text that is written on the image
        text_info = {
            "joint": {"text": joint_name, "pos": (490, 220), "size": 50},
            "current_r": {
                "text": "[ {}, {} [  A".format(
                    str(group_keys[0][0]), str(group_keys[0][1])
                ),
                "pos": (490, 315),
                "size": 46,
            },
            "temp_r": {
                "text": "[ {}, {} [  °C".format(
                    str(group_keys[1][0]), str(group_keys[1][1])
                ),
                "pos": (490, 415),
                "size": 46,
            },
            "load_d": {"text": "Both", "pos": (490, 530), "size": 38},
            "data_size": {
                "text": "{} sine waves".format(len(group_ids)),
                "pos": (490, 630),
                "size": 46,
            },
        }
        # The following dict contains all the images that are pasted on the image
        img_info = {
            "bode": {"img": bode_img, "pos": (250, 800), "resize": 1.05},
            "max_current": {"img": maxcur_img, "pos": (1050, 2050), "resize": 1},
            "offset": {"img": offset_img, "pos": (250, 2050), "resize": 1},
        }

        # Add each piece of text on the image
        for item in text_info.values():
            imutils.add_txt_on_img(
                synthesis_im, item["text"], item["pos"], item["size"]
            )
        # Add each sub-image on the image
        for item in img_info.values():
            imutils.add_img_on_img(
                synthesis_im, item["img"], item["pos"], item["resize"]
            )
        # Save the image in a file
        if save_dir:
            base_dir = os.path.join(save_dir, joint_name)
            # Create the directories if they do not exist
            os.makedirs(base_dir, exist_ok=True)

            base_name = (
                "synthesis_sine_{joint}_{current[0]}-{current[1]}_"
                "{temp[0]}-{temp[1]}_{load}"
            ).format(
                joint=joint_name,
                current=group_keys[0],
                temp=group_keys[1],
                load=group_keys[2],
            )
            # Save the image
            filename = "{}.png".format(base_name)
            im_path = os.path.join(base_dir, filename)
            synthesis_im.save(im_path, format="png")

            # Save the pickle file
            filename = "{}.pickle".format(base_name)
            file_path = os.path.join(base_dir, filename)
            datautils.store_op_data(
                file_path,
                "sine",
                joint_name,
                group_results,
                group_keys,
                offset_dist=offset_values,
                maximum_current_dist=maxcur_values,
            )

        # Add the image to the returned list
        all_img.append(synthesis_im)

    # Display the image if it is asked for
    if show:
        for img in all_img:
            display(img)

    return all_img
