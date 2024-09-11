#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys
import os

plots = {
    "latencies": "Plot data from the combined latencies using the x.latencies.csv analysis output",
    "windowed-frame-stats": "Plot frame stats using the 'x.windowed.stats.csv' analysis output",
    "frame-duration-hist": "Plot histogram using 'x.frame.duration.csv' analysis output",
    "avsync": "Plot AV sync using the 'x.avsync.csv' analysis output",
    "video-latency": "Plot video latency using the 'x.video.latency.csv' analysis output",
    "audio-latency": "Plot audio latency and related data using the 'x.audio.latency.csv' analysis output",
    "measurement-quality": "Plot measurement quality using the 'x.measurement.quality.csv' analysis output (or the all.measurement.quality.csv)",
}

extensions = {
    "latencies": "latencies.csv",
    "windowed-frame-stats": "windowed.stats.csv",
    "frame-duration-hist": "frame.duration.csv",
    "avsync": "avsync.csv",
    "video-latency": "video.latency.csv",
    "audio-latency": "audio.latency.csv",
    "measurement-quality": "measurement.quality.csv",
}


def configure_axis(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()


def plot_latencies(data, options):
    # frame_num | audio_latency_sec | video_latency_sec | av_sync_sec
    if (
        len(
            [
                col
                for col in [
                    "audio_latency_sec",
                    "video_latency_sec",
                    "av_sync_sec",
                    "frame_num",
                ]
                if col in data.columns
            ]
        )
        == 0
    ):
        print("Error: data does not contain correct columns")
        exit(0)

    if options.aggregate:
        pass

    title = options.title if options.title else "Latencies"
    if options.rolling and not options.aggregate:
        title = f"{title} (rolling window: {options.rolling} values)"
    plot_columns(
        data,
        "frame_num",
        ["audio_latency_sec", "video_latency_sec", "av_sync_sec"],
        ["Frame number"] * 3,
        ["Latency (sec)"] * 3,
        title,
        options,
    )


def plot_video_latency(data, options):
    # frame_num | timestamp | frame_num_read | original_frame | video_latency_sec
    if (
        len([col for col in ["video_latency_sec", "timestamp"] if col in data.columns])
        == 0
    ):
        print("Error: data does not contain correct columns")
        exit(0)

    title = options.title if options.title else "Video latency"
    if options.rolling and not options.aggregate:
        title = f"{title} (rolling window: {options.rolling} values)"
    plot_columns(
        data,
        "timestamp",
        ["video_latency_sec"],
        ["Time (sec)"],
        ["AV Sync (sec)"],
        title,
        options,
    )


def plot_audio_latency(data, options):
    # frame_num | timestamp | frame_num_read | original_frame | video_latency_sec
    if (
        len([col for col in ["audio_latency_sec", "timestamp"] if col in data.columns])
        == 0
    ):
        print("Error: data does not contain correct columns")
        exit(0)

    title = options.title if options.title else "Audio latency"
    if options.rolling and not options.aggregate:
        title = f"{title} (rolling window: {options.rolling} values)"
    plot_columns(
        data,
        "timestamp1",
        ["audio_latency_sec", "cor1", "cor2"],
        ["Time (sec)"] * 3,
        ["AV Sync (sec)", "Correlation 1", "Correlation 2"],
        title,
        options,
    )


def plot_av_sync(data, options):
    # frame_num | timestamp | frame_num_read | original_frame | av_sync_sec
    if len([col for col in ["av_sync_sec", "timestamp"] if col in data.columns]) == 0:
        print("Error: data does not contain correct columns")
        exit(0)
    title = options.title if options.title else "AV Sync"
    if options.rolling and not options.aggregate:
        title = f"{title} (rolling window: {options.rolling} values)"
    plot_columns(
        data,
        "timestamp",
        ["av_sync_sec"],
        ["Time (sec)"],
        ["AV Sync (sec)"],
        title,
        options,
    )


def plot_measurement_quality(data, options):
    # video_frames_metiq_errors_percentage | video_frames_metiq_error.no_valid_tag | video_frames_metiq_error.invalid_graycode | video_frames_metiq_error.single_graycode_bit | video_frames_metiq_error.unknown | signal_distance_sec | max_correlation | min_correlation | mean_correlation | index

    if (
        len(
            [
                col
                for col in ["video_frames_metiq_errors_percentage"]
                if col in data.columns
            ]
        )
        == 0
    ):
        print("Error: data does not contain correct columns")
        exit(0)

    title = options.title if options.title else "Measurement quality - Parsing errors"
    if options.rolling and not options.aggregate:
        title = f"{title} (rolling window: {options.rolling} frames)"
    plot_columns(
        data,
        "file",
        [
            "video_frames_metiq_errors_percentage",
            "video_frames_metiq_error.no_valid_tag",
            "video_frames_metiq_error.invalid_graycode",
            "video_frames_metiq_error.single_graycode_bit",
            "video_frames_metiq_error.unknown",
        ],
        ["File"] * 6,
        [
            "Percentage",
            "count",
            "count",
            "count",
            "count",
        ],
        title,
        options,
    )


def plot_frame_duration_hist(data, args):
    # | count | frame_count |   time | capture_fps | ref_fps
    # Verify that data contain correct columns
    if len([col for col in ["count", "time"] if col in data.columns]) == 0:
        print("Error: data does not contain correct columns")
        exit(0)

    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(111)
    data["time ms"] = (data["time"] * 1000).astype(int)
    data["pct"] = (data["count"] / data["count"].sum() * 100).astype(int)
    ref_duration_ms = int(1000.0 / data["ref_fps"].unique()[0])
    if args.aggregate:
        ax.bar(data["time ms"], data["pct"])
    else:
        for file in data["file"].unique():
            _data = pd.DataFrame(data[data["file"] == file])
            ax.bar(x=_data["time ms"], height=_data["pct"], label=file)

    ax.vlines(
        ref_duration_ms,
        0,
        data["pct"].max(),
        color="red",
        alpha=0.6,
        linewidth=4,
        linestyle="dotted",
        label=f"Reference frame duration: {ref_duration_ms}ms",
    )
    configure_axis(ax, "Frame duration histogram", "duration in msec", "count")

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.2),
        ncol=1,
        fancybox=True,
        shadow=True,
        borderaxespad=0.0,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if args.output:
        plt.savefig(args.output)
    if args.show:
        plt.show()


def plot_columns(data, xname, columns, xtitles, ytitles, title, options):
    figw = options.width
    figh = options.height

    switch_col_row = False
    if options.aggregate:
        ncol = len(columns)
        fig = plt.figure(figsize=(figw, figh))

        group = data.groupby("file")
        mean = group.mean()

        columns_data = []
        labels_data = []
        for col in columns:
            mean_col = mean[col]
            mean_col.dropna(inplace=True)
            columns_data.append(mean_col)
            labels_data.append(col)

        for i in range(len(columns_data)):
            ax = None
            if switch_col_row:
                ax = fig.add_subplot(2, ncol, i + 1)
            else:
                ax = fig.add_subplot(ncol + 2, ncol, i + 1)
            ax.boxplot(columns_data[i : i + 1], labels=labels_data[i : i + 1])
            configure_axis(ax, "Aggregated frame stats", "", ytitles[i])

        ax.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        columns_data = {}
        labels_data = []
        for col in columns:
            columns_data[col] = []

        for file in data["file"].unique():
            _data = data[data["file"] == file]
            column_data = []
            for num, col in enumerate(columns):
                columns_data[col].append(_data[col])
            labels_data.append(file)

        for num, col in enumerate(columns):
            ax = None
            if switch_col_row:
                ax = fig.add_subplot(2, ncol, num + ncol + 1)
            else:
                ax = fig.add_subplot(ncol + 2, 1, num + 2)
            ax.boxplot(columns_data[col], labels=labels_data)
            configure_axis(ax, f"{col}", "", ytitles[num])
            if num < len(columns) - 1 and not switch_col_row:
                ax.set_xticklabels([])
                ax.set_xlabel(None)
        ax.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.xticks(rotation=30, ha="right", rotation_mode="anchor")
    else:
        ncol = len(columns)
        fig = plt.figure(figsize=(figw, figh))
        axes = []

        for num, col in enumerate(columns):
            if switch_col_row:
                axes.append(fig.add_subplot(1, ncol + 1, num + 1))
            else:
                axes.append(fig.add_subplot(ncol + 1, 1, num + 1))

        for file in data["file"].unique():
            _data = pd.DataFrame(data[data["file"] == file])

            for num, col in enumerate(columns):
                field = col
                if options.rolling:
                    _data[field] = _data[field].rolling(window=options.rolling).mean()
                ax = axes[num]
                ax.plot(
                    _data[xname],
                    _data[field],
                    label=f"{col}: " + file + f" {num} {field}",
                )
                plt.legend()
        ax = None
        for num, col in enumerate(columns):
            ax = axes[num]
            configure_axis(ax, col, xtitles[num], ytitles[num])
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, -0.4),
            ncol=2,
            fancybox=True,
            shadow=True,
            borderaxespad=0.0,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(title)
    if options.output:
        plt.savefig(options.output)
    if options.show:
        plt.show()


def plot_windowed_framestats(data, options):
    # | frame | frames | shown | drops | window

    # Verify that data contain correct columns
    if len([col for col in ["frames", "shown", "drops"] if col in data.columns]) == 0:
        print("Error: data does not contain correct columns")
        exit(0)

    frames_field = "frames"
    shown_field = "shown"
    drops_field = "drops"
    title = options.title if options.title else "Windowed frame stats"
    if options.rolling and not options.aggregate:
        title = f"{title} (rolling window: {options.rolling} frames)"

    plot_columns(
        data,
        "frame",
        [frames_field, shown_field, drops_field],
        ["Time (sec)"] * 3,
        ["Fps"] * 3,
        title,
        options,
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description="Plot media data"
    )
    parser.add_argument(
        "-i", "--input", type=str, help="File(s) to plot", required=True, nargs="+"
    )
    parser.add_argument("-o", "--output", type=str, help="Output file")
    parser.add_argument("--title", type=str, help="Title of the plot")
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        help=f"Type of the plot: \n"
        + "".join([f"\t{k} : {v}\n" for k, v in plots.items()]),
        required=False,
    )
    parser.add_argument("-s", "--show", action="store_true", help="Show the plot")
    parser.add_argument(
        "-r", "--rolling", type=int, default=0, help="Rolling window size"
    )
    parser.add_argument("--aggregate", action="store_true", help="Aggregate the data")
    parser.add_argument("--width", type=int, default=12, help="Width of the plot")
    parser.add_argument("--height", type=int, default=12, help="Height of the plot")

    args = parser.parse_args()
    _data = []

    for file in args.input:
        # Guess the type
        if args.type == None or len(args.type) == 0:
            # find the type
            for key, ending in extensions.items():
                if file.endswith(ending):
                    args.type = key
                    break
        if file[-3:] != "csv":
            # Assume this is the videofile and instead add default extensions
            if args.type != None and len(args.type) > 0:
                extension = extensions[args.type]
            file = f"{file}.{extension}"

        if file[-3:] == "csv" and os.path.exists(file):
            _tmp = pd.read_csv(file)
            _tmp["file"] = file
            _data.append(_tmp)

        data = pd.concat(_data)
    if args.type == "windowed-frame-stats":
        plot_windowed_framestats(data, args)
    elif args.type == "frame-duration-hist":
        plot_frame_duration_hist(data, args)
    elif args.type == "latencies":
        plot_latencies(data, args)
    elif args.type == "avsync":
        plot_av_sync(data, args)
    elif args.type == "video-latency":
        plot_video_latency(data, args)
    elif args.type == "audio-latency":
        plot_audio_latency(data, args)
    elif args.type == "measurement-quality":
        plot_measurement_quality(data, args)


if __name__ == "__main__":
    main()
