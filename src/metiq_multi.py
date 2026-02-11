#! /usr/bin/env python3


"""
This script is used to run multiple METIQ instances in parallel. It is a wrapper script that calls the metiq.py script
using common arguments.
"""


import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd
import metiq
import audio_parse
import media_parse
import media_analyze
import multiprocessing as mp
import video_tag_coordinates
import metiq_reader

VIDEO_ENDING = ".video.csv"


def combined_calculations_json(source_files, outfile):
    """Build aggregated JSON from per-file .results.json files."""
    import subprocess

    files_data = []
    all_avsync_values = []
    all_audio_latency_values = []
    all_video_latency_values = []
    all_quality_error_pcts = []

    for file in source_files:
        if file.endswith(VIDEO_ENDING):
            file = file[: -len(VIDEO_ENDING)]

        results_path = file + ".results.json"
        if not os.path.isfile(results_path):
            print(f"Warning: {results_path} not found, skipping")
            continue

        with open(results_path) as f:
            file_data = json.load(f)
        files_data.append(file_data)

        # Collect values for aggregated stats
        avsync = file_data.get("avsync", {})
        if "data" in avsync:
            for row in avsync["data"]:
                if "avsync_sec" in row and row["avsync_sec"] is not None:
                    all_avsync_values.append(row["avsync_sec"])

        audio = file_data.get("audio", {})
        if "latency" in audio and "data" in audio["latency"]:
            for row in audio["latency"]["data"]:
                if "audio_latency_sec" in row and row["audio_latency_sec"] is not None:
                    all_audio_latency_values.append(row["audio_latency_sec"])

        video = file_data.get("video", {})
        if "latency" in video and "data" in video["latency"]:
            for row in video["latency"]["data"]:
                if "video_latency_sec" in row and row["video_latency_sec"] is not None:
                    all_video_latency_values.append(row["video_latency_sec"])

        if "quality" in video and "data" in video["quality"]:
            for row in video["quality"]["data"]:
                if "video_frames_metiq_errors_percentage" in row:
                    all_quality_error_pcts.append(row["video_frames_metiq_errors_percentage"])

    if not files_data:
        return

    # Build aggregated stats
    aggregated = {}
    if all_avsync_values:
        aggregated["avsync_sec"] = compute_stats(all_avsync_values)
    if all_audio_latency_values:
        aggregated["audio_latency_sec"] = compute_stats(all_audio_latency_values)
    if all_video_latency_values:
        aggregated["video_latency_sec"] = compute_stats(all_video_latency_values)
    if all_quality_error_pcts:
        aggregated["quality"] = {
            "mean_error_pct": float(np.mean(all_quality_error_pcts)),
            "std": float(np.std(all_quality_error_pcts)),
            "min": float(np.min(all_quality_error_pcts)),
            "max": float(np.max(all_quality_error_pcts)),
        }

    # Build metiq info
    metiq_info = {}
    try:
        git_version = subprocess.check_output(
            ["/usr/bin/git", "describe", "HEAD"],
            cwd=os.path.dirname(__file__),
            text=True,
        ).strip()
        metiq_info["version"] = git_version
    except (subprocess.CalledProcessError, FileNotFoundError):
        metiq_info["version"] = "unknown"
    metiq_info["command"] = " ".join(sys.argv)

    result = {
        "metiq": metiq_info,
        "num_files": len(files_data),
        "files": files_data,
        "aggregated": aggregated,
    }

    results_path = f"{outfile}.results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)


def combined_calculations(options):
    # video latency and avsync latency share original frame
    # video latency and audio latency share timestamp
    source_files = options.infile_list
    outfile = options.output
    output_format = getattr(options, "output_format", "csv")

    if output_format == "json":
        combined_calculations_json(source_files, outfile)
        return

    all_audio_latency = pd.DataFrame()
    all_video_latency = pd.DataFrame()
    all_av_sync = pd.DataFrame()
    all_combined = pd.DataFrame()
    all_quality_stats = pd.DataFrame()
    all_frame_duration = pd.DataFrame()
    all_avsyncs = pd.DataFrame()
    all_windowed_stats = pd.DataFrame()

    for file in source_files:
        if file.endswith(VIDEO_ENDING):
            file = file[: -len(VIDEO_ENDING)]

        # This will be the root of the file name
        # Assuming default naming scheme
        audio_latency = pd.DataFrame()
        video_latency = pd.DataFrame()
        av_sync = pd.DataFrame()
        quality_stats = pd.DataFrame()
        frame_duration = pd.DataFrame()
        windowed_frame_stats = pd.DataFrame()

        if os.path.isfile(file + ".audio.latency.csv"):
            try:
                audio_latency = pd.read_csv(file + ".audio.latency.csv")
            except pd.errors.EmptyDataError:
                print("Empty audio latency file: " + file + ".audio.latency.csv")
                pass
        if os.path.isfile(file + ".video.latency.csv"):
            try:
                video_latency = pd.read_csv(file + ".video.latency.csv")
            except pd.errors.EmptyDataError:
                print("Empty video latency file: " + file + ".video.latency.csv")
                pass
        if os.path.isfile(file + ".avsync.csv"):
            try:
                av_sync = pd.read_csv(file + ".avsync.csv")
            except pd.errors.EmptyDataError:
                print("Empty avsync file: " + file + ".avsync.csv")
                pass
        if os.path.isfile(file + ".measurement.quality.csv"):
            try:
                quality_stats = pd.read_csv(file + ".measurement.quality.csv")
            except pd.errors.EmptyDataError:
                print("Empty quality stats file:" + file + ".measurement.quality.csv")
                pass
        if os.path.isfile(file + ".frame.duration.csv"):
            try:
                frame_duration = pd.read_csv(file + ".frame.duration.csv")
            except pd.errors.EmptyDataError:
                pass
        if os.path.isfile(file + ".windowed.stats.csv"):
            try:
                windowed_frame_stats = pd.read_csv(file + ".windowed.stats.csv")
            except pd.errors.EmptyDataError:
                pass

        combined = []
        # If all three latency measure exists
        if audio_latency is None or audio_latency.empty:
            print("Warning. No audio latency values calculated.")
        if video_latency is None or video_latency.empty:
            print("Warning. No video latency values calculated.")
        if av_sync is None or av_sync.empty:
            print("Warning. No av sync values calculated.")
        if not video_latency.empty and not audio_latency.empty and not av_sync.empty:
            frames = video_latency["original_frame"].values
            for frame in frames:
                video_latency_row = video_latency.loc[
                    video_latency["original_frame"] == frame
                ]
                video_latency_sec = video_latency_row["video_latency_sec"].values[0]
                timestamp = video_latency_row["timestamp"].values[0]
                audio_latency_row = audio_latency.loc[
                    audio_latency["timestamp1"] == timestamp
                ]

                if len(audio_latency_row) == 0:
                    continue
                audio_latency_sec = audio_latency_row["audio_latency_sec"].values[0]
                av_sync_sec_row = av_sync.loc[av_sync["original_frame"] == frame]
                if len(av_sync_sec_row) == 0:
                    continue

                av_sync_sec = av_sync_sec_row["avsync_sec"].values[0]
                combined.append(
                    [frame, audio_latency_sec, video_latency_sec, av_sync_sec]
                )

        combined = pd.DataFrame(
            combined,
            columns=[
                "frame_num",
                "audio_latency_sec",
                "video_latency_sec",
                "av_sync_sec",
            ],
        )

        if len(combined) > 0:
            path = f"{file}.latencies.csv"
            combined.to_csv(path, index=False)

        if not frame_duration.empty:
            frame_duration["file"] = filter
            all_frame_duration = pd.concat([all_frame_duration, frame_duration])

        if not quality_stats.empty:
            quality_stats["file"] = file
            all_quality_stats = pd.concat([all_quality_stats, quality_stats])

        if not windowed_frame_stats.empty:
            windowed_frame_stats["file"] = file
            all_windowed_stats = pd.concat([all_windowed_stats, windowed_frame_stats])

        # Maybe a combined avsync
        if not av_sync.empty:
            av_sync["file"] = file
            all_av_sync = pd.concat([all_av_sync, av_sync])

        # only create the combined stat file
        if len(combined) > 0:
            combined["file"] = file
            all_combined = pd.concat([all_combined, combined])
            if audio_latency is not None:
                audio_latency["file"] = file
                all_audio_latency = pd.concat([all_audio_latency, audio_latency])
            if video_latency is not None:
                video_latency["file"] = file
                all_video_latency = pd.concat([all_video_latency, video_latency])
            if av_sync is not None:
                av_sync["file"] = file
                all_av_sync = pd.concat([all_av_sync, av_sync])

    if len(source_files) > 0:
        aggregated_string = "Aggregated stats"
        per_file_string = " -- per file stats --"
        if options.stats:
            print("\n *** All stats **")
        if len(all_audio_latency) > 0:
            path = f"{outfile}.audio_latency.csv"
            all_audio_latency.to_csv(path, index=False)

            # Calc stats
            simple = (
                all_audio_latency[["file", "audio_latency_sec"]]
                .groupby("file")
                .agg(
                    [
                        "mean",
                        "std",
                        "min",
                        "max",
                        lambda x: np.percentile(x, q=50),
                        lambda x: np.percentile(x, q=90),
                    ]
                )
            )
            simple = simple.droplevel(0, axis=1)
            simple.columns = ["mean", "std", "min", "max", "p50", "p90"]
            path = f"{outfile}.audio_latency.stats.csv"
            simple.to_csv(path)
            if options.stats:
                mean = all_audio_latency["audio_latency_sec"].mean()
                std = all_audio_latency["audio_latency_sec"].std()
                min = all_audio_latency["audio_latency_sec"].min()
                max = all_audio_latency["audio_latency_sec"].max()
                p50 = all_audio_latency["audio_latency_sec"].quantile(0.5)
                p90 = all_audio_latency["audio_latency_sec"].quantile(0.9)

                # Print error stats
                descr = "\nAudio latency: "
                aggregated_string += f"{descr:<24} {mean:+.2f} std dev: {std:+.2f}, min/max: {min:+.2f}/{max:+.2f}"

                if len(source_files) > 1:
                    per_file_string += "\n* audio latency *"
                    for file in simple.index:
                        per_file_string += (
                            f"\n{file:<30} mean: {simple.loc[file]['mean']:+.3f}, std: {simple.loc[file]['std']:+.3f},"
                            f"p50: {simple.loc[file]['p50']:+.3f}, p90 {simple.loc[file]['p90']:+.3f},  min: {simple.loc[file]['min']:+.3f}, max: {simple.loc[file]['max']:+.3f}"
                        )

        if len(all_video_latency) > 0:
            path = f"{outfile}.video_latency.csv"
            all_video_latency.to_csv(path, index=False)

            # Calc stats
            simple = (
                all_video_latency[["file", "video_latency_sec"]]
                .groupby("file")
                .agg(
                    [
                        "mean",
                        "std",
                        "min",
                        "max",
                        lambda x: np.percentile(x, q=50),
                        lambda x: np.percentile(x, q=90),
                    ]
                )
            )
            simple = simple.droplevel(0, axis=1)
            simple.columns = ["mean", "std", "min", "max", "p50", "p90"]
            path = f"{outfile}.video_latency.stats.csv"
            simple.to_csv(path)
            if options.stats:
                mean = all_video_latency["video_latency_sec"].mean()
                std = all_video_latency["video_latency_sec"].std()
                min = all_video_latency["video_latency_sec"].min()
                max = all_video_latency["video_latency_sec"].max()
                p50 = all_video_latency["video_latency_sec"].quantile(0.5)
                p90 = all_video_latency["video_latency_sec"].quantile(0.9)
                descr = "\nVideo latency:: "
                aggregated_string += f"{descr:<24} {mean:+.2f} std dev: {std:+.2f}, p50: {p50:+.2f}, p90 {p90:+.2f} `min/max: {min:+.2f}/{max:+.2f}"

                if len(source_files) > 1:
                    per_file_string += "\n* Video latency *"
                    for file in simple.index:
                        per_file_string += (
                            f"\n{file:<30} mean: {simple.loc[file]['mean']:+.3f}, std: {simple.loc[file]['std']:+.3f},"
                            f"p50: {simple.loc[file]['p50']:+.3f}, p90 {simple.loc[file]['p90']:+.3f}, min: {simple.loc[file]['min']:+.3f}, max: {simple.loc[file]['max']:+.3f}"
                        )

        if len(all_av_sync) > 0:
            path = f"{outfile}.avsync.csv"
            all_av_sync.to_csv(path, index=False)

            # Calc stats and make an aggregated summary
            simple = (
                all_av_sync[["file", "avsync_sec"]]
                .groupby("file")
                .agg(
                    [
                        "mean",
                        "std",
                        "min",
                        "max",
                        lambda x: np.percentile(x, q=50),
                        lambda x: np.percentile(x, q=90),
                    ]
                )
            )
            simple = simple.droplevel(0, axis=1)
            simple.columns = ["mean", "std", "min", "max", "p50", "p90"]
            simple.to_csv(path)

            if options.stats:
                mean = all_av_sync["avsync_sec"].mean()
                std = all_av_sync["avsync_sec"].std()
                min = all_av_sync["avsync_sec"].min()
                max = all_av_sync["avsync_sec"].max()
                p50 = all_av_sync["avsync_sec"].quantile(0.5)
                p90 = all_av_sync["avsync_sec"].quantile(0.9)

                # Print error stats
                descr = "\nAudio/Video sync: "
                aggregated_string += f"{descr:<24} {mean:+.2f} std dev: {std:+.2f}, min/max: {min:+.2f}/{max:+.2f}"

                if len(source_files) > 1:
                    per_file_string += "\n* Av sync *"
                    for file in simple.index:
                        per_file_string += (
                            f"\n{file:<30} mean: {simple.loc[file]['mean']:+.3f}, std: {simple.loc[file]['std']:+.3f}"
                            f", p50: {simple.loc[file]['p50']:+.3f}, p90 {simple.loc[file]['p90']:+.3f}, min: {simple.loc[file]['min']:+.3f}, max: {simple.loc[file]['max']:+.3f}"
                        )

        if len(all_combined) > 0:
            path = f"{outfile}.latencies.csv"
            all_combined.to_csv(path, index=False)

            # Calc stats
            simple = (
                all_combined[
                    ["file", "audio_latency_sec", "video_latency_sec", "av_sync_sec"]
                ]
                .groupby("file")
                .agg(
                    [
                        "mean",
                        "std",
                        "min",
                        "max",
                        lambda x: np.percentile(x, q=50),
                        lambda x: np.percentile(x, q=90),
                    ]
                )
            )
            simple = simple.droplevel(0, axis=1)
            simple.columns = [
                "audio_latency_sec_mean",
                "audio_latency_sec_std",
                "audio_latency_sec_min",
                "audio_latency_sec_max",
                "audio_latency_sec_p50",
                "audio_latency_sec_p90",
                "video_latency_sec_mean",
                "video_latency_sec_std",
                "video_latency_sec_min",
                "video_latency_sec_max",
                "video_latency_sec_p50",
                "video_latency_sec_p90",
                "av_sync_sec_mean",
                "av_sync_sec_std",
                "av_sync_sec_min",
                "av_sync_sec_max",
                "av_sync_sec_p50",
                "av_sync_sec_p90",
            ]
            path = f"{outfile}.latencies.stats.csv"
            simple.to_csv(path)

        if len(all_windowed_stats) > 0:
            path = f"{outfile}.windowed.stats.data.csv"
            all_windowed_stats.to_csv(path, index=False)

            # Calc stats and make an aggregated summary
            fields = ["frames", "shown", "drops", "window"]
            all_data = pd.DataFrame()
            for field in fields:
                simple = (
                    all_windowed_stats[["file", field]]
                    .groupby("file")
                    .agg(["median", "mean", "std", "min", "max"])
                )
                simple = simple.droplevel(0, axis=1)
                simple["field"] = field
                all_data = pd.concat([all_data, simple])

            path = f"{outfile}.windowed.aggr.stats.csv"
            all_data.to_csv(path)

        if len(all_quality_stats) > 0:
            path = f"{outfile}.measurement.quality.csv"
            all_quality_stats.to_csv(path, index=False)

            if options.stats:
                mean = all_quality_stats["video_frames_metiq_errors_percentage"].mean()
                std = all_quality_stats["video_frames_metiq_errors_percentage"].std()
                min = all_quality_stats["video_frames_metiq_errors_percentage"].min()
                max = all_quality_stats["video_frames_metiq_errors_percentage"].max()
                descr = "\nMean parsing error: "
                aggregated_string += f"{descr:<24} {mean:+.2f} std dev: {std:+.2f}, min/max: {min:+.2f}/{max:+.2f}"

                if len(source_files) > 1:
                    per_file_string += "\n* Parsing quality *"
                    for file in all_quality_stats["file"].unique():
                        per_file_string += f"\n{file:<30} parsing error: {all_quality_stats[all_quality_stats['file'] == file]['video_frames_metiq_errors_percentage'].mean():+.3f} %"

        if len(all_frame_duration) > 0:
            path = f"{outfile}.frame_duration.csv"
            all_frame_duration.to_csv(path, index=False)

        if options.stats:
            print(aggregated_string)

            if len(source_files) > 1:
                print("-" * 20)
                print(per_file_string)


def get_options(argv):
    """Generic option parser.

    Args:
        argv: list containing arguments

    Returns:
        Namespace - An argparse.ArgumentParser-generated option object
    """
    # init parser
    parser = argparse.ArgumentParser(
        description="Run multiple METIQ instances in parallel"
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="count",
        dest="debug",
        default=0,
        help="Increase verbosity (use multiple times for more)",
    )
    parser.add_argument(
        "--quiet",
        action="store_const",
        dest="debug",
        const=-1,
        help="Zero verbosity",
    )
    parser.add_argument("infile_list", nargs="+", type=str, help="Input file(s)")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="all",
        help="Output file. This is the aggregated output file name. ",
    )
    parser.add_argument(
        "-pa",
        "--parse-audio",
        action="store_true",
        dest="parse_audio",
        help="Reparse audio",
    )
    parser.add_argument(
        "-pv",
        "--parse-video",
        action="store_true",
        dest="parse_video",
        help="Reparse video",
    )
    parser.add_argument(
        "-ao", "--audio-offset", type=float, default=0.0, help="Audio offset in seconds"
    )
    parser.add_argument(
        "--max-parallel",
        dest="max_parallel",
        type=int,
        default=1,
        help="Maximum number of parallel processes",
    )
    parser.add_argument(
        "--filter-all-echoes",
        dest="filter_all_echoes",
        action="store_true",
        help="Filter all echoes from the audio, essentially only do avsync analysis,",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print stats in the console.",
    )
    parser.add_argument(
        "--surpress-video-cleanup",
        action="store_false",
        dest="surpress_cleanup_video",
        help="Do not cleanup parsed values.",
    )
    parser.add_argument(
        "-z",
        "--z-filter",
        type=float,
        default=3.0,
        help="Z-filter threshold for audio",
        dest="z_filter",
    )
    parser.add_argument(
        "--min-match-threshold",
        type=float,
        default=metiq.default_values["min_match_threshold"],
        dest="min_match_threshold",
        help="Minimum audio correlation threshold",
    )
    parser.add_argument(
        "-bp",
        "--bandpass-filter",
        dest="bandpass_filter",
        action="store_true",
        default=audio_parse.default_values["bandpass_filter"],
        help="Gentle butterworth bandpass filter. Sometimes low correlation hits can improve. Before lowering correlation threshold try filtering.",
    )
    parser.add_argument(
        "--sharpen",
        action="store_true",
        help="Sharpen the image before calculating",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=1,
        help="Contrast value. Keep this value positive and less than 2 (most likely). It is a multiplication of the actual pixel values so for anything above 127 a contrast of 2 would clip.",
    )
    parser.add_argument(
        "--brightness",
        type=int,
        default=0,
        help="Brightness value. Keep this value between -255 and 255 for 8bit, probaly much less i.e. +/20. It is a simple addition to the pixel values.",
    )
    parser.add_argument(
        "--tag-manual",
        dest="tag_manual",
        action="store_true",
        help="Find tags manually",
    )
    parser.add_argument(
        "--output-format",
        dest="output_format",
        type=str,
        choices=["csv", "json"],
        default="csv",
        help="Output format: csv (default) or json",
    )
    parser.add_argument(
        "--video-reader",
        dest="video_reader",
        type=str,
        choices=list(metiq_reader.VIDEO_READERS.keys()),
        default=metiq_reader.DEFAULT_VIDEO_READER,
        help=f"Video reader to use. Available: {', '.join(metiq_reader.VIDEO_READERS.keys())}. Default: {metiq_reader.DEFAULT_VIDEO_READER}",
    )
    parser.add_argument(
        "--audio-reader",
        dest="audio_reader",
        type=str,
        choices=list(metiq_reader.AUDIO_READERS.keys()),
        default=metiq_reader.DEFAULT_AUDIO_READER,
        help=f"Audio reader to use. Available: {', '.join(metiq_reader.AUDIO_READERS.keys())}. Default: {metiq_reader.DEFAULT_AUDIO_READER}",
    )
    options = parser.parse_args()
    return options


def compute_stats(values):
    """Compute summary statistics for a numeric array."""
    return {
        "average": float(np.mean(values)),
        "stddev": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "size": len(values),
    }


def read_csv_as_dicts(path):
    """Read a CSV file and return list of row dicts, or None if missing/empty."""
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df.to_dict(orient="records")
    except (pd.errors.EmptyDataError, Exception):
        return None


def build_per_file_json(file, stats_json_path):
    """Merge stats JSON with per-file analysis CSVs into a single .results.json."""
    # Read the stats JSON as base
    if os.path.isfile(stats_json_path):
        with open(stats_json_path) as f:
            result = json.load(f)
    else:
        result = {}

    # Ensure top-level sections exist
    result.setdefault("avsync", {})
    result.setdefault("audio", {})
    result.setdefault("video", {})

    # Merge avsync data
    avsync_data = read_csv_as_dicts(file + ".avsync.csv")
    if avsync_data is not None:
        result["avsync"]["data"] = avsync_data

    # Merge audio latency data with stats
    audio_lat_data = read_csv_as_dicts(file + ".audio.latency.csv")
    if audio_lat_data is not None:
        latency_values = [r["audio_latency_sec"] for r in audio_lat_data if "audio_latency_sec" in r]
        result["audio"]["latency"] = {"data": audio_lat_data}
        if latency_values:
            result["audio"]["latency"]["stats"] = compute_stats(latency_values)

    # Merge video latency data with stats
    video_lat_data = read_csv_as_dicts(file + ".video.latency.csv")
    if video_lat_data is not None:
        latency_values = [r["video_latency_sec"] for r in video_lat_data if "video_latency_sec" in r]
        result["video"]["latency"] = {"data": video_lat_data}
        if latency_values:
            result["video"]["latency"]["stats"] = compute_stats(latency_values)

    # Merge measurement quality data
    quality_data = read_csv_as_dicts(file + ".measurement.quality.csv")
    if quality_data is not None:
        result["video"]["quality"] = {"data": quality_data}

    # Merge windowed stats data
    windowed_data = read_csv_as_dicts(file + ".windowed.stats.csv")
    if windowed_data is not None:
        result["video"]["windowed_stats"] = {"data": windowed_data}

    # Merge frame duration data
    frame_dur_data = read_csv_as_dicts(file + ".frame.duration.csv")
    if frame_dur_data is not None:
        result["video"]["frame_duration"] = {"data": frame_dur_data}

    # Merge video playout data
    playout_data = read_csv_as_dicts(file + ".video.playout.csv")
    if playout_data is not None:
        result["video"]["playout"] = {"data": playout_data}

    # Write the combined results JSON
    results_path = file + ".results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    # Delete intermediate files
    intermediate_files = [
        stats_json_path,
        file + ".avsync.csv",
        file + ".audio.latency.csv",
        file + ".video.latency.csv",
        file + ".measurement.quality.csv",
        file + ".windowed.stats.csv",
        file + ".frame.duration.csv",
        file + ".video.playout.csv",
    ]
    for path in intermediate_files:
        if os.path.isfile(path):
            os.remove(path)


def run_file(kwargs):
    file = kwargs.get("file", None)
    parse_audio = kwargs.get("parse_audio", False)
    parse_video = kwargs.get("parse_video", False)
    audio_offset = kwargs.get("audio_offset", 0.0)
    filter_all_echoes = kwargs.get("filter_all_echoes", False)
    cleanup_video = kwargs.get("cleanup_video", False)
    z_filter = kwargs.get("z_filter", 3.0)
    debug = kwargs.get("debug", 0)
    output_format = kwargs.get("output_format", "csv")
    video_reader = kwargs.get("video_reader", metiq_reader.DEFAULT_VIDEO_READER)
    audio_reader = kwargs.get("audio_reader", metiq_reader.DEFAULT_AUDIO_READER)
    video_reader_class = metiq_reader.VIDEO_READERS[video_reader]
    audio_reader_class = metiq_reader.AUDIO_READERS[audio_reader]
    min_match_threshold = kwargs.get(
        "min_match_threshold", metiq.default_values["min_match_threshold"]
    )
    sharpen = kwargs.get("sharpen", False)
    contrast = kwargs.get("contrast", 1)
    brightness = kwargs.get("brightness", 0)
    # We assume default settings on/ everything.
    # TODO(johan): expose more settings to the user
    width = metiq.default_values["width"]
    height = metiq.default_values["height"]
    pre_samples = metiq.default_values["pre_samples"]
    samplerate = metiq.default_values["samplerate"]
    beep_freq = metiq.default_values["beep_freq"]
    beep_period_sec = metiq.default_values["beep_period_sec"]
    beep_duration_samples = metiq.default_values["beep_duration_samples"]
    bandpass_filter = kwargs.get(
        "bandpass_filter", audio_parse.default_values["bandpass_filter"]
    )
    scale = metiq.default_values["scale"]
    pixel_format = metiq.default_values["pixel_format"]
    luma_threshold = metiq.default_values["luma_threshold"]
    num_frames = -1
    kwargs = {
        "lock_layout": True,
        "threaded": False,
        "tag_manual": kwargs.get("tag_manual"),
        "video_reader_class": video_reader_class,
        "audio_reader_class": audio_reader_class,
    }

    min_separation_msec = metiq.default_values["min_separation_msec"]
    audio_sample = metiq.default_values["audio_sample"]
    vft_id = metiq.default_values["vft_id"]

    # TODO(johan): remove
    force_fps = 30
    windowed_stats_sec = metiq.default_values["windowed_stats_sec"]
    analysis_type = "all"

    videocsv = file + VIDEO_ENDING
    audiocsv = file + ".audio.csv"
    # Allow us to run a reanalysis of a fiel without reprocessing the video
    if not file.endswith(VIDEO_ENDING):
        # files exist
        audio_result = None
        if not os.path.exists(audiocsv) or parse_audio:
            # 1. parse the audio stream
            audio_result = media_parse.media_parse_audio(
                pre_samples,
                samplerate,
                beep_freq,
                beep_duration_samples,
                beep_period_sec,
                scale,
                file,
                audiocsv,
                bandpass_filter=bandpass_filter,
                min_match_threshold=min_match_threshold,
                debug=debug,
                **kwargs,
            )

            if audio_result is None:
                print("Audio parsing failed. Quitting.")
                return

        if not os.path.exists(videocsv) or parse_video:
            # 2. parse the video stream
            media_parse.media_parse_video(
                width,
                height,
                num_frames,
                pixel_format,
                luma_threshold,
                pre_samples,
                samplerate,
                beep_freq,
                beep_duration_samples,
                beep_period_sec,
                scale,
                file,
                videocsv,
                sharpen=sharpen,
                contrast=contrast,
                brightness=brightness,
                debug=debug,
                **kwargs,
            )
    else:
        videocsv = file
        audiocsv = file[: -len(VIDEO_ENDING)] + ".audio.csv"

    if not os.path.exists(audiocsv) or not os.path.exists(videocsv):
        print(f"Error: {audiocsv} or {videocsv} does not exist")
        return None

    # Analyze the video and audio files
    stats_json_path = file + ".stats.json" if output_format == "json" else None
    try:
        media_analyze.media_analyze(
            analysis_type,
            pre_samples,
            samplerate,
            beep_freq,
            beep_duration_samples,
            beep_period_sec,
            videocsv,
            audiocsv,
            None,  # options.output,
            stats_json_path,
            force_fps,
            audio_offset,
            z_filter=z_filter,
            windowed_stats_sec=windowed_stats_sec,
            filter_all_echoes=filter_all_echoes,
            cleanup_video=cleanup_video,
            min_match_threshold=min_match_threshold,
            debug=debug,
        )
    except Exception as e:
        print(f"Error: {e}")
        return None

    if output_format == "json":
        build_per_file_json(file, stats_json_path)


def main(argv):
    # parse options
    options = get_options(argv)
    parse_video = options.parse_video
    parse_audio = options.parse_audio
    audio_offset = options.audio_offset

    # TODO(johan): Add more options
    kwargs_list = [
        (
            {
                "file": infile,
                "parse_audio": parse_audio,
                "parse_video": parse_video,
                "audio_offset": audio_offset,
                "filter_all_echoes": options.filter_all_echoes,
                "cleanup_video": not options.surpress_cleanup_video,
                "z_filter": options.z_filter,
                "min_match_threshold": options.min_match_threshold,
                "bandpass_filter": options.bandpass_filter,
                "sharpen": options.sharpen,
                "contrast": options.contrast,
                "brightness": options.brightness,
                "tag_manual": options.tag_manual,
                "output_format": options.output_format,
                "video_reader": options.video_reader,
                "audio_reader": options.audio_reader,
                "debug": options.debug,
            }
        )
        for infile in options.infile_list
    ]

    start_index = 0
    video_tag_coordinates.use_cache()
    video_tag_coordinates.clear_cache()
    if options.tag_manual:
        # Run the first file manually, save the parsed positions and use that for the subsequent ones.
        # remove .tag_freeze if existing (aborted previous run?).
        if os.path.exists(".tag_freeze"):
            os.remove(".tag_freeze")
        start_index = 1
        results = run_file(kwargs_list[0])

    if options.max_parallel == 0:
        # do not use multiprocessing
        for kwargs in kwargs_list[start_index:]:
            results = run_file(kwargs)
    else:
        with mp.Pool(processes=options.max_parallel) as p:
            results = p.map(run_file, kwargs_list[start_index:], chunksize=1)

    combined_calculations(options)
    video_tag_coordinates.clear_cache()


if __name__ == "__main__":
    main(sys.argv)
