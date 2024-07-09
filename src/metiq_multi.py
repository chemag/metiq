#! /usr/bin/env python3


"""
This script is used to run multiple METIQ instances in parallel. It is a wrapper script that calls the metiq.py script
using common arguments.
"""


import os
import sys
import argparse
import time
import pandas as pd
import metiq
import audio_parse
import media_parse
import media_analyze
import multiprocessing as mp

VIDEO_ENDING = ".video.csv"


def combined_calculations(options):
    # video latency and avsync latency share original frame
    # video latency and audio latency share timestamp
    source_files = options.infile_list
    outfile = options.output

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

                av_sync_sec = av_sync_sec_row["av_sync_sec"].values[0]
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
                .agg(["mean", "std", "min", "max"])
            )
            simple = simple.droplevel(0, axis=1)
            path = f"{outfile}.audio_latency.stats.csv"
            simple.to_csv(path)
            if options.stats:
                mean = all_audio_latency["audio_latency_sec"].mean()
                std = all_audio_latency["audio_latency_sec"].std()
                min = all_audio_latency["audio_latency_sec"].min()
                max = all_audio_latency["audio_latency_sec"].max()
                # Print error stats
                descr = "\nAudio latency: "
                aggregated_string += f"{descr:<24} {mean:+.2f} std dev: {std:+.2f}, min/max: {min:+.2f}/{max:+.2f}"

                if len(source_files) > 1:
                    per_file_string += "\n* audio latency *"
                    for file in simple.index:
                        per_file_string += f"\n{file:<30} av_sync mean: {simple.loc[file]['mean']:+.3f}, std: {simple.loc[file]['std']:+.3f}, min: {simple.loc[file]['min']:+.3f}, max: {simple.loc[file]['max']:+.3f}"

        if len(all_video_latency) > 0:
            path = f"{outfile}.video_latency.csv"
            all_video_latency.to_csv(path, index=False)

            # Calc stats
            simple = (
                all_video_latency[["file", "video_latency_sec"]]
                .groupby("file")
                .agg(["mean", "std", "min", "max"])
            )
            simple = simple.droplevel(0, axis=1)
            path = f"{outfile}.video_latency.stats.csv"
            simple.to_csv(path)
            if options.stats:
                mean = all_video_latency["video_latency_sec"].mean()
                std = all_video_latency["video_latency_sec"].std()
                min = all_video_latency["video_latency_sec"].min()
                max = all_video_latency["video_latency_sec"].max()
                descr = "\nVideo latency:: "
                aggregated_string += f"{descr:<24} {mean:+.2f} std dev: {std:+.2f}, min/max: {min:+.2f}/{max:+.2f}"

                if len(source_files) > 1:
                    per_file_string += "\n* Video latency *"
                    for file in simple.index:
                        per_file_string += f"\n{file:<30} av_sync mean: {simple.loc[file]['mean']:+.3f}, std: {simple.loc[file]['std']:+.3f}, min: {simple.loc[file]['min']:+.3f}, max: {simple.loc[file]['max']:+.3f}"

        if len(all_av_sync) > 0:
            path = f"{outfile}.avsync.csv"
            all_av_sync.to_csv(path, index=False)

            # Calc stats and make an aggregated summary
            simple = (
                all_av_sync[["file", "av_sync_sec"]]
                .groupby("file")
                .agg(["mean", "std", "min", "max"])
            )
            simple = simple.droplevel(0, axis=1)
            simple.to_csv(path)

            if options.stats:
                mean = all_av_sync["av_sync_sec"].mean()
                std = all_av_sync["av_sync_sec"].std()
                min = all_av_sync["av_sync_sec"].min()
                max = all_av_sync["av_sync_sec"].max()
                # Print error stats
                descr = "\nAudio/Video sync: "
                aggregated_string += f"{descr:<24} {mean:+.2f} std dev: {std:+.2f}, min/max: {min:+.2f}/{max:+.2f}"

                if len(source_files) > 1:
                    per_file_string += "\n* Av sync *"
                    for file in simple.index:
                        per_file_string += f"\n{file:<30} av_sync mean: {simple.loc[file]['mean']:+.3f}, std: {simple.loc[file]['std']:+.3f}, min: {simple.loc[file]['min']:+.3f}, max: {simple.loc[file]['max']:+.3f}"

        if len(all_combined) > 0:
            path = f"{outfile}.latencies.csv"
            all_combined.to_csv(path, index=False)

            # Calc stats
            simple = (
                all_combined[
                    ["file", "audio_latency_sec", "video_latency_sec", "av_sync_sec"]
                ]
                .groupby("file")
                .agg(["mean", "std", "min", "max"])
            )
            simple = simple.droplevel(0, axis=1)
            simple.columns = [
                "audio_latency_sec_mean",
                "audio_latency_sec_std",
                "audio_latency_sec_min",
                "audio_latency_sec_max",
                "video_latency_sec_mean",
                "video_latency_sec_std",
                "video_latency_sec_min",
                "video_latency_sec_max",
                "av_sync_sec_mean",
                "av_sync_sec_std",
                "av_sync_sec_min",
                "av_sync_sec_max",
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
                        per_file_string += f"\n{file:<30} parsing error: {all_quality_stats[all_quality_stats['file'] == file]['video_frames_metiq_errors_percentage'].mean():+.3f}"

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
    options = parser.parse_args()
    return options


def run_file(kwargs):
    file = kwargs.get("file", None)
    parse_audio = kwargs.get("parse_audio", False)
    parse_video = kwargs.get("parse_video", False)
    audio_offset = kwargs.get("audio_offset", 0.0)
    filter_all_echoes = kwargs.get("filter_all_echoes", False)
    cleanup_video = kwargs.get("cleanup_video", False)
    z_filter = kwargs.get("z_filter", 3.0)
    debug = kwargs.get("debug", 0)
    min_match_threshold = kwargs.get(
        "min_match_threshold", metiq.default_values["min_match_threshold"]
    )
    print(f"{min_match_threshold=}")
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
    kwargs = {"lock_layout": True, "threaded": False}

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
        if not os.path.exists(audiocsv) or parse_audio:
            # 1. parse the audio stream
            media_parse.media_parse_audio(
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
                debug,
                **kwargs,
            )
    else:
        videocsv = file
        audiocsv = file[: -len(VIDEO_ENDING)] + ".audio.csv"

    if not os.path.exists(audiocsv) or not os.path.exists(videocsv):
        print(f"Error: {audiocsv} or {videocsv} does not exist")
        return None

    # Analyze the video and audio files
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
                "debug": options.debug,
            }
        )
        for infile in options.infile_list
    ]
    if options.max_parallel == 0:
        # do not use multiprocessing
        for kwargs in kwargs_list:
            results = run_file(kwargs)
    else:
        with mp.Pool(processes=options.max_parallel) as p:
            results = p.map(run_file, kwargs_list, chunksize=1)

    combined_calculations(options)


if __name__ == "__main__":
    main(sys.argv)
