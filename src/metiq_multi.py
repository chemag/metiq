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
import media_parse
import media_analyze


def combined_calculations(source_files, outfile):
    # video latency and avsync latency share original frame
    # video latency and audio latency share timestamp

    all_audio_latency = pd.DataFrame()
    all_video_latency = pd.DataFrame()
    all_av_sync = pd.DataFrame()
    all_combined = pd.DataFrame()
    all_quality_stats = pd.DataFrame()
    all_frame_duration = pd.DataFrame()

    for file in source_files:
        # This will be the root of the file name
        # Assuming default naming scheme
        audio_latency = pd.read_csv(file + ".audio.latency.csv")
        video_latency = pd.read_csv(file + ".video.latency.csv")
        av_sync = pd.read_csv(file + ".avsync.csv")
        quality_stats = pd.read_csv(file + ".measurement.quality.csv")
        frame_duration = pd.read_csv(file + ".frame.duration.csv")

        combined = []
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
            combined.append([frame, audio_latency_sec, video_latency_sec, av_sync_sec])

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

        frame_duration["file"] = file

        all_frame_duration = pd.concat([all_frame_duration, frame_duration])

        quality_stats["file"] = file
        all_quality_stats = pd.concat([all_quality_stats, quality_stats])

        # only create the combined stat file
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

    if len(source_files) > 1:
        if len(all_audio_latency) > 0:
            path = f"all.audio_latency.csv"
            if outfile is not None and len(outfile) > 0:
                path = f"{outfile}.all.audio_latency.csv"
            all_audio_latency.to_csv(path, index=False)

        if len(all_video_latency) > 0:
            path = f"all.video_latency.csv"
            if outfile is not None and len(outfile) > 0:
                path = f"{outfile}.all.video_latency.csv"
            all_video_latency.to_csv(path, index=False)

        if len(all_av_sync) > 0:
            path = f"all.avsync.csv"
            if outfile is not None and len(outfile) > 0:
                path = f"{outfile}.all.avsync.csv"
            all_av_sync.to_csv(path, index=False)

        if len(all_combined) > 0:
            path = f"all.combined.csv"
            if outfile is not None and len(outfile) > 0:
                path = f"{outfile}.all.latencies.csv"
            all_combined.to_csv(path, index=False)

        if len(all_quality_stats) > 0:
            path = f"all.quality_stats.csv"
            if outfile is not None and len(outfile) > 0:
                path = f"{outfile}.all.measurement.quality.csv"
            all_quality_stats.to_csv(path, index=False)

        if len(all_frame_duration) > 0:
            path = f"all.frame_duration.csv"
            if outfile is not None and len(outfile) > 0:
                path = f"{outfile}.all.frame_duration.csv"
            all_frame_duration.to_csv(path, index=False)


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
    options = parser.parse_args()
    return options


def main(argv):
    # parse options
    options = get_options(argv)

    # We assume default settings on everything.
    # TODO(johan): expose more settings to the user
    width = metiq.default_values["width"]
    height = metiq.default_values["height"]
    pre_samples = metiq.default_values["pre_samples"]
    samplerate = metiq.default_values["samplerate"]
    beep_freq = metiq.default_values["beep_freq"]
    beep_period_sec = metiq.default_values["beep_period_sec"]
    beep_duration_samples = metiq.default_values["beep_duration_samples"]
    scale = metiq.default_values["scale"]
    pixel_format = metiq.default_values["pixel_format"]
    luma_threshold = metiq.default_values["luma_threshold"]
    num_frames = -1
    kwargs = {"lock_layout": True, "threaded": True}

    min_match_threshold = metiq.default_values["min_match_threshold"]
    min_separation_msec = metiq.default_values["min_separation_msec"]
    audio_sample = metiq.default_values["audio_sample"]
    vft_id = metiq.default_values["vft_id"]
    force_fps = 30  # TODO: remove
    z_filter = 3
    windowed_stats_sec = metiq.default_values["windowed_stats_sec"]
    analysis_type = "all"

    debug = 0
    for file in options.infile_list:
        videocsv = file + ".video.csv"
        audiocsv = file + ".audio.csv"

        # files exist
        if not os.path.exists(audiocsv) or options.parse_audio:
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
                debug,
                **kwargs,
            )

        if not os.path.exists(videocsv) or options.parse_video:
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
        # Analyze the video and audio files
        for file in options.infile_list:
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
                options.audio_offset,
                z_filter,
                windowed_stats_sec,
                debug,
            )
    combined_calculations(options.infile_list, options.output)


if __name__ == "__main__":
    main(sys.argv)
