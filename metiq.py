#!/usr/bin/env python3

"""tool.py module description."""


import argparse
import math
import numpy as np
import os
import sys
import tempfile

import audio_common
import audio_generate
import audio_analyze
import video_common
import video_generate
import video_analyze
import common


from _version import __version__

FUNC_CHOICES = {
    "help": "show help options",
    "generate": "generate reference video",
    "analyze": "analyze distorted video",
}

DEFAULT_NUM_FRAMES = 1800

default_values = {
    "debug": 0,
    # video parameters
    "num_frames": DEFAULT_NUM_FRAMES,
    "fps": video_common.DEFAULT_FPS,
    "width": video_common.DEFAULT_WIDTH,
    "height": video_common.DEFAULT_HEIGHT,
    "luma_threshold": video_common.DEFAULT_LUMA_THRESHOLD,
    "pixel_format": video_common.DEFAULT_PIXEL_FORMAT,
    # audio parameters
    "pre_samples": audio_common.DEFAULT_PRE_SAMPLES,
    "samplerate": audio_common.DEFAULT_SAMPLERATE,
    "beep_freq": audio_common.DEFAULT_BEEP_FREQ,
    "beep_duration_samples": audio_common.DEFAULT_BEEP_DURATION_SAMPLES,
    "beep_period_sec": audio_common.DEFAULT_BEEP_PERIOD_SEC,
    "scale": audio_common.DEFAULT_SCALE,
    # common parameters
    "func": "help",
    "infile": None,
    "outfile": None,
}


def media_file_generate(
    width,
    height,
    fps,
    num_frames,
    pre_samples,
    samplerate,
    beep_freq,
    beep_duration_samples,
    beep_period_sec,
    scale,
    outfile,
    debug,
):
    # generate the (raw) video input
    video_filename = tempfile.NamedTemporaryFile().name + ".rgb24"
    rem = f"period: {beep_period_sec} freq_hz: {beep_freq} samples: {beep_duration_samples}"
    video_generate.video_generate(
        width, height, fps, num_frames, video_filename, "default", rem, debug
    )
    duration_sec = num_frames / fps
    # generate the (raw) audio input
    audio_filename = tempfile.NamedTemporaryFile().name + ".wav"
    audio_generate.audio_generate(
        duration_sec,
        audio_filename,
        pre_samples=pre_samples,
        samplerate=samplerate,
        beep_freq=beep_freq,
        beep_duration_samples=beep_duration_samples,
        beep_period_sec=beep_period_sec,
        scale=scale,
        debug=debug,
    )
    # put them together
    command = "ffmpeg -y "
    command += f"-f rawvideo -pixel_format rgb24 -s {width}x{height} -r {fps} -i {video_filename} "
    command += f"-i {audio_filename} "
    command += f"-c:v libx264 -pix_fmt yuv420p -c:a aac {outfile}"
    ret, stdout, stderr = common.run(command, debug=debug)
    assert ret == 0, f"error: {stderr}"
    # clean up raw files
    os.remove(video_filename)
    os.remove(audio_filename)


def estimate_audio_frame_num(
    prev_video_frame_num, video_frame_num, prev_video_ts, video_ts, audio_ts
):
    audio_frame_num = prev_video_frame_num + (
        video_frame_num - prev_video_frame_num
    ) * ((audio_ts - prev_video_ts) / (video_ts - prev_video_ts))
    return audio_frame_num


def estimate_avsync(video_results, fps, audio_results, beep_period_sec):
    video_index = 0
    # get the video frame_num corresponding to each audio timestamp
    audio_frame_num_list = []
    for _audio_sample, audio_ts, _audio_correlation in audio_results:
        if video_index >= len(video_results):
            break
        # find the video matches whose timestamps surround the audio one
        prev_video_ts, prev_video_frame_num = video_results[video_index][2:]
        video_index += 1
        get_next_audio_ts = False
        while not get_next_audio_ts and video_index < len(video_results):
            video_ts, video_frame_num = video_results[video_index][2:]
            if video_frame_num is None:
                video_index += 1
                continue
            if audio_ts >= prev_video_ts and audio_ts <= video_ts:
                # found a match
                audio_frame_num = estimate_audio_frame_num(
                    prev_video_frame_num,
                    video_frame_num,
                    prev_video_ts,
                    video_ts,
                    audio_ts,
                )
                audio_frame_num_list.append(audio_frame_num)
                get_next_audio_ts = True
            elif audio_ts < prev_video_ts:
                # could not find the audio timestamp
                get_next_audio_ts = True
            prev_video_ts, prev_video_frame_num = video_ts, video_frame_num
            # linearize the video frame number
            video_index += 1
    # get the closest video_frame_num
    video_frame_num_period = fps * beep_period_sec
    avsync_sec_list = []
    for audio_frame_num in audio_frame_num_list:
        candidate_1 = (
            math.floor(audio_frame_num / video_frame_num_period)
            * video_frame_num_period
        )
        candidate_2 = (
            math.ceil(audio_frame_num / video_frame_num_period) * video_frame_num_period
        )
        if (audio_frame_num - candidate_1) < (candidate_2 - audio_frame_num):
            video_frame_num = candidate_1
        else:
            video_frame_num = candidate_2
        # Following ITU-R BT.1359-1 rec here: "positive value indicates that
        # sound is advanced with respect to vision"
        avsync_sec = (video_frame_num - audio_frame_num) / fps
        avsync_sec_list.append(avsync_sec)
    return avsync_sec_list


def media_file_analyze(
    width,
    height,
    fps,
    num_frames,
    pixel_format,
    luma_threshold,
    pre_samples,
    samplerate,
    beep_freq,
    beep_duration_samples,
    beep_period_sec,
    scale,
    infile,
    outfile,
    debug,
):
    # analyze the video stream
    video_results = video_analyze.video_analyze(
        infile,
        width,
        height,
        fps,
        pixel_format,
        luma_threshold,
        debug,
    )
    # 1. analyze the video smoothness
    video_delta = video_analyze.get_video_delta(video_results)
    # 2. analyze the audio stream
    audio_results = audio_analyze.audio_analyze(
        infile,
        pre_samples=pre_samples,
        samplerate=samplerate,
        beep_freq=beep_freq,
        beep_duration_samples=beep_duration_samples,
        beep_period_sec=beep_period_sec,
        scale=scale,
    )
    # 2. estimate a/v sync
    avsync_sec_list = estimate_avsync(
        video_results, fps, audio_results, beep_period_sec
    )
    return video_delta, avsync_sec_list


def get_options(argv):
    """Generic option parser.

    Args:
        argv: list containing arguments

    Returns:
        Namespace - An argparse.ArgumentParser-generated option object
    """
    # init parser
    # usage = 'usage: %prog [options] arg1 arg2'
    # parser = argparse.OptionParser(usage=usage)
    # parser.print_help() to get argparse.usage (large help)
    # parser.print_usage() to get argparse.usage (just usage line)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        dest="version",
        default=False,
        help="Print version",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="count",
        dest="debug",
        default=default_values["debug"],
        help="Increase verbosity (use multiple times for more)",
    )
    parser.add_argument(
        "--quiet",
        action="store_const",
        dest="debug",
        const=-1,
        help="Zero verbosity",
    )
    # 2-parameter setter using argparse.Action
    parser.add_argument(
        "--width",
        action="store",
        type=int,
        dest="width",
        default=default_values["width"],
        metavar="WIDTH",
        help=("use WIDTH width (default: %i)" % default_values["width"]),
    )
    parser.add_argument(
        "--height",
        action="store",
        type=int,
        dest="height",
        default=default_values["height"],
        metavar="HEIGHT",
        help=("HEIGHT height (default: %i)" % default_values["height"]),
    )

    class VideoSizeAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            namespace.width, namespace.height = [int(v) for v in values[0].split("x")]

    parser.add_argument(
        "--video-size",
        action=VideoSizeAction,
        nargs=1,
        help="use <width>x<height>",
    )
    parser.add_argument(
        "--fps",
        action="store",
        type=int,
        dest="fps",
        default=default_values["fps"],
        metavar="FPS",
        help=("use FPS fps (default: %i)" % default_values["fps"]),
    )
    parser.add_argument(
        "--num-frames",
        action="store",
        type=int,
        dest="num_frames",
        default=default_values["num_frames"],
        metavar="NUM_FRAMES",
        help=("use NUM_FRAMES frames (default: %i)" % default_values["num_frames"]),
    )
    parser.add_argument(
        "--pixel-format",
        action="store",
        type=str,
        dest="pixel_format",
        default=default_values["pixel_format"],
        choices=video_common.PIXEL_FORMAT_CHOICES,
        metavar="[%s]"
        % (
            " | ".join(
                video_common.PIXEL_FORMAT_CHOICES,
            )
        ),
        help="pixel format",
    )
    parser.add_argument(
        "--luma-threshold",
        action="store",
        type=int,
        dest="luma_threshold",
        default=default_values["luma_threshold"],
        metavar="LUMA_THRESHOLD",
        help=(
            "detection luma_threshold (default: %i)" % default_values["luma_threshold"]
        ),
    )

    parser.add_argument(
        "--pre-samples",
        action="store",
        type=int,
        dest="pre_samples",
        default=default_values["pre_samples"],
        metavar="pre_samples",
        help=("use pre_samples (default: %i)" % default_values["pre_samples"]),
    )
    parser.add_argument(
        "--samplerate",
        action="store",
        type=int,
        dest="samplerate",
        default=default_values["samplerate"],
        metavar="samplerate",
        help=("use samplerate Hz (default: %i)" % default_values["samplerate"]),
    )
    parser.add_argument(
        "--beep-freq",
        action="store",
        type=int,
        dest="beep_freq",
        default=default_values["beep_freq"],
        metavar="beep_freq",
        help=("use beep_freq Hz (default: %i)" % default_values["beep_freq"]),
    )
    parser.add_argument(
        "--beep-duration-samples",
        action="store",
        type=int,
        dest="beep_duration_samples",
        default=default_values["beep_duration_samples"],
        metavar="beep_duration_samples",
        help=(
            "use beep_duration_samples (default: %i)"
            % default_values["beep_duration_samples"]
        ),
    )
    parser.add_argument(
        "--beep-period-sec",
        action="store",
        type=float,
        dest="beep_period_sec",
        default=default_values["beep_period_sec"],
        metavar="beep_period_sec",
        help=("use beep_period_sec (default: %i)" % default_values["beep_period_sec"]),
    )
    parser.add_argument(
        "--scale",
        action="store",
        type=float,
        dest="scale",
        default=default_values["scale"],
        metavar="scale",
        help=("use scale [0-1] (default: %i)" % default_values["scale"]),
    )

    parser.add_argument(
        "func",
        type=str,
        nargs="?",
        default=default_values["func"],
        choices=FUNC_CHOICES.keys(),
        help="%s"
        % (" | ".join("{}: {}".format(k, v) for k, v in FUNC_CHOICES.items())),
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        dest="infile",
        default=default_values["infile"],
        metavar="input-file",
        help="input file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        dest="outfile",
        default=default_values["outfile"],
        metavar="output-file",
        help="output file",
    )
    # do the parsing
    options = parser.parse_args(argv[1:])
    if options.version:
        return options
    # implement help
    if options.func == "help":
        parser.print_help()
        sys.exit(0)
    return options


def main(argv):
    # parse options
    options = get_options(argv)
    if options.version:
        print("version: %s" % __version__)
        sys.exit(0)

    # print results
    if options.debug > 0:
        print(options)

    # do something
    if options.func == "generate":
        # get outfile
        if options.outfile is None or options.outfile == "-":
            options.outfile = "/dev/fd/1"
        # do something
        media_file_generate(
            options.width,
            options.height,
            options.fps,
            options.num_frames,
            options.pre_samples,
            options.samplerate,
            options.beep_freq,
            options.beep_duration_samples,
            options.beep_period_sec,
            options.scale,
            options.outfile,
            options.debug,
        )

    elif options.func == "analyze":
        # get infile
        if options.infile is None or options.infile == "-":
            options.infile = "/dev/fd/0"
        # get outfile
        if options.outfile is None or options.outfile == "-":
            options.outfile = "/dev/fd/1"
        video_delta, avsync_sec_list = media_file_analyze(
            options.width,
            options.height,
            options.fps,
            options.num_frames,
            options.pixel_format,
            options.luma_threshold,
            options.pre_samples,
            options.samplerate,
            options.beep_freq,
            options.beep_duration_samples,
            options.beep_period_sec,
            options.scale,
            options.infile,
            options.outfile,
            options.debug,
        )
        if options.debug > 0:
            print(f"{avsync_sec_list = }")
        # print statistics
        print(f"{video_delta = }")
        avsync_sec_average = np.average(avsync_sec_list)
        avsync_sec_stddev = np.std(avsync_sec_list)
        print(
            f"avsync_sec average: {avsync_sec_average} stddev: {avsync_sec_stddev} size: {len(avsync_sec_list)}"
        )


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
