#!/usr/bin/env python

"""metiq.py module description."""


import argparse
import sys

import audio_common
import audio_parse
import common
import media_analyze
import media_generate
import media_parse
import vft
import video_common

from _version import __version__

FUNC_CHOICES = {
    "help": "show help options",
    "generate": "generate reference video",
    "parse": "parse distorted video",
    "analyze": "analyze distorted video",
}


default_values = {
    "debug": common.DEFAULT_DEBUG,
    # vft parameters
    "vft_id": vft.DEFAULT_VFT_ID,
    "vft_tag_border_size": vft.DEFAULT_TAG_BORDER_SIZE,
    "luma_threshold": vft.DEFAULT_LUMA_THRESHOLD,
    # video parameters
    "num_frames": video_common.DEFAULT_NUM_FRAMES,
    "fps": video_common.DEFAULT_FPS,
    "width": video_common.DEFAULT_WIDTH,
    "height": video_common.DEFAULT_HEIGHT,
    "pixel_format": video_common.DEFAULT_PIXEL_FORMAT,
    # audio parameters
    "pre_samples": audio_common.DEFAULT_PRE_SAMPLES,
    "samplerate": audio_common.DEFAULT_SAMPLERATE,
    "beep_freq": audio_common.DEFAULT_BEEP_FREQ,
    "beep_duration_samples": audio_common.DEFAULT_BEEP_DURATION_SAMPLES,
    "beep_period_sec": audio_common.DEFAULT_BEEP_PERIOD_SEC,
    "scale": audio_common.DEFAULT_SCALE,
    "min_separation_msec": audio_parse.DEFAULT_MIN_SEPARATION_MSEC,
    "min_match_threshold": audio_parse.DEFAULT_MIN_MATCH_THRESHOLD,
    # common parameters
    "func": "help",
    "infile": None,
    "input_audio": None,
    "input_video": None,
    "outfile": None,
    "output_audio": None,
    "output_video": None,
    "audio_offset": 0,
    "lock_layout": False,
    "audio_sample": audio_common.DEFAULT_AUDIO_SAMPLE,
}


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
        "--force-fps",
        action="store",
        type=int,
        dest="force_fps",
        default=-1,
        metavar="ForceFPS",
        help=(
            "If the auto detection mechanism failes, this value can be used to override te measured value."
        ),
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
        "--audio-sample",
        type=str,
        dest="audio_sample",
        default=default_values["audio_sample"],
        help="use a sample as source for signal",
    )
    parser.add_argument(
        "--pixel-format",
        action="store",
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
        "--vft-id",
        type=str,
        nargs="?",
        default=default_values["vft_id"],
        choices=vft.VFT_IDS,
        help="%s (default: %s)"
        % (" | ".join("{}".format(k) for k in vft.VFT_IDS), default_values["vft_id"]),
    )
    parser.add_argument(
        "--vft-tag-border-size",
        action="store",
        type=int,
        dest="vft_tag_border_size",
        default=default_values["vft_tag_border_size"],
        metavar="BORDER_SIZE",
        help=(
            "vft tag border size (default: %i)" % default_values["vft_tag_border_size"]
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
        "--min-separation-msec",
        dest="min_separation_msec",
        default=default_values["min_separation_msec"],
        help="Sets a minimal distance between two adjacent signals and sets the shortest detectable time difference in ms. Default is set to half the needle length.",
    )
    parser.add_argument(
        "--min-match-threshold",
        dest="min_match_threshold",
        default=default_values["min_match_threshold"],
        help=f"Sets the threshold for detecting audio matches. Default is {default_values['min_match_threshold']}, ratio between the highest correlation and the lower threshold for triggering hits.",
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
        "--input-audio",
        type=str,
        required=False,
        dest="input_audio",
        default=default_values["input_audio"],
        metavar="input-audio-file",
        help="input audio file",
    )
    parser.add_argument(
        "--input-video",
        type=str,
        required=False,
        dest="input_video",
        default=default_values["input_video"],
        metavar="input-video-file",
        help="input video file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        dest="outfile",
        default=default_values["outfile"],
        metavar="output-file",
        help="output file",
    )
    parser.add_argument(
        "--output-audio",
        type=str,
        required=False,
        dest="output_audio",
        default=default_values["output_audio"],
        metavar="output-audio-file",
        help="output audio file",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        required=False,
        dest="output_video",
        default=default_values["output_video"],
        metavar="output-video-file",
        help="output video file",
    )
    parser.add_argument(
        "--no-cache",
        action="store_false",
        dest="cache_both",
        default=True,
        help="Recalculate both audio and video parsing",
    )
    parser.add_argument(
        "--no-cache-audio",
        action="store_false",
        dest="cache_audio",
        default=True,
        help="Recalculate audio parsing",
    )
    parser.add_argument(
        "--no-cache-video",
        action="store_false",
        dest="cache_video",
        default=True,
        help="Recalculate video parsing",
    )
    parser.add_argument(
        "--no-hw-decode",
        action="store_true",
        dest="no_hw_decode",
        default=False,
        help="Do not try to enable hardware decoding",
    )
    parser.add_argument(
        "--audio-offset",
        type=float,
        dest="audio_offset",
        default=default_values["audio_offset"],
        metavar="audio_offset",
        help="Adjust problem in sync for either source of measuring device",
    )
    parser.add_argument(
        "--lock-layout",
        action="store_true",
        dest="lock_layout",
        help="Reuse video frame layout location from the first frame to subsequent frames. This reduces the complexity of the parsing when the camera and DUT are set in a fixed setup",
    )
    parser.add_argument(
        "--calc-all",
        action="store_true",
        dest="calc_all",
        help="Calculate all possible derived data",
    )
    parser.add_argument(
        "--audio-latency",
        action="store_true",
        dest="audio_latency",
        help="Calculate audio latency.",
    )
    parser.add_argument(
        "--video-latency",
        action="store_true",
        dest="video_latency",
        help="Calculate video latency.",
    )
    parser.add_argument(
        "--av-sync",
        action="store_true",
        dest="av_sync",
        help="Calculate audio/video synchronization using audio timestamps and video frame numbers.",
    )
    parser.add_argument(
        "--quality-stats",
        action="store_true",
        dest="quality_stats",
        help="Calculate quality stats.",
    )
    parser.add_argument(
        "--windowed-stats-sec",
        type=float,
        dest="windowed_stats_sec",
        default=None,
        help="Calculate video frames shown/dropped per unit sec.",
    )
    parser.add_argument(
        "--calc-frame-durations",
        action="store_true",
        dest="calculate_frame_durations",
        help="Calculate source frame durations.",
    )
    parser.add_argument(
        "--noise-video",
        action="store_true",
        dest="noise_video",
        help="For 'generate', create a noise video with tags but without audio. For 'parse', calculate percentage of identified video.",
    )
    parser.add_argument(
        "--tag-manual",
        action="store_true",
        dest="tag_manual",
        default=False,
        help="Mouse click tag positions",
    )
    parser.add_argument(
        "--threaded",
        action="store_true",
    )
    parser.add_argument(
        "-z",
        "--z-filter",
        dest="z_filter",
        type=float,
        default=0,
        help="Filter latency outliers by calculating z-scores and filter above this value. Typical value is 3.`",
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
    if options.debug > 2:
        print(options)

    if options.func == "generate":
        # get outfile
        if options.outfile == "-":
            options.outfile = "/dev/fd/1"
        # do something
        if options.noise_video:
            media_generate.media_generate_noise_video(
                outfile=options.outfile,
                width=options.width,
                height=options.height,
                fps=options.fps,
                num_frames=options.num_frames,
                vft_id=options.vft_id,
                debug=options.debug,
            )

        else:
            media_generate.media_generate(
                width=options.width,
                height=options.height,
                fps=options.fps,
                num_frames=options.num_frames,
                vft_id=options.vft_id,
                pre_samples=options.pre_samples,
                samplerate=options.samplerate,
                beep_freq=options.beep_freq,
                beep_duration_samples=options.beep_duration_samples,
                beep_period_sec=options.beep_period_sec,
                scale=options.scale,
                outfile=options.outfile,
                debug=options.debug,
                audio_sample=options.audio_sample,
            )

    elif options.func == "parse":
        if options.output_audio is None:
            options.output_audio = f"{options.infile}.audio.csv"
        if options.output_video is None:
            options.output_video = f"{options.infile}.video.csv"
        media_parse.media_parse(
            width=options.width,
            height=options.height,
            num_frames=options.num_frames,
            pixel_format=options.pixel_format,
            luma_threshold=options.luma_threshold,
            pre_samples=options.pre_samples,
            samplerate=options.samplerate,
            beep_freq=options.beep_freq,
            beep_duration_samples=options.beep_duration_samples,
            beep_period_sec=options.beep_period_sec,
            scale=options.scale,
            infile=options.infile,
            output_video=options.output_video,
            output_audio=options.output_audio,
            debug=options.debug,
        )

    elif options.func == "analyze":
        media_analyze.media_analyze(
            width=options.width,
            height=options.height,
            num_frames=options.num_frames,
            pixel_format=options.pixel_format,
            luma_threshold=options.luma_threshold,
            pre_samples=options.pre_samples,
            samplerate=options.samplerate,
            beep_freq=options.beep_freq,
            beep_duration_samples=options.beep_duration_samples,
            beep_period_sec=options.beep_period_sec,
            scale=options.scale,
            input_video=options.input_video,
            input_audio=options.input_audio,
            outfile=options.outfile,
            vft_id=options.vft_id,
            cache_video=options.cache_video,
            cache_audio=options.cache_audio,
            cache_both=options.cache_both,
            min_separation_msec=options.min_separation_msec,
            min_match_threshold=options.min_match_threshold,
            audio_sample=options.audio_sample,
            lock_layout=options.lock_layout,
            tag_manual=options.tag_manual,
            force_fps=options.force_fps,
            threaded=options.threaded,
            audio_offset=options.audio_offset,
            audio_latency=options.audio_latency,
            video_latency=options.video_latency,
            calc_all=options.calc_all,
            z_filter=options.z_filter,
            av_sync=options.av_sync,
            quality_stats=options.quality_stats,
            windowed_stats_sec=options.windowed_stats_sec,
            calculate_frame_durations=options.calculate_frame_durations,
            no_hw_decode=options.no_hw_decode,
            debug=options.debug,
        )


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
