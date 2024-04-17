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


PARSE = "parse"
GENERATE = "generate"
ANALYZE = "analyze"

FUNC_CHOICES = {
    "help": "show help options",
    GENERATE: "generate reference video",
    PARSE: "parse distorted video",
    ANALYZE: "analyze distorted video",
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
    "analysis_type": None,
    "infile": None,
    "input_audio": None,
    "input_video": None,
    "outfile": None,
    "output_audio": None,
    "output_video": None,
    "audio_offset": 0,
    "windowed_stats_sec": 1,
    "lock_layout": False,
    "audio_sample": audio_common.DEFAULT_AUDIO_SAMPLE,
    "z_filter": 3,
    "force_fps": 30,
}


class VideoSizeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.width, namespace.height = [int(v) for v in values[0].split("x")]


input_args = {
    "version": {
        "func": f"{GENERATE}, {PARSE}, {ANALYZE}",
        "short": "-v",
        "long": "--version",
        "args": {
            "help": "show version",
        },
    },
    "debug": {
        "func": f"{GENERATE}, {PARSE}, {ANALYZE}",
        "short": "-d",
        "long": "--debug",
        "args": {
            "action": "count",
            "help": "Increase verbosity (use multiple times for more)",
            "default": default_values["debug"],
        },
    },
    "quiet": {
        "func": f"{GENERATE}, {PARSE}, {ANALYZE}",
        "short": "-q",
        "long": "--quiet",
        "args": {
            "action": "store_true",
            "help": "Suppress all output",
        },
    },
    "width": {
        "func": f"{GENERATE}, {PARSE}",
        "short": "",
        "long": "--width",
        "args": {
            "type": int,
            "help": "Width of video",
            "default": default_values["width"],
        },
    },
    "height": {
        "func": f"{GENERATE}, {PARSE}",
        "short": "",
        "long": "--height",
        "args": {
            "type": int,
            "help": "Height of video",
            "default": default_values["height"],
        },
    },
    "num_frames": {
        "func": f"{GENERATE}, {PARSE}",
        "short": "",
        "long": "--num-frames",
        "dest": "num_frames",
        "args": {
            "type": int,
            "help": "Number of frames in video",
            "default": default_values["num_frames"],
        },
    },
    "vft_id": {
        "func": GENERATE,
        "short": "",
        "long": "--vft-id",
        "args": {
            "type": str,
            "help": "VFT ID",
            "default": default_values["vft_id"],
        },
    },
    "vft_tag_border_size": {
        "func": GENERATE,
        "short": "",
        "long": "--vft-tag-border-size",
        "args": {
            "type": int,
            "help": "VFT tag border size",
            "default": default_values["vft_tag_border_size"],
        },
    },
    "luma_threshold": {
        "func": PARSE,
        "short": "",
        "long": "--luma-threshold",
        "args": {
            "type": float,
            "help": "Luma threshold",
            "default": default_values["luma_threshold"],
        },
    },
    "fps": {
        "func": GENERATE,
        "short": "",
        "long": "--fps",
        "args": {
            "type": float,
            "help": "Frames per second",
            "default": default_values["fps"],
        },
    },
    "pixel_format": {
        "func": f"{GENERATE}, {PARSE}",
        "short": "",
        "long": "--pixel-format",
        "args": {
            "type": str,
            "help": "Pixel format",
            "default": default_values["pixel_format"],
        },
    },
    "pre_samples": {
        "func": f"{GENERATE}. {PARSE}, {ANALYZE}",
        "short": "",
        "long": "--pre-samples",
        "args": {
            "type": int,
            "help": "Pre samples",
            "default": default_values["pre_samples"],
        },
    },
    "samplerate": {
        "func": f"{GENERATE}.{PARSE}, {ANALYZE}",
        "short": "",
        "long": "--samplerate",
        "args": {
            "type": int,
            "help": "Samplerate",
            "default": default_values["samplerate"],
        },
    },
    "beep_freq": {
        "func": f"{GENERATE}.{PARSE}, {ANALYZE}",
        "short": "",
        "long": "--beep-freq",
        "args": {
            "type": int,
            "help": "Beep frequency",
            "default": default_values["beep_freq"],
        },
    },
    "beep_duration_samples": {
        "func": f"{GENERATE}.{PARSE}, {ANALYZE}",
        "short": "",
        "long": "--beep-duration-samples",
        "args": {
            "type": int,
            "help": "Beep duration samples",
            "default": default_values["beep_duration_samples"],
        },
    },
    "beep_period_sec": {
        "func": f"{GENERATE}.{PARSE}, {ANALYZE}",
        "short": "",
        "long": "--beep-period-sec",
        "args": {
            "type": int,
            "help": "Beep period seconds",
            "default": default_values["beep_period_sec"],
        },
    },
    "scale": {
        "func": f"{GENERATE}, {PARSE}",
        "short": "",
        "long": "--scale",
        "args": {
            "type": int,
            "help": "Scale",
            "default": default_values["scale"],
        },
    },
    "min_separation_msec": {
        "func": PARSE,
        "short": "",
        "long": "--min-separation-msec",
        "args": {
            "type": int,
            "help": "Minimum separation milliseconds",
            "default": default_values["min_separation_msec"],
        },
    },
    "lock_layout": {
        "func": PARSE,
        "short": "",
        "long": "--lock-layout",
        "args": {
            "action": "store_true",
            "help": "Lock layout",
        },
    },
    "analysis_type": {
        "func": ANALYZE,
        "short": "-a",
        "long": "--analysis-type",
        "args": {
            "type": str,
            "help": "Analysis type",
            "default": default_values["analysis_type"],
        },
    },
    "input": {
        "func": PARSE,
        "short": "-i",
        "long": "--input",
        "args": {
            "type": str,
            "help": "Input media file",
            "dest": "infile",
            "default": default_values["infile"],
        },
    },
    "output": {
        "func": f"{GENERATE}, {PARSE}, {ANALYZE}",
        "short": "-o",
        "long": "--output",
        "args": {
            "type": str,
            "help": "Output media file or output csv base name",
            "dest": "outfile",
            "default": default_values["outfile"],
        },
    },
    "input_audio": {
        "func": ANALYZE,
        "short": "",
        "long": "--input-audio",
        "dest": "input_audio",
        "args": {
            "type": str,
            "help": "Input audio parsed csv file",
            "default": default_values["input_audio"],
        },
    },
    "input_video": {
        "func": ANALYZE,
        "short": "",
        "long": "--input-video",
        "dest": "input_video",
        "args": {
            "type": str,
            "help": "Input video parsed csv file",
            "default": default_values["input_video"],
        },
    },
    "output_audio": {
        "func": PARSE,
        "short": "",
        "long": "--output-audio",
        "dest": "output_audio",
        "args": {
            "type": str,
            "help": "Output audio parsed csv file",
            "default": default_values["output_audio"],
        },
    },
    "output_video": {
        "func": PARSE,
        "short": "",
        "long": "--output-video",
        "args": {
            "type": str,
            "help": "Output video parsed csv file",
            "default": default_values["output_video"],
        },
    },
    "no_hw_decode": {
        "func": PARSE,
        "short": "",
        "long": "--no-hw-decode",
        "args": {
            "action": "store_true",
            "help": "Disable hardware decoding",
        },
    },
    "tag_manual": {
        "func": PARSE,
        "short": "",
        "long": "--tag-manual",
        "args": {
            "action": "store_true",
            "help": "Tag manual",
        },
    },
    "threaded": {
        "func": PARSE,
        "short": "",
        "long": "--threaded",
        "args": {
            "action": "store_true",
            "help": "Threaded",
        },
    },
    "audio_offset": {
        "func": ANALYZE,
        "short": "-f",
        "long": "--audio-offset",
        "args": {
            "type": float,
            "help": "Audio offset",
            "default": default_values["audio_offset"],
        },
    },
    "z_filter": {
        "func": ANALYZE,
        "short": "",
        "long": "--z-filter",
        "args": {
            "type": str,
            "help": "Z filter",
            "default": default_values["z_filter"],
        },
    },
    "analysis_type": {
        "func": ANALYZE,
        "short": "-a",
        "long": "--analysis-type",
        "args": {
            "type": str,
            "dest": "analysis_type",
            "help": "%s"
            % (
                " | ".join(
                    "{}: {}".format(k, v[1])
                    for k, v in media_analyze.MEDIA_ANALYSIS.items()
                )
            ),
            "default": default_values["analysis_type"],
        },
    },
    "audio_sample": {
        "func": PARSE,
        "short": "",
        "long": "--audio-sample",
        "args": {
            "type": str,
            "help": "Audio sample",
            "default": default_values["audio_sample"],
        },
    },
    "force_fps": {
        "func": f"{PARSE}, {ANALYZE}",
        "short": "",
        "long": "--force-fps",
        "args": {
            "type": int,
            "help": "Force fps",
            "default": default_values["force_fps"],
        },
    },
    "video_size": {
        "func": PARSE,
        "short": "",
        "long": "--video-size",
        "args": {
            "type": str,
            "action": VideoSizeAction,
            "nargs": 1,
            "help": "use <width>x<height>",
        },
    },
    "windowed_stats_sec": {
        "func": ANALYZE,
        "short": "",
        "long": "--windowed-stats-sec",
        "args": {
            "type": int,
            "help": "Windowed stats seconds",
            "default": default_values["windowed_stats_sec"],
        },
    },
}


def add_arg(func, parser):
    for key, value in input_args.items():
        if func in value["func"]:
            if value["short"]:
                parser.add_argument(
                    value["short"],
                    value["long"],
                    **value["args"],
                )
            else:
                parser.add_argument(
                    value["long"],
                    **value["args"],
                )


def get_options(argv):
    """Generic option parser.

    Args:
        argv: list containing arguments

    Returns:
        Namesypace - An argparse.ArgumentParser-generated option object
    """
    # init parser
    # usage = 'usage: %prog [options] arg1 arg2'
    # parser = argparse.OptionParser(usage=usage)
    # parser.print_help() to get argparse.usage (large help)
    # parser.print_usage() to get argparse.usage (just usage line)
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="func", help="function to perform")

    gen_parser = subparsers.add_parser(GENERATE, help="generate reference video")
    parse_parser = subparsers.add_parser(PARSE, help="parse distorted video")
    analyze_parser = subparsers.add_parser(ANALYZE, help="analyze distorted video")

    add_arg(GENERATE, gen_parser)
    add_arg(PARSE, parse_parser)
    add_arg(ANALYZE, analyze_parser)

    # 2-parameter setter using argparse.Action

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

    if options.func == GENERATE:
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

    elif options.func == PARSE:
        print(f"parse: {options}")
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
            lock_layout=options.lock_layout,
            threaded=options.threaded,
            debug=options.debug,
        )

    elif options.func == ANALYZE:
        media_analyze.media_analyze(
            analysis_type=options.analysis_type,
            pre_samples=options.pre_samples,
            samplerate=options.samplerate,
            beep_freq=options.beep_freq,
            beep_duration_samples=options.beep_duration_samples,
            beep_period_sec=options.beep_period_sec,
            input_video=options.input_video,
            input_audio=options.input_audio,
            outfile=options.outfile,
            force_fps=options.force_fps,
            audio_offset=options.audio_offset,
            z_filter=options.z_filter,
            windowed_stats_sec=options.windowed_stats_sec,
            debug=options.debug,
        )


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
