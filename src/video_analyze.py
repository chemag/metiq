#!/usr/bin/env python3

"""video_analyze.py module description."""


import argparse
import cv2
import math
import sys
import numpy as np
import scipy
import pandas as pd
import video_common
import vft
from _version import __version__


COLOR_BLACK = (0, 0, 0)
COLOR_BACKGROUND = (128, 128, 128)
COLOR_WHITE = (255, 255, 255)

ERROR_NO_VALID_TAG = 1
ERROR_INVALID_GRAYCODE = 2
ERROR_SINGLE_GRAYCODE_BIT = 3
ERROR_UNKNOWN = 100

ERROR_TYPES = {
    # error_id: ("short message", "long message"),
    ERROR_NO_VALID_TAG: ("no_valid_tag", "Frame has no valid set of tags"),
    ERROR_INVALID_GRAYCODE: ("invalid_graycode", "Invalid gray code read"),
    ERROR_SINGLE_GRAYCODE_BIT: (
        "single_graycode_bit",
        "Single non-read bit not in gray position",
    ),
    ERROR_UNKNOWN: ("unknown", "Unknown error"),
}

default_values = {
    "debug": 0,
    "width": video_common.DEFAULT_WIDTH,
    "height": video_common.DEFAULT_HEIGHT,
    "luma_threshold": vft.DEFAULT_LUMA_THRESHOLD,
    "pixel_format": video_common.DEFAULT_PIXEL_FORMAT,
    "infile": None,
    "outfile": None,
}


# VideoCapture-compatible raw (yuv) reader
class VideoCaptureYUV:
    def __init__(self, filename, width, height, pixel_format):
        self.width = width
        self.height = height
        # assume 4:2:0 here
        self.frame_len = math.ceil(self.width * self.height * 3 / 2)
        self.shape = (int(self.height * 1.5), self.width)
        self.pixel_format = pixel_format
        self.f = open(filename, "rb")  # noqa: P201

    def __del__(self):
        self.f.close()

    def release(self):
        self.__del__()

    def isOpened(self):
        return True

    def readRaw(self):
        try:
            raw = self.f.read(self.frame_len)
            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape(self.shape)
        except Exception as ex:
            print(str(ex))
            return False, None
        return True, yuv

    def read(self):
        ret, yuv = self.readRaw()
        if not ret:
            return ret, yuv
        bgr = cv2.cvtColor(yuv, self.pixel_format_to_color_format(self.pixel_format))
        return ret, bgr

    @classmethod
    def pixel_format_to_color_format(cls, pix_fmt):
        if pix_fmt == "yuv420p":
            return cv2.COLOR_YUV2BGR_I420
        elif pix_fmt == "nv12":
            return cv2.COLOR_YUV2BGR_NV12
        elif pix_fmt == "nv21":
            return cv2.COLOR_YUV2BGR_NV21
        raise AssertionError(f"error: invalid {pix_fmt = }")


def get_video_capture(input_file, width, height, pixel_format):
    video_capture = None
    if pixel_format is not None:
        video_capture = VideoCaptureYUV(input_file, width, height, pixel_format)
    else:
        # TODO(chema): fix CAP_FFMPEG open mode
        # right now it does not open the file
        # video_capture = cv2.VideoCapture(input_file, cv2.CAP_FFMPEG)
        # video_capture = cv2.VideoCapture(input_file, cv2.CAP_GSTREAMER)
        video_capture = cv2.VideoCapture(input_file)
    return video_capture


def round_to_nearest_half(value):
    return round(2 * value) / 2


def estimate_video_smoothness(video_results, fps):
    # video_results = [[frame_num, timestamp, frame_num_expected, value_read]*]
    # note that <timestamps> = k * <frame_num>
    # * in the ideal case, the distance between <value_read>
    #   in 2x consecutive frames (  in the ideal case, the distance between <value_read>
    #
    #                      <timestamp>_{i+1} - <timestamp>_i
    # video_speed_{i+1} = -----------------------------------
    #                     <value_read>_{i+1} - <value_read>_i
    #
    video_speed_list = []
    for vr1, vr0 in zip(video_results[1:], video_results[:-1]):
        if vr1[2] is None or vr0[2] is None:
            # invalid reading: skip
            continue
        if vr1[2] == vr0[2]:
            # divide by zero: skip
            continue
        video_speed_list.append((vr1[1] - vr0[1]) / (vr1[2] - vr0[2]))
    return video_speed_list


def video_parse(
    infile,
    width,
    height,
    pixel_format,
    luma_threshold,
    lock_layout=False,
    debug=0,
):
    # If we do ot know the source fps we can still do the parsing.
    # Ignore delta calculations and guessing frames, i.e. the analysis
    return video_analyze(
        infile,
        width,
        height,
        -1,
        pixel_format,
        luma_threshold,
        lock_layout,
        debug=debug,
    )


# Returns a list with one tuple per frame in the distorted video
# stream. Each tuple consists of the following elements:
# * (a) `frame_num`: the frame number (correlative values) in
#   the distorted file,
# * (b) `timestamp`: the timestamp, calculated from `frame_num`
#   and the distorted fps value,
# * (c) `frame_num_expected`: the expected frame number based on
#   `timestamp` and the reference fps (`timestamp * reference_fps`),
# * (d) `status`: whether metiq managed to read the value_read
# * (e) `value_read`: the frame number read in the frame (None
#   if it cannot read it).
# * (f) `delta_frame`: `value_read - delta_mode` (None if
#   `value_read` is not readable).
def video_analyze(
    infile,
    width,
    height,
    ref_fps,
    pixel_format,
    luma_threshold,
    lock_layout=False,
    debug=0,
):
    video_capture = get_video_capture(infile, width, height, pixel_format)
    if not video_capture.isOpened():
        print(f"error: {infile = } is not open")
        sys.exit(-1)
    # 1. analyze the video image-by-image
    in_fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_num = -1
    video_results = pd.DataFrame(
        columns=("frame_num", "timestamp", "frame_num_expected", "status", "value_read")
    )
    while True:
        # get image
        status, img = video_capture.read()
        if not status:
            break
        frame_num += 1
        # this (wrongly) assumes frames are perfectly separated
        timestamp_alt = frame_num / in_fps
        # cv2.CAP_PROP_POS_MSEC returns the right timestamp
        timestamp = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        # the frame_num we expect to see for this timestamp
        frame_num_expected = timestamp * ref_fps
        if debug > 2:
            print(
                f"video_analyze: parsing {frame_num = } {timestamp = } {ref_fps = } {in_fps = }"
            )
        # analyze image
        value_read = None
        try:
            if width > 0 and height > 0:
                dim = (width, height)
                img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            value_read = image_analyze(img, luma_threshold, lock_layout = lock_layout, debug = debug)
            status = 0
        except vft.NoValidTag as ex:
            status = ERROR_NO_VALID_TAG
        except vft.InvalidGrayCode as ex:
            status = ERROR_INVALID_GRAYCODE
        except vft.SingleGraycodeBitError as ex:
            status = ERROR_SINGLE_GRAYCODE_BIT
        except Exception as ex:
            if debug > 0:
                print(f"{frame_num = } {str(ex)}")
            status = ERROR_UNKNOWN
            continue
        if debug > 2:
            print(f"video_analyze: read image value: {value_read}")
            if value_read is None:
                cv2.imwrite(f"debug/{infile}_{frame_num}.png", img)
        video_results.loc[len(video_results.index)] = (
            frame_num,
            timestamp,
            frame_num_expected,
            status,
            value_read,
        )

    # 2. clean up
    try:
        video_capture.release()
    except Exception as exc:
        print(f"error: {exc = }")
        pass
    # 3. calculate the delta mode
    # note that <timestamps> = k * <frame_num>
    num_frames = len(video_results)
    # remove the None values in order to calculate the delta between frames
    delta_list = round_to_nearest_half(
        video_results["value_read"] - video_results["frame_num_expected"]
    )
    # calculate the mode of the delta between frames
    delta_mode = scipy.stats.mode(delta_list, keepdims=True).mode[0]
    # 4. calculate the delta column
    # simplify the results: substract the mode, and keep the None
    # * value_read - (frame_num_expected + delta_mode)  # if value_read is not None
    # * None  # otherwise
    video_results["delta_frame"] = round_to_nearest_half(
        video_results["value_read"] - video_results["frame_num_expected"] - delta_mode
    )
    return video_results


# distill video_delta_info from video_results
def video_analyze_delta_info(video_results):
    num_frames = len(video_results)
    # remove the None values in order to calculate the delta between frames
    delta_list = round_to_nearest_half(
        video_results["value_read"] - video_results["frame_num_expected"]
    )
    # calculate the mode of the delta between frames
    delta_mode = scipy.stats.mode(delta_list, keepdims=True).mode[0]
    ok_frames = (video_results["delta_frame"] == 0.0).sum()
    sok_frames = (video_results["delta_frame"].between(-0.5, 0.5)).sum()
    unknown_frames = video_results["delta_frame"].isna().sum()
    nok_frames = num_frames - sok_frames - unknown_frames
    stddev = video_results["delta_frame"].std()
    if num_frames == 0:
        return None
    video_delta_info = pd.DataFrame(
        columns=(
            "mode",
            "stddev",
            "ok_ratio",
            "sok_ratio",
            "nok_ratio",
            "unknown_ratio",
        )
    )
    video_delta_info.loc[len(video_delta_info.index)] = (
        delta_mode,
        stddev,
        ok_frames / num_frames,
        sok_frames / num_frames,
        nok_frames / num_frames,
        unknown_frames / num_frames,
    )
    return video_delta_info


def image_analyze(img, luma_threshold, lock_layout=False, debug=0):
    num_read, vft_id = vft.analyze_graycode(img, luma_threshold, lock_layout = lock_layout, debug = debug)
    return num_read


def dump_video_results(video_results, outfile, debug):
    if video_results is None or len(video_results) == 0:
        print(f"No video results: {video_results = }")
        return
    # write the output as a csv file
    video_results.to_csv(outfile, index=False)


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
        metavar="THRESHOLD",
        help=(
            "luma detection threshold (default: %i)" % default_values["luma_threshold"]
        ),
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
    return options


def main(argv):
    # parse options
    options = get_options(argv)
    if options.version:
        print("version: %s" % __version__)
        sys.exit(0)
    # get infile/outfile
    if options.infile is None or options.infile == "-":
        options.infile = "/dev/fd/0"
    if options.outfile is None or options.outfile == "-":
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(options)
    # do something
    video_results = video_analyze(
        options.infile,
        options.width,
        options.height,
        30,
        options.pixel_format,
        options.luma_threshold,
        options.debug,
    )
    dump_video_results(video_results, options.outfile, options.debug)
    # print the delta info
    video_delta_info = video_analyze_delta_info(video_results)
    print(f"score for {options.infile = } {video_delta_info = }")


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
