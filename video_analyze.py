#!/usr/bin/env python3

"""video_analyze.py module description."""


import argparse
import cv2
import math
import sys
import numpy as np
import scipy

import video_common
import aruco_common
from _version import __version__


# use fiduciarial markers from this dictionary
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

COLOR_BLACK = (0, 0, 0)
COLOR_BACKGROUND = (128, 128, 128)
COLOR_WHITE = (255, 255, 255)


default_values = {
    "debug": 0,
    "width": video_common.DEFAULT_WIDTH,
    "height": video_common.DEFAULT_HEIGHT,
    "luma_threshold": video_common.DEFAULT_LUMA_THRESHOLD,
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


def detect_markers(img, debug):
    corners, ids = aruco_common.detect_aruco_tags(img)
    if len(ids) != 3:
        if debug > 2:
            print(f"error: image has {len(ids)} marker(s) (should have 3)")
        return None
    ids.shape = 3
    if set(ids) != {0, 1, 2}:
        print(f"error: invalid set of markers: {set(ids)}")
        return None
    marker_locations = []
    for marker_id in range(3):
        index = list(ids).index(marker_id)
        assert corners[index].shape == (
            1,
            4,
            2,
        ), f"error: invalid corners[{index}]: {corners[index]}"
        xt = 0.0
        yt = 0.0
        for corner in corners[index][0]:
            (x, y) = corner
            xt += x
            yt += y
        xt /= 4
        yt /= 4
        marker_locations.append((xt, yt))
    return marker_locations


def affine_transformation(img, marker_locations, marker_expected_locations, debug):
    # process the image
    s0, s1, s2 = marker_locations
    d0, d1, d2 = marker_expected_locations
    src_trio = np.array([s0, s1, s2]).astype(np.float32)
    dst_trio = np.array([d0, d1, d2]).astype(np.float32)
    transform_matrix = cv2.getAffineTransform(src_trio, dst_trio)
    if debug > 2:
        print(f"transform_matrix: {transform_matrix}")
    outimg = cv2.warpAffine(img, transform_matrix, (img.shape[1], img.shape[0]))
    return outimg


def parse_gray_code(img, image_info, luma_threshold, debug):
    # extract the luma
    img_luma = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # read the bit string
    bit_stream = []
    for box_id in range(image_info.gb_num_bits):
        x0 = image_info.gb_x[box_id]
        x1 = x0 + image_info.gb_boxsize
        yc = image_info.gb_y[box_id]
        yt = yc - image_info.gb_boxsize
        yb = yc + image_info.gb_boxsize
        img_luma_top = img_luma[yt:yc, x0:x1]
        luma_top = np.mean(img_luma_top)
        img_luma_bot = img_luma[yc:yb, x0:x1]
        luma_bot = np.mean(img_luma_bot)
        if abs(luma_top - luma_bot) < luma_threshold:
            bit = "X"
        elif luma_top > luma_bot:
            bit = 1
        else:
            bit = 0
        bit_stream.append(bit)
    # convert it to a gray number
    num_read = video_common.gray_bitstream_to_num(bit_stream)
    return num_read


def get_video_delta(video_results):
    # video_results = [[frame_num, frame_num_effective, timestamp, frame_num_read]*]
    # note that <timestamps> = k * <frame_num>
    num_frames = len(video_results)
    # remove the None values in order to calculate the delta between frames
    delta_list = list(
        delta
        for delta in map(
            lambda t: t[3] - t[1] if t[3] is not None else None, video_results
        )
        if delta is not None
    )
    # calculate the mode of the delta between frames
    delta_mode = scipy.stats.mode(delta_list, keepdims=True).mode[0]
    # simplify the results
    # * t[3] - (t[1] + delta_mode)  # if t[3] is not None
    # * None  # otherwise
    delta_results = list(
        map(
            lambda t: (t[3] - (t[1] + delta_mode) if t[3] is not None else None),
            video_results,
        )
    )
    ok_frames = delta_results.count(0.0)
    unknown_frames = delta_results.count(None)
    nok_frames = num_frames - ok_frames - unknown_frames
    stddev = np.std(list(delta for delta in delta_results if delta is not None))
    return {
        "mode": delta_mode,
        "stddev": stddev,
        "ok_ratio": ok_frames / num_frames,
        "nok_ratio": nok_frames / num_frames,
        "unknown_ratio": unknown_frames / num_frames,
    }


def estimate_video_smoothness(video_results, fps):
    # video_results = [[frame_num, frame_num_effective, timestamp, frame_num_read]*]
    # note that <timestamps> = k * <frame_num>
    # * in the ideal case, the distance between <frame_num_read>
    #   in 2x consecutive frames (  in the ideal case, the distance between <frame_num_read>
    #
    #                          <timestamp>_{i+1} - <timestamp>_i
    # video_speed_{i+1} = -------------------------------------------
    #                     <frame_num_read>_{i+1} - <frame_num_read>_i
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


# Returns a list with one tuple per frame in the distorted video
# stream. Each tuple consists of 4x elements:
# * (a) `frame_num`: the frame number (correlative values)
# * (b) `frame_num_effective`: the effective frame number
#   (`frame_num` whenc considering the reference fps),
# * (c) `timestamp`: the timestamp (calculated from `frame_num`
#   and the `in_fps` value), and
# * (d) `frame_num_read`: the frame number read in the frame (None
#   if it cannot read it).
def video_analyze(infile, width, height, fps, pixel_format, luma_threshold, debug):
    image_info = video_common.ImageInfo(width, height, video_common.NUM_BITS)
    video_capture = get_video_capture(infile, width, height, pixel_format)
    if not video_capture.isOpened():
        print(f"error: {infile = } is not open")
        sys.exit(-1)

    # analyze the video image-by-image
    in_fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_num = -1
    video_results = []
    while True:
        # get image
        status, img = video_capture.read()
        if not status:
            break
        if debug > 1:
            print(f"video_analyze: parsing {frame_num = }")
        frame_num += 1
        frame_num_effective = frame_num * fps / in_fps
        timestamp = frame_num / in_fps
        timestamp_alt = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        # analyze image
        marker_locations = detect_markers(img, debug)
        if marker_locations is None:
            # could not read the 3x markers properly: stop here
            video_results.append((frame_num, frame_num_effective, timestamp, None))
            continue
        marker_expected_locations = [
            image_info.get_marker_center(marker_id) for marker_id in range(3)
        ]
        # apply affine transformation to source image
        img_affine = affine_transformation(
            img, marker_locations, marker_expected_locations, debug
        )
        # read the gray code
        frame_num_read = parse_gray_code(img_affine, image_info, luma_threshold, debug)
        if debug > 2:
            print(f"{timestamp = } {timestamp_alt = }")
        video_results.append(
            (frame_num, frame_num_effective, timestamp, frame_num_read)
        )

    # clean up
    try:
        video_capture.release()
    except Exception as exc:
        print(f"error: {exc = }")
        pass

    return video_results


def dump_results(video_results, outfile, debug):
    # write the output as a csv file
    with open(outfile, "w") as fd:
        fd.write("frame_num,timestamp,frame_num_read\n")
        for frame_num, timestamp, frame_num_read in video_results:
            fd.write(f"{frame_num},{timestamp},{frame_num_read}\n")


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
        "infile",
        type=str,
        default=default_values["infile"],
        metavar="input-file",
        help="input file",
    )
    parser.add_argument(
        "outfile",
        type=str,
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
        options.pixel_format,
        options.luma_threshold,
        options.debug,
    )
    dump_results(video_results, options.outfile, options.debug)
    # provide a score
    video_delta = get_video_delta(video_results)
    print(f"score for {options.infile = } {video_delta = }")


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
