#!/usr/bin/env python3

"""video_parse.py module description."""


import argparse
import cv2
import math
import sys
import numpy as np
import scipy
import pandas as pd
import video_common
import audio_common
import video_tag_coordinates as vtc
import vft
import time
from shapely.geometry import Polygon
from _version import __version__
import timeit
import threading
import queue
import time

COLOR_BLACK = (0, 0, 0)
COLOR_BACKGROUND = (128, 128, 128)
COLOR_WHITE = (255, 255, 255)

HW_DECODER_ENABLE = True
TEN_TO_NINE = 1000000000.0
COMMON_FPS = [7.0, 15.0, 29.97, 30.0, 59.94, 60.0, 119.88, 120.0, 239.76]

default_values = {
    "debug": 0,
    "width": video_common.DEFAULT_WIDTH,
    "height": video_common.DEFAULT_HEIGHT,
    "luma_threshold": vft.DEFAULT_LUMA_THRESHOLD,
    "pixel_format": video_common.DEFAULT_PIXEL_FORMAT,
    "infile": None,
    "outfile": None,
}

# OpenCV have memory issues when used with threads causing a crash on deallocation
# If we keep the name global in the file we can release it just fine without a crash
video_capture = None


# Wrap the VideoCapture
# Use a threaded decode to parallelize the work
class VideoCaptureWrapper(cv2.VideoCapture):
    decode = True
    frames = None
    thread = None
    # Max numbers of decoded frames in the queue
    frameLimit = threading.Semaphore(5)
    current_time = 0

    def __init__(self, filename, api=0, flags=0):
        super(VideoCaptureWrapper, self).__init__(filename, api, flags)
        self.frames = queue.Queue(maxsize=5)
        self.thread = threading.Thread(target=self.decode_video)
        self.thread.start()

    # Override
    def read(self):
        if self.decode or self.frames.qsize() > 0:
            frame, timestamp = self.frames.get()
            self.current_time = timestamp
        else:
            return False, None

        self.frameLimit.release()
        return True, frame

    # Override
    def get(self, propId):
        if propId == cv2.CAP_PROP_POS_MSEC:
            return self.current_time
        else:
            return super(VideoCaptureWrapper, self).get(propId)

    def decode_video(self):
        while self.decode:
            ret, frame = super(VideoCaptureWrapper, self).read()
            current_time = super(VideoCaptureWrapper, self).get(cv2.CAP_PROP_POS_MSEC)
            if not ret:
                self.decode = False
                break
            self.frameLimit.acquire()
            self.frames.put((frame, current_time))

    def release(self):
        self.decode = False
        if self.frames.qsize() > 0:
            self.frames.join()
        self.thread.join()

        super(VideoCaptureWrapper, self).release()


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


def get_video_capture(input_file, width, height, pixel_format, threaded=False):
    video_capture = None
    if pixel_format is not None:
        video_capture = VideoCaptureYUV(input_file, width, height, pixel_format)
    else:
        # Auto detect API and try to open hw acceleration
        if HW_DECODER_ENABLE:
            if threaded:
                video_capture = VideoCaptureWrapper(
                    input_file,
                    0,
                    (cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY),
                )
            else:
                video_capture = cv2.VideoCapture(
                    input_file,
                    0,
                    (cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY),
                )
        else:
            if threaded:
                video_capture = VideoCaptureWrapper(input_file)
            else:
                video_capture = cv2.VideoCapture(input_file)
        # throw error

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


def image_parse(
    img,
    frame_id,
    luma_threshold,
    vft_id,
    tag_center_locations,
    tag_expected_center_locations,
    debug,
):
    value_read, status = image_parse_raw(
        img,
        frame_id,
        luma_threshold,
        vft_id=vft_id,
        tag_center_locations=tag_center_locations,
        tag_expected_center_locations=tag_expected_center_locations,
        debug=debug,
    )
    return value_read, status


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
def video_parse(
    infile,
    width,
    height,
    ref_fps,
    pixel_format,
    luma_threshold,
    lock_layout=False,
    tag_manual=False,
    threaded=False,
    debug=0,
):
    # reset and latch onto a fresh layout config
    vft_id = None
    tag_center_locations = None
    tag_expected_center_locations = None

    # Open a window and mouse click coordinates?
    if tag_manual:
        vft_id = vft.DEFAULT_VFT_ID
        tag_center_locations = vtc.tag_video(infile, width, height)
        vft_layout = vft.VFTLayout(width, height, vft_id)
        tag_expected_center_locations = vft_layout.get_tag_expected_center_locations()
        lock_layout = True
    # With lock_layout go through the file until valid tags has been identified.
    # Save settings and use those for tranformation and gray code parsing.
    if lock_layout and vft_id is None:
        (
            vft_id,
            tag_center_locations,
            tag_expected_center_locations,
        ) = find_first_valid_tag(infile, width, height, pixel_format, debug)
    video_capture = get_video_capture(infile, width, height, pixel_format, threaded)
    if not video_capture.isOpened():
        print(f"error: {infile = } is not open")
        sys.exit(-1)
    # 1. parse the video image-by-image
    in_fps = video_capture.get(cv2.CAP_PROP_FPS)
    in_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    in_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if in_width < width:
        width = 0
    if in_height < height:
        height = 0

    frame_num = -1
    columns = ("frame_num", "timestamp", "frame_num_expected", "status", "value_read")
    video_results = pd.DataFrame(columns=columns)

    previous_value = -1
    total_nbr_of_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    start = time.monotonic_ns()
    accumulated_decode_time = 0
    failed_parses = 0
    vft_layout = None
    while True:
        # get image
        decstart = time.monotonic_ns()
        status, img = video_capture.read()
        if not status:
            break
        frame_num += 1
        accumulated_decode_time += time.monotonic_ns() - decstart
        # this (wrongly) assumes frames are perfectly separated
        timestamp_alt = frame_num / in_fps
        # cv2.CAP_PROP_POS_MSEC returns the right timestamp
        timestamp = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        # the frame_num we expect to see for this timestamp
        frame_num_expected = timestamp * ref_fps
        if debug > 2:
            print(
                f"video_parse: parsing {frame_num = } {timestamp = } {ref_fps = } {in_fps = }"
            )
        # parse image
        value_read = None
        if width > 0 and height > 0:
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            status = vft.VFTReading.other
            threshold = luma_threshold
            while (
                not vft.VFTReading.readable(status)
                and threshold > 1
                and previous_value >= 0
            ):
                # The slow part is the decode so let us spend some more time on failures.
                # To prevent the `first value from being garbage wait until there is something stable.
                # We can accept some instability later on since the check on large jumps will prevent
                # errors (small errors are unlikely - large ones very likely).

                frame_id = f"{infile}.frame_{frame_num}"
                value_read, status = image_parse(
                    img,
                    frame_id,
                    threshold,
                    vft_id,
                    tag_center_locations,
                    tag_expected_center_locations,
                    debug,
                )
                if not vft.VFTReading.readable(status):
                    threshold = threshold / 2

        current_time = time.monotonic_ns()
        time_per_iteration = (current_time - start) / (frame_num + 1)
        time_left_sec = (
            time_per_iteration * (total_nbr_of_frames - frame_num) / TEN_TO_NINE
        )
        estimation = f" estimated time left: {time_left_sec:6.1f} sec"
        speed_text = f", processing {1/(time_per_iteration/TEN_TO_NINE):.2f} fps"
        error_ratio = ""
        if failed_parses > 0:
            error_ratio = f"- errors: {100*failed_parses/(frame_num + 1):5.2f}% "
        if debug > 0:
            decode_time_per_iteration = accumulated_decode_time / (frame_num + 1)
            speed_text = f"{speed_text}, dec. time:{decode_time_per_iteration/1000000:=5.2f} ms, calc, time: {(time_per_iteration - decode_time_per_iteration)/1000000:5.2f} ms"

        print(
            f"-- {round(100 * frame_num/total_nbr_of_frames, 2):5.2f} % frame_num: {frame_num} {estimation}{speed_text} {error_ratio}{' ' * 20}",
            end="\r",
        )
        if status != vft.VFTReading.ok:
            if status != vft.VFTReading.single_graycode:
                # parse image
                _vft_id = None
                _ids = None
                if not tag_manual:
                    _vft_id, _tag_center_locations, _borders, _ids = vft.detect_tags(
                        img, debug=0
                    )
                if _vft_id:
                    vft_layout = vft.VFTLayout(width, height, _vft_id)
                    vft_id = _vft_id
                    tag_center_locations = _tag_center_locations

                elif not vtc.are_tags_frozen() and tag_manual:
                    tag_center_locations = vtc.tag_frame(img)

                if vft_layout:
                    tag_expected_center_locations = (
                        vft_layout.get_tag_expected_center_locations()
                    )
                if tag_expected_center_locations == None and tag_center_locations == None:
                    print("No tags found")
                    failed_parses += 1
                    continue
                if (
                    len(tag_expected_center_locations) == 3
                    or len(tag_center_locations) == 3
                ) and _ids is not None:
                    tag_expected_center_locations = sort_tag_expected_center_locations(
                        tag_expected_center_locations, vft_layout, _ids
                    )
                frame_id = f"{infile}.frame_{frame_num}"
                value_read, status = image_parse(
                    img,
                    frame_id,
                    luma_threshold,
                    vft_id,
                    tag_center_locations,
                    tag_expected_center_locations,
                    debug,
                )

        if not vft.VFTReading.readable(status):
            print(f"failed parsing frame {frame_num=}")
            failed_parses += 1
            if status == vft.VFTReading.other:
                continue

        # Filter huge leaps (indicating erronous parsing)
        leap_max = 20  # secs
        if (
            isinstance(value_read, int)
            and value_read >= 0
            and previous_value >= 0
            and ref_fps > 0
        ):
            # OK we had something before, compare the diff
            if abs(value_read - previous_value) > ref_fps * leap_max:
                print(
                    f"Big leap. {previous_value=}, {value_read=}, {abs(value_read - previous_value)}, {ref_fps=}"
                )
                value_read = None
                status = vft.VFTReading.large_delta
        if isinstance(value_read, int):
            previous_value = value_read

        if debug > 2:
            print(f"video_parse: read image value: {value_read}")
            if not isinstance(value_read, int):
                cv2.imwrite(f"debug/{infile}_{frame_num}.png", img)
        video_results.loc[len(video_results.index)] = (
            frame_num,
            timestamp,
            frame_num_expected,
            status.value,
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

    current_time = time.monotonic_ns()
    total_time = current_time - start
    calc_time = total_time - accumulated_decode_time
    error_ratio = f"- errors: {100*failed_parses/(total_nbr_of_frames):5.2f}% "
    if debug > 0:
        decode_time_per_iteration = accumulated_decode_time / (frame_num + 1)
        print(f"{' ' * 120}")
        print(
            f"Total time: {total_time/1000000000:.2f} sec, total decoding time: {accumulated_decode_time/1000000000:.2f} sec"
        )
        print(
            f"Processing  {total_nbr_of_frames/total_time*1000000000:.2f} fps, per frame: {decode_time_per_iteration/1000000:=5.2f} ms,"
            f"calc: {(time_per_iteration - decode_time_per_iteration)/1000000:5.2f} ms {error_ratio}"
        )
    else:
        print(f"{' ' * 120}")
        print(
            f"Total time: {total_time/1000000000:.2f} sec, processing {total_nbr_of_frames/total_time*1000000000:.2f} fps {' '*30} - {error_ratio}"
        )
    # fix the column types
    video_results = video_results.astype({"frame_num": int, "status": int})
    return video_results


# distill video_delta_info from video_results
def video_parse_delta_info(video_results):
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


def image_parse_raw(
    img,
    frame_id,
    luma_threshold,
    vft_id=None,
    tag_center_locations=None,
    tag_expected_center_locations=None,
    debug=0,
):
    num_read, status, vft_id = vft.graycode_parse(
        img,
        frame_id,
        luma_threshold,
        vft_id=vft_id,
        tag_center_locations=tag_center_locations,
        tag_expected_center_locations=tag_expected_center_locations,
        debug=debug,
    )
    return num_read, status


def dump_video_results(video_results, outfile, debug):
    if video_results is None or len(video_results) == 0:
        print(f"No video results: {video_results = }")
        return
    # write the output as a csv file
    video_results.to_csv(outfile, index=False)


def calc_alignment(infile, width, height, pixel_format, debug):
    # With 'lock--layout' we only need one sample, for now let us asume this is the case always...
    video_capture = get_video_capture(infile, width, height, pixel_format)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not video_capture.isOpened():
        print(f"error: {infile = } is not open")
        sys.exit(-1)

    tag_center_locations = None
    while tag_center_locations is None:
        status, img = video_capture.read()
        if not status:
            print(f"error: {infile = } could not read frame")
            sys.exit(-1)

        # parse image
        vft_id, tag_center_locations, borders, ids = vft.detect_tags(img, debug=0)

    # transform
    vft_layout = vft.VFTLayout(width, height, vft_id)
    tag_expected_center_locations = vft_layout.get_tag_expected_center_locations()

    measured_area = 0
    if len(tag_center_locations) == 3 and ids is not None:
        tag_order = [nbr for nbr, id_ in enumerate(vft_layout.tag_ids) if id_ in ids]
        tag_expected_center_locations = [
            tag_expected_center_locations[i] for i in tag_order
        ]

        # assume we are in the plane and no angle (not much we can do if not)
        triangle = Polygon(tag_center_locations)
        measured_area = triangle.area * 2
        expected_triangle = Polygon(tag_expected_center_locations)
        expected_area = expected_triangle.area
    else:
        # the order is wrong for shapely
        tag_center_locations = [tag_center_locations[i] for i in [0, 1, 3, 2]]
        rectangle = Polygon(tag_center_locations)
        measured_area = rectangle.area
        tag_expected_center_locations = [
            tag_expected_center_locations[i] for i in [0, 1, 3, 2]
        ]
        expected = Polygon(tag_expected_center_locations)
        expected_area = expected.area

    ratio = measured_area / expected_area
    perc = ratio * 100
    print(f"Coverage: {round(perc,2)} %")

    if debug > 2:
        lastpoint = None
        for location in tag_center_locations:
            cv2.circle(img, (int(location[0]), int(location[1])), 10, (0, 0, 255), 10)
            if lastpoint:
                cv2.line(
                    img,
                    (int(location[0]), int(location[1])),
                    (int(lastpoint[0]), int(lastpoint[1])),
                    (0, 0, 255),
                    10,
                )
            lastpoint = location
        lastpoint = None
        for location in tag_expected_center_locations:
            cv2.circle(img, (int(location[0]), int(location[1])), 10, (0, 255, 0), 10)
            if lastpoint:
                cv2.line(
                    img,
                    (int(location[0]), int(location[1])),
                    (int(lastpoint[0]), int(lastpoint[1])),
                    (0, 255, 0),
                    10,
                )
            lastpoint = location
        cv2.imshow("img", img)
        cv2.waitKey(0)
    return perc


def sort_tag_expected_center_locations(tag_expected_center_locations, vft_layout, ids):
    tag_order = [nbr for nbr, id_ in enumerate(vft_layout.tag_ids) if id_ in ids]
    tag_expected_center_locations = [
        tag_expected_center_locations[i] for i in tag_order
    ]
    return tag_expected_center_locations


def find_first_valid_tag(infile, width, height, pixel_format, debug):
    video_capture = get_video_capture(infile, width, height, pixel_format)
    if not video_capture.isOpened():
        print(f"error: {infile = } is not open")
        sys.exit(-1)

    tag_center_locations = None
    tag_expected_center_locations = None
    ids = None
    vft_id = None
    while tag_center_locations is None:
        status, img = video_capture.read()
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        if not status:
            print(f"error: {infile = } could not read frame")
            sys.exit(-1)
        # parse image
        vft_id, tag_center_locations, borders, ids = vft.detect_tags(img, debug=0)

        # bail if we are reading to far ahead (three times the beep?)
        current_time = video_capture.get(cv2.CAP_PROP_POS_MSEC)
        # compare with beep time
        if current_time > audio_common.DEFAULT_BEEP_PERIOD_SEC * 3000:
            print(f"error: {infile = } could not find a tag in the first {current_time} ms")
            break
    if vft_id != None:
        vft_layout = vft.VFTLayout(width, height, vft_id)
        tag_expected_center_locations = vft_layout.get_tag_expected_center_locations()

        if len(tag_center_locations) == 3 and ids is not None:
            tag_expected_center_locations = sort_tag_expected_center_locations(
                tag_expected_center_locations, vft_layout, ids
            )
    try:
        video_capture.release()
    except Exception as exc:
        print(f"error: {exc = }")
        pass
    video_capture = None

    if tag_center_locations is None:
        raise ValueError(f"error: {infile = } could not find a tag")

    if debug > 0:
        print(f"Found tags: {len(tag_center_locations) = }")
    return vft_id, tag_center_locations, tag_expected_center_locations


def config_decoder(**options):
    global HW_DECODER_ENABLE
    HW_DECODER_ENABLE = options.get("HW_DECODER_ENABLE", False)


# estimates the framerate (fps) of a video
def estimate_fps(video_results, use_common_fps_vals=True):
    # Estimate source and capture fps by looking at video timestamps
    video_results = video_results.replace([np.inf, -np.inf], np.nan)
    video_results = video_results.dropna(subset=["value_read"])

    if len(video_results) == 0:
        raise Exception("Failed to estimate fps")
    capture_fps = len(video_results) / (
        video_results["timestamp"].max() - video_results["timestamp"].min()
    )

    video_results["value_read_int"] = video_results["value_read"].astype(int)
    min_val = video_results["value_read_int"].min()
    min_ts = video_results.loc[video_results["value_read_int"] == min_val][
        "timestamp"
    ].values[0]
    max_val = video_results["value_read_int"].max()
    max_ts = video_results.loc[video_results["value_read_int"] == max_val][
        "timestamp"
    ].values[0]
    vals = video_results["value_read_int"].unique()

    min_val = np.min(vals)
    max_val = np.max(vals)

    ref_fps = (max_val - min_val) / (max_ts - min_ts)

    if use_common_fps_vals:
        ref_fps = COMMON_FPS[np.argmin([abs(x - ref_fps) for x in COMMON_FPS])]
        capture_fps = COMMON_FPS[np.argmin([abs(x - capture_fps) for x in COMMON_FPS])]
    return ref_fps, capture_fps


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
    parser.add_argument(
        "--calc-alignment",
        action="store_true",
        dest="calc_alignment",
        default=False,
        help="Calculate alignment",
    )
    parser.add_argument(
        "--no-hw-decode",
        action="store_true",
        dest="no_hw_decode",
        default=False,
        help="Do not try to enable hardware decoding",
    )
    parser.add_argument(
        "--lock-layout",
        action="store_true",
        dest="lock_layout",
        help="Reuse video frame layout location from the first frame to subsequent frames. This reduces the complexity of the parsing when the camera and DUT are set in a fixed setup",
    )
    parser.add_argument(
        "--tag-manual",
        action="store_true",
        dest="tag_manual",
        default=False,
        help="Mous click tag positions",
    )
    parser.add_argument(
        "--threaded",
        action="store_true",
    )

    # do the parsing
    options = parser.parse_args(argv[1:])
    if options.version:
        return options
    return options


def main(argv):
    global HW_DECODER_ENABLE
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

    if options.no_hw_decode:
        HW_DECODER_ENABLE = False
    # do something
    if options.calc_alignment:
        calc_alignment(
            options.infile,
            options.width,
            options.height,
            options.pixel_format,
            options.debug,
        )
        return
    video_results = video_parse(
        options.infile,
        options.width,
        options.height,
        30,
        options.pixel_format,
        options.luma_threshold,
        options.lock_layout,
        tag_manual=options.tag_manual,
        threaded=options.threaded,
        debug=options.debug,
    )
    dump_video_results(video_results, options.outfile, options.debug)
    # print the delta info
    video_delta_info = video_parse_delta_info(video_results)
    print(f"score for {options.infile = } {video_delta_info = }")


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
