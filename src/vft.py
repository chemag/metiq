#!/usr/bin/env python3

"""vft.py module description.

A VFT (Video Fine-Grained Timing) 2D Barcode Library.
"""


import argparse
import copy
import cv2
import dataclasses
import enum
import graycode
import itertools
import json
import numpy as np
import operator
import os
import random
import subprocess
import sys
import time
import typing

import aruco_common

__version__ = "0.1"


class VFTReading(enum.Enum):
    # all VFT blocks were read correctly (ok)
    ok = 0
    # all VFT blocks but 1 were read correctly (ok)
    single_graycode = 1
    # could not read the bit stream in the VFT
    invalid_graycode = 2
    # could not fix a 1-block issue in the VFT
    single_graycode_unfixable = 3
    # large difference between 2x consecutive frames [parsing error]
    large_delta = 4
    # could not read the VFT
    no_input = 5
    # could not read the VFT tags
    no_tags = 6
    # other issue
    other = 7

    @classmethod
    def readable(cls, val):
        return val in (cls.ok, cls.single_graycode)


VFT_IDS = ("9x8", "9x6", "7x5", "5x4")
DEFAULT_VFT_ID = "7x5"
DEFAULT_TAG_BORDER_SIZE = 2
DEFAULT_LUMA_THRESHOLD = 20
DEFAULT_TAG_NUMBER = 4
DEFAULT_VIDEO_PARSE_MODE = "median"

VFT_LAYOUT = {
    # "vft_id": [numcols, numrows, (aruco_tag_0, aruco_tag_1, aruco_tag_2)],
    "7x5": [7, 5, (0, 1, 2, 7)],  # 15 bits (7*5-4=31 -> 15 bits)
    "5x4": [5, 4, (0, 1, 3, 8)],  # 8 bits (5x4-4=16 -> 8 bits)
    "9x8": [9, 8, (0, 1, 4, 9)],  # 34 bits (9x8-4=68 -> 34 bits)
    "9x6": [9, 6, (0, 1, 5, 10)],  # 25 bits (9x6-4=50 -> 25 bits)
}


# use fiduciarial markers ("tags") from this dictionary
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

COLOR_BLACK = (0, 0, 0)
COLOR_BACKGROUND = (128, 128, 128)
COLOR_WHITE = (255, 255, 255)

# Debug visualization colors
COLOR_FIDUCIAL_CIRCLE = (0, 0, 255)  # red
COLOR_FIDUCIAL_ARROW = (255, 255, 255)  # white
COLOR_FIDUCIAL_TEXT = (255, 0, 0)  # blue
COLOR_BLOCK_BORDER = (0, 255, 0)  # green
COLOR_BLOCK_TEXT = (255, 0, 0)  # blue

MIN_TAG_BORDER_SIZE = 2
MIN_SIZE = 64


FUNC_CHOICES = {
    "help": "show help options",
    "generate": "generate VFT tag",
    "parse": "parse VFT tag in image",
}

default_values = {
    "debug": 0,
    "width": 1280,
    "height": 720,
    "vft_id": DEFAULT_VFT_ID,
    "tag_border_size": DEFAULT_TAG_BORDER_SIZE,
    "luma_threshold": DEFAULT_LUMA_THRESHOLD,
    "value": 0,
    "func": "help",
    "infile": None,
    "outfile": None,
}


# Gray-code based API
def generate_graycode(width, height, vft_id, tag_border_size, value, debug):
    # convert value to gray code
    graycode_value = graycode.tc_to_gray_code(value)
    return generate(width, height, vft_id, tag_border_size, graycode_value, debug=debug)


def draw_tags(img, vft_id, tag_border_size, debug):
    global vft_layout
    width = img.shape[1]
    height = img.shape[0]
    if vft_layout is None:
        vft_layout = VFTLayout(width, height, vft_id, tag_border_size)
    # 2. add fiduciary markers (tags) in the top-left, top-right,
    # and bottom-left corners
    for tag_number in range(DEFAULT_TAG_NUMBER):
        img = generate_add_tag(img, vft_layout, tag_number, debug=debug)
    return img


vft_layout = None


def graycode_parse(
    img,
    infile,
    frame_num,
    luma_threshold,
    vft_id=None,
    tag_center_locations=None,
    tag_expected_center_locations=None,
    frame_num_debug_output=-1,
    frame_debug_mode="all",
    video_parse_mode=DEFAULT_VIDEO_PARSE_MODE,
    debug=0,
):
    global vft_layout
    bit_stream = None
    if vft_id:
        if vft_layout is None:
            vft_layout = VFTLayout(img.shape[1], img.shape[0], vft_id)

        bit_stream, vft_id = locked_parse(
            img,
            infile,
            frame_num,
            luma_threshold,
            vft_id,
            vft_layout,
            tag_center_locations,
            tag_expected_center_locations,
            frame_num_debug_output=frame_num_debug_output,
            frame_debug_mode=frame_debug_mode,
            video_parse_mode=video_parse_mode,
            debug=debug,
        )
    else:
        bit_stream, vft_id = do_parse(
            img,
            infile,
            frame_num,
            luma_threshold,
            frame_num_debug_output=frame_num_debug_output,
            frame_debug_mode=frame_debug_mode,
            video_parse_mode=video_parse_mode,
            debug=debug,
        )
    # convert gray code in bit_stream to a number
    if bit_stream is not None:
        num_read, status = gray_bitstream_to_num(bit_stream)

        return num_read, status, vft_id
    return None, VFTReading.invalid_graycode, vft_id


# File-based API
def generate_file(width, height, vft_id, tag_border_size, value, outfile, debug):
    # create the tag
    img_luma = generate_graycode(
        width, height, vft_id, tag_border_size, value, debug=debug
    )
    assert img_luma is not None, "error generating VFT"
    # get a full color image
    img = cv2.cvtColor(img_luma, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(outfile, img)


def parse_file(infile, frame_num, luma_threshold, width=0, height=0, debug=0):
    img = cv2.imread(cv2.samples.findFile(infile))
    if width > 0 and height > 0:
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # Img stats
    if debug > 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gmin = np.min(gray)
        gmax = np.max(gray)
        gmean = int(np.mean(gray))
        gstd = int(np.std(gray))
        print(f"min/max luminance: {gmin}/{gmax}, mean: {gmean} +/- {gstd}")
    return graycode_parse(img, infile, frame_num, luma_threshold, debug=debug)


# Generic Number-based API
def generate(width, height, vft_id, tag_border_size, value, debug):
    # 0. start with an empty image with the right background color
    img = np.zeros((height, width, 1), np.uint8)
    pts = np.array([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]])
    cv2.fillPoly(img, pts=[pts], color=COLOR_BACKGROUND)
    # 1. set the layout
    vft_layout = VFTLayout(width, height, vft_id, tag_border_size)
    # 2. add fiduciary markers (tags) in the top-left, top-right,
    # and bottom-left corners
    for tag_number in range(DEFAULT_TAG_NUMBER):
        img = generate_add_tag(img, vft_layout, tag_number, debug=debug)
    # 3. add number code
    # we print <value> starting with the LSB
    value_bit_position = 0
    first_block = True
    if value > 2**vft_layout.numbits:
        raise Exception(
            f"ERROR: {value = } does not fit in {vft_layout.numbits = } (per {vft_id = })"
        )
    for row, col in itertools.product(
        range(vft_layout.numrows), range(vft_layout.numcols)
    ):
        block_id = (row * vft_layout.numcols) + col
        if block_id in vft_layout.tag_block_ids:
            # this is a tag: skip it
            continue
        bit_value = (value >> value_bit_position) & 0x1
        color_white = operator.xor(bit_value == 1, first_block)
        img = generate_add_block(img, vft_layout, block_id, color_white, debug=debug)
        # prepare next block
        first_block = not first_block
        if first_block:
            value_bit_position += 1
            if value_bit_position >= vft_layout.numbits:
                break
    return img


def locked_parse(
    img,
    infile,
    frame_num,
    luma_threshold,
    vft_id=None,
    vft_layout=None,
    tag_center_locations=None,
    tag_expected_center_locations=None,
    frame_num_debug_output=-1,
    frame_debug_mode="all",
    video_parse_mode=DEFAULT_VIDEO_PARSE_MODE,
    debug=0,
):
    img_transformed = None

    # Convert dictionary to list of values in sorted key order
    # (tag_center_locations is now a dict mapping tag_id -> (x, y))
    if isinstance(tag_center_locations, dict):
        tag_center_list = [
            tag_center_locations[k] for k in sorted(tag_center_locations.keys())
        ]
    else:
        tag_center_list = tag_center_locations

    if len(tag_center_list) == 3:
        img_transformed = affine_transformation(
            img, tag_center_list, tag_expected_center_locations, debug=debug
        )

    elif len(tag_center_list) == 4:
        img_transformed = perspective_transformation(
            img, tag_center_list, tag_expected_center_locations, debug=debug
        )
    else:
        return None, vft_id

    if debug > 2:
        cv2.imshow("img", img)
        cv2.imshow("Transformed", img_transformed)
        k = cv2.waitKey(-1)

    bit_stream = parse_read_bits(
        img,
        img_transformed,
        infile,
        frame_num,
        vft_layout,
        luma_threshold,
        frame_num_debug_output=frame_num_debug_output,
        frame_debug_mode=frame_debug_mode,
        tag_center_locations=tag_center_locations,
        tag_expected_center_locations=tag_expected_center_locations,
        video_parse_mode=video_parse_mode,
        debug=debug,
    )
    return bit_stream, vft_id


def do_parse(
    img,
    infile,
    frame_num,
    luma_threshold,
    frame_num_debug_output=-1,
    frame_debug_mode="all",
    video_parse_mode=DEFAULT_VIDEO_PARSE_MODE,
    debug=0,
):
    ids = None
    # 1. get VFT id and tag locations
    vft_id, tag_center_locations, borders, ids = detect_tags(img, debug=debug)
    if tag_center_locations is None:
        if debug > 0:
            print(f"{vft_id=} {tag_center_locations=} {borders=}")

        # could not read the 3x tags properly: stop here
        return None, None

    # 2. set the layout
    height, width, _ = img.shape
    vft_layout = VFTLayout(width, height, vft_id)
    if debug > 2:
        # Convert dictionary to list for iteration
        tag_locs = (
            [tag_center_locations[k] for k in sorted(tag_center_locations.keys())]
            if isinstance(tag_center_locations, dict)
            else tag_center_locations
        )
        for tag in tag_locs:
            cv2.circle(img, (int(tag[0]), int(tag[1])), 5, (0, 255, 0), 2)
        cv2.imshow("Source", img)
        k = cv2.waitKey(-1)

    # 3. apply affine transformation to source image
    # Convert dictionary to list of values in sorted key order
    if isinstance(tag_center_locations, dict):
        tag_center_list = [
            tag_center_locations[k] for k in sorted(tag_center_locations.keys())
        ]
    else:
        tag_center_list = tag_center_locations

    tag_expected_center_locations = vft_layout.get_tag_expected_center_locations()
    if len(tag_center_list) == 3:
        # if we do not have ids tags at this point but we have had them earlier
        # let us just assume all is well.
        # If we have ids points by all means sort them...
        if ids is not None:
            tag_order = [
                nbr for nbr, id_ in enumerate(vft_layout.tag_ids) if id_ in ids
            ]
            tag_expected_center_locations = [
                tag_expected_center_locations[i] for i in tag_order
            ]

        img_transformed = affine_transformation(
            img, tag_center_list, tag_expected_center_locations, debug=debug
        )

    elif len(tag_center_list) == 4:
        img_transformed = perspective_transformation(
            img, tag_center_list, tag_expected_center_locations, debug=debug
        )
    else:
        return None, None

    if debug > 2:
        cv2.imshow("Transformed", img_transformed)
        k = cv2.waitKey(-1)

    # 4. read the bits
    bit_stream = parse_read_bits(
        img,
        img_transformed,
        infile,
        frame_num,
        vft_layout,
        luma_threshold,
        frame_num_debug_output=frame_num_debug_output,
        frame_debug_mode=frame_debug_mode,
        video_parse_mode=video_parse_mode,
        debug=debug,
    )
    return bit_stream, vft_id


@dataclasses.dataclass
class VFTLayout:
    vft_id: str
    width: int
    height: int
    numcols: int
    numrows: int
    numbits: int
    tag_ids: typing.List[int]
    tag_block_ids: typing.List[int]
    x: typing.Dict[int, int]
    y: typing.Dict[int, int]
    block_width: int
    block_height: int
    tag_size: int
    tag_border_size: int

    def __init__(self, width, height, vft_id, tag_border_size=DEFAULT_TAG_BORDER_SIZE):
        self.vft_id = vft_id
        self.numcols, self.numrows, self.tag_ids = VFT_LAYOUT[vft_id]
        # fiduciary markers (tags) located in the top-left, top-right,
        # and bottom-left corners
        self.tag_block_ids = (
            0,
            self.numcols - 1,
            (self.numrows - 1) * self.numcols,
            (self.numrows - 1) * self.numcols + self.numcols - 1,
        )  # why minus -1
        self.numbits = (self.numcols * self.numrows - 3) // 2
        usable_width = (width // self.numcols) * self.numcols
        usable_height = (height // self.numrows) * self.numrows
        self.width = usable_width
        self.height = usable_height
        self.x = [int(i * self.width / self.numcols) for i in range(self.numcols)]
        self.y = [int(i * self.height / self.numrows) for i in range(self.numrows)]
        self.block_width = self.x[1] - self.x[0]
        self.block_height = self.y[1] - self.y[0]
        self.tag_size = min(self.block_width, self.block_height)
        assert (
            tag_border_size >= MIN_TAG_BORDER_SIZE
        ), f"error: tag border size must be at least {MIN_TAG_BORDER_SIZE} ({tag_border_size = })"
        self.tag_border_size = tag_border_size

    def get_colrow(self, block_id):
        col = block_id % self.numcols
        row = block_id // self.numcols
        return col, row

    def get_tag_expected_center_locations(self):
        # top-left
        x0 = self.x[0] + self.block_width / 2
        y0 = self.y[0] + self.block_height / 2
        # top-right
        x1 = self.x[-1] + self.block_width / 2
        y1 = self.y[0] + self.block_height / 2
        # bottom-left
        x2 = self.x[0] + self.block_width / 2
        y2 = self.y[-1] + self.block_height / 2
        # bottom-right
        x3 = self.x[-1] + self.block_width / 2
        y3 = self.y[-1] + self.block_height / 2
        return [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]


def generate_add_tag(img, vft_layout, tag_number, debug=1):
    tag_id = vft_layout.tag_ids[tag_number]
    img_tag = aruco_common.generate_aruco_tag(
        vft_layout.tag_size, tag_id, vft_layout.tag_border_size
    )
    block_id = vft_layout.tag_block_ids[tag_number]
    # get the coordinates
    col, row = vft_layout.get_colrow(block_id)

    x0 = vft_layout.x[col]
    x1 = x0 + vft_layout.tag_size
    y0 = vft_layout.y[row]
    y1 = y0 + vft_layout.tag_size

    # center the coordinates
    # XXX: sure you don't want to move them to the extremes?
    if vft_layout.tag_size < vft_layout.block_width:
        shift = (vft_layout.block_width - vft_layout.tag_size) // 2
        x0 += shift
        x1 += shift
    if vft_layout.tag_size < vft_layout.block_height:
        shift = (vft_layout.block_height - vft_layout.tag_size) // 2
        y0 += shift
        y1 += shift
    # copy it into the main image
    if debug > 1:
        print(
            f"adding tag: {tag_number = } {block_id = } {tag_id = } x = {x0}:{x1} y = {y0}:{y1}"
        )
    img[y0:y1, x0:x1] = img_tag
    return img


def generate_add_block(img, vft_layout, block_id, color_white, debug):
    # get the block coordinates
    col, row = vft_layout.get_colrow(block_id)
    x0 = vft_layout.x[col]
    x1 = x0 + vft_layout.block_width
    y0 = vft_layout.y[row]
    y1 = y0 + vft_layout.block_height
    # color the block
    pts = np.array([[x0, y0], [x0, y1 - 1], [x1 - 1, y1 - 1], [x1 - 1, y0]])
    color = COLOR_WHITE if color_white else COLOR_BLACK
    if debug > 1:
        print(
            f"adding block: {block_id = } {col = } {row = } x = {x0}:{x1} y = {y0}:{y1} {color = }"
        )
    cv2.fillPoly(img, pts=[pts], color=color)
    return img


def get_vft_id(ids):
    for vft_id, value in VFT_LAYOUT.items():
        tag_ids = set(value[2])
        if set(ids).issubset(tag_ids):
            return vft_id
    return None


def get_tag_center_locations(ids, corners, debug=0):
    tag_center_locations = {}
    expected_corner_shape = (1, 4, 2)

    # Debug: print detected IDs
    if debug > 0:
        print(f"DEBUG get_tag_center_locations: detected IDs = {sorted(ids)}")

    for tag_id in sorted(ids):
        i = list(ids).index(tag_id)
        assert (
            corners[i].shape == expected_corner_shape
        ), f"error: invalid corners[{i}]: {corners[i]}"
        # use the center point as tag location
        xt = 0.0
        yt = 0.0
        for corner in corners[i][0]:
            (x, y) = corner
            xt += x
            yt += y
        xt /= 4
        yt /= 4
        tag_center_locations[tag_id] = (xt, yt)

        if debug > 0:
            print(f"  ID {tag_id} -> center location ({xt:.1f}, {yt:.1f})")

    return tag_center_locations


def detect_tags(img, cached_ids=None, cached_corners=None, debug=0):
    """Detect tags in img with optional known tags

    The cached_ids and cached_cornes are lists containing earlier finds.
    This way knowledge can be accumulated (assuming a fixed position).
    """

    # 1. detect tags
    if cached_ids == None or len(cached_ids) < 4:
        corners, ids = aruco_common.detect_aruco_tags(img)
    else:
        corners = cached_corners
        ids = np.asarray(cached_ids, dtype=np.int32).reshape(-1)

    if debug > 2:
        print(f"{corners=} {ids=}")

    if ids is None:
        if debug > 2:
            print("error: cannot detect any tags in image")
        return None, None, None, None

    if cached_ids != None and corners != None:
        for i, corner in enumerate(corners):
            id_ = int(ids[i])
            if id_ not in cached_ids and id_ < 10:
                cached_ids.append(id_)
                cached_corners.append(corner)

        if debug > 1:
            img2 = img.copy()
            if len(ids) > 0:
                img2 = cv2.aruco.drawDetectedMarkers(
                    img2, corners, ids.reshape(-1, 1), (255, 255, 0)
                )

            # draw already detected
            img2 = cv2.aruco.drawDetectedMarkers(
                img2,
                cached_corners,
                np.asarray(cached_ids, dtype=np.int32).reshape(-1, 1),
                (255, 0, 255),
            )
            cv2.imshow("found", img2)
            cv2.waitKey(1)

    if len(ids) < 3:
        if debug > 2:
            print(f"error: image has {len(ids)} tag(s) (should have 3)")
        return None, None, None, None
    else:
        # check tag list last number VFT_LAYOUT last number 2-5
        ids = [int(id_) for id_ in ids if id_ in [0, 1, 2, 3, 4, 5, 6, 7]]

    # 2. make sure they are a valid set
    vft_id = get_vft_id(list(ids))
    if vft_id is None:
        if debug > 0:
            print(f"error: image has invalid tag ids: {set(ids)}")
        return None, None, None, None
    # 3. get the locations
    tag_center_locations = get_tag_center_locations(ids, corners, debug=debug)
    # 4. get the borders
    x0 = x1 = y0 = y1 = None
    for corner in corners:
        for x, y in corner[0]:
            if x0 is None or x0 > x:
                x0 = x
            if x1 is None or x1 < x:
                x1 = x
            if y0 is None or y0 > y:
                y0 = y
            if y1 is None or y1 < y:
                y1 = y
    borders = ((x0, y0), (x1, y1))

    return vft_id, tag_center_locations, borders, ids


def perspective_transformation(
    img, tag_center_locations, tag_expected_center_locations, debug
):
    # process the image
    s0, s1, s2, s3 = tag_center_locations
    d0, d1, d2, d3 = tag_expected_center_locations
    src_locs = np.array([s0, s1, s2, s3]).astype(np.float32)
    dst_locs = np.array([d0, d1, d2, d3]).astype(np.float32)
    transform_matrix = cv2.getPerspectiveTransform(src_locs, dst_locs)
    if debug > 3:
        print(f"  transform_matrix: [{transform_matrix[0]} {transform_matrix[1]}]")
    outimg = cv2.warpPerspective(img, transform_matrix, (img.shape[1], img.shape[0]))
    return outimg


def affine_transformation(
    img, tag_center_locations, tag_expected_center_locations, debug
):
    # process the image
    s0, s1, s2 = tag_center_locations
    # Should never be less than three but occasionally four - skip the last
    d0, d1, d2 = tag_expected_center_locations[0:3]
    src_trio = np.array([s0, s1, s2]).astype(np.float32)
    dst_trio = np.array([d0, d1, d2]).astype(np.float32)
    transform_matrix = cv2.getAffineTransform(src_trio, dst_trio)
    if debug > 3:
        print(f"  transform_matrix: [{transform_matrix[0]} {transform_matrix[1]}]")
    outimg = cv2.warpAffine(img, transform_matrix, (img.shape[1], img.shape[0]))
    return outimg


last_min_diff = -1


def parse_read_bits(
    img_original,
    img_transformed,
    infile,
    frame_num,
    vft_layout,
    luma_threshold,
    frame_num_debug_output=-1,
    frame_debug_mode="all",
    tag_center_locations=None,
    tag_expected_center_locations=None,
    video_parse_mode=DEFAULT_VIDEO_PARSE_MODE,
    debug=0,
):
    global last_min_diff

    # Check if this frame should generate debug output
    if frame_num == frame_num_debug_output and frame_num_debug_output >= 0:
        debug = 2

    # 1. extract the luma
    if len(img_transformed.shape) == 3:
        img_luma = cv2.cvtColor(img_transformed, cv2.COLOR_BGR2GRAY)
    else:
        img_luma = img_transformed

    # 2. read the per-block luma average value
    block_luma_avgs = []
    block_luma_medians = []
    pixels_per_block = vft_layout.block_width * vft_layout.block_height
    for row, col in itertools.product(
        range(vft_layout.numrows), range(vft_layout.numcols)
    ):
        block_id = (row * vft_layout.numcols) + col
        if block_id in vft_layout.tag_block_ids:
            # this is a tag: skip it
            continue
        # get the coordinates
        col, row = vft_layout.get_colrow(block_id)
        x0 = vft_layout.x[col]
        x1 = x0 + vft_layout.block_width
        y0 = vft_layout.y[row]
        y1 = y0 + vft_layout.block_height
        img_luma_block = img_luma[y0:y1, x0:x1]
        # calling np.mean is slower
        block_luma_avg = np.sum(img_luma_block) / pixels_per_block
        block_luma_avgs.append(block_luma_avg)
        block_luma_median = np.nanmedian(img_luma_block)
        block_luma_medians.append(block_luma_median)

    # 3. convert per-block luma averages to bits
    # TODO(chema): what we really want here is an adaptive luma
    # threshold system: If we are getting luma avg values close
    # 0 and 255, we can infer the image quality is pretty good,
    # and therefore use a large threshold. Otherwise, we should
    # resort to a smaller threshold.
    bit_stream = []
    diff = 255

    # Choose which luma values to use based on video_parse_mode
    if video_parse_mode == "median":
        block_luma_values = block_luma_medians
    else:  # default to "average"
        block_luma_values = block_luma_avgs

    # Bit reading decision logic:
    # For each pair of consecutive blocks, compare the block luma values.
    # If the difference is below luma_threshold, the bit is unreadable ("X").
    # Otherwise, bit=1 if luma2>luma1, or bit=0 if luma2<luma1.
    # This threshold-based approach ensures robustness against noise
    # and compression artifacts.
    # We use 2 different parsing modes to get the block luma value:
    # * (1) average: A block luma is the average of the lumas of all
    #   its pixels
    # * (2) median: A block luma is the median of the lumas of all
    #   its pixels
    for luma1, luma2 in zip(block_luma_values[0::2], block_luma_values[1::2]):
        if abs(luma2 - luma1) < luma_threshold:
            bit = "X"
            if debug:
                if abs(luma2 - luma1) < diff:
                    diff = abs(luma2 - luma1)
        elif luma2 > luma1:
            bit = 1
        else:
            bit = 0
        bit_stream.append(bit)
    bit_stream.reverse()

    if debug > 1:
        if diff != last_min_diff:
            print(f"minimum diff was {diff}")
            last_min_diff = diff
        # Compute VFT reading value for debug output
        num_read, status = gray_bitstream_to_num(bit_stream)
        write_annotated_image(
            img_transformed,
            img_original,
            infile,
            frame_num,
            vft_layout,
            frame_debug_mode,
            tag_center_locations,
            tag_expected_center_locations,
            num_read,
            bit_stream,
        )
    return bit_stream


def write_annotated_image_zoom(
    img_transformed,
    infile,
    frame_num,
    vft_layout,
    tag_center_locations,
    tag_expected_center_locations,
    num_read=None,
    bit_stream=None,
):
    """Write annotated VFT image in zoom mode."""
    outfile_zoom = f"{infile}.vft_debug.frame_{frame_num}.zoom.png"
    write_annotated_tag(
        img_transformed,
        vft_layout,
        outfile_zoom,
        mode="zoom",
        tag_center_locations=tag_center_locations,
        tag_expected_center_locations=tag_expected_center_locations,
        num_read=num_read,
        bit_stream=bit_stream,
    )


def write_annotated_image_original(
    img_original,
    infile,
    frame_num,
    vft_layout,
    tag_center_locations,
    tag_expected_center_locations,
    num_read=None,
    bit_stream=None,
):
    """Write annotated VFT image in original mode - shows detected fiducials at full video resolution."""
    outfile_original = f"{infile}.vft_debug.frame_{frame_num}.original.png"

    # Get the original video resolution using ffprobe
    try:
        ffprobe_cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "v:0",
            infile,
        ]
        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
        video_info = json.loads(result.stdout)
        orig_width = video_info["streams"][0]["width"]
        orig_height = video_info["streams"][0]["height"]
    except Exception as e:
        print(f"Warning: Could not get original video resolution: {e}")
        print(f"  Falling back to img_original dimensions")
        orig_height, orig_width = img_original.shape[:2]

    # Extract the specific frame at full resolution using ffmpeg
    try:
        ffmpeg_cmd = [
            "ffmpeg",
            "-i",
            infile,
            "-vf",
            f"select=eq(n\\,{frame_num}),scale={orig_width}:{orig_height}",
            "-vframes",
            "1",
            "-f",
            "image2pipe",
            "-pix_fmt",
            "gray",
            "-vcodec",
            "rawvideo",
            "-",
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        # Read the raw grayscale frame
        img_fullres = np.frombuffer(result.stdout, dtype=np.uint8).reshape(
            (orig_height, orig_width)
        )
    except Exception as e:
        print(f"Warning: Could not extract full-resolution frame: {e}")
        print(f"  Falling back to img_original")
        img_fullres = img_original.copy()
        orig_height, orig_width = img_fullres.shape[:2]

    # Convert grayscale to BGR for color drawing
    if len(img_fullres.shape) == 2:
        img_output = cv2.cvtColor(img_fullres, cv2.COLOR_GRAY2BGR)
    else:
        img_output = img_fullres.copy()

    # Calculate scaling factors from processing resolution to original resolution
    scale_x = orig_width / vft_layout.width
    scale_y = orig_height / vft_layout.height

    # Calculate transformation matrix from detected fiducials if available
    transform_matrix = None
    if tag_center_locations is not None and tag_expected_center_locations is not None:
        # Prepare detected and expected points
        detected_points = []
        expected_points = []

        # Get the full unfiltered expected locations for proper indexing by tag_id
        full_expected_locations = vft_layout.get_tag_expected_center_locations()
        tag_ids = vft_layout.tag_ids

        for tag_id in sorted(tag_center_locations.keys()):
            if tag_id in tag_ids:
                # Map tag_id to its index in the tag_ids list
                idx = tag_ids.index(tag_id)
                detected_points.append(tag_center_locations[tag_id])
                expected_points.append(full_expected_locations[idx])

        print(f"DEBUG Transformation setup:")
        print(f"  Detected fiducials: {len(detected_points)}")
        print(f"  Expected points: {expected_points}")
        print(f"  Detected points: {detected_points}")

        # Print detailed fiducial locations
        if tag_center_locations is not None:
            fiducial_str = ", ".join(
                f"({tag_id}, {int(tag_center_locations[tag_id][0])}, {int(tag_center_locations[tag_id][1])})"
                for tag_id in sorted(tag_center_locations.keys())
            )
            print(f"  Fiducials (processing res): {fiducial_str}")

        if len(detected_points) >= 4:
            # Use perspective transformation for 4 points
            detected_pts = np.array(detected_points, dtype=np.float32)
            expected_pts = np.array(expected_points, dtype=np.float32)
            transform_matrix = cv2.getPerspectiveTransform(expected_pts, detected_pts)
            print(f"  Using perspective transformation (4 points)")
            print(f"  Transform matrix:\n{transform_matrix}")
        elif len(detected_points) == 3:
            # Use affine transformation for 3 points
            detected_pts = np.array(detected_points, dtype=np.float32)
            expected_pts = np.array(expected_points, dtype=np.float32)
            transform_matrix = cv2.getAffineTransform(expected_pts, detected_pts)
            print(f"  Using affine transformation (3 points)")
            print(f"  Transform matrix:\n{transform_matrix}")
        else:
            print(f"  Not enough fiducials for transformation (need at least 3)")

    # Draw all blocks from VFT layout
    block_corners_list = []  # Collect block corners for debug output
    for row, col in itertools.product(
        range(vft_layout.numrows), range(vft_layout.numcols)
    ):
        block_id = (row * vft_layout.numcols) + col
        # Get the block coordinates (at processing resolution) directly from row/col
        x0 = vft_layout.x[col]
        x1 = vft_layout.x[col] + vft_layout.block_width
        y0 = vft_layout.y[row]
        y1 = vft_layout.y[row] + vft_layout.block_height

        # Transform block corners if we have a transformation matrix
        if transform_matrix is not None:
            # Create corner points
            corners = np.array(
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
            )

            # Apply transformation
            if transform_matrix.shape[0] == 3 and transform_matrix.shape[1] == 3:
                # Perspective transformation
                corners_homogeneous = np.hstack([corners, np.ones((4, 1))])
                transformed = corners_homogeneous @ transform_matrix.T
                transformed_corners = transformed[:, :2] / transformed[:, 2:3]
            else:
                # Affine transformation
                corners_homogeneous = np.hstack([corners, np.ones((4, 1))])
                transformed_corners = corners_homogeneous @ transform_matrix.T

            # Scale to original resolution
            transformed_corners[:, 0] *= scale_x
            transformed_corners[:, 1] *= scale_y
            transformed_corners = transformed_corners.astype(np.int32)
        else:
            # No transformation, just scale coordinates
            transformed_corners = np.array(
                [
                    [int(x0 * scale_x), int(y0 * scale_y)],
                    [int(x1 * scale_x), int(y0 * scale_y)],
                    [int(x1 * scale_x), int(y1 * scale_y)],
                    [int(x0 * scale_x), int(y1 * scale_y)],
                ],
                dtype=np.int32,
            )

        # Save first few blocks for debug output (to avoid too much output)
        if block_id < 5:  # Only save first 5 blocks
            block_corners_list.append(
                (
                    block_id,
                    transformed_corners[0][0],
                    transformed_corners[0][1],
                    transformed_corners[2][0],
                    transformed_corners[2][1],
                )
            )

        if block_id in vft_layout.tag_block_ids:
            # This is a fiducial block - draw circle with X if detected
            if tag_center_locations is not None:
                # Find which tag ID this block corresponds to
                fiducial_index = vft_layout.tag_block_ids.index(block_id)
                tag_id = vft_layout.tag_ids[fiducial_index]

                # Check if this tag was actually detected
                if tag_id in tag_center_locations:
                    center_x, center_y = tag_center_locations[tag_id]

                    # Scale coordinates from processing resolution to original resolution
                    scaled_x = int(center_x * scale_x)
                    scaled_y = int(center_y * scale_y)

                    # Use a reasonable circle radius based on image size
                    circle_radius = max(20, min(orig_height, orig_width) // 50)

                    # Draw filled circle at detected position
                    cv2.circle(
                        img_output,
                        (scaled_x, scaled_y),
                        circle_radius,
                        COLOR_FIDUCIAL_CIRCLE,
                        -1,
                    )

                    # Draw X through the circle
                    line_len = int(circle_radius * 1.5)
                    line_thickness = max(3, circle_radius // 10)
                    cv2.line(
                        img_output,
                        (scaled_x - line_len, scaled_y - line_len),
                        (scaled_x + line_len, scaled_y + line_len),
                        COLOR_FIDUCIAL_ARROW,
                        line_thickness,
                    )
                    cv2.line(
                        img_output,
                        (scaled_x - line_len, scaled_y + line_len),
                        (scaled_x + line_len, scaled_y - line_len),
                        COLOR_FIDUCIAL_ARROW,
                        line_thickness,
                    )

                    # Draw tag ID label
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = circle_radius / 20.0
                    font_thickness = max(2, circle_radius // 15)
                    cv2.putText(
                        img_output,
                        f"ID:{tag_id}",
                        (scaled_x + line_len, scaled_y - line_len),
                        font,
                        font_scale,
                        COLOR_FIDUCIAL_TEXT,
                        font_thickness,
                    )
        else:
            # This is a data block - draw rectangle using transformed corners
            rect_thickness = 1
            cv2.polylines(
                img_output,
                [transformed_corners],
                isClosed=True,
                color=COLOR_BLOCK_BORDER,
                thickness=rect_thickness,
            )

            # Add bit value annotation for each data block
            if bit_stream is not None:
                # Count how many data blocks (non-tag blocks) come before this block_id
                data_block_index = sum(
                    1 for i in range(block_id) if i not in vft_layout.tag_block_ids
                )
                bit_index_before_reversal = data_block_index // 2
                if bit_index_before_reversal < len(bit_stream):
                    # bit_stream string is formatted MSB-first (leftmost) to LSB-last (rightmost)
                    # bit 0 (LSB) is the last character: bit_stream[-1]
                    # bit 1 is second-to-last: bit_stream[-2], etc.
                    bit_position = bit_index_before_reversal
                    bit_value = bit_stream[-(bit_position + 1)]

                    # Calculate center of block for text placement
                    center_x = int(
                        (transformed_corners[0][0] + transformed_corners[2][0]) / 2
                    )
                    center_y = int(
                        (transformed_corners[0][1] + transformed_corners[2][1]) / 2
                    )

                    # Scale text based on image size with larger, bolder font
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = min(orig_height, orig_width) / 1600.0
                    font_thickness = max(2, int(font_scale * 5))

                    # Draw bit annotation
                    cv2.putText(
                        img_output,
                        f"{bit_position}:{bit_value}",
                        (center_x, center_y),
                        font,
                        font_scale,
                        COLOR_BLOCK_TEXT,
                        font_thickness,
                    )

    # Add VFT value and bit stream overlay at the bottom-left of the image
    if num_read is not None and bit_stream is not None:
        img_height, img_width = img_output.shape[:2]

        # Format the text
        text_line1 = f"Value: {num_read}"
        text_line2 = f"Bits: {bit_stream}"

        # Set font parameters - match the per-block annotation style
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Use same scaling as per-block annotations for original mode
        font_scale = min(orig_height, orig_width) / 1600.0
        font_thickness = max(2, int(font_scale * 5))
        # Use same color as per-block annotations
        font_color = COLOR_BLOCK_TEXT

        # Get text size for positioning
        (text_width1, text_height1), baseline1 = cv2.getTextSize(
            text_line1, font, font_scale, font_thickness
        )
        (text_width2, text_height2), baseline2 = cv2.getTextSize(
            text_line2, font, font_scale, font_thickness
        )

        # Position text at bottom-left with padding to avoid overlap
        padding = 15
        x1 = padding
        y1 = img_height - text_height2 - baseline2 - padding * 2
        x2 = padding
        y2 = img_height - padding

        # Draw text with black background for visibility
        # Background rectangles
        cv2.rectangle(
            img_output,
            (x1 - 5, y1 - text_height1 - 5),
            (x1 + text_width1 + 5, y1 + baseline1 + 5),
            (0, 0, 0),
            -1,
        )
        cv2.rectangle(
            img_output,
            (x2 - 5, y2 - text_height2 - 5),
            (x2 + text_width2 + 5, y2 + baseline2 + 5),
            (0, 0, 0),
            -1,
        )

        # Text
        cv2.putText(
            img_output,
            text_line1,
            (x1, y1),
            font,
            font_scale,
            font_color,
            font_thickness,
        )
        cv2.putText(
            img_output,
            text_line2,
            (x2, y2),
            font,
            font_scale,
            font_color,
            font_thickness,
        )

    print(f"DEBUG write_annotated_image_original:")
    print(f"  VFT reading value: {num_read}")
    if bit_stream is not None:
        bit_stream_str = "".join(str(b) for b in bit_stream)
        print(f"  VFT bit stream: {bit_stream_str}")
    print(f"  Original video resolution: {orig_width}x{orig_height}")
    print(f"  Processing resolution: {vft_layout.width}x{vft_layout.height}")
    print(f"  Scale factors: {scale_x:.2f}x, {scale_y:.2f}x")
    print(f"  Output image shape: {img_output.shape}")

    # Print block corners debug info
    if len(block_corners_list) > 0:
        blocks_str = ", ".join(
            f"({bid}, {x0}, {y0}, {x1}, {y1})"
            for bid, x0, y0, x1, y1 in block_corners_list
        )
        print(f"  Blocks (first 5, original res): {blocks_str}")

    cv2.imwrite(outfile_original, img_output)


def write_annotated_image(
    img_transformed,
    img_original,
    infile,
    frame_num,
    vft_layout,
    frame_debug_mode,
    tag_center_locations,
    tag_expected_center_locations,
    num_read=None,
    bit_stream=None,
):
    """Write annotated VFT image based on debug mode."""
    if frame_debug_mode in ("zoom", "all"):
        write_annotated_image_zoom(
            img_transformed,
            infile,
            frame_num,
            vft_layout,
            tag_center_locations,
            tag_expected_center_locations,
            num_read,
            bit_stream,
        )
    if frame_debug_mode in ("original", "all"):
        write_annotated_image_original(
            img_original,
            infile,
            frame_num,
            vft_layout,
            tag_center_locations,
            tag_expected_center_locations,
            num_read,
            bit_stream,
        )


def write_annotated_tag(
    img_transformed,
    vft_layout,
    outfile,
    mode="zoom",
    tag_center_locations=None,
    tag_expected_center_locations=None,
    num_read=None,
    bit_stream=None,
):
    """Write annotated VFT tag image with fiducials and data blocks highlighted.

    Args:
        img_transformed: Input image (already transformed)
        vft_layout: VFT layout object
        outfile: Output filename
        mode: "zoom" or "original" - controls annotation style
        tag_center_locations: Actual detected fiducial positions
        tag_expected_center_locations: Expected fiducial positions
        num_read: VFT integer value read
        bit_stream: Bit stream string with values 0, 1, or X
    """
    # Debug: print fiducial mapping info
    print(f"DEBUG write_annotated_tag:")
    print(f"  vft_layout.tag_ids = {vft_layout.tag_ids}")
    print(f"  vft_layout.tag_block_ids = {vft_layout.tag_block_ids}")
    if tag_center_locations:
        print(f"  tag_center_locations has {len(tag_center_locations)} entries")
    if tag_expected_center_locations:
        print(
            f"  tag_expected_center_locations has {len(tag_expected_center_locations)} entries"
        )

    # Work with a copy to avoid modifying input
    img_output = img_transformed.copy()

    # Convert grayscale to BGR for color drawing
    if len(img_output.shape) == 2:
        img_output = cv2.cvtColor(img_output, cv2.COLOR_GRAY2BGR)

    # Calculate circle radius based on block dimensions (33% of block size)
    circle_radius = min(vft_layout.block_width, vft_layout.block_height) // 3

    # Add VFT value and bit stream overlay at the top of the image
    if num_read is not None and bit_stream is not None:
        img_height, img_width = img_output.shape[:2]

        # Format the text
        text_line1 = f"Value: {num_read}"
        text_line2 = f"Bits: {bit_stream}"

        # Set font parameters - match the per-block annotation style
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Use same scaling as per-block annotations for zoom mode
        font_scale = min(vft_layout.block_width, vft_layout.block_height) / 150.0
        font_thickness = max(2, int(font_scale * 5))
        # Use same color as per-block annotations
        font_color = COLOR_BLOCK_TEXT

        # Get text size for positioning
        (text_width1, text_height1), baseline1 = cv2.getTextSize(
            text_line1, font, font_scale, font_thickness
        )
        (text_width2, text_height2), baseline2 = cv2.getTextSize(
            text_line2, font, font_scale, font_thickness
        )

        # Position text at bottom-left with padding to avoid overlap
        padding = 15
        x1 = padding
        y1 = img_height - text_height2 - baseline2 - padding * 2
        x2 = padding
        y2 = img_height - padding

        # Draw text with black background for visibility
        # Background rectangles
        cv2.rectangle(
            img_output,
            (x1 - 5, y1 - text_height1 - 5),
            (x1 + text_width1 + 5, y1 + baseline1 + 5),
            (0, 0, 0),
            -1,
        )
        cv2.rectangle(
            img_output,
            (x2 - 5, y2 - text_height2 - 5),
            (x2 + text_width2 + 5, y2 + baseline2 + 5),
            (0, 0, 0),
            -1,
        )

        # Text
        cv2.putText(
            img_output,
            text_line1,
            (x1, y1),
            font,
            font_scale,
            font_color,
            font_thickness,
        )
        cv2.putText(
            img_output,
            text_line2,
            (x2, y2),
            font,
            font_scale,
            font_color,
            font_thickness,
        )

    for row, col in itertools.product(
        range(vft_layout.numrows), range(vft_layout.numcols)
    ):
        block_id = (row * vft_layout.numcols) + col
        # get the coordinates
        col, row = vft_layout.get_colrow(block_id)
        x0 = int(vft_layout.x[col])
        x1 = int(vft_layout.x[col] + vft_layout.block_width)
        y0 = int(vft_layout.y[row])
        y1 = int(vft_layout.y[row] + vft_layout.block_height)

        if block_id in vft_layout.tag_block_ids:
            # Only draw fiducials that were actually detected
            if tag_center_locations is not None:
                # Find which tag ID this block corresponds to
                fiducial_index = vft_layout.tag_block_ids.index(block_id)
                tag_id = vft_layout.tag_ids[fiducial_index]

                # Check if this tag was actually detected
                if tag_id in tag_center_locations:
                    # This fiducial was detected - draw circle with X
                    center_x = (x0 + x1) // 2
                    center_y = (y0 + y1) // 2

                    # Draw filled circle
                    cv2.circle(
                        img_output,
                        (center_x, center_y),
                        circle_radius,
                        COLOR_FIDUCIAL_CIRCLE,
                        -1,
                    )

                    # Draw X through the circle (two diagonal lines)
                    line_len = int(circle_radius * 1.5)
                    line_thickness = max(3, circle_radius // 10)
                    cv2.line(
                        img_output,
                        (center_x - line_len, center_y - line_len),
                        (center_x + line_len, center_y + line_len),
                        COLOR_FIDUCIAL_ARROW,
                        line_thickness,
                    )  # White X
                    cv2.line(
                        img_output,
                        (center_x - line_len, center_y + line_len),
                        (center_x + line_len, center_y - line_len),
                        COLOR_FIDUCIAL_ARROW,
                        line_thickness,
                    )  # White X
            else:
                # No tag_center_locations provided - draw all fiducials
                center_x = (x0 + x1) // 2
                center_y = (y0 + y1) // 2
                cv2.circle(
                    img_output,
                    (center_x, center_y),
                    circle_radius,
                    COLOR_FIDUCIAL_CIRCLE,
                    -1,
                )
                line_len = int(circle_radius * 1.5)
                line_thickness = max(3, circle_radius // 10)
                cv2.line(
                    img_output,
                    (center_x - line_len, center_y - line_len),
                    (center_x + line_len, center_y + line_len),
                    COLOR_FIDUCIAL_ARROW,
                    line_thickness,
                )
                cv2.line(
                    img_output,
                    (center_x - line_len, center_y + line_len),
                    (center_x + line_len, center_y - line_len),
                    COLOR_FIDUCIAL_ARROW,
                    line_thickness,
                )
        else:
            # This is a data block: draw green rectangle
            rect_thickness = 1
            cv2.rectangle(
                img_output, (x0, y0), (x1, y1), COLOR_BLOCK_BORDER, rect_thickness
            )

            # Add bit value annotation for each data block
            if bit_stream is not None:
                # Count how many data blocks (non-tag blocks) come before this block_id
                data_block_index = sum(
                    1 for i in range(block_id) if i not in vft_layout.tag_block_ids
                )
                bit_index_before_reversal = data_block_index // 2
                if bit_index_before_reversal < len(bit_stream):
                    # bit_stream string is formatted MSB-first (leftmost) to LSB-last (rightmost)
                    # bit 0 (LSB) is the last character: bit_stream[-1]
                    # bit 1 is second-to-last: bit_stream[-2], etc.
                    bit_position = bit_index_before_reversal
                    bit_value = bit_stream[-(bit_position + 1)]

                    # Calculate center of block for text placement
                    center_x = (x0 + x1) // 2
                    center_y = (y0 + y1) // 2

                    # Scale text based on block size with larger, bolder font
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # Use smaller font for zoom mode
                    scale_divisor = 150.0 if mode == "zoom" else 40.0
                    font_scale = (
                        min(vft_layout.block_width, vft_layout.block_height)
                        / scale_divisor
                    )
                    font_thickness = max(2, int(font_scale * 5))

                    # Draw bit annotation
                    cv2.putText(
                        img_output,
                        f"{bit_position}:{bit_value}",
                        (center_x, center_y),
                        font,
                        font_scale,
                        COLOR_BLOCK_TEXT,
                        font_thickness,
                    )

    cv2.imwrite(outfile, img_output)


def bit_stream_to_number(bit_stream):
    num = 0
    for bit in bit_stream:
        num = num << 1 | bit
    return num


previous_value = -1


def gray_bitstream_to_num(bit_stream):
    """
    @brief Converts a bitstream read from a metiq image into a number.

    @param[in] bit_stream Bit stream resulting from parsing a metiq image.
      Parameter is a string, so value can be "10001101001010", "1X0010010",
      etc.

    @return Conversion output.

    If the bit_stream does not have any failed bits (no "X" values), the
    function converts the bitstream to a gray code number, and then to a
    conventional number.

    If the bit_stream has 2+ failed bits, then it gives up and returns
    "invalid_graycode".

    If there is exactly 1 failed bit, then it checks the previous value,
    and tests replacing the "X" with both "0" and "1". It chooses the value
    with the smallest absolute difference to the previous value, and marks
    it as "single_graycode". If no previous value, it returns the average
    of the 2 possible values.
    """
    global previous_value
    if bit_stream is None:
        # broken case: no bit stream at all
        return None, VFTReading.no_input
    if bit_stream.count("X") == 0:
        # perfect case: can read all bits
        gray_num = bit_stream_to_number(bit_stream)
        previous_value = graycode.gray_code_to_tc(gray_num)
        return previous_value, VFTReading.ok
    elif bit_stream.count("X") > 1:
        # broken case: cannot read 2+ bits
        return None, VFTReading.invalid_graycode
    # slightly degenerated case: a single non-read bit
    b0 = [0 if b == "X" else b for b in bit_stream]
    g0 = bit_stream_to_number(b0)
    n0 = graycode.gray_code_to_tc(g0)
    b1 = [1 if b == "X" else b for b in bit_stream]
    g1 = bit_stream_to_number(b1)
    n1 = graycode.gray_code_to_tc(g1)
    if previous_value > -1:  # no previous value
        d0 = abs(n0 - previous_value)
        d1 = abs(n1 - previous_value)
        if d0 < d1 and d0 <= 1:
            previous_value = n0
            return n0, VFTReading.single_graycode
        elif d1 <= 1:
            previous_value = n1
            return n1, VFTReading.single_graycode
    elif abs(n0 - n1) == 1:
        # error produces consecutive number
        previous_value = (n1 + n0) / 2
        return previous_value, VFTReading.single_graycode
    return None, VFTReading.single_graycode_unfixable


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

    class ImageSizeAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            namespace.width, namespace.height = [int(v) for v in values[0].split("x")]

    parser.add_argument(
        "--size",
        action=ImageSizeAction,
        nargs=1,
        help="use <width>x<height>",
    )
    parser.add_argument(
        "--vft-id",
        type=str,
        nargs="?",
        default=default_values["vft_id"],
        choices=VFT_IDS,
        help="%s" % (" | ".join("{}".format(k) for k in VFT_IDS)),
    )
    parser.add_argument(
        "--border-size",
        action="store",
        type=int,
        dest="tag_border_size",
        default=default_values["tag_border_size"],
        metavar="BORDER_SIZE",
        help=("tag border size (default: %i)" % default_values["tag_border_size"]),
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
        "--value",
        action="store",
        type=int,
        dest="value",
        default=default_values["value"],
        metavar="VALUE",
        help=("use VALUE value (width/height) (default: %i)" % default_values["value"]),
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
        if options.outfile == "-":
            options.outfile = "/dev/fd/1"
        assert options.outfile is not None, "error: need a valid output file"
        # do something
        generate_file(
            options.width,
            options.height,
            options.vft_id,
            options.tag_border_size,
            options.value,
            options.outfile,
            debug=options.debug,
        )

    elif options.func == "parse":
        # get infile
        if options.infile == "-":
            options.infile = "/dev/fd/0"
        assert options.infile is not None, "error: need a valid input file"
        num_read, status, vft_id = parse_file(
            options.infile,
            0,  # frame_num,
            options.luma_threshold,
            options.width,
            options.height,
            debug=options.debug,
        )
        print(f"read: {num_read = } {status = } ({vft_id = })")


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
