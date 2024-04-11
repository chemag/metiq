#!/usr/bin/env python3

"""vft.py module description.

A VFT (Video Fine-Grained Timing) 2D Barcode Library.
"""


import argparse
import cv2
import dataclasses
import graycode
import itertools
import operator
import random
import sys
import typing
import numpy as np

import aruco_common
import time
import copy

__version__ = "0.1"


class NoValidTag(Exception):
    def __init__(self):
        super().__init__("error: frame has no valid set of tags")


class InvalidGrayCode(Exception):
    def __init__(self, message):
        super().__init__(message)


class SingleGraycodeBitError(Exception):
    def __init__(self, message):
        super().__init__(message)


VFT_IDS = ("9x8", "9x6", "7x5", "5x4")
DEFAULT_VFT_ID = "7x5"
DEFAULT_TAG_BORDER_SIZE = 2
DEFAULT_LUMA_THRESHOLD = 20
DEFAULT_TAG_NUMBER = 4

VFT_LAYOUT = {
    # "vft_id": [numcols, numrows, (aruco_tag_0, aruco_tag_1, aruco_tag_2)],
    "7x5": [7, 5, (0, 1, 2, 7)],  # 16 bits
    "5x4": [5, 4, (0, 1, 3, 8)],  # 8 bits
    "9x8": [9, 8, (0, 1, 4, 9)],  # 34 bits
    "9x6": [9, 6, (0, 1, 5, 10)],  # 25 bits
}


# use fiduciarial markers ("tags") from this dictionary
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

COLOR_BLACK = (0, 0, 0)
COLOR_BACKGROUND = (128, 128, 128)
COLOR_WHITE = (255, 255, 255)

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
    return generate(width, height, vft_id, tag_border_size, graycode_value, debug)


def draw_tags(img, vft_id, tag_border_size, debug):
    global vft_layout
    width = img.shape[1]
    height = img.shape[0]
    if vft_layout is None:
        vft_layout = VFTLayout(width, height, vft_id, tag_border_size)
    # 2. add fiduciary markers (tags) in the top-left, top-right,
    # and bottom-left corners
    for tag_number in range(DEFAULT_TAG_NUMBER):
        img = generate_add_tag(img, vft_layout, tag_number, debug)

    return img


vft_layout = None


def graycode_parse(
    img,
    luma_threshold,
    vft_id=None,
    tag_center_locations=None,
    tag_expected_center_locations=None,
    debug=0,
):
    global vft_layout
    bit_stream = None
    if vft_id:
        if vft_layout is None:
            vft_layout = VFTLayout(img.shape[1], img.shape[0], vft_id)

        bit_stream = locked_parse(
            img,
            luma_threshold,
            vft_id,
            vft_layout,
            tag_center_locations,
            tag_expected_center_locations,
            debug,
        )
    else:
        bit_stream, vft_id = do_parse(img, luma_threshold, debug=debug)
    # convert gray code in bit_stream to a number
    num_read = gray_bitstream_to_num(bit_stream)

    return num_read, vft_id


# File-based API
def generate_file(width, height, vft_id, tag_border_size, value, outfile, debug):
    # create the tag
    img_luma = generate_graycode(width, height, vft_id, tag_border_size, value, debug)
    assert img_luma is not None, "error generating VFT"
    # get a full color image
    img = cv2.cvtColor(img_luma, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(outfile, img)


def parse_file(infile, luma_threshold, width=0, height=0, debug=0):
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
    return graycode_parse(img, luma_threshold, debug)


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
        img = generate_add_tag(img, vft_layout, tag_number, debug)
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
        img = generate_add_block(img, vft_layout, block_id, color_white, debug)
        # prepare next block
        first_block = not first_block
        if first_block:
            value_bit_position += 1
            if value_bit_position >= vft_layout.numbits:
                break
    return img


def locked_parse(
    img,
    luma_threshold,
    vft_id=None,
    vft_layout=None,
    tag_center_locations=None,
    tag_expected_center_locations=None,
    debug=0,
):
    img_transformed = None
    if len(tag_center_locations) == 3:
        img_transformed = affine_transformation(
            img, tag_center_locations, tag_expected_center_locations, debug=debug
        )

    elif len(tag_center_locations) == 4:
        img_transformed = perspective_transformation(
            img, tag_center_locations, tag_expected_center_locations, debug=debug
        )
    else:
        return None, vft_id

    if debug > 2:
        cv2.imshow("img", img)
        cv2.imshow("Transformed", img_transformed)
        k = cv2.waitKey(-1)

    bit_stream = parse_read_bits(
        img_transformed, vft_layout, luma_threshold, debug=debug
    )
    return bit_stream


def do_parse(img, luma_threshold, debug=0):
    ids = None

    # 1. get VFT id and tag locations
    vft_id, tag_center_locations, borders, ids = detect_tags(img, debug=debug)
    if tag_center_locations is None:
        if debug > 0:
            print(f"{vft_id=} {tag_center_locations=} {borders=}")

        # could not read the 3x tags properly: stop here
        raise NoValidTag()
        return None, None

    # 2. set the layout
    height, width, _ = img.shape
    vft_layout = VFTLayout(width, height, vft_id)
    if debug > 2:
        for tag in tag_center_locations:
            cv2.circle(img, (int(tag[0]), int(tag[1])), 5, (0, 255, 0), 2)
        cv2.imshow("Source", img)
        k = cv2.waitKey(-1)

    # 3. apply affine transformation to source image
    tag_expected_center_locations = vft_layout.get_tag_expected_center_locations()
    if len(tag_center_locations) == 3:
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
            img, tag_center_locations, tag_expected_center_locations, debug=debug
        )

    elif len(tag_center_locations) == 4:
        img_transformed = perspective_transformation(
            img, tag_center_locations, tag_expected_center_locations, debug=debug
        )
    else:
        return None, None

    if debug > 2:
        cv2.imshow("Transformed", img_transformed)
        k = cv2.waitKey(-1)

    # 4. read the bits
    bit_stream = parse_read_bits(
        img_transformed, vft_layout, luma_threshold, debug=debug
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
    tag_center_locations = []
    expected_corner_shape = (1, 4, 2)
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
        tag_center_locations.append((xt, yt))
    return tag_center_locations


def detect_tags(img, debug):
    # 1. detect tags
    corners, ids = aruco_common.detect_aruco_tags(img)
    if debug > 2:
        print(f"{corners=} {ids=}")

    if ids is None:
        if debug > 2:
            print("error: cannot detect any tags in image")
        return None, None, None, None
    if len(ids) < 3:
        if debug > 2:
            print(f"error: image has {len(ids)} tag(s) (should have 3)")
        if debug > 2:
            tag_center_locations = get_tag_center_locations(ids, corners, debug)
            for tag in tag_center_locations:
                cv2.circle(img, (int(tag[0]), int(tag[1])), 5, (0, 255, 0), 2)
                cv2.imshow("Failed identification, showing succesful ids", img)
                k = cv2.waitKey(-1)
        return None, None, None, None
    if len(ids) >= 3:
        if debug > 2:
            print(f"error: image has {len(ids)} tag(s) (should have 3)")
        # check tag list last number VFT_LAYOUT last number 2-5
        ids = [id[0] for id in ids if id in [0, 1, 2, 3, 4, 5, 6, 7]]

    # 2. make sure they are a valid set
    vft_id = get_vft_id(list(ids))
    if vft_id is None:
        if debug > 0:
            print(f"error: image has invalid tag ids: {set(ids)}")
        return None, None, None, None
    # 3. get the locations
    tag_center_locations = get_tag_center_locations(ids, corners, debug)
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
    img, tag_center_locations, tag_expected_locations, debug
):
    # process the image
    s0, s1, s2, s3 = tag_center_locations
    d0, d1, d2, d3 = tag_expected_locations
    src_locs = np.array([s0, s1, s2, s3]).astype(np.float32)
    dst_locs = np.array([d0, d1, d2, d3]).astype(np.float32)
    transform_matrix = cv2.getPerspectiveTransform(src_locs, dst_locs)
    if debug > 3:
        print(f"  transform_matrix: [{transform_matrix[0]} {transform_matrix[1]}]")
    outimg = cv2.warpPerspective(img, transform_matrix, (img.shape[1], img.shape[0]))
    return outimg


def affine_transformation(img, tag_center_locations, tag_expected_locations, debug):
    # process the image
    s0, s1, s2 = tag_center_locations
    d0, d1, d2 = tag_expected_locations
    src_trio = np.array([s0, s1, s2]).astype(np.float32)
    dst_trio = np.array([d0, d1, d2]).astype(np.float32)
    transform_matrix = cv2.getAffineTransform(src_trio, dst_trio)
    if debug > 3:
        print(f"  transform_matrix: [{transform_matrix[0]} {transform_matrix[1]}]")
    outimg = cv2.warpAffine(img, transform_matrix, (img.shape[1], img.shape[0]))
    return outimg


last_min_diff = -1


def parse_read_bits(img, vft_layout, luma_threshold, debug):
    global last_min_diff
    # 1. extract the luma
    img_luma = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2. read the per-block luma average value
    block_luma_avgs = []
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
        block_luma_avg = np.mean(img_luma_block)
        block_luma_avgs.append(block_luma_avg)
    # 3. convert per-block luma averages to bits
    # TODO(chema): what we really want here is an adaptive luma
    # threshold system: If we are getting luma avg values close
    # 0 and 255, we can infer the image quality is pretty good,
    # and therefore use a large threshold. Otherwise, we should
    # resort to a smaller threshold.
    bit_stream = []
    diff = 255
    for luma1, luma2 in zip(block_luma_avgs[0::2], block_luma_avgs[1::2]):
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
        # 4. write annotated image to file
        outfile = "/tmp/vft_debug." + "".join(str(bit) for bit in bit_stream) + ".png"
        if diff != last_min_diff:
            print(f"minimum diff was {diff}")
            last_min_diff = diff
        write_annotated_tag(img, vft_layout, outfile)
    return bit_stream


def write_annotated_tag(img, vft_layout, outfile):
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
        color = (
            random.randrange(256),
            random.randrange(256),
            random.randrange(256),
        )
        # pts = np.array([[x0, y0], [x0, y1 - 1], [x1 - 1, y1 - 1], [x1 - 1, y0]])
        # cv2.fillPoly(img, pts=[pts], color=color)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 10)
    cv2.imwrite(outfile, img)


def bit_stream_to_number(bit_stream):
    num = 0
    for bit in bit_stream:
        num = num << 1 | bit
    return num


def gray_bitstream_to_num(bit_stream):
    if bit_stream is None:
        return None
    if bit_stream.count("X") == 0:
        gray_num = bit_stream_to_number(bit_stream)
        return graycode.gray_code_to_tc(gray_num)
    elif bit_stream.count("X") > 1:
        raise InvalidGrayCode(f"{bit_stream = }")
        return None
    # slightly degenerated case: a single non-read bit
    b0 = [0 if b == "X" else b for b in bit_stream]
    g0 = bit_stream_to_number(b0)
    n0 = graycode.gray_code_to_tc(g0)
    b1 = [1 if b == "X" else b for b in bit_stream]
    g1 = bit_stream_to_number(b1)
    n1 = graycode.gray_code_to_tc(g1)
    if abs(n0 - n1) == 1:
        # error produces consecutive numbers
        return (n1 + n0) / 2
    raise SingleGraycodeBitError(f"{bit_stream = }")
    return None


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
        assert options.infile is not None, "error: need a valid in file"
        num_read, vft_id = parse_file(
            options.infile,
            options.luma_threshold,
            options.width,
            options.height,
            debug=options.debug,
        )
        print(f"read: {num_read = } ({vft_id = })")


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
