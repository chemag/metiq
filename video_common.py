#!/usr/bin/env python3

"""video_common.py module description."""

import dataclasses


DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_LUMA_THRESHOLD = 20
PIXEL_FORMAT_CHOICES = ["yuv420p", "nv12", "nv21"]
DEFAULT_PIXEL_FORMAT = None


NUM_BITS = 16
MIN_TAG_SIZE = 16


def binary_to_int(binary_array):
    rval = 0
    for val in binary_array:
        rval = (rval << 1) | int(val)
    return rval


def num_to_gray(num, bits):
    bnum = bin(num)[2:]
    bnum = bnum.zfill(bits)
    done = False
    position = bits - 2
    gnum = [0] * bits
    while not done and position > 0:
        val = int(bnum[position])
        if val == 1:
            val = int(bnum[position + 1])
            gnum[position + 1] = 1 - val
        else:
            gnum[position + 1] = bnum[position + 1]
        position -= 1
    return binary_to_int(gnum)


def gray_to_num(num, bits):
    gnum = bin(num)[2:]
    gnum = gnum.zfill(bits)
    done = False
    position = bits - 1
    bnum = [0] * bits
    while not done and position > 0:
        val = 0
        for posval in gnum[0:position]:
            val = (val + int(posval)) % 2
        if val == 1:
            val = int(bnum[position])
            bnum[position] = 1 - int(gnum[position])
        else:
            bnum[position] = gnum[position]
        position -= 1
    return binary_to_int(bnum)


def bit_stream_to_number(bit_stream):
    num = 0
    for bit in bit_stream:
        num = num << 1 | bit
    return num


def gray_bitstream_to_num(bit_stream, bits):
    if bit_stream.count("X") == 0:
        gray_num = bit_stream_to_number(bit_stream)
        return gray_to_num(gray_num, bits)
    elif bit_stream.count("X") == 1:
        # support slightly degenerated cases
        b0 = [0 if b == "X" else b for b in bit_stream]
        g0 = bit_stream_to_number(b0)
        n0 = gray_to_num(g0, bits)
        b1 = [1 if b == "X" else b for b in bit_stream]
        g1 = bit_stream_to_number(b1)
        n1 = gray_to_num(g1, bits)
        if abs(n0 - n1) != 1:
            # error does not produce consecutive numbers
            return None
        return (n1 + n0) / 2
    elif "X" in bit_stream:
        print(f"warn: invalid gray code read ({bit_stream = }")
        return None


@dataclasses.dataclass
class ImageInfo:
    height: int
    width: int
    tag_size: int
    tag_border_size: int
    tag_x: "dict(int -> int)"
    tag_y: "dict(int -> int)"
    gb_num_bits: int
    gb_x: "dict(int -> int)"
    gb_y: "dict(int -> int)"
    gb_boxsize: int

    def __init__(self, width, height, num_bits, border_size=0):
        self.width = width
        self.height = height
        # fiduciary markers' size is 1/8th of the smaller resolution
        self.tag_size = min(self.height >> 3, self.width >> 3)
        assert (
            self.tag_size > MIN_TAG_SIZE
        ), f"error: resolution ({self.width}x{self.height} too small to add markers"
        self.tag_border_size = border_size
        # fiduciary markers locations
        #  +--------------------------+
        #  |                          |
        #  | (x0, y0)        (x1, y1) |
        #  |   []               []    |
        #  |    [   gray code   ]     |
        #  |   []                     |
        #  | (x2, y2)                 |
        #  |                          |
        #  +--------------------------+
        x0 = self.width >> 3
        self.tag_x = {
            0: x0,
            1: self.width - x0 - self.tag_size,
            2: x0,
        }
        y0 = self.height >> 3
        self.tag_y = {
            0: y0,
            1: y0,
            2: self.height - y0 - self.tag_size,
        }
        # gray block location
        self.gb_num_bits = num_bits
        self.gb_boxsize = int(self.width // self.gb_num_bits)
        x = 0
        yc = self.height // 2
        self.gb_x = {}
        self.gb_y = {}
        for i in range(self.gb_num_bits):
            self.gb_x[i] = x
            x += self.gb_boxsize
            self.gb_y[i] = yc

    def get_marker_center(self, marker_id):
        # note that the marker's center
        xc = self.tag_x[marker_id] + ((self.tag_size - 1) / 2)
        yc = self.tag_y[marker_id] + ((self.tag_size - 1) / 2)
        return xc, yc

    def get_gray_block_location(self):
        x0 = self.gb_x[0]
        xn = self.gb_x[self.gb_num_bits - 1] + self.gb_boxsize
        yt = self.gb_y[0] - self.gb_boxsize
        yc = self.gb_y[0]
        yb = self.gb_y[0] + self.gb_boxsize
        return x0, xn, yt, yc, yb
