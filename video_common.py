#!/usr/bin/env python3

"""video_common.py module description."""

import dataclasses
import typing


DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_LUMA_THRESHOLD = 20
DEFAULT_VFT_BORDER_SIZE = 2
PIXEL_FORMAT_CHOICES = ["yuv420p", "nv12", "nv21"]
DEFAULT_PIXEL_FORMAT = None


MIN_VFT_SIZE = 16


@dataclasses.dataclass
class ImageInfo:
    height: int
    width: int
    vft_margin: int
    vft_border_size: int
    vft_x: typing.List[int]
    vft_y: typing.List[int]

    def __init__(self, width, height, vft_border_size=DEFAULT_VFT_BORDER_SIZE):
        self.width = width
        self.height = height
        # VFT margin size is 1/8th of each resolution component
        # VFT code location
        #  +--------------------------+
        #  |                     ^    |
        #  |   vft_margin_height |    |
        #  | vft_margin_width    |    |
        #  |<->                  v    |
        #  |   [-----------------]    |
        #  |   [     VFT code    ]    |
        #  |   [_________________]    |
        #  |                          |
        #  |                          |
        #  +--------------------------+
        vft_margin_width = self.width >> 3
        self.vft_x = [
            vft_margin_width,
            self.width - vft_margin_width,
        ]
        vft_margin_height = self.height >> 3
        self.vft_y = [
            vft_margin_height,
            self.height - vft_margin_height,
        ]
        assert (
            vft_margin_width > MIN_VFT_SIZE
        ), f"error: resolution ({self.width}x{self.height} too narrow to add a VFT code"
        assert (
            vft_margin_height > MIN_VFT_SIZE
        ), f"error: resolution ({self.width}x{self.height} too short to add a VFT code"
        self.vft_border_size = vft_border_size
