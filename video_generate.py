#!/usr/bin/env python3

"""video_generate.py module description."""


import argparse
import cv2
import sys
import numpy as np

import video_common
import aruco_common
from _version import __version__


# for 8-bits/component, the ffmpeg pix fmt is 'rgb24'

# use fiduciarial markers from this dictionary
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

COLOR_BLACK = (0, 0, 0)
COLOR_BACKGROUND = (128, 128, 128)
COLOR_WHITE = (255, 255, 255)


default_values = {
    "debug": 0,
    "fps": 30,
    "num_frames": 150,
    "width": video_common.DEFAULT_WIDTH,
    "height": video_common.DEFAULT_HEIGHT,
    "outfile": None,
}


def generate_image(image_info, gray_num, text1, text2, font, debug):
    # 0. start with an empty image
    img = np.zeros((image_info.height, image_info.width, 3), np.uint8)
    # 1. paint the original image
    x0 = 0
    x1 = image_info.width
    y0 = 0
    y1 = image_info.height
    pts = np.array([[x0, y0], [x0, y1 - 1], [x1 - 1, y1 - 1], [x1 - 1, y0]])
    cv2.fillPoly(img, pts=[pts], color=COLOR_BACKGROUND)
    # 2. write the text(s)
    if text1:
        x0 = 32
        y0 = 32
        cv2.putText(img, text1, (x0, y0), font, 1, COLOR_BLACK, 16, cv2.LINE_AA)
        cv2.putText(img, text1, (x0, y0), font, 1, COLOR_WHITE, 2, cv2.LINE_AA)
    if text2:
        x0 = 32
        y0 = image_info.height - 32
        cv2.putText(img, text2, (x0, y0), font, 1, COLOR_BLACK, 16, cv2.LINE_AA)
        cv2.putText(img, text2, (x0, y0), font, 1, COLOR_WHITE, 2, cv2.LINE_AA)
    # 3. add fiduciary markers
    for aruco_id in range(3):
        img_tag = aruco_common.generate_aruco_tag(
            image_info.tag_size, aruco_id, image_info.tag_border_size
        )
        # copy it into the main image
        xpos1 = image_info.tag_x[aruco_id]
        xpos2 = xpos1 + image_info.tag_size
        ypos1 = image_info.tag_y[aruco_id]
        ypos2 = ypos1 + image_info.tag_size
        img[ypos1:ypos2, xpos1:xpos2] = img_tag
    # 4. add gray code
    generate_gray_code(img, image_info, gray_num, debug)
    return img


def generate_gray_code(img, image_info, gray_num, debug):
    # 1. add box surrounding the gray code
    x0, xn, yt, yc, yb = image_info.get_gray_block_location()
    pts = np.array([[x0, yt], [x0, yb - 1], [xn - 1, yb - 1], [xn - 1, yt]])
    cv2.fillPoly(img, pts=[pts], color=COLOR_WHITE)
    # cv2.rectangle(img, (0, y - 1), (image_info.width, int(y + ph + 1)), COLOR_WHITE, 2)
    # 2. add the gray code
    img = draw_gray_code(img, image_info, gray_num)


# draw boxes in black&white that represent a Gray code.
# Always draw mirrored boxes around the horizontal axis.
def draw_gray_code(img, image_info, gray_num):
    # print the bit string
    for i in range(image_info.gb_num_bits):
        box_id = image_info.gb_num_bits - i - 1
        x0 = image_info.gb_x[box_id]
        x1 = x0 + image_info.gb_boxsize
        yc = image_info.gb_y[box_id]
        yt = yc - image_info.gb_boxsize
        yb = yc + image_info.gb_boxsize
        if gray_num % 2 == 1:
            # rectangle in top part
            pts = np.array([[x0, yc], [x0, yb - 1], [x1 - 1, yb - 1], [x1 - 1, yc]])
        else:
            # rectangle in bottom part
            pts = np.array([[x0, yt], [x0, yc - 1], [x1 - 1, yc - 1], [x1 - 1, yt]])
        cv2.fillPoly(img, pts=[pts], color=COLOR_BLACK)
        gray_num >>= 1
    return img


def video_generate(width, height, fps, num_frames, outfile, metiq_id, rem, debug):
    image_info = video_common.ImageInfo(width, height, video_common.NUM_BITS)

    with open(outfile, "wb") as rawstream:
        font = cv2.FONT_HERSHEY_SIMPLEX
        # original image
        for frame_num in range(0, num_frames, 1):
            img = np.zeros((height, width, 3), np.uint8)
            time = (frame_num // fps) + (frame_num % fps) / fps
            gray_num = video_common.num_to_gray(frame_num, image_info.gb_num_bits)
            text1 = f"id: {metiq_id} frame: {frame_num} time: {time:.03f} gray_num: {gray_num:0{image_info.gb_num_bits}b}"
            text2 = f"fps: {fps:.2f} resolution: {img.shape[1]}x{img.shape[0]} {rem}"
            img = generate_image(image_info, gray_num, text1, text2, font, debug)
            rawstream.write(img)


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
        "--video_size",
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

    # get outfile
    if options.outfile is None or options.outfile == "-":
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(options)
    # do something
    video_generate(
        options.width,
        options.height,
        options.fps,
        options.num_frames,
        options.outfile,
        "default",
        "",
        options.debug,
    )
    if options.debug > 0:
        print(
            f"run: ffmpeg -y -f rawvideo -pixel_format rgb24 -s {options.width}x{options.height} -r {options.fps} -i {options.outfile} {options.outfile}.mp4"
        )


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
