#!/usr/bin/env python3

"""video_generate.py module description."""


import argparse
import cv2
import graycode
import math
import subprocess
import sys
import numpy as np

import common
import video_common
import vft
import _version


VFT_ID = "7x5"

COLOR_BLACK = (0, 0, 0)
COLOR_BACKGROUND = (128, 128, 128)
COLOR_BACKGROUND_ALT = COLOR_BLACK
COLOR_WHITE = (255, 255, 255)


default_values = {
    "debug": 0,
    "fps": 30,
    "num_frames": 150,
    "beep_frame_period": 30,
    "width": video_common.DEFAULT_WIDTH,
    "height": video_common.DEFAULT_HEIGHT,
    "outfile": None,
}


def _pipe_frames_to_ffmpeg(width, height, fps, outfile, frame_generator):
    """Helper to pipe frames from a generator to ffmpeg.

    Args:
        width: Video width in pixels
        height: Video height in pixels
        fps: Frames per second
        outfile: Output file path
        frame_generator: Iterator/generator yielding numpy arrays (frames)
    """
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        # keyframe period (in frames)
        "-g",
        "30",
        "-pix_fmt",
        "yuv420p",
        outfile,
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout = b""
    stderr = b""
    try:
        for img in frame_generator:
            # Handle both numpy arrays and raw bytes
            if hasattr(img, "tobytes"):
                proc.stdin.write(img.tobytes())
            else:
                proc.stdin.write(img)

        proc.stdin.close()
        # Python 3.12's communicate() tries to flush stdin even after the caller
        # closes it, so detach the pipe and wait for ffmpeg explicitly.
        proc.stdin = None
        stdout, stderr = proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")
    except Exception as e:
        proc.kill()
        proc.wait()
        raise e


def image_generate(
    image_info,
    frame_num,
    text0,
    text1,
    text2,
    beep_color,
    font,
    vft_id,
    fontscale=1.0,
    debug=0,
):
    # 0. start with an empty image
    img = np.zeros((image_info.height, image_info.width, 3), np.uint8)
    # 1. paint the original image
    x0 = 0
    x1 = image_info.width
    y0 = 0
    y1 = image_info.height
    pts = np.array([[x0, y0], [x0, y1 - 1], [x1 - 1, y1 - 1], [x1 - 1, y0]])
    color_background = COLOR_BACKGROUND if not beep_color else COLOR_BACKGROUND_ALT
    cv2.fillPoly(img, pts=[pts], color=color_background)
    # 2. write the text(s)
    if text0:
        x0 = 32
        y0 = 32
        cv2.putText(img, text0, (x0, y0), font, fontscale, COLOR_BLACK, 12, cv2.LINE_AA)
        cv2.putText(img, text0, (x0, y0), font, fontscale, COLOR_WHITE, 2, cv2.LINE_AA)
    if text1:
        x0 = 32
        y0 = 64
        cv2.putText(img, text1, (x0, y0), font, fontscale, COLOR_BLACK, 12, cv2.LINE_AA)
        cv2.putText(img, text1, (x0, y0), font, fontscale, COLOR_WHITE, 2, cv2.LINE_AA)
    if text2:
        x0 = 32
        y0 = image_info.height - 32
        cv2.putText(img, text2, (x0, y0), font, fontscale, COLOR_BLACK, 12, cv2.LINE_AA)
        cv2.putText(img, text2, (x0, y0), font, fontscale, COLOR_WHITE, 2, cv2.LINE_AA)
    # 3. add VFT code
    x0, x1 = image_info.vft_x
    y0, y1 = image_info.vft_y
    vft_width = x1 - x0
    vft_height = y1 - y0
    img_vft = vft.generate_graycode(
        vft_width, vft_height, vft_id, image_info.vft_border_size, frame_num, debug
    )
    # copy it into the main image
    img[y0:y1, x0:x1] = img_vft
    return img


def video_generate_noise(width, height, fps, num_frames, outfile, vft_id, debug):
    """Generate video with random noise and VFT ArUco markers (for encoder stress testing)."""
    image_info = video_common.ImageInfo(width, height)
    vft_layout = vft.VFTLayout(width, height, vft_id)

    m = (160, 160, 160)
    s = (80, 80, 80)

    def generate_noise_frames():
        """Generator that yields noise frames with VFT tags."""
        for frame_num in range(num_frames):
            img = np.zeros((height, width, 3), np.uint8)
            time = (frame_num // fps) + (frame_num % fps) / fps
            img = cv2.randn(img, m, s)
            # 3. add VFT code
            x0, x1 = image_info.vft_x
            y0, y1 = image_info.vft_y
            vft_width = x1 - x0
            vft_height = y1 - y0
            img = vft.draw_tags(img, vft_id, image_info.vft_border_size, debug)
            yield img

    _pipe_frames_to_ffmpeg(width, height, fps, outfile, generate_noise_frames())


def video_generate(
    width,
    height,
    fps,
    num_frames,
    beep_frame_period,
    frame_period,
    outfile,
    vft_id,
    rem,
    debug,
):
    """Generate video with VFT timing codes, text overlays, and beep indicators."""
    image_info = video_common.ImageInfo(width, height)
    vft_layout = vft.VFTLayout(width, height, vft_id)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1.0
    # Need to make sure we can fit the second text not knowing the actual rem text
    while True:
        textwidth = cv2.getTextSize(
            f"fps: fps:30 resolution: 123x123 {rem}", font, fontscale, 2
        )
        fontscale = fontscale * 0.9
        if textwidth[0][0] <= width:
            break

    def generate_frames():
        """Generator that yields video frames with VFT codes and text."""
        for frame_num in range(num_frames):
            img = np.zeros((height, width, 3), np.uint8)
            time = (frame_num // fps) + (frame_num % fps) / fps
            actual_frame_num = frame_num % frame_period
            gray_num = graycode.tc_to_gray_code(actual_frame_num)
            # VFT 7x5 has 16 bits, use that for display
            num_bits = 16
            # Calculate beep timing
            frames_since_beep = frame_num % beep_frame_period
            frames_to_next_beep = (
                beep_frame_period - frames_since_beep
                if frames_since_beep > 0
                else beep_frame_period
            )
            time_since_beep = (
                -frames_since_beep / fps
            )  # negative because it's in the past
            time_to_next_beep = frames_to_next_beep / fps
            # Calculate avoffset: positive while approaching beep, negative while leaving
            # Range: +half_period (far before beep) -> 0 (at beep) -> -half_period (far after beep)
            half_period_sec = (beep_frame_period / fps) / 2
            if -time_since_beep <= half_period_sec:
                # First half: leaving beep (0 to -half_period)
                avoffset = time_since_beep
            else:
                # Second half: approaching beep (+half_period to 0)
                avoffset = time_to_next_beep
            text0 = f"version: {_version.__version__} vft_id: {vft_id} url: {common.METIQ_URL}"
            text1 = f"frame: {actual_frame_num} gray_num: {gray_num:0{num_bits}b} time: {time:.03f} avoffset: {avoffset:+.03f}"
            text2 = f"fps: {fps:.2f} resolution: {img.shape[1]}x{img.shape[0]} {rem}"
            beep_color = (frame_num % beep_frame_period) == 0
            img = image_generate(
                image_info,
                actual_frame_num,
                text0,
                text1,
                text2,
                beep_color,
                font,
                vft_id,
                fontscale,
                debug,
            )
            yield img

    _pipe_frames_to_ffmpeg(width, height, fps, outfile, generate_frames())


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
        "--num-frames",
        action="store",
        type=int,
        dest="num_frames",
        default=default_values["num_frames"],
        metavar="NUM_FRAMES",
        help=("use NUM_FRAMES frames (default: %i)" % default_values["num_frames"]),
    )
    parser.add_argument(
        "--beep-frame-period",
        action="store",
        type=int,
        dest="beep_frame_period",
        default=default_values["beep_frame_period"],
        metavar="BEEP_FRAME_PERIOD",
        help=(
            "use BEEP_FRAME_PERIOD frames (default: %i)"
            % default_values["beep_frame_period"]
        ),
    )
    parser.add_argument(
        "--frame-period",
        action="store",
        type=int,
        dest="frame_period",
        default=default_values["frame_period"],
        metavar="FRAME_PERIOD",
        help=(
            "use FRAME_PERIOD frame period (default: %i)"
            % default_values["frame_period"]
        ),
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        dest="noise",
        default=False,
        help="Special mode where noise images are generated with tags.",
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
        print("version: %s" % _version.__version__)
        sys.exit(0)

    # get outfile
    if options.outfile is None or options.outfile == "-":
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(options)
    # do something
    if options.noise:
        video_generate_noise(
            options.width,
            options.height,
            options.fps,
            options.num_frames,
            options.outfile,
            VFT_ID,
            options.debug,
        )
    else:
        video_generate(
            options.width,
            options.height,
            options.fps,
            options.num_frames,
            options.beep_frame_period,
            options.frame_period,
            options.outfile,
            VFT_ID,
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
