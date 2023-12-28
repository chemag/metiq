#!/usr/bin/env python3

"""Generate and run test content"""
import sys
metiq_path = "../src"
sys.path.append(metiq_path)

import common
import video_generate
import audio_generate
import audio_common
import video_common
import vft
import pandas as pd
import argparse
import tempfile
import cv2
import numpy as np
import graycode
import math
import os
import scipy
import inspect
import glob
import re
import verify_tests


BEEP_PERIOD_SEC = 3.0
WIDTH = 720
HEIGHT = 536
FPS = 30
VFT_LAYOUT = vft.VFT_LAYOUT[vft.DEFAULT_VFT_ID]
BEEP_PERIOD_FRAMES = BEEP_PERIOD_SEC * FPS

DURATION_SEC = 12
NUM_FRAMES = int(DURATION_SEC * FPS)
SAMPLE_RATE = 16000
DEBUG = 0
PREC = 0.01  # sec
KEEP_FILES = False


def audio_generate(duration_sec, output_filename, **settings):
    audio_delay = settings.get("audio_delay", 0)
    video_delay = settings.get("video_delay", 0)
    samplerate = settings.get("samplerate", SAMPLE_RATE)

    # normal default signal
    aud = audio_common.generate_chirp(duration_sec, **settings)

    audio_signal = None
    auddio_filtered = None
    if audio_delay > 0:
        delay = [0.0] * int(audio_delay * samplerate)
        audio_filtered = np.concatenate((delay, aud))[: len(aud)]
    elif audio_delay < 0:
        audio_filtered = aud[-int(audio_delay * samplerate) :]
    else:
        audio_filtered = aud

    # video delay cannot be negative...
    if video_delay > 0:
        delay = [0.0] * int(video_delay * samplerate)
        tmp = aud.copy()[int(video_delay * samplerate) :]
        audio_filtered = audio_filtered // 4
        audio_signal = np.concatenate((tmp, delay))[: len(aud)]

        aud = (audio_filtered + audio_signal) // 2
    else:
        aud = audio_filtered

    scipy.io.wavfile.write(
        output_filename, audio_common.DEFAULT_SAMPLERATE, aud.astype(np.int16)
    )


OLD_FRAME = None
BLACK_FRAME = None


def write_frame(rawstream, frame, frame_num, freeze_frames, black_frames):
    global OLD_FRAME
    global BLACK_FRAME

    if frame_num in freeze_frames:
        frame = OLD_FRAME
    elif frame_num in black_frames:
        frame = BLACK_FRAME

    rawstream.write(frame)
    OLD_FRAME = frame


def generate_test_file(**settings):
    global BLACK_FRAME

    audio_delay = settings.get("audio_delay", 0)
    video_delay = settings.get("video_delay", 0)

    outfile = settings.get("outfile", f"test.mov")
    descr = settings.get("descr", "test")
    num_frames = settings.get("num_frames", NUM_FRAMES)
    fps = settings.get("fps", FPS)
    output_fps = settings.get("output_fps", 30)
    width = settings.get("width", WIDTH)
    height = settings.get("height", HEIGHT)
    samplerate = settings.get("sample_rate", SAMPLE_RATE)
    vft_id = settings.get("vft_id", vft.DEFAULT_VFT_ID)
    # frozen frames are stuck frames
    freeze_frames = settings.get("freeze_frames", [])
    # black frames simulate failed parsing and should not contribute to lost
    # frames but instead be treated as good frames and interpolated (or the measurement dropped).
    black_frames = settings.get("black_frames", [])
    extra_frames = output_fps / fps - 1
    BLACK_FRAME = np.zeros((height, width, 3), np.uint8)

    vft_layout = vft.VFTLayout(width, height, vft_id)
    max_frame_num = 2**vft_layout.numbits
    frame_period = BEEP_PERIOD_FRAMES * (max_frame_num // BEEP_PERIOD_FRAMES)
    beep_freq = 300
    beep_duration_samples = int(samplerate * BEEP_PERIOD_SEC)
    beep_period_sec = BEEP_PERIOD_SEC
    debug = DEBUG

    # generate the (raw) video input
    video_filename = tempfile.NamedTemporaryFile().name + ".rgb24"

    image_info = video_common.ImageInfo(width, height)
    vft_layout = vft.VFTLayout(width, height, vft_id)
    metiq_id = "default"
    rem = f"default chirp"

    output_frame_num = 0

    # todo: introduce distorted parts
    with open(video_filename, "wb") as rawstream:
        font = cv2.FONT_HERSHEY_SIMPLEX
        # original image
        for frame_num in range(0, num_frames, 1):
            img = np.zeros((height, width, 3), np.uint8)
            time = (frame_num // fps) + (frame_num % fps) / fps
            actual_frame_num = int(frame_num % frame_period)
            gray_num = graycode.tc_to_gray_code(actual_frame_num)
            num_bits = math.ceil(math.log2(num_frames))
            text1 = f"id: {metiq_id} frame: {actual_frame_num} time: {time:.03f} gray_num: {gray_num:0{num_bits}b}"
            text2 = f"fps: {fps:.2f} resolution: {img.shape[1]}x{img.shape[0]} {rem}"
            beep_color = (frame_num % BEEP_PERIOD_FRAMES) == 0
            img = video_generate.image_generate(
                image_info,
                actual_frame_num,
                text1,
                text2,
                beep_color,
                font,
                vft_id,
                DEBUG,
            )
            write_frame(rawstream, img, output_frame_num, freeze_frames, black_frames)
            output_frame_num += 1
            # We are generating at 30fps
            for i in range(int(extra_frames)):
                write_frame(
                    rawstream, img, output_frame_num, freeze_frames, black_frames
                )
                output_frame_num += 1

    duration_sec = num_frames / fps
    # generate the (raw) audio input
    audio_filename = tempfile.NamedTemporaryFile().name + ".wav"
    pre_samples = 0

    audio_generate(
        duration_sec,
        audio_filename,
        pre_samples=pre_samples,
        samplerate=samplerate,
        beep_freq=beep_freq,
        beep_duration_samples=beep_duration_samples,
        beep_period_sec=beep_period_sec,
        scale=1,
        audio_delay=audio_delay,
        video_delay=video_delay,
        debug=debug,
    )

    # speed up
    ws = int(width / 4)
    hs = int(height / 4)
    # put them together
    command = "ffmpeg -y "
    command += f"-y -f rawvideo -pixel_format rgb24 -s {width}x{height} -r {output_fps} -i {video_filename} "
    command += f"-i {audio_filename} "
    command += f"-c:v libx264 -pix_fmt yuv420p -c:a pcm_s16le -s {ws}x{hs} {outfile}"

    ret, stdout, stderr = common.run(command, debug=debug)
    assert ret == 0, f"error: {stderr}"
    # clean up raw files
    os.remove(video_filename)
    os.remove(audio_filename)


def run_metiq_cli(**settings):
    filename = settings.get("outfile", "")
    audio_offset = settings.get("audio_offset", 0)
    command = f"python3 {metiq_path}/metiq.py -i {filename} --audio-offset {audio_offset} --calc-all analyze --no-cache"
    ret, stdout, stderr = common.run(command, debug=DEBUG)
    print(f"{command=}")
    assert ret == 0, f"error: {stderr}"



def verify_metiq_cli(**settings):
    failed = False
    doc = settings.get("doc", "")
    filename = settings.get("outfile")
    audio_offset = settings.get("audio_offset", 0.0)
    av_sync = -settings.get("audio_delay", 0) + audio_offset
    video_delay = settings.get("video_delay", 0)
    audio_delay = video_delay - av_sync

    print(f"\n{'-'*20}\n{filename}\n")
    print(f"{doc}")
    print("Audio delay: ", audio_delay)
    print("Video delay: ", video_delay)
    print("A/V sync: ", av_sync)
    print("Audio offset: ", audio_offset)
    # read the files and compare
    if video_delay > 0:
        videolat = pd.read_csv(f"{filename}.video.latency.csv")
        meanval = videolat["video_latency_sec"].mean()
        result = meanval < PREC + video_delay and meanval > video_delay - PREC
        if not result:
            failed = True
            print(f"Video delay measurement failed: video delay: {meanval}")

        audiolat = pd.read_csv(f"{filename}.audio.latency.csv")
        meanval = audiolat["audio_latency_sec"].mean()
        result = meanval < PREC + audio_delay and meanval > audio_delay - PREC
        if not result:
            failed = True
            print(f"Audio delay measurement failed: audio delay: {meanval}")

    avsync = pd.read_csv(f"{filename}.avsync.csv")
    meanval = avsync["av_sync_sec"].mean()
    result = meanval < PREC + av_sync and meanval > av_sync - PREC
    if not result:
        failed = True
        print(f"Audio/video synchronization measurement failed: audio delay: {meanval}")

    if failed:
        print(f"{filename}")
        print(f"!!! FAILED\n---\n")
        # Keep files if broken test
        return False
    else:
        print(f"PASS\n{'-'*20}\n")
        # remove all test files
        if not KEEP_FILES:
            for file in glob.glob(f"{filename}*"):
                os.remove(file)

    return True


def run_test(**settings):
    settings["doc"] = get_caller_doc()[1]
    generate_test_file(**settings)
    run_metiq_cli(**settings)

    return verify_metiq_cli(**settings)


def get_caller_doc():
    outerframe = inspect.currentframe().f_back.f_back
    name = outerframe.f_code.co_name
    doc = outerframe.f_globals[name].__doc__
    return name, doc


def run_all_tests():
    tests = list_all_tests()
    failed = []
    testcount = len(tests)
    print(f"Total number of tests to run: {testcount}")
    for test in tests:
        result = test[1]()
        if not result:
            print(f"{test[1]} failed, append to failed tests")
            failed.append(test[1])

    passcount = len(tests) - len(failed)
    print(f"Done: {passcount}/{testcount}")

    if len(failed) > 0:
        print("Failed tests:")
        for test in failed:
            print(f"\t{test.__name__}")


def list_all_tests():
    module = sys.modules["verify_tests"]
    functions = inspect.getmembers(module, inspect.isfunction)
    tests = [
        (int(re.search(r"[0-9]+", str(f))[0]), f)
        for f in functions
        if f[0].startswith("test")
    ]

    tests_sorted = sorted(tests, key=lambda x: int(x[0]))
    return [x[1] for x in tests_sorted]


def print_all_tests():
    tests = list_all_tests()
    for test in tests:
        print(f"{test[0]}: {test[1].__doc__}")


def find_test(testname):
    tests = list_all_tests()
    for test in tests:
        if re.search(f"{testname}$", test[0]):
            return test[1]

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
        "--keep",
        action="store_true",
        help="keep all files after a run",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="list all tests",
    )
    parser.add_argument("--run", type=str, help="Run a specific test number")

    options = parser.parse_args(argv[1:])

    return options


def main(argv):
    global KEEP_FILES
    # parse options
    options = get_options(argv)

    KEEP_FILES = options.keep
    if options.list:
        print_all_tests()
        exit(0)

    if options.run:
        test = find_test(options.run)
        if test:
            test()
        exit(0)
    run_all_tests()


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
