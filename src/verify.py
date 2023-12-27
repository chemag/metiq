#!/usr/bin/env python3

"""Generate and run test content"""

import common
import video_generate
import audio_generate
import audio_common
import video_common
import vft
import pandas as pd
import sys
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

    # put them together
    command = "ffmpeg -y "
    command += f"-y -f rawvideo -pixel_format rgb24 -s {width}x{height} -r {output_fps} -i {video_filename} "
    command += f"-i {audio_filename} "
    command += f"-c:v libx264 -pix_fmt yuv420p -c:a pcm_s16le {outfile}"

    ret, stdout, stderr = common.run(command, debug=debug)
    assert ret == 0, f"error: {stderr}"
    # clean up raw files
    os.remove(video_filename)
    os.remove(audio_filename)


def run_metiq_cli(filename, audio_offset=0.0):
    command = f"python3 metiq.py -i {filename} --audio-offset {audio_offset} --calc-all analyze --no-cache"
    ret, stdout, stderr = common.run(command, debug=DEBUG)


def verify_metiq_cli(filename, label, audio_delay=0, video_delay=0, av_sync=0):
    failed = False
    print(f"\n-----\n{label}\n")
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
    else:
        print(f"PASS\n---\n")
        # remove all test files
        if not KEEP_FILES:
            for file in glob.glob(f"{filename}*"):
                os.remove(file)


def test1():
    # -------------------------------------------------------
    # avsync of perfect file @ 60fps
    audio = 0.0
    video = 0.0
    descr = f"{inspect.currentframe().f_code.co_name}_vd.{video}s.ad.{audio}s"
    label = f"Video delay:{video}s, Audio delay:{audio}s"
    settings = {
        "outfile": f"{descr}.mov",
        "descr": f"{descr}",
        "output_fps": 60,
        "video_delay": video,
        "audio_delay": audio,
    }

    generate_test_file(**settings)
    run_metiq_cli(settings["outfile"])
    verify_metiq_cli(settings["outfile"], label)


def test2():
    # -------------------------------------------------------
    # audio delay 30 ms
    # positive delay is audio leading, negative audio is late
    audio = -0.030
    video = 0.0
    descr = f"{inspect.currentframe().f_code.co_name}_vd.{video}s.ad.{audio}s"
    label = f"Video delay:{video}s, Audio delay:{audio}s"
    settings = {
        "outfile": f"{descr}.mov",
        "descr": f"{descr}",
        "output_fps": 60,
        "video_delay": video,
        "audio_delay": -audio,
    }
    generate_test_file(**settings)
    run_metiq_cli(settings["outfile"])
    verify_metiq_cli(settings["outfile"], label, 0, 0, audio)


def test3():
    # -------------------------------------------------------
    # compensate the delay of 30 ms
    audio = -0.030
    video = 0.0
    offset = -audio
    descr = f"{inspect.currentframe().f_code.co_name}_vd.{video}s.ad.{audio}s"
    label = f"Video delay:{video}s, Audio delay:{audio}s, compensate 30ms playback/recording offset"
    settings = {
        "outfile": f"{descr}.mov",
        "descr": f"{descr}",
        "output_fps": 60,
        "video_delay": video,
        "audio_delay": -audio,
    }
    generate_test_file(**settings)
    run_metiq_cli(settings["outfile"], offset)
    verify_metiq_cli(settings["outfile"], label)


def test4():

    # -------------------------------------------------------
    # audio early 30 ms
    audio = 0.030
    video = 0.0
    descr = f"{inspect.currentframe().f_code.co_name}_vd.{video}s.ad.{audio}s"
    label = f"Audio early, Video delay:{video}s, Audio delay:{audio}s"
    settings = {
        "outfile": f"{descr}.mov",
        "descr": f"{descr}",
        "output_fps": 60,
        "video_delay": video,
        "audio_delay": -audio,
    }
    generate_test_file(**settings)
    run_metiq_cli(settings["outfile"])
    verify_metiq_cli(settings["outfile"], label, 0, 0, audio)


def test5():

    # -------------------------------------------------------
    # audio sync perfect, video delay 100ms
    audio = 0.0
    video = 0.600
    descr = f"{inspect.currentframe().f_code.co_name}_vd.{video}s.ad.{audio}s"
    label = f"Video delay:{video}s, Audio delay:{audio}s"

    settings = {
        "outfile": f"{descr}.mov",
        "descr": "vd.100ms",
        "output_fps": 60,
        "video_delay": video,
        "audio_delay": -audio,
    }
    generate_test_file(**settings)
    run_metiq_cli(settings["outfile"])
    verify_metiq_cli(settings["outfile"], label, video - audio, video, audio)


def test6():
    # -------------------------------------------------------
    # audio late 60ms, video delay 100ms
    audio = -0.060
    video = 0.100
    descr = f"v{inspect.currentframe().f_code.co_name}_vd.{video}s.ad.{audio}s"
    label = f"Video delay:{video}s, Audio delay:{audio}s"
    settings = {
        "outfile": f"{descr}.mov",
        "descr": f"{descr}",
        "output_fps": 60,
        "video_delay": video,
        "audio_delay": -audio,
    }
    generate_test_file(**settings)
    run_metiq_cli(settings["outfile"])
    verify_metiq_cli(settings["outfile"], label, video - audio, video, audio)


def test7():
    #
    # Test black frames outside of sync position
    audio = 0.0
    video = 0.0
    descr = f"{inspect.currentframe().f_code.co_name}_vd.{video}s.ad.{audio}s"
    label = f"Video delay:{video}s, Audio delay:{audio}s"
    # In 60 fps number scheme
    black_frames = [*range(15, 20), *range(70, 74)]
    settings = {
        "outfile": f"{descr}.mov",
        "descr": f"{descr}",
        "output_fps": 60,
        "video_delay": video,
        "audio_delay": audio,
        "black_frames": black_frames,
    }

    generate_test_file(**settings)
    run_metiq_cli(settings["outfile"])
    verify_metiq_cli(settings["outfile"], label)


def test8():
    #
    # Test black frames at sync position
    audio = 0.0
    video = 0.0
    descr = f"{inspect.currentframe().f_code.co_name}_vd.{video}s.ad.{audio}s"
    label = f"Video delay:{video}s, Audio delay:{audio}s"
    # In 60 fps number scheme
    frames = [*range(160, 190)]
    settings = {
        "outfile": f"{descr}.mov",
        "descr": f"{descr}",
        "output_fps": 60,
        "video_delay": video,
        "audio_delay": audio,
        "black_frames": frames,
    }

    generate_test_file(**settings)
    run_metiq_cli(settings["outfile"])
    verify_metiq_cli(settings["outfile"], label)


def test9():
    #
    # Test black frames at sync position with video delay
    audio = 0.0
    video = 0.200
    descr = f"{inspect.currentframe().f_code.co_name}_vd.{video}s.ad.{audio}s"
    label = f"Video delay:{video}s, Audio delay:{audio}s"
    # In 60 fps number scheme
    frames = [*range(160, 190)]
    settings = {
        "outfile": f"{descr}.mov",
        "descr": f"{descr}",
        "output_fps": 60,
        "video_delay": video,
        "audio_delay": audio,
        "black_frames": frames,
    }

    generate_test_file(**settings)
    run_metiq_cli(settings["outfile"])
    verify_metiq_cli(settings["outfile"], label)


def test10():
    #
    # Test frozen frames at sync position
    audio = 0.0
    video = 0.0
    descr = f"{inspect.currentframe().f_code.co_name}_vd.{video}s.ad.{audio}s"
    label = f"Video delay:{video}s, Audio delay:{audio}s"
    # In 60 fps number scheme
    frames = [*range(160, 190)]
    settings = {
        "outfile": f"{descr}.mov",
        "descr": f"{descr}",
        "output_fps": 60,
        "video_delay": video,
        "audio_delay": audio,
        "freeze_frames": frames,
    }

    generate_test_file(**settings)
    run_metiq_cli(settings["outfile"])
    verify_metiq_cli(settings["outfile"], label)


def test11():
    #
    # Test frozen frames at sync position with video delay
    audio = 0.0
    video = 0.200
    descr = f"{inspect.currentframe().f_code.co_name}_vd.{video}s.ad.{audio}s"
    label = f"Video delay:{video}s, Audio delay:{audio}s"
    # In 60 fps number scheme
    frames = [*range(160, 190)]
    settings = {
        "outfile": f"{descr}.mov",
        "descr": f"{descr}",
        "output_fps": 60,
        "video_delay": video,
        "audio_delay": audio,
        "freeze_frames": frames,
    }

    generate_test_file(**settings)
    run_metiq_cli(settings["outfile"])
    verify_metiq_cli(settings["outfile"], label)


def test():
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()
    test8()
    test9()
    test10()
    test11()


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
    parser.add_argument("--run", type=str, help="Run a specific test number")

    options = parser.parse_args(argv[1:])

    return options


def main(argv):
    global KEEP_FILES
    # parse options
    options = get_options(argv)

    KEEP_FILES = options.keep
    if options.run:
        print(f"Running {options.run}")
        if options.run.isdigit():
            globals()[f"test{options.run}"]()
        else:
            globals()[options.run]()
        exit(0)
    test()


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
