#!/usr/bin/env python3

"""Test definitions"""
from verify import run_test
import inspect
import verify_config

# Tests are created by varying audio and video delay.
# Generally both audio and video delay transltes to audio delay and
# video is static.
# For example:
# 1)
# A video delay of 100ms will create a pair of audio signals
# 100ms apart. The second delay will be aligned with the beep signal frame.
# The first signal will be early by 100ms as this simulates the sound of the
# playout signal in sending side.
# This is the transmission scenario with tx and rx audio signals present.
#
# 2)
# If only a audio delay is specified, then there will only be one audio signal present.
# This is a simple audio/video sync scenario.
#
# In addition distortion can be added to the video in the form of freeze frames or black frames
# This simulates sending irregularities and quality/parsing issues respectively.
#
# Verification is done by observing:
# 1) Audio delay will be the audio delay
# 2) Video delay will be the video delay set directly
# 3) Audio offset will be the video delay - audio delay
#
# Please not that the delays going into the generator are all in the same direction which for audio is the opposite sign
# compared to avync standard


def test1():
    """
    Plain file with no delays and perfect a/v sync. File @ 60fps
    """
    audio = 0.0
    video = 0.0
    descr = f"{inspect.currentframe().f_code.co_name}"

    settings = {
        "outfile": f"{descr}.mov",
        "descr": f"{descr}",
        "output_fps": 60,
        "video_delay": video,
        "audio_delay": audio,
    }

    return run_test(**settings)


def test2():
    """
    Audio late 30 ms
    """
    audio = 0.030
    video = 0.0
    descr = f"{inspect.currentframe().f_code.co_name}"
    settings = {
        "outfile": f"{descr}.mov",
        "descr": f"{descr}",
        "output_fps": 60,
        "video_delay": video,
        "audio_delay": audio,
    }

    return run_test(**settings)


def test3():
    """
    Audio late 30 ms, compensate the delay with audio offset.
    """
    audio = 0.030
    video = 0.0
    offset = 0.030
    descr = f"{inspect.currentframe().f_code.co_name}"
    settings = {
        "outfile": f"{descr}.mov",
        "descr": f"{descr}",
        "output_fps": 60,
        "video_delay": video,
        "audio_delay": audio,
        "audio_offset": offset,
    }

    return run_test(**settings)


def test4():
    """
    audio early 30 ms
    """
    audio = -0.030
    video = 0.0
    descr = f"{inspect.currentframe().f_code.co_name}"
    settings = {
        "outfile": f"{descr}.mov",
        "descr": f"{descr}",
        "output_fps": 60,
        "video_delay": video,
        "audio_delay": audio,
    }

    return run_test(**settings)


def test5():

    """
    Video delay 100ms, a/v sync perfect
    """
    audio = 0.200
    video = 0.200
    offset = 0.0
    descr = f"{inspect.currentframe().f_code.co_name}"

    settings = {
        "outfile": f"{descr}.mov",
        "descr": "vd.100ms",
        "output_fps": 60,
        "video_delay": video,
        "audio_delay": audio,
        "audio_offset": offset,
    }

    return run_test(**settings)


def test6():
    """
    Audio late 200ms, video delay 100ms
    """
    audio = 0.200
    video = 0.100
    descr = f"{inspect.currentframe().f_code.co_name}"
    settings = {
        "outfile": f"{descr}.mov",
        "descr": f"{descr}",
        "output_fps": 60,
        "video_delay": video,
        "audio_delay": audio,
    }

    return run_test(**settings)


def test7():
    """
    Black frames outside of sync position. No delays and perfect sync.
    """
    audio = 0.0
    video = 0.0
    descr = f"{inspect.currentframe().f_code.co_name}"
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

    return run_test(**settings)


def test8():
    """
    Black frames at sync position. No delays and perfect sync.
    """
    audio = 0.0
    video = 0.0
    descr = f"{inspect.currentframe().f_code.co_name}"
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

    return run_test(**settings)


def test9():
    """
    Black frames at sync position with 200 ms video delay. Pefect sync.
    """
    audio = 0.200
    video = 0.200
    descr = f"{inspect.currentframe().f_code.co_name}"
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

    return run_test(**settings)


def test10():
    """
    Frozen frames at sync position. No delays and perfect sync.
    """
    audio = 0.0
    video = 0.0
    descr = f"{inspect.currentframe().f_code.co_name}"
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

    return run_test(**settings)


def test11():
    """
    Frozen frames at sync position with 500 ms video delay. Audio 300ms early.
    """
    audio = 0.200
    video = 0.500
    descr = f"{inspect.currentframe().f_code.co_name}"
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

    return run_test(**settings)


def test12():
    """
    Offset is close to the video delay.
    """
    audio = 0.5
    video = 0.5
    offset = 0.5
    descr = f"{inspect.currentframe().f_code.co_name}"
    settings = {
        "outfile": f"{descr}.mov",
        "descr": f"{descr}",
        "output_fps": 60,
        "video_delay": video,
        "audio_delay": audio,
        "audio_offset": offset,
    }

    return run_test(**settings)
