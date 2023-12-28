#!/usr/bin/env python3
import sys

metiq_path = "../src"
sys.path.append(metiq_path)

import common
import video_generate
import audio_generate
import audio_common
import video_common
import vft
import numpy as np
import tempfile
import cv2
import graycode
import math
import scipy
import os

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


def write_frame(
    rawstream, frame, frame_num, freeze_frames, black_frames, old_frame, black_frame
):
    if frame_num in freeze_frames:
        frame = old_frame
    elif frame_num in black_frames:
        frame = black_frame

    rawstream.write(frame)
    return frame


def generate_test_file(**settings):
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
    black_frame = np.zeros((height, width, 3), np.uint8)

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
    old_frame = black_frame
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
            old_frame = write_frame(
                rawstream,
                img,
                output_frame_num,
                freeze_frames,
                black_frames,
                old_frame,
                black_frame,
            )
            output_frame_num += 1
            # We are generating at 30fps
            for i in range(int(extra_frames)):
                old_frame = write_frame(
                    rawstream,
                    img,
                    output_frame_num,
                    freeze_frames,
                    black_frames,
                    old_frame,
                    black_frame,
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
