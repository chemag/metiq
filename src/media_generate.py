#!/usr/bin/env python

"""media_generate.py module description."""


import os
import tempfile

import audio_common
import audio_generate
import common
import vft
import video_common
import video_generate
import common


def media_generate_noise_video(outfile, **kwarg):
    width = kwarg.get("width", video_common.DEFAULT_WIDTH)
    height = kwarg.get("height", video_common.DEFAULT_HEIGHT)
    fps = kwarg.get("fps", video_common.DEFAULT_FPS)
    num_frames = kwarg.get("num_frames", video_common.DEFAULT_NUM_FRAMES)
    vft_id = kwarg.get("vft_id", vft.DEFAULT_VFT_ID)
    pre_samples = kwarg.get("pre_samples", audio_common.DEFAULT_PRE_SAMPLES)
    debug = kwarg.get("debug", common.DEFAULT_DEBUG)

    video_filename = tempfile.NamedTemporaryFile().name + ".rgb24"
    video_generate.video_generate_noise(
        width, height, fps, num_frames, video_filename, vft_id, debug
    )
    duration_sec = num_frames / fps

    if outfile[-3:] != "y4m":
        print(
            f"Warning! {outfile[-3:]} should be y4m for an uncompressed original. Noise is hard to encode."
        )
    command = "ffmpeg -y "
    command += f"-f rawvideo -pixel_format rgb24 -s {width}x{height} -r {fps} -i {video_filename} "
    command += f" -pix_fmt yuv420p {outfile}"
    ret, stdout, stderr = common.run(command, debug=debug)
    assert ret == 0, f"error: {stderr}"
    # clean up raw files
    os.remove(video_filename)


def media_generate(outfile, **kwarg):
    width = kwarg.get("width", video_common.DEFAULT_WIDTH)
    height = kwarg.get("height", video_common.DEFAULT_HEIGHT)
    fps = kwarg.get("fps", video_common.DEFAULT_FPS)
    num_frames = kwarg.get("num_frames", video_common.DEFAULT_NUM_FRAMES)
    vft_id = kwarg.get("vft_id", vft.DEFAULT_VFT_ID)
    pre_samples = kwarg.get("pre_samples", audio_common.DEFAULT_PRE_SAMPLES)
    samplerate = kwarg.get("samplerate", audio_common.DEFAULT_SAMPLERATE)
    beep_freq = kwarg.get("beep_freq", audio_common.DEFAULT_BEEP_FREQ)
    beep_duration_samples = kwarg.get(
        "beep_duration_samples", audio_common.DEFAULT_BEEP_DURATION_SAMPLES
    )
    beep_period_sec = kwarg.get("beep_period_sec", audio_common.DEFAULT_BEEP_PERIOD_SEC)
    scale = kwarg.get("scale", audio_common.DEFAULT_SCALE)
    debug = kwarg.get("debug", common.DEFAULT_DEBUG)
    audio_sample = kwarg.get("audio_sample", audio_common.DEFAULT_AUDIO_SAMPLE)

    # calculate the frame period
    beep_period_frames = beep_period_sec * fps
    vft_layout = vft.VFTLayout(width, height, vft_id)
    max_frame_num = 2**vft_layout.numbits
    frame_period = beep_period_frames * (max_frame_num // beep_period_frames)
    # generate the (raw) video input
    video_filename = tempfile.NamedTemporaryFile().name + ".rgb24"
    rem = f"period: {beep_period_sec} freq_hz: {beep_freq} samples: {beep_duration_samples}"
    if len(audio_sample) > 0:
        rem = f"period: {beep_period_sec} signal: {audio_sample}"

    video_generate.video_generate(
        width,
        height,
        fps,
        num_frames,
        beep_period_frames,
        frame_period,
        video_filename,
        "default",
        vft_id,
        rem,
        debug,
    )
    duration_sec = num_frames / fps
    # generate the (raw) audio input
    audio_filename = tempfile.NamedTemporaryFile().name + ".wav"
    audio_generate.audio_generate(
        duration_sec,
        audio_filename,
        pre_samples=pre_samples,
        samplerate=samplerate,
        beep_freq=beep_freq,
        beep_duration_samples=beep_duration_samples,
        beep_period_sec=beep_period_sec,
        scale=scale,
        audio_sample=audio_sample,
        debug=debug,
    )
    # put them together
    command = "ffmpeg -y "
    command += f"-f rawvideo -pixel_format rgb24 -s {width}x{height} -r {fps} -i {video_filename} "
    command += f"-i {audio_filename} "
    command += f"-c:v libx264 -pix_fmt yuv420p -c:a aac {outfile}"
    ret, stdout, stderr = common.run(command, debug=debug)
    assert ret == 0, f"error: {stderr}"
    # clean up raw files
    os.remove(video_filename)
    os.remove(audio_filename)
