#!/usr/bin/env python

"""media_generate.py module description."""


import os
import tempfile

import audio_generate
import vft
import video_generate
import common


def media_generate_noise_video(outfile, **kwarg):
    width = kwarg.get("width", default_values["width"])
    height = kwarg.get("height", default_values["height"])
    fps = kwarg.get("fps", default_values["fps"])
    num_frames = kwarg.get("num_frames", default_values["num_frames"])
    vft_id = kwarg.get("vft_id", default_values["vft_id"])
    pre_samples = kwarg.get("pre_samples", default_values["pre_samples"])
    debug = kwarg.get("debug", default_values["debug"])

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
    width = kwarg.get("width", default_values["width"])
    height = kwarg.get("height", default_values["height"])
    fps = kwarg.get("fps", default_values["fps"])
    num_frames = kwarg.get("num_frames", default_values["num_frames"])
    vft_id = kwarg.get("vft_id", default_values["vft_id"])
    pre_samples = kwarg.get("pre_samples", default_values["pre_samples"])
    samplerate = kwarg.get("samplerate", default_values["samplerate"])
    beep_freq = kwarg.get("beep_freq", default_values["beep_freq"])
    beep_duration_samples = kwarg.get(
        "beep_duration_samples", default_values["beep_duration_samples"]
    )
    beep_period_sec = kwarg.get("beep_period_sec", default_values["beep_period_sec"])
    scale = kwarg.get("scale", default_values["scale"])
    debug = kwarg.get("debug", default_values["debug"])
    audio_sample = kwarg.get("audio_sample", default_values["audio_sample"])

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
