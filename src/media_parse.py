#!/usr/bin/env python

"""media_parse.py module description."""


import audio_parse
import video_parse


def media_parse_coverage_video(
    infile,
    **kwarg,
):
    return video_parse.calc_coverage(
        infile,
        width=kwarg.get("width", 0),
        height=kwarg.get("height", 0),
        pixel_format=kwarg.get("pixel_format"),
        debug=kwarg.get("debug", False),
    )


def media_parse_audio(
    pre_samples,
    samplerate,
    beep_freq,
    beep_duration_samples,
    beep_period_sec,
    scale,
    infile,
    outfile,
    debug,
    **kwargs,
):
    min_separation_msec = kwargs.get("min_separation_msec", 50)
    min_match_threshold = kwargs.get("min_match_threshold", 10)
    audio_sample = kwargs.get("audio_sample", "")

    audio_results = audio_parse.audio_parse(
        infile,
        pre_samples=pre_samples,
        samplerate=samplerate,
        beep_freq=beep_freq,
        beep_duration_samples=beep_duration_samples,
        beep_period_sec=beep_period_sec,
        scale=scale,
        min_separation_msec=min_separation_msec,
        min_match_threshold=min_match_threshold,
        audio_sample=audio_sample,
        debug=debug,
    )
    if audio_results is None or len(audio_results) == 0:
        # without audio there is not point in running the video parsing
        raise Exception(
            "ERROR: audio calculation failed. Verify that there are signals in audio stream."
        )
    # write up the results to disk
    audio_results.to_csv(outfile, index=False)


def media_parse_video(
    width,
    height,
    num_frames,
    pixel_format,
    luma_threshold,
    pre_samples,
    samplerate,
    beep_freq,
    beep_duration_samples,
    beep_period_sec,
    scale,
    infile,
    outfile,
    debug,
    **kwargs,
):
    ref_fps = kwargs.get("ref_fps", -1)
    lock_layout = kwargs.get("lock_layout", False)
    tag_manual = kwargs.get("tag_manual", False)
    threaded = kwargs.get("threaded", False)
    # recalculate the video results
    video_results = video_parse.video_parse(
        infile,
        width,
        height,
        ref_fps,
        pixel_format,
        luma_threshold,
        lock_layout=lock_layout,
        tag_manual=tag_manual,
        threaded=threaded,
        debug=debug,
    )
    if debug > 0:
        print(f"Done parsing, write csv, size: {len(video_results)} to {path_video}")
    # write up the results to disk
    video_results.to_csv(outfile, index=False)


def media_parse(
    width,
    height,
    num_frames,
    pixel_format,
    luma_threshold,
    pre_samples,
    samplerate,
    beep_freq,
    beep_duration_samples,
    beep_period_sec,
    scale,
    infile,
    output_video,
    output_audio,
    debug,
    **kwargs,
):
    # 1. parse the audio stream
    media_parse_audio(
        pre_samples,
        samplerate,
        beep_freq,
        beep_duration_samples,
        beep_period_sec,
        scale,
        infile,
        output_audio,
        debug,
        **kwargs,
    )

    # 2. parse the video stream
    media_parse_video(
        width,
        height,
        num_frames,
        pixel_format,
        luma_threshold,
        pre_samples,
        samplerate,
        beep_freq,
        beep_duration_samples,
        beep_period_sec,
        scale,
        infile,
        output_video,
        debug,
        **kwargs,
    )
