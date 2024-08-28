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
    min_separation_msec = kwargs.get(
        "min_separation_msec", audio_parse.DEFAULT_MIN_SEPARATION_MSEC
    )
    min_match_threshold = kwargs.get(
        "min_match_threshold", audio_parse.DEFAULT_MIN_MATCH_THRESHOLD
    )
    audio_sample = kwargs.get("audio_sample", "")
    bandpass_filter = kwargs.get(
        "bandpass_filter", audio_parse.default_values["bandpass_filter"]
    )

    try:
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
            bandpass_filter=bandpass_filter,
            debug=debug,
        )
        if audio_results is None or len(audio_results) == 0:
            # without audio there is not point in running the video parsing
            raise Exception(
                "ERROR: audio calculation failed. Verify that there are signals in audio stream."
            )
        # write up the results to disk
        audio_results.to_csv(outfile, index=False)
        return audio_results
    except Exception as e:
        print(f"ERROR: failed parsing audio, {e}")
        return None


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
    debug=False,
    **kwargs,
):
    ref_fps = kwargs.get("ref_fps", -1)
    if ref_fps == -1:
        print("Warning: ref_fps is not set. Using the default value of 30 fps.")
        ref_fps = 30
    lock_layout = kwargs.get("lock_layout", False)
    tag_manual = kwargs.get("tag_manual", False)
    threaded = kwargs.get("threaded", False)
    sharpen = kwargs.get("sharpen", False)
    contrast = kwargs.get("contrast", 1)
    brightness = kwargs.get("brightness", 0)

    try:
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
            sharpen=sharpen,
            contrast=contrast,
            brightness=brightness,
            debug=debug,
        )
        if debug > 0:
            print(f"Done parsing, write csv, size: {len(video_results)} to {outfile}")
        # write up the results to disk
        video_results.to_csv(outfile, index=False)
        return video_results
    except Exception as e:
        print(f"Error: failed parsing video, {e}")
        return None


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
    if output_audio is None:
        output_audio = infile + ".audio.csv"
    audio_results = media_parse_audio(
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

    if audio_results is None:
        print("ERROR: audio parsing failed. Exiting.")
        return None

    # 2. parse the video stream
    if output_video is None:
        output_video = infile + ".video.csv"
    video_results = media_parse_video(
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
