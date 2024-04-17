#! /usr/bin/env python3


'''
This script is used to run multiple METIQ instances in parallel. It is a wrapper script that calls the metiq.py script
using common arguments.
'''



import os
import sys
import argparse
import time
import metiq
import media_parse
import media_analyze 

def main():
    parser = argparse.ArgumentParser(description='Run multiple METIQ instances in parallel')
    parser.add_argument("files", nargs="+", type=str, help='Input file(s)')
    parser.add_argument("-o", "--output", type=str, default="all", help='Output file. This is the aggregated output file name. ')
    parser.add_argument("-pa", "--parse-audio", action="store_true", dest="parse_audio", help="Reparse audio")
    parser.add_argument("-pv", "--parse-video", action="store_true", dest="parse_video", help="Reparse video")
    parser.add_argument("-ao", "--audio-offset", type=float, default=0.0, help="Audio offset in seconds")
    args = parser.parse_args()

    # We assume default settings on everything. TODO: expose more settings to the user
    width = metiq.default_values["width"]
    height = metiq.default_values["height"]
    pre_samples = metiq.default_values['pre_samples']
    samplerate = metiq.default_values['samplerate']
    beep_freq = metiq.default_values['beep_freq']
    beep_period_sec = metiq.default_values['beep_period_sec']
    beep_duration_samples = metiq.default_values['beep_duration_samples']
    scale = metiq.default_values['scale']
    pixel_format = metiq.default_values['pixel_format']
    luma_threshold = metiq.default_values['luma_threshold']
    num_frames = -1
    kwargs = {"lock_layout": True, "threaded": True}

    min_match_threshold = metiq.default_values['min_match_threshold']
    min_separation_msec = metiq.default_values['min_separation_msec']
    audio_sample = metiq.default_values['audio_sample']
    vft_id = metiq.default_values['vft_id']
    force_fps = 30 #TODO: remove
    z_filter = 3
    windowed_stats_sec = metiq.default_values['windowed_stats_sec']
    analysis_type = "all"


    debug = 0
    for file in args.files:
        videocsv = file + ".video.csv"
        audiocsv = file + ".audio.csv"

        # files exist
        if not os.path.exists(audiocsv) or args.parse_audio:
            # 1. parse the audio stream
            media_parse.media_parse_audio(
                pre_samples,
                samplerate,
                beep_freq,
                beep_duration_samples,
                beep_period_sec,
                scale,
                file,
                audiocsv,
                debug,
                **kwargs,
            )

        if not os.path.exists(videocsv) or args.parse_video:
            # 2. parse the video stream
            media_parse.media_parse_video(
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
                file,
                videocsv,
                debug,
                **kwargs,
        )

        # Analyze the video and audio files
        for file in args.files:

            media_analyze.media_analyze(
                analysis_type,
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
                None,
                videocsv,
                audiocsv,
                None, #args.output,
                vft_id,
                False, #cache_video,
                False, #cache_audio,
                False, #cache_both,
                min_separation_msec,
                min_match_threshold,
                audio_sample,
                False, #lock_layout,
                False, #tag_manual,
                force_fps,
                True, #threaded,
                args.audio_offset,
                z_filter,
                windowed_stats_sec,
                False, #no_hw_decode,
                debug,
            )
    media_analyze.combined_calculations(args.files, args.output)
if __name__ == '__main__':
    main()

