#!/usr/bin/env python

"""media_analyze.py module description."""


import numpy as np
import pandas as pd
import os

import audio_parse
import media_parse
import vft
import video_parse
import time
import sys


def calculate_frames_moving_average(video_results, windowed_stats_sec):
    # frame, ts, video_result_frame_num_read_int
    video_results = pd.DataFrame(video_results.dropna(subset=["value_read"]))

    if len(video_results) == 0:
        return pd.DataFrame()
    # only one testcase and one situation so no filter is needed.
    startframe = video_results.iloc[0]["value_read"]
    endframe = video_results.iloc[-1]["value_read"]

    frame = int(startframe)
    window_sum = 0
    tmp = 0
    average = []
    while frame < endframe:
        current = video_results.loc[video_results["value_read"].astype(int) == frame]

        if len(current) == 0:
            frame += 1
            continue
        nextframe = video_results.loc[
            video_results["timestamp"]
            >= (current.iloc[0]["timestamp"] + windowed_stats_sec)
        ]
        if len(nextframe) == 0:
            break

        nextframe_num = nextframe.iloc[0]["value_read"]

        windowed_data = video_results.loc[
            (video_results["value_read"] >= frame)
            & (video_results["value_read"] < nextframe_num)
        ]
        window_sum = len(np.unique(windowed_data["value_read"]))
        distance = nextframe_num - frame
        drops = distance - window_sum
        average.append(
            {
                "frame": frame,
                "frames": distance,
                "shown": window_sum,
                "drops": drops,
                "window": (
                    nextframe.iloc[0]["timestamp"] - current.iloc[0]["timestamp"]
                ),
            }
        )
        frame += 1

    return pd.DataFrame(average)


def calculate_frame_durations(video_results):
    # Calculate how many times a source frame is shown in capture frames/time
    video_results = pd.DataFrame(video_results.replace([np.inf, -np.inf], np.nan))
    video_results = video_results.dropna(subset=["value_read"])

    ref_fps, capture_fps = video_parse.estimate_fps(video_results)
    video_results["value_read_int"] = video_results["value_read"].astype(int)
    capt_group = video_results.groupby("value_read_int")
    cg = capt_group.count()["value_read"]
    cg = cg.value_counts().sort_index().to_frame()
    cg.index.rename("consecutive_frames", inplace=True)
    cg["frame_count"] = np.arange(1, len(cg) + 1)
    cg["time"] = cg["frame_count"] / capture_fps
    cg["capture_fps"] = capture_fps
    cg["ref_fps"] = ref_fps
    return cg


def calculate_measurement_quality_stats(audio_results, video_results):
    stats = {}

    # Count only unrecoverable errors
    readable = [
        (val in (vft.VFTReading.single_graycode.value, vft.VFTReading.ok.value))
        for val in video_results["status"]
    ].count(True)
    video_frames_capture_total = len(video_results)
    frame_errors = video_frames_capture_total - readable

    stats["video_frames_metiq_errors_percentage"] = round(
        100 * frame_errors / video_frames_capture_total, 2
    )

    # video metiq errors
    for vft_error in vft.VFTReading:
        stats["video_frames_metiq_error." + vft_error.name] = len(
            video_results.loc[video_results["status"] == vft_error.value]
        )

    # Audio signal
    audio_duration = audio_results["timestamp"].max() - audio_results["timestamp"].min()
    audio_sig_detected = len(audio_results)
    if audio_sig_detected == 0:
        audio_sig_detected = -1  # avoid division by zero
    stats["signal_distance_sec"] = audio_duration / audio_sig_detected
    stats["max_correlation"] = audio_results["correlation"].max()
    stats["min_correlation"] = audio_results["correlation"].min()
    stats["mean_correlation"] = audio_results["correlation"].mean()
    stats["index"] = 0

    return pd.DataFrame(stats, index=[0])


def calculate_stats(
    audio_latency_results,
    video_latency_results,
    av_syncs,
    video_results,
    audio_duration_samples,
    audio_duration_seconds,
    inputfile,
    debug=False,
):
    stats = {}
    ignore_latency = False
    if len(av_syncs) == 0 or len(video_results) == 0:
        print(f"Failure - no data")
        return None, None

    # 1. basic file statistics
    stats["file"] = inputfile
    video_frames_capture_duration = (
        video_results["timestamp"].max() - video_results["timestamp"].min()
    )
    stats["video_frames_capture_duration_sec"] = video_frames_capture_duration
    video_frames_capture_total = (
        video_results["frame_num"].max() - video_results["frame_num"].min()
    )
    stats["video_frames_capture_total"] = video_frames_capture_total
    stats["audio_frames_capture_duration_frames"] = audio_duration_seconds
    stats["audio_frames_capture_duration_samples"] = audio_duration_samples

    # 2. video latency statistics
    stats["video_latency_sec.num_samples"] = len(video_latency_results)
    stats["video_latency_sec.mean"] = (
        np.nan
        if len(video_latency_results) == 0
        else np.mean(video_latency_results["video_latency_sec"])
    )
    stats["video_latency_sec.std_dev"] = (
        np.nan
        if len(video_latency_results) == 0
        else np.std(video_latency_results["video_latency_sec"].values)
    )

    # 3. video latency statistics
    stats["audio_latency_sec.num_samples"] = len(audio_latency_results)
    stats["audio_latency_sec.mean"] = (
        np.nan
        if len(video_latency_results) == 0
        else np.mean(audio_latency_results["audio_latency_sec"])
    )
    stats["audio_latency_sec.std_dev"] = (
        np.nan
        if len(audio_latency_results) == 0
        else np.std(audio_latency_results["audio_latency_sec"].values)
    )

    # 4. av_sync statistics
    stats["av_sync_sec.num_samples"] = len(av_syncs)
    stats["av_sync_sec.mean"] = np.mean(av_syncs["av_sync_sec"])
    stats["av_sync_sec.std_dev"] = np.std(av_syncs["av_sync_sec"].values)

    # 5. video source (metiq) stats
    video_results["value_read_int"] = video_results["value_read"].dropna().astype(int)
    dump_frame_drops(video_results, inputfile)
    # 5.1. which source (metiq) frames have been show
    video_frames_sources_min = int(video_results["value_read_int"].min())
    video_frames_sources_max = int(video_results["value_read_int"].max())
    stats["video_frames_source_min"] = video_frames_sources_min
    stats["video_frames_source_max"] = video_frames_sources_max
    (
        video_frames_source_count,
        video_frames_source_unseen,
    ) = calculate_dropped_frames_stats(video_results)
    stats["video_frames_source_total"] = video_frames_source_count
    stats["video_frames_source_seen"] = (
        video_frames_source_count - video_frames_source_unseen
    )
    stats["video_frames_source_unseen"] = video_frames_source_unseen
    stats["video_frames_source_unseen_percentage"] = round(
        100 * video_frames_source_unseen / video_frames_source_count, 2
    )
    # 6. metiq processing statistics
    # TODO(chema): use video.csv information to calculate errors
    # stats["video_frames_metiq_errors"] = video_frames_metiq_errors
    # stats["video_frames_metiq_errors_percentage"] = round(
    #    100 * video_frames_metiq_errors / video_frames_capture_total, 2
    # )
    # video metiq errors
    # for vft_error in vft.VFTReading:
    #     stats["video_frames_metiq_error." + vft_error.name] = len(
    #         video_results.loc[video_results["status"] == vft_error.value]
    #     )
    # 7. calculate consecutive frame distribution
    capt_group = video_results.groupby("value_read_int")  # .count()
    cg = capt_group.count()["value_read"]
    cg = cg.value_counts().sort_index().to_frame()
    cg.index.rename("consecutive_frames", inplace=True)
    cg = cg.reset_index()
    # 7.2. times each source (metiq) frame been show
    stats["video_frames_source_appearances.mean"] = capt_group.size().mean()
    stats["video_frames_source_appearances.std_dev"] = capt_group.size().std()

    # TODO match gaps with source frame numbers?
    return pd.DataFrame(stats, columns=stats.keys(), index=[0]), cg


# Function searches for the video_results row whose timestamp
# is closer to ts
# It returns a tuple containing:
# (a) the frame_num of the selected row,
# (b) the searched (input) timestamp,
# (c) the value read in the selected frame,
# (d) the frame_num of the next frame where a beep is expected,
# (e) the latency assuming the initial frame_time.
# (f) if the value read is extrapolated (or perfect match)
def match_video_to_sources_beep(
    ts,
    video_results,
    beep_period_frames,
    frame_time,
    previous_matches,
    closest=False,
    match_distance_frames=-1,
    debug=1,
):
    video_results = video_results.copy()
    # The algorithm is as follows:
    # 1) find the frame in video that match the closest to ts
    # 2) Check the value parsed and compare to the expected beep frame time
    #    given the value just read.
    # 3) Find the frame matching the beep number
    # 3) If the match in (1) was not exact adjust for it.

    # Limit errors (audio offset and errors)
    if match_distance_frames < 0:
        # somewhat arbitrary +/- 1 frame i.e. 33ms at 30fps
        match_distance_frames = 4

    closematch = None
    # Just find the closes match to the timestamp
    video_results["distance"] = np.abs(video_results["timestamp"] - ts)
    # The purpose of this is just to find the beep source
    closematch = video_results.loc[video_results["distance"] < beep_period_frames]

    # remove non valid values
    closematch = closematch.loc[closematch["value_read"].notna()]
    if len(closematch) == 0:
        print(f"Warning. No match for {ts} within a beep period is found")
        return None

    # sort by time difference
    closematch = closematch.sort_values("distance")
    closematch.bfill(inplace=True)
    best_match = closematch.iloc[0]

    # 2) Check the value parsed and compare to the expected beep frame
    matched_value_read = best_match["value_read"]
    # estimate the frame for the next beep based on the frequency
    next_beep_frame = (
        int(matched_value_read / beep_period_frames) + 1
    ) * beep_period_frames
    if (
        next_beep_frame - matched_value_read > beep_period_frames / 2
        and (next_beep_frame - beep_period_frames) not in previous_matches
    ):
        next_beep_frame -= beep_period_frames

    if next_beep_frame in previous_matches:
        # This one has already been seen, this is latency beyond a beep
        if debug > 0:
            print("Warning. latency beyond beep period.")
        next_beep_frame += beep_period_frames

    # Find the beep
    if closest:
        video_results["distance_frames"] = np.abs(
            video_results["value_read"] - next_beep_frame
        )
        closematch = video_results.loc[
            video_results["distance_frames"] < match_distance_frames
        ]
    else:
        video_results["distance_frames"] = video_results["value_read"] - next_beep_frame
        video_results.sort_values("distance_frames", inplace=True)
        closematch = video_results.loc[
            (video_results["distance_frames"] >= 0)
            & (video_results["distance_frames"] < match_distance_frames)
        ]

    # remove non valid values
    closematch = closematch.loc[closematch["value_read"].notna()]
    if len(closematch) == 0:
        print(f"Warning. No match for {ts} is found")
        return None

    # sort by time difference
    closematch = closematch.sort_values("distance_frames")
    closematch.bfill(inplace=True)
    best_match = closematch.iloc[0]

    # get offset if not perfect match
    offset = best_match["value_read"] - next_beep_frame
    latency = best_match["timestamp"] - ts - offset * frame_time

    # Find the closest frame to the expected beep

    if not closest and latency < 0:
        if debug > 0:
            print("ERROR: negative latency")
    else:
        vlat = [
            best_match["frame_num"],
            ts,
            best_match["value_read"],
            next_beep_frame,
            latency,
        ]
        return vlat
    return None


def calculate_audio_latency(
    audio_results,
    video_results,
    beep_period_sec,
    fps=30,
    ignore_match_order=True,
    debug=False,
):
    # audio is {sample, ts, cor}
    # video is (frame, ts, expected, status, read, delta)
    # audio latency is the time between two correlated values where one should be higher

    prev = None
    beep_period_frames = int(beep_period_sec * fps)  # fps
    frame_time = 1 / fps
    # run audio_results looking for audio latency matches,
    # defined as 2x audio correlations that are close and
    # where the correlation value goes down
    audio_latency_results = pd.DataFrame(
        columns=[
            "audio_sample1",
            "timestamp1",
            "audio_sample2",
            "timestamp2",
            "audio_latency_sec",
            "cor1",
            "cor2",
        ],
    )

    for index in range(len(audio_results)):
        if prev is not None:
            match = audio_results.iloc[index]
            ts_diff = match["timestamp"] - prev["timestamp"]
            # correlation indicates that match is an echo (if ts_diff < period)
            if not ignore_match_order and prev["correlation"] < match["correlation"]:
                # This skip does not move previoua but the next iteration will
                # test agains same prev match
                continue
            # ensure the 2x correlations are close enough
            if ts_diff >= beep_period_sec * 0.5:
                # default 3 sec -> 1.5 sec, max detected audio delay
                prev = match
                continue
            audio_latency_results.loc[len(audio_latency_results.index)] = [
                prev["audio_sample"],
                prev["timestamp"],
                match["audio_sample"],
                match["timestamp"],
                ts_diff,
                prev["correlation"],
                match["correlation"],
            ]
        prev = audio_results.iloc[index]
    # Remove echoes.
    audio_latency_results["diff"] = audio_latency_results["timestamp1"].diff()
    too_close = len(
        audio_latency_results.loc[audio_latency_results["diff"] < beep_period_sec * 0.5]
    )
    if too_close > 0:
        print(f"WARNING. Potential echoes detected - {too_close} counts")
    audio_latency_results.fillna(beep_period_sec, inplace=True)
    audio_latency_results = audio_latency_results.loc[
        audio_latency_results["diff"] > beep_period_sec * 0.5
    ]
    audio_latency_results = audio_latency_results.drop(columns=["diff"])
    return audio_latency_results


def filter_echoes(audiodata, beep_period_sec, margin):
    """
    The DataFrame audiodata have a timestamp in seconds, margin is 0 to 1.

    Filter everything that is closer than margin * beep_period_sec
    This puts the limit on the combined length of echoes in order not
    to prevent identifying the first signal too.
    """

    audiodata["timestamp_diff"] = audiodata["timestamp"].diff()
    # keep first signal even if it could be an echo - we cannot tell.
    audiodata.fillna(beep_period_sec, inplace=True)
    return audiodata.loc[audiodata["timestamp_diff"] > beep_period_sec * margin]


def calculate_video_relation(
    audio_results,
    video_results,
    audio_anchor,
    closest_reference,
    beep_period_sec,
    fps=30,
    ignore_match_order=True,
    debug=False,
):
    # video is (frame, ts, expected, status, read, delta)
    # video latency is the time between the frame shown when a signal is played
    # and the time when it should be played out
    prev = None
    video_latency_results = []
    beep_period_frames = int(beep_period_sec * fps)  # fps
    frame_time = 1 / fps

    previous_matches = []
    video_latency_results = pd.DataFrame(
        columns=[
            "frame_num",
            "timestamp",
            "frame_num_read",
            "original_frame",
            "video_latency_sec",
        ],
    )

    for index in range(len(audio_results)):
        match = audio_results.iloc[index]
        # calculate video latency based on the
        # timestamp of the first (prev) audio match
        # vs. the timestamp of the video frame.
        vmatch = match_video_to_sources_beep(
            match[audio_anchor],
            video_results,
            beep_period_frames,
            frame_time,
            previous_matches,
            closest=closest_reference,
        )

        if vmatch is not None and (
            vmatch[4] >= 0 or closest_reference
        ):  # av_sync can be negative
            video_latency_results.loc[len(video_latency_results.index)] = vmatch
            previous_matches.append(vmatch[3])
        elif vmatch is None:
            print(f"ERROR: no match found for video latency calculation")
        else:
            print(
                f"ERROR: negative video latency - period length needs to be increased, {vmatch}"
            )

    return video_latency_results


def calculate_video_latency(
    audio_results,
    video_results,
    beep_period_sec,
    fps=30,
    ignore_match_order=True,
    debug=False,
):
    # video latency is the time between the frame shown when a signal is played
    # In the case of a transmission we look at the time from the first played out source
    # and when it is shown on the screen on the rx side.
    return calculate_video_relation(
        audio_results,
        video_results,
        "timestamp",
        False,
        beep_period_sec=beep_period_sec,
        fps=fps,
        ignore_match_order=ignore_match_order,
        debug=debug,
    )


def calculate_av_sync(
    audio_results,
    video_results,
    beep_period_sec,
    fps=30,
    ignore_match_order=True,
    debug=False,
):

    # av sync is the difference between when a signal is heard and when the frame is shown
    # If there is a second ssignal, use that one.
    timefield = "timestamp2"
    if timefield not in audio_results.columns:
        timefield = "timestamp"
    av_sync_results = calculate_video_relation(
        audio_results,
        video_results,
        timefield,
        True,
        beep_period_sec=beep_period_sec,
        fps=fps,
        ignore_match_order=ignore_match_order,
        debug=debug,
    )
    av_sync_results = av_sync_results.rename(
        columns={"video_latency_sec": "av_sync_sec"}
    )
    return av_sync_results


def z_filter_function(data, field, z_val):
    mean = data[field].mean()
    std = data[field].std()
    return data.drop(data[data[field] > mean + z_val * std].index)


def create_output_filename(input_filename, analysis_name):
    # We either have a XX.mov/mp4 or a XX.mov.video.csv
    name = input_filename
    if name[-10:].lower() == ".video.csv":
        name = name[:-10]
    name = f"{name}{MEDIA_ANALYSIS[analysis_name][2]}"
    return name


def all_analysis_function(**kwargs):
    outfile = kwargs.get("outfile", None)
    if not outfile:
        outfile = kwargs.get("input_video", None)

    for analysis_name in MEDIA_ANALYSIS:
        if analysis_name == "all":
            # prevent a loop :)
            continue
        kwargs["outfile"] = create_output_filename(outfile, analysis_name)
        results = MEDIA_ANALYSIS[analysis_name][0](**kwargs)


def audio_latency_function(**kwargs):
    audio_results = kwargs.get("audio_results")
    video_results = kwargs.get("video_results")
    ref_fps = kwargs.get("ref_fps")
    beep_period_sec = kwargs.get("beep_period_sec")
    debug = kwargs.get("debug")
    outfile = kwargs.get("outfile")

    if not outfile:
        infile = kwargs.get("input_video", None)
        outfile = create_output_filename(infile, "audio_latency")

    audio_latency_results = calculate_audio_latency(
        audio_results,
        video_results,
        fps=ref_fps,
        beep_period_sec=beep_period_sec,
        debug=debug,
    )
    if len(audio_latency_results) > 0:
        audio_latency_results.to_csv(outfile, index=False)
    else:
        if debug > 0:
            print("Warning. No audio latency results")


def remove_non_doubles(audio_results, clean_audio):
    # Find all echoes and match with the original signal
    # If any clean signal has a more than one match, remove the furthest
    # match from the audio_results.

    residue = pd.concat([audio_results, clean_audio]).drop_duplicates(keep=False)
    closest = []
    for index, match in clean_audio.iterrows():
        closest_match = -1
        try:
            closest_match = (
                residue.loc[residue.index > index]["timestamp"] - match["timestamp"]
            ).idxmin()
        except:
            # could be that there are no signals > index for the actual ts
            pass
        closest.append(closest_match)

    # Find matches with multiple references
    multis = {}
    drop = []
    for source_index, matching_index in enumerate(closest):
        if closest.count(matching_index) > 1:
            first = clean_audio.iloc[source_index]["timestamp"]
            second = None
            try:
                second = residue.loc[residue.index == matching_index][
                    "timestamp"
                ].values[0]
                diff = abs(first - second)
                if matching_index in multis:
                    match = multis[matching_index]
                    if diff < match[1]:
                        # remove the previous match
                        drop.append(clean_audio.index[match[0]])
                        multis[matching_index] = (source_index, diff)
                    else:
                        # remove this match
                        clean_audio.drop(
                            clean_audio.index[[source_index]], inplace=True
                        )
                        drop.append(clean_audio.index[source_index])
                else:
                    # First match for this row
                    multis[matching_index] = (source_index, diff, first)

            except Exception as ex:
                print(f"ERROR: not match for residue (remove non doubles)")
    return clean_audio.drop(drop)


def video_latency_function(**kwargs):
    audio_results = kwargs.get("audio_results")
    video_results = kwargs.get("video_results")
    ref_fps = kwargs.get("ref_fps")
    beep_period_sec = kwargs.get("beep_period_sec")
    debug = kwargs.get("debug")
    z_filter = kwargs.get("z_filter")
    outfile = kwargs.get("outfile")

    if len(audio_results) == 0:
        print("Warning. No audio signals present")
        return

    if not outfile:
        infile = kwargs.get("input_video", None)
        outfile = create_output_filename(infile, "video_latency")

    # Assuming that the source frame is played out when the audio signal
    # is first heard, video latency is the difference between the video frame
    # of the soruce and video frame shown on rx

    # First filter all echoes and keep only source signal
    clean_audio = filter_echoes(audio_results, beep_period_sec, 0.7)

    signal_ratio = len(clean_audio) / len(audio_results)
    if len(clean_audio) == 0:
        print("Warning. No source signals present")
        return
    elif signal_ratio < 1:
        print("Warning: unmatched echo/source signal. Removing unmatched.")
        clean_audio = remove_non_doubles(audio_results, clean_audio)

    # calculate the video latencies
    video_latency_results = calculate_video_latency(
        clean_audio,
        video_results,
        fps=ref_fps,
        beep_period_sec=beep_period_sec,
        debug=debug,
    )
    # filter the video latencies
    if z_filter > 0:
        video_latency_results = z_filter_function(
            video_latency_results, "video_latency_sec", z_filter
        )

    if len(video_latency_results) > 0:
        video_latency_results.to_csv(outfile, index=False)
    else:
        if debug > 0:
            print("Warning. No video latency results")


def av_sync_function(**kwargs):
    audio_results = kwargs.get("audio_results")
    video_results = kwargs.get("video_results")
    ref_fps = kwargs.get("ref_fps")
    beep_period_sec = kwargs.get("beep_period_sec")
    debug = kwargs.get("debug")
    z_filter = kwargs.get("z_filter")
    outfile = kwargs.get("outfile")

    # av sync is the time from the signal until the video is shown
    # for tests that include a transmission the signal of interest is
    # the first echo and not the source.

    if len(audio_results) == 0:
        print("No audio results, skipping av sync calculation")
        return

    if not outfile:
        infile = kwargs.get("input_video", None)
        outfile = create_output_filename(infile, "av_sync")

    margin = 0.7
    clean_audio = filter_echoes(audio_results, beep_period_sec, margin)
    # Check residue
    signal_ratio = len(clean_audio) / len(audio_results)
    if signal_ratio < 1:
        print(f"\nRemoved {signal_ratio * 100:.2f}% echoes, transmission use case. Video latency can be calculated.\n")
        if signal_ratio < 0.2:
            print("Few echoes, recheck thresholds")

        # Filter residues to get echoes
        residue = audio_results[~audio_results.index.isin(clean_audio.index)]
        clean_audio = filter_echoes(pd.DataFrame(residue), beep_period_sec, margin)

    else:
        print("\nWarning, no echoes, simple source use case. No video latency calculation possible.\n")

    av_sync_results = calculate_av_sync(
        clean_audio,
        video_results,
        fps=ref_fps,
        beep_period_sec=beep_period_sec,
        debug=debug,
    )
    # filter the a/v sync values
    if z_filter > 0:
        av_sync_results = z_filter_function(av_sync_results, "av_sync_sec", z_filter)
    if len(av_sync_results) > 0:
        av_sync_results.to_csv(outfile, index=False)

    # print statistics
    avsync_sec_average = np.average(av_sync_results["av_sync_sec"])
    avsync_sec_stddev = np.std(av_sync_results["av_sync_sec"])
    print(
        f"avsync_sec average: {avsync_sec_average} stddev: {avsync_sec_stddev} size: {len(av_sync_results)}"
    )


def calculate_video_playouts(video_results):
    # 1 add value_read_delta (difference between the value read between consecutive frames)
    video_results["value_read_delta"] = [
        0,
    ] + list(
        y - x
        for (x, y) in zip(video_results.value_read[:-1], video_results.value_read[1:])
    )
    # 2. remove consecutive frames with the same value read
    video_results = video_results.drop(
        video_results[video_results.value_read_delta == 0].index
    )
    # 3. add value_read_delta_minus_mean (column assuming the average delta)
    average_delta = round(video_results.value_read_delta.mean())
    video_results["value_read_delta_minus_mean"] = (
        video_results.value_read_delta - average_delta
    )
    # 4. remove unused columns
    # TODO: or keep wanted?
    unused_col_names = (
        "frame_num_expected",
        "delta_frame",
        "value_before_clean",
        "value_read_delta",
    )
    unused_col_names = [
        col for col in unused_col_names if col in video_results.columns.values
    ]
    video_results = video_results.drop(
        columns=unused_col_names,
        axis=1,
    )
    return video_results


def filter_halfsteps(video_results):
    # Halfsteps are the result of the video signal being read in between frames
    # We cannot know what it really should be. Let us do the following:
    # If the value is .5 from previous value, use previous value.
    # If the value is .5 from next value, use next value.
    # else use round up (time moves forward most of the time).
    video_results = pd.DataFrame(video_results)
    half_values = video_results.loc[video_results["value_read"].mod(1) == 0.5]
    if len(half_values) == 0:
        return video_results
    for index, row in half_values.iterrows():
        if index == 0 or index == len(video_results) - 1:
            continue
        if abs(row["value_read"] - video_results.at[index - 1, "value_read"]) == 0.5:
            video_results.at[index, "value_read"] = video_results.at[
                index - 1, "value_read"
            ]
        elif abs(row["value_read"] - video_results.at[index + 1, "value_read"]) == 0.5:
            video_results.at[index, "value_read"] = video_results.at[
                index + 1, "value_read"
            ]
        else:
            video_results.at[index, "value_read"] = math.floor(row["value_read"])
    return video_results


def filter_ambiguous_framenumber(video_results):
    video_results = filter_halfsteps(video_results)
    # one frame cannot have a different value than two adjacent frames.
    # this is only true if the capture fps is at least twice the draw frame rate (i.e. 240fps at 120Hz display).
    # Use next value

    # no holes please
    video_results["value_read"].ffill(inplace=True)
    # Maybe some values in the beginning are bad as well.
    video_results["value_read"].bfill(inplace=True)
    video_results["value_clean"] = video_results["value_read"].astype(int)
    video_results["val_m1"] = video_results["value_clean"].shift(-1)
    video_results["val_p1"] = video_results["value_clean"].shift(1)
    video_results["val_m1"].ffill(inplace=True)
    video_results["val_p1"].bfill(inplace=True)

    video_results["singles"] = (
        video_results["value_clean"] != video_results["val_m1"]
    ) & (video_results["value_clean"] != video_results["val_p1"])
    video_results.loc[video_results["singles"], "value_clean"] = np.NaN
    video_results["value_clean"].ffill(inplace=True)
    video_results["value_clean"] = video_results["value_clean"].astype(int)
    video_results["value_before_clean"] = video_results["value_read"]

    # use the new values in subsequent analysis
    video_results["value_read"] = video_results["value_clean"]
    video_results.drop(
        columns=["val_m1", "val_p1", "singles", "value_clean"], inplace=True
    )
    return video_results


def quality_stats_function(**kwargs):
    audio_results = kwargs.get("audio_results")
    video_results = kwargs.get("video_results")
    outfile = kwargs.get("outfile")

    if not outfile:
        infile = kwargs.get("input_video", None)
        outfile = create_output_filename(infile, "quality_stats")

    quality_stats_results = calculate_measurement_quality_stats(
        audio_results, video_results
    )
    quality_stats_results.to_csv(outfile, index=False)


def windowed_stats_function(**kwargs):
    video_results = kwargs.get("video_results")
    windowed_stats_sec = kwargs.get("windowed_stats_sec")
    outfile = kwargs.get("outfile")

    if not outfile:
        infile = kwargs.get("input_video", None)
        outfile = create_output_filename(infile, "windowed_stats")

    windowed_stats_results = calculate_frames_moving_average(
        video_results, windowed_stats_sec
    )
    if len(windowed_stats_results) > 0:
        windowed_stats_results.to_csv(outfile, index=False)
    else:
        print("No windowed stats to write")


def frame_duration_function(**kwargs):
    video_results = kwargs.get("video_results")
    outfile = kwargs.get("outfile")

    if not outfile:
        infile = kwargs.get("input_video", None)
        outfile = create_output_filename(infile, "frame_duration")

    frame_duration_results = calculate_frame_durations(video_results)
    if len(frame_duration_results) > 0:
        frame_duration_results.to_csv(outfile, index=False)
    else:
        print("No frame durations to write")


def video_playout_function(**kwargs):
    video_results = kwargs.get("video_results")
    outfile = kwargs.get("outfile")

    if not outfile:
        infile = kwargs.get("input_video", None)
        outfile = create_output_filename(infile, "video_playout")

    video_playout_results = calculate_video_playouts(video_results)
    if len(video_playout_results) > 0:
        video_playout_results.to_csv(outfile, index=False)
    else:
        print("No video playouts to write")


def media_analyze(
    analysis_type,
    pre_samples,
    samplerate,
    beep_freq,
    beep_duration_samples,
    beep_period_sec,
    input_video,
    input_audio,
    outfile,
    force_fps,
    audio_offset,
    filter_all_echoes,
    z_filter,
    windowed_stats_sec,
    cleanup_video=False,
    min_match_threshold=None,
    debug=0,
):
    # read inputs
    video_results = None
    try:
        video_results = pd.read_csv(input_video)
        # Remove obvious errors
        if cleanup_video:
            video_results = filter_ambiguous_framenumber(video_results)
    except ValueError:
        # ignore in case the analysis does not need it
        if debug > 0:
            print("No video data")
        pass
    audio_results = None
    try:
        audio_results = pd.read_csv(input_audio)
    except ValueError:
        # ignore in case the analysis does not need it
        pass

    # filter audio thresholds
    if audio_results is not None and min_match_threshold is not None:
        audio_results = audio_results.loc[
            audio_results["correlation"] >= min_match_threshold
        ]
    # estimate the video framerate
    # TODO: capture fps should be available
    ref_fps, capture_fps = video_parse.estimate_fps(video_results)
    if force_fps > 0:
        ref_fps = force_fps

    # adjust the audio offset
    if audio_offset is not None:
        video_results["timestamp"] += audio_offset

    if filter_all_echoes:
        audio_results = filter_echoes(audio_results, beep_period_sec, 0.7)

    assert analysis_type is not None, f"error: need to specify --analysis-type"
    analysis_function = MEDIA_ANALYSIS[analysis_type][0]
    analysis_function(
        audio_results=audio_results,
        video_results=video_results,
        fps=ref_fps,  # TODO(chema): only one
        ref_fps=ref_fps,
        beep_period_sec=beep_period_sec,
        debug=debug,
        outfile=outfile,
        z_filter=z_filter,
        windowed_stats_sec=windowed_stats_sec,
        input_video=input_video,
    )


MEDIA_ANALYSIS = {
    "audio_latency": (
        audio_latency_function,
        "Calculate audio latency",
        ".audio.latency.csv",
    ),
    "video_latency": (
        video_latency_function,
        "Calculate video latency",
        ".video.latency.csv",
    ),
    "av_sync": (
        av_sync_function,
        "Calculate audio/video synchronization offset using audio timestamps and video frame numbers",
        ".avsync.csv",
    ),
    "quality_stats": (
        quality_stats_function,
        "Calculate quality stats",
        ".measurement.quality.csv",
    ),
    "windowed_stats": (
        windowed_stats_function,
        "Calculate video frames shown/dropped per unit sec",
        ".windowed.stats.csv",
    ),
    "frame_duration": (
        frame_duration_function,
        "Calculate source frame durations",
        ".frame.duration.csv",
    ),
    "video_playout": (
        video_playout_function,
        "Analyze video playout issues",
        ".video.playout.csv",
    ),
    "all": (all_analysis_function, "Calculate all media analysis", None),
}
