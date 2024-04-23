#!/usr/bin/env python

"""media_analyze.py module description."""


import numpy as np
import pandas as pd
import os

import audio_parse
import media_parse
import video_parse
import time
import sys


def calculate_frames_moving_average(video_results, windowed_stats_sec):
    # frame, ts, video_result_frame_num_read_int

    video_results = video_results.dropna()
    if len(video_results) == 0:
        return pd.DataFrame()
    # only one testcase and one situation so no filter is needed.
    startframe = video_results.iloc[0]["value_read"]
    endframe = video_results.iloc[-1]["value_read"]

    frame = startframe
    window_sum = 0
    tmp = 0
    average = []
    while frame < endframe:
        current = video_results.loc[video_results["value_read"] == frame]
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
    video_results = video_results.replace([np.inf, -np.inf], np.nan)
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
    frame_errors = video_results.loc[video_results["status"] > 0]
    video_frames_capture_total = len(video_results)

    stats["video_frames_metiq_errors_percentage"] = round(
        100 * len(frame_errors) / video_frames_capture_total, 2
    )

    # video metiq errors
    for error, (short, _) in video_parse.ERROR_TYPES.items():
        stats["video_frames_metiq_error." + short] = len(
            video_results.loc[video_results["status"] == error]
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
    # for error, (short, _) in video_parse.ERROR_TYPES.items():
    #     stats["video_frames_metiq_error." + short] = len(
    #         video_metiq_errors.loc[video_metiq_errors["error_type"] == error]
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
    closematch.fillna(method="bfill", inplace=True)
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
    closematch.fillna(method="bfill", inplace=True)
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


def all_analysis_function(**kwargs):
    outfile = kwargs.get("outfile", None)
    if not outfile:
        infile = kwargs.get("input_video", None)
        # It could be MOV.video.csv or X.csv
        split_text = infile.split(".")
        if split_text[-3].lower() == "mov":
            outfile = ".".join(split_text[:-2])

    for function in MEDIA_ANALYSIS:
        if function == "all":
            # prevent a loop :)
            continue
        kwargs["outfile"] = f"{outfile}{MEDIA_ANALYSIS[function][2]}"
        results = MEDIA_ANALYSIS[function][0](**kwargs)


def audio_latency_function(**kwargs):
    audio_results = kwargs.get("audio_results")
    video_results = kwargs.get("video_results")
    ref_fps = kwargs.get("ref_fps")
    beep_period_sec = kwargs.get("beep_period_sec")
    debug = kwargs.get("debug")
    outfile = kwargs.get("outfile")

    audio_latency_results = calculate_audio_latency(
        audio_results,
        video_results,
        fps=ref_fps,
        beep_period_sec=beep_period_sec,
        debug=debug,
    )
    audio_latency_results.to_csv(outfile, index=False)


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

    # Assuming that the source frame is played out when the audio signal
    # is first heard, video latency is the difference between the video frame
    # of the soruce and video frame shown on rx

    # Frist filter all echoes and keep only source signal
    clean_audio = filter_echoes(audio_results, beep_period_sec, 0.7)

    signal_ratio = len(clean_audio) / len(audio_results)
    if signal_ratio == 0:
        print(audio_results)
        print("Warning. No source signals present")
        return

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
    video_latency_results.to_csv(outfile, index=False)


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

    margin = 0.7
    clean_audio = filter_echoes(audio_results, beep_period_sec, margin)
    # Check residue
    signal_ratio = len(clean_audio) / len(audio_results)
    if signal_ratio < 1:
        print(f"Removed {signal_ratio * 100:.2f}% echoes, transmission use case")
        if signal_ratio < 0.2:
            print("Few echoes, recheck thresholds")

        # Filter residues to get echoes
        residue = audio_results[~audio_results.index.isin(clean_audio.index)]
        clean_audio = filter_echoes(pd.DataFrame(residue), beep_period_sec, margin)

    else:
        print("No echoes, simple source use case")

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
    # calculate the delta of the value read between consecutive frames
    video_results["value_read_delta"] = [
        0,
    ] + list(
        y - x
        for (x, y) in zip(video_results.value_read[:-1], video_results.value_read[1:])
    )
    # remove consecutive frames with the same value read
    video_results = video_results.drop(
        video_results[video_results.value_read_delta == 0].index
    )
    # add column assuming the average delta
    average_delta = round(video_results.value_read_delta.mean())
    video_results["value_read_delta_minus_mean"] = (
        video_results.value_read_delta - average_delta
    )
    # remove unused columns
    # TODO: or keep wanted?
    unused = (
        "frame_num_expected",
        "status",
        "value_read",
        "delta_frame",
        "value_before_clean",
        "value_read_delta",
    )

    unused = [col for col in unused if col in video_results.columns.values]

    video_results = video_results.drop(
        columns=unused,
        axis=1,
    )
    return video_results


def quality_stats_function(**kwargs):
    audio_results = kwargs.get("audio_results")
    video_results = kwargs.get("video_results")
    outfile = kwargs.get("outfile")

    quality_stats_results = calculate_measurement_quality_stats(
        audio_results, video_results
    )
    quality_stats_results.to_csv(outfile, index=False)


def windowed_stats_function(**kwargs):
    video_results = kwargs.get("video_results")
    windowed_stats_sec = kwargs.get("windowed_stats_sec")
    outfile = kwargs.get("outfile")

    windowed_stats_results = calculate_frames_moving_average(
        video_results, windowed_stats_sec
    )
    windowed_stats_results.to_csv(outfile, index=False)


def frame_duration_function(**kwargs):
    video_results = kwargs.get("video_results")
    outfile = kwargs.get("outfile")

    frame_duration_results = calculate_frame_durations(video_results)
    frame_duration_results.to_csv(outfile, index=False)


def video_playout_fun(**kwargs):
    video_results = kwargs.get("video_results")
    outfile = kwargs.get("outfile")

    video_playout_results = calculate_video_playouts(video_results)
    video_playout_results.to_csv(outfile, index=False)


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
    debug,
):
    # read inputs
    video_results = None
    try:
        video_results = pd.read_csv(input_video)
    except ValueError:
        # ignore in case the analysis does not need it
        pass
    audio_results = None
    try:
        audio_results = pd.read_csv(input_audio)
    except ValueError:
        # ignore in case the analysis does not need it
        pass

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
        video_playout_fun,
        "Analyze video playout issues",
        ".video.playout.csv",
    ),
    "all": (all_analysis_function, "Calculate all media analysis", None),
}
